"""
5-year regime-adaptive backtest with full charge accounting.

Run:  python backtest_runner.py
"""

import warnings
warnings.filterwarnings("ignore")

import sys
import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

import config
from models.feature_engine import FeatureEngine
from models.regime_hmm import RegimeHMM
from utils.helpers import setup_logger, est_charges

logger = setup_logger("backtest")

# ── Configuration ─────────────────────────────────────────────────────────────
BACKTEST_YEARS  = 5
CAPITAL         = config.CAPITAL
RISK_PCT        = config.RISK_PER_TRADE_PCT
MAX_POS         = config.MAX_POSITIONS
SIGNAL_THRESH   = config.SIGNAL_THRESHOLD
ATR_MULT_SL     = 1.5
ATR_MULT_TP     = 3.0          # 1:2 R:R minimum
MAX_DAILY_LOSS  = config.MAX_DAILY_LOSS
CHARGES_PER     = config.CHARGES_PER_TRADE
TARGET_GROSS    = config.TARGET_GROSS_DAILY

SYMBOLS = config.WATCHLIST


# ── Data loader ───────────────────────────────────────────────────────────────

def load_data(symbol: str, years: int = BACKTEST_YEARS) -> pd.DataFrame:
    """Load daily OHLCV. Try cache → yfinance."""
    cache_path = config.DATA_DIR / f"{symbol}_bt_{years}y.parquet"
    if cache_path.exists():
        df = pd.read_parquet(cache_path)
        logger.info(f"  {symbol}: loaded {len(df)} bars from cache")
        return df

    logger.info(f"  {symbol}: downloading {years}yr daily data...")
    try:
        raw = yf.download(
            f"{symbol}.NS",
            period=f"{years}y",
            interval="1d",
            progress=False,
            auto_adjust=True,
        )
        if raw.empty:
            raise ValueError("Empty response")
        raw.columns = [c.lower() for c in raw.columns]
        df = raw[["open", "high", "low", "close", "volume"]].dropna()
        df.to_parquet(cache_path)
        logger.info(f"  {symbol}: {len(df)} bars saved")
        return df
    except Exception as exc:
        logger.warning(f"  {symbol}: download failed ({exc})")
        return pd.DataFrame()


# ── Signal simulation ─────────────────────────────────────────────────────────

def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    prev_c = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_c).abs(),
        (df["low"]  - prev_c).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def compute_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Add all indicator columns used for entry scoring."""
    fe = FeatureEngine()
    df = fe.compute(df)
    df["atr14"] = _atr(df)
    return df


def _signal_score(row: pd.Series) -> float:
    """Simplified signal score from pre-computed features (0–1)."""
    score = 0.0

    # RSI
    rsi = row.get("rsi_14", 50)
    if 30 <= rsi < 50:   score += 0.18
    elif 50 <= rsi <= 65: score += 0.12
    elif rsi < 30:        score += 0.20

    # MACD histogram
    hist = row.get("macd_hist", 0)
    if hist > 0:  score += 0.18
    elif hist < 0: score -= 0.05

    # EMA cross
    score += 0.15 if row.get("ema_9_20_cross", -1) == 1 else 0

    # ADX
    adx = row.get("adx_14", 20)
    score += 0.15 if adx > 25 else (0.08 if adx > 20 else 0)

    # Volume spike
    vr = row.get("volume_ratio", 1)
    score += 0.12 if vr > 1.5 else (0.06 if vr > 1.0 else 0)

    # BB %B oversold
    bb = row.get("bb_pct_b", 0.5)
    score += 0.10 if bb < 0.25 else (0.05 if bb < 0.40 else 0)

    # Stoch oversold
    sk = row.get("stoch_k", 50)
    score += 0.08 if sk < 25 else 0

    return round(min(score, 1.0), 4)


# ── Backtest engine ───────────────────────────────────────────────────────────

class BacktestEngine:
    def __init__(self, use_hmm: bool = True) -> None:
        self.use_hmm = use_hmm
        self._hmm = RegimeHMM() if use_hmm else None
        self._fe  = FeatureEngine()

        # Results storage
        self.trades: list[dict] = []
        self.daily_records: list[dict] = []

    def run(self, all_data: dict[str, pd.DataFrame]) -> dict:
        """Run full backtest across all symbols and dates."""
        # Get common date range
        all_dates: set = set()
        for df in all_data.values():
            all_dates.update(df.index.tolist())
        sorted_dates = sorted(all_dates)

        logger.info(f"Backtesting {len(sorted_dates)} trading days across {len(all_data)} symbols")

        # Pre-compute signals + regimes
        signals: dict[str, pd.DataFrame] = {}
        for sym, df in all_data.items():
            if len(df) < 60:
                continue
            try:
                signals[sym] = compute_signals(df)
            except Exception as exc:
                logger.warning(f"Signal compute failed for {sym}: {exc}")

        # Regime series (use first symbol as market proxy)
        regime_series = self._compute_regime_series(all_data)

        # Daily simulation loop
        capital = CAPITAL
        open_positions: dict[str, dict] = {}  # symbol → position dict
        daily_pnl = 0.0
        daily_trades = 0
        daily_gross = 0.0
        prev_date: pd.Timestamp | None = None

        for dt in sorted_dates:
            # New day reset
            if prev_date is None or dt.date() != prev_date.date():
                if prev_date is not None:
                    self.daily_records.append({
                        "date":        prev_date.date(),
                        "gross_pnl":   daily_gross,
                        "charges":     daily_trades * CHARGES_PER,
                        "net_pnl":     daily_gross - daily_trades * CHARGES_PER,
                        "num_trades":  daily_trades,
                        "capital":     capital,
                        "regime":      regime_series.get(prev_date, "Sideways"),
                    })
                    capital = max(capital + daily_gross - daily_trades * CHARGES_PER, 0)
                daily_pnl = 0.0
                daily_gross = 0.0
                daily_trades = 0

            regime = regime_series.get(dt, "Sideways")
            threshold = 0.72 if regime == "Sideways" else SIGNAL_THRESH

            if regime == "Bear":
                prev_date = dt
                continue

            # Check and close open positions at today's prices
            to_close = []
            for sym, pos in open_positions.items():
                df_sym = signals.get(sym)
                if df_sym is None or dt not in df_sym.index:
                    continue
                row = df_sym.loc[dt]
                hi, lo = row["high"], row["low"]

                # Check SL hit
                if lo <= pos["sl"]:
                    pnl = (pos["sl"] - pos["entry"]) * pos["qty"]
                    self._record_trade(pos, pos["sl"], pnl, "SL", dt)
                    daily_gross += pnl
                    daily_trades += 1
                    daily_pnl += pnl - CHARGES_PER
                    to_close.append(sym)
                    continue

                # Check target hit
                if hi >= pos["target"]:
                    pnl = (pos["target"] - pos["entry"]) * pos["qty"]
                    self._record_trade(pos, pos["target"], pnl, "TARGET", dt)
                    daily_gross += pnl
                    daily_trades += 1
                    daily_pnl += pnl - CHARGES_PER
                    to_close.append(sym)

            for sym in to_close:
                del open_positions[sym]

            # Daily loss cap
            if daily_pnl <= -MAX_DAILY_LOSS:
                prev_date = dt
                continue

            # New entry scan
            if len(open_positions) < MAX_POS:
                for sym, df_sym in signals.items():
                    if sym in open_positions:
                        continue
                    if len(open_positions) >= MAX_POS:
                        break
                    if dt not in df_sym.index:
                        continue

                    row = df_sym.loc[dt]
                    score = _signal_score(row)
                    if score < threshold:
                        continue

                    atr = row.get("atr14", 0)
                    if atr <= 0:
                        continue

                    entry = row["close"]
                    sl    = entry - atr * ATR_MULT_SL
                    tgt   = entry + atr * ATR_MULT_SL * (ATR_MULT_TP / ATR_MULT_SL)

                    risk_amount = capital * RISK_PCT
                    stop_dist   = entry - sl
                    qty = int(risk_amount / stop_dist) if stop_dist > 0 else 0
                    max_qty = int(capital * 0.20 / entry)
                    qty = max(1, min(qty, max_qty))

                    open_positions[sym] = {
                        "symbol":  sym,
                        "entry":   entry,
                        "sl":      sl,
                        "target":  tgt,
                        "qty":     qty,
                        "date":    dt,
                        "score":   score,
                        "regime":  regime,
                    }

            prev_date = dt

        # Force-close any positions still open at end
        for sym, pos in open_positions.items():
            df_sym = signals.get(sym)
            if df_sym is not None and len(df_sym) > 0:
                last_price = float(df_sym["close"].iloc[-1])
                pnl = (last_price - pos["entry"]) * pos["qty"]
                self._record_trade(pos, last_price, pnl, "EXPIRY", sorted_dates[-1])

        return self._compile_results()

    def _record_trade(
        self, pos: dict, exit_price: float, gross_pnl: float,
        status: str, exit_dt: pd.Timestamp
    ) -> None:
        self.trades.append({
            "symbol":    pos["symbol"],
            "entry_date": pos["date"].date() if hasattr(pos["date"], "date") else pos["date"],
            "exit_date": exit_dt.date() if hasattr(exit_dt, "date") else exit_dt,
            "entry":     pos["entry"],
            "exit":      exit_price,
            "qty":       pos["qty"],
            "status":    status,
            "regime":    pos["regime"],
            "score":     pos["score"],
            "gross_pnl": round(gross_pnl, 2),
            "charges":   CHARGES_PER,
            "net_pnl":   round(gross_pnl - CHARGES_PER, 2),
        })

    def _compute_regime_series(
        self, all_data: dict[str, pd.DataFrame]
    ) -> dict[pd.Timestamp, str]:
        """Compute rolling regime label for each date using HMM."""
        if not self.use_hmm or self._hmm is None:
            return {}

        # Use NIFTY as market proxy
        try:
            raw = yf.download("^NSEI", period=f"{BACKTEST_YEARS}y", interval="1d", progress=False)
            raw.columns = [c.lower() for c in raw.columns]
            nifty = raw[["open", "high", "low", "close", "volume"]].dropna()
        except Exception:
            nifty = list(all_data.values())[0] if all_data else pd.DataFrame()

        if nifty.empty or not self._hmm.is_trained():
            # Train on available data
            if not nifty.empty and len(nifty) > 100:
                try:
                    self._hmm.train(nifty)
                except Exception as exc:
                    logger.warning(f"HMM train failed: {exc}")
                    return {}

        regime_map: dict[pd.Timestamp, str] = {}
        window = 60
        for i in range(window, len(nifty)):
            chunk = nifty.iloc[i - window: i + 1]
            dt = nifty.index[i]
            try:
                regime_map[dt] = self._hmm.predict(chunk)
            except Exception:
                regime_map[dt] = "Sideways"

        return regime_map

    def _compile_results(self) -> dict:
        if not self.trades:
            return {"error": "No trades generated"}

        df_t = pd.DataFrame(self.trades)
        df_d = pd.DataFrame(self.daily_records)

        total_trades  = len(df_t)
        wins          = (df_t["net_pnl"] > 0).sum()
        losses        = (df_t["net_pnl"] <= 0).sum()
        win_rate      = round(wins / total_trades * 100, 1)

        gross_total   = df_t["gross_pnl"].sum()
        charges_total = df_t["charges"].sum()
        net_total     = df_t["net_pnl"].sum()

        avg_win       = df_t.loc[df_t["net_pnl"] > 0, "net_pnl"].mean()
        avg_loss      = df_t.loc[df_t["net_pnl"] <= 0, "net_pnl"].mean()
        avg_rr        = abs(avg_win / avg_loss) if avg_loss != 0 else 0

        trading_days  = len(df_d) if len(df_d) > 0 else 1
        avg_gross_day = round(gross_total / trading_days, 2)
        avg_net_day   = round(net_total / trading_days, 2)
        avg_charges_d = round(charges_total / trading_days, 2)

        # Sharpe ratio
        if len(df_d) > 1:
            daily_returns = df_d["net_pnl"].values
            sharpe = np.sqrt(252) * daily_returns.mean() / (daily_returns.std() + 1e-9)
            sharpe = round(sharpe, 2)
        else:
            sharpe = 0.0

        # Max drawdown
        if len(df_d) > 0:
            equity = df_d["net_pnl"].cumsum()
            running_max = equity.cummax()
            drawdown = equity - running_max
            max_dd = round(float(drawdown.min()), 2)
        else:
            max_dd = 0.0

        # Target achievement rate
        target_days = (df_d["gross_pnl"] >= TARGET_GROSS).sum() if len(df_d) > 0 else 0
        target_rate = round(target_days / trading_days * 100, 1)

        # Monthly breakdown
        monthly = None
        if len(df_d) > 0:
            df_d["month"] = pd.to_datetime(df_d["date"]).dt.to_period("M")
            monthly = df_d.groupby("month").agg(
                trading_days=("date", "count"),
                gross_pnl=("gross_pnl", "sum"),
                charges=("charges", "sum"),
                net_pnl=("net_pnl", "sum"),
                num_trades=("num_trades", "sum"),
            ).round(2)

        # Regime breakdown
        regime_stats = df_t.groupby("regime")["net_pnl"].agg(["count", "mean", "sum"]).round(2)

        return {
            "total_trades":    total_trades,
            "wins":            int(wins),
            "losses":          int(losses),
            "win_rate":        win_rate,
            "avg_rr":          round(avg_rr, 2),
            "gross_total":     round(gross_total, 2),
            "charges_total":   round(charges_total, 2),
            "net_total":       round(net_total, 2),
            "avg_gross_day":   avg_gross_day,
            "avg_net_day":     avg_net_day,
            "avg_charges_day": avg_charges_d,
            "sharpe":          sharpe,
            "max_drawdown":    max_dd,
            "trading_days":    trading_days,
            "target_rate":     target_rate,
            "monthly":         monthly,
            "regime_stats":    regime_stats,
            "df_trades":       df_t,
            "df_daily":        df_d,
        }


# ── Reporter ──────────────────────────────────────────────────────────────────

def print_report(r: dict, label: str = "WITH HMM") -> None:
    bar = "═" * 52
    print(f"\n{bar}")
    print(f"  BACKTEST RESULTS — {label}")
    print(f"  {BACKTEST_YEARS}-year | {len(SYMBOLS)} symbols | ₹{CAPITAL:,.0f} capital")
    print(bar)
    print(f"  Total trades:        {r['total_trades']:>6}")
    print(f"  Win rate:            {r['win_rate']:>5.1f}%")
    print(f"  Avg R:R:             {r['avg_rr']:>5.2f}")
    print(f"  Trading days:        {r['trading_days']:>6}")
    print()
    print(f"  GROSS P&L total:    ₹{r['gross_total']:>10,.2f}")
    print(f"  Total charges:      ₹{r['charges_total']:>10,.2f}")
    print(f"  NET P&L total:      ₹{r['net_total']:>10,.2f}")
    print()
    print(f"  Avg GROSS/day:      ₹{r['avg_gross_day']:>8,.2f}")
    print(f"  Avg charges/day:    ₹{r['avg_charges_day']:>8,.2f}")
    print(f"  Avg NET/day:        ₹{r['avg_net_day']:>8,.2f}")
    print()
    print(f"  Sharpe ratio:        {r['sharpe']:>6.2f}")
    print(f"  Max drawdown:       ₹{r['max_drawdown']:>8,.2f}")
    print(f"  Target hit rate:     {r['target_rate']:>5.1f}%  (days ≥ ₹{TARGET_GROSS:,.0f} gross)")
    print(bar)

    if r.get("regime_stats") is not None:
        print("\n  By Regime:")
        print(r["regime_stats"].to_string())

    if r.get("monthly") is not None:
        print("\n  Monthly Breakdown (last 12 months):")
        monthly = r["monthly"].tail(12)
        for period, row in monthly.iterrows():
            print(
                f"  {period}: "
                f"Gross=₹{row['gross_pnl']:>8,.0f}  "
                f"Charges=₹{row['charges']:>6,.0f}  "
                f"Net=₹{row['net_pnl']:>8,.0f}  "
                f"Trades={int(row['num_trades'])}"
            )

    print()
    if r["avg_net_day"] >= TARGET_GROSS - 500:
        print(f"  ✅ System meets ₹{TARGET_GROSS-500:,.0f}/day net target")
    else:
        print(f"  ⚠️  Net/day ₹{r['avg_net_day']:,.0f} below ₹{TARGET_GROSS-500:,.0f} target")
        print("     → Tune signal weights / thresholds and re-run")
    print()


# ── Comparison: with HMM vs without ──────────────────────────────────────────

def run_comparison(all_data: dict[str, pd.DataFrame]) -> None:
    print("\n" + "=" * 52)
    print("  Running WITH-HMM backtest...")
    engine_hmm = BacktestEngine(use_hmm=True)
    r_hmm = engine_hmm.run(all_data)
    print_report(r_hmm, label="WITH HMM REGIME FILTER")

    print("  Running WITHOUT-HMM backtest...")
    engine_noh = BacktestEngine(use_hmm=False)
    r_noh = engine_noh.run(all_data)
    print_report(r_noh, label="WITHOUT HMM (all trades)")

    # Delta comparison
    print("  HMM IMPACT:")
    print(f"    Net P&L improvement:  ₹{r_hmm['net_total'] - r_noh['net_total']:+,.2f}")
    print(f"    Win rate change:      {r_hmm['win_rate'] - r_noh['win_rate']:+.1f}%")
    print(f"    Sharpe improvement:   {r_hmm['sharpe'] - r_noh['sharpe']:+.2f}")
    print(f"    Max DD reduction:     ₹{r_noh['max_drawdown'] - r_hmm['max_drawdown']:+,.2f}")
    print()

    # Save results
    out = {
        "run_date": datetime.now().isoformat(),
        "backtest_years": BACKTEST_YEARS,
        "symbols": SYMBOLS,
        "capital": CAPITAL,
        "with_hmm": {k: v for k, v in r_hmm.items()
                     if not isinstance(v, (pd.DataFrame, pd.Series))},
        "without_hmm": {k: v for k, v in r_noh.items()
                        if not isinstance(v, (pd.DataFrame, pd.Series))},
    }
    out_path = config.BASE_DIR / "backtest_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"  Results saved → {out_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    print("\n" + "=" * 52)
    print(f"  UPSTOX {BACKTEST_YEARS}-YEAR BACKTEST")
    print(f"  Symbols: {', '.join(SYMBOLS)}")
    print(f"  Capital: ₹{CAPITAL:,.0f}")
    print(f"  Charges: ₹{CHARGES_PER:.0f}/trade (round-trip)")
    print("=" * 52)

    print("\nStep 1: Loading historical data...")
    all_data: dict[str, pd.DataFrame] = {}
    for sym in SYMBOLS:
        df = load_data(sym, years=BACKTEST_YEARS)
        if not df.empty:
            all_data[sym] = df

    if not all_data:
        print("❌ No data loaded. Check internet connection.")
        sys.exit(1)

    print(f"\nLoaded {len(all_data)} symbols.")

    print("\nStep 2: Checking HMM model...")
    hmm = RegimeHMM()
    if not hmm.is_trained():
        print("  HMM not trained — training now on Nifty 5yr data...")
        hmm.train()
        print("  HMM trained.")
    else:
        print("  HMM model loaded.")

    print("\nStep 3: Running backtest comparison...")
    run_comparison(all_data)


if __name__ == "__main__":
    main()
