"""
5-year regime-adaptive backtest with full charge accounting.
Run: python backtest_runner.py

HOW TO RUN (avoids Claude Code stream timeout):
  # Option 1 - run in background, tail the log:
  nohup python backtest_runner.py > backtest.log 2>&1 &
  tail -f backtest.log

  # Option 2 - quick 1-year smoke test (finishes in ~60s):
  python backtest_runner.py --quick --no-hmm

ROOT CAUSE OF 'Stream idle timeout':
  Claude Code has a hard ~2-3 min API stream timeout.
  A 5-year backtest over 15 symbols takes 5-15 min.
  Solution: the backtest saves results to backtest_results.json
  and backtest.log continuously, so you can ask Claude Code
  to READ the results file after the run, not watch it live.
"""
import warnings
warnings.filterwarnings("ignore")

import sys
import json
import argparse
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

import config
from models.feature_engine import FeatureEngine
from models.regime_hmm import RegimeHMM
from utils.helpers import setup_logger, est_charges

# Write ALL output to both console AND a log file so results survive timeouts
LOG_FILE = config.BASE_DIR / "backtest.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, mode="w"),
    ],
)
log = logging.getLogger("backtest")

def p(msg: str) -> None:
    """Print + log + flush in one call."""
    log.info(msg)
    sys.stdout.flush()

logger = setup_logger("backtest")

# -- Configuration -----------------------------------------------------------
BACKTEST_YEARS = 5
CAPITAL        = config.CAPITAL
RISK_PCT       = config.RISK_PER_TRADE_PCT
MAX_POS        = config.MAX_POSITIONS
SIGNAL_THRESH  = config.SIGNAL_THRESHOLD
ATR_MULT_SL    = 1.5
ATR_MULT_TP    = 3.0
MAX_DAILY_LOSS = config.MAX_DAILY_LOSS
CHARGES_PER    = config.CHARGES_PER_TRADE
TARGET_GROSS   = config.TARGET_GROSS_DAILY
SYMBOLS        = config.WATCHLIST

# -- Parallel data loader ----------------------------------------------------
def _download_one(symbol: str, years: int) -> tuple:
    cache_path = config.DATA_DIR / f"{symbol}_bt_{years}y.parquet"
    if cache_path.exists():
        df = pd.read_parquet(cache_path)
        p(f"  cache {symbol}: {len(df)} bars")
        return symbol, df
    p(f"  downloading {symbol} {years}yr ...")
    try:
        raw = yf.download(
            f"{symbol}.NS", period=f"{years}y",
            interval="1d", progress=False, auto_adjust=True,
        )
        if raw.empty:
            raise ValueError("Empty response")
        # Fix for yfinance >= 0.2.40 MultiIndex columns
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = [col[0].lower() for col in raw.columns]
        else:
            raw.columns = [c.lower() if isinstance(c, str) else str(c[0]).lower() for c in raw.columns]
        df = raw[["open", "high", "low", "close", "volume"]].dropna()
        df.to_parquet(cache_path)
        p(f"  saved {symbol}: {len(df)} bars")
        return symbol, df
    except Exception as exc:
        p(f"  FAILED {symbol}: {exc}")
        return symbol, pd.DataFrame()


def load_all_data(symbols: list, years: int = BACKTEST_YEARS) -> dict:
    """Download all symbols in parallel (max 6 threads)."""
    all_data = {}
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = {pool.submit(_download_one, sym, years): sym for sym in symbols}
        for fut in as_completed(futures):
            sym, df = fut.result()
            if not df.empty:
                all_data[sym] = df
    return all_data


# -- Signal computation ------------------------------------------------------
def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    prev_c = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_c).abs(),
        (df["low"]  - prev_c).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def compute_signals(df: pd.DataFrame) -> pd.DataFrame:
    fe = FeatureEngine()
    df = fe.compute(df)
    df["atr14"] = _atr(df)
    return df


def _signal_score(row: pd.Series) -> float:
    score = 0.0
    rsi = row.get("rsi_14", 50)
    if   30 <= rsi < 50:  score += 0.18
    elif 50 <= rsi <= 65: score += 0.12
    elif rsi < 30:        score += 0.20
    hist = row.get("macd_hist", 0)
    if   hist > 0: score += 0.18
    elif hist < 0: score -= 0.05
    score += 0.15 if row.get("ema_9_20_cross", -1) == 1 else 0
    adx = row.get("adx_14", 20)
    score += 0.15 if adx > 25 else (0.08 if adx > 20 else 0)
    vr = row.get("volume_ratio", 1)
    score += 0.12 if vr > 1.5 else (0.06 if vr > 1.0 else 0)
    bb = row.get("bb_pct_b", 0.5)
    score += 0.10 if bb < 0.25 else (0.05 if bb < 0.40 else 0)
    sk = row.get("stoch_k", 50)
    score += 0.08 if sk < 25 else 0
    return round(min(score, 1.0), 4)


# -- Backtest engine ---------------------------------------------------------
class BacktestEngine:
    def __init__(self, use_hmm: bool = True) -> None:
        self.use_hmm = use_hmm
        self._hmm = RegimeHMM() if use_hmm else None
        self._fe  = FeatureEngine()
        self.trades        = []
        self.daily_records = []

    def run(self, all_data: dict) -> dict:
        all_dates = set()
        for df in all_data.values():
            all_dates.update(df.index.tolist())
        sorted_dates = sorted(all_dates)
        p(f"  {len(sorted_dates)} trading days x {len(all_data)} symbols")

        # Pre-compute signals
        p("  Computing signals...")
        signals = {}
        for i, (sym, df) in enumerate(all_data.items(), 1):
            if len(df) < 60:
                continue
            try:
                signals[sym] = compute_signals(df)
                if i % 5 == 0:
                    p(f"    signals {i}/{len(all_data)}")
            except Exception as exc:
                logger.warning(f"Signal compute failed for {sym}: {exc}")

        # Vectorised regime (weekly step)
        p("  Computing regime series...")
        regime_series = self._compute_regime_series_fast(all_data)

        # Daily simulation
        p("  Running simulation...")
        capital        = CAPITAL
        open_positions = {}
        daily_pnl      = 0.0
        daily_trades   = 0
        daily_gross    = 0.0
        prev_date      = None
        total_days     = len(sorted_dates)

        for idx, dt in enumerate(sorted_dates):
            if idx % 250 == 0:          # log every ~1 year of data
                p(f"    day {idx}/{total_days} ({dt.date()}) ...")

            if prev_date is None or dt.date() != prev_date.date():
                if prev_date is not None:
                    self.daily_records.append({
                        "date":       prev_date.date(),
                        "gross_pnl":  daily_gross,
                        "charges":    daily_trades * CHARGES_PER,
                        "net_pnl":    daily_gross - daily_trades * CHARGES_PER,
                        "num_trades": daily_trades,
                        "capital":    capital,
                        "regime":     regime_series.get(prev_date, "Sideways"),
                    })
                    capital = max(capital + daily_gross - daily_trades * CHARGES_PER, 0)
                daily_pnl    = 0.0
                daily_gross  = 0.0
                daily_trades = 0

            regime    = regime_series.get(dt, "Sideways")
            threshold = 0.72 if regime == "Sideways" else SIGNAL_THRESH

            if regime == "Bear":
                prev_date = dt
                continue

            to_close = []
            for sym, pos in open_positions.items():
                df_sym = signals.get(sym)
                if df_sym is None or dt not in df_sym.index:
                    continue
                row = df_sym.loc[dt]
                hi, lo = row["high"], row["low"]
                if lo <= pos["sl"]:
                    pnl = (pos["sl"] - pos["entry"]) * pos["qty"]
                    self._record_trade(pos, pos["sl"], pnl, "SL", dt)
                    daily_gross  += pnl; daily_trades += 1
                    daily_pnl    += pnl - CHARGES_PER
                    to_close.append(sym); continue
                if hi >= pos["target"]:
                    pnl = (pos["target"] - pos["entry"]) * pos["qty"]
                    self._record_trade(pos, pos["target"], pnl, "TARGET", dt)
                    daily_gross  += pnl; daily_trades += 1
                    daily_pnl    += pnl - CHARGES_PER
                    to_close.append(sym)
            for sym in to_close:
                del open_positions[sym]

            if daily_pnl <= -MAX_DAILY_LOSS:
                prev_date = dt; continue

            if len(open_positions) < MAX_POS:
                for sym, df_sym in signals.items():
                    if sym in open_positions or len(open_positions) >= MAX_POS:
                        continue
                    if dt not in df_sym.index: continue
                    row   = df_sym.loc[dt]
                    score = _signal_score(row)
                    if score < threshold: continue
                    atr = row.get("atr14", 0)
                    if atr <= 0: continue
                    entry       = row["close"]
                    sl          = entry - atr * ATR_MULT_SL
                    tgt         = entry + atr * ATR_MULT_SL * (ATR_MULT_TP / ATR_MULT_SL)
                    stop_dist   = entry - sl
                    qty         = int(capital * RISK_PCT / stop_dist) if stop_dist > 0 else 0
                    qty         = max(1, min(qty, int(capital * 0.20 / entry)))
                    open_positions[sym] = {
                        "symbol": sym, "entry": entry, "sl": sl,
                        "target": tgt, "qty": qty, "date": dt,
                        "score": score, "regime": regime,
                    }
            prev_date = dt

        for sym, pos in open_positions.items():
            df_sym = signals.get(sym)
            if df_sym is not None and len(df_sym) > 0:
                last_price = float(df_sym["close"].iloc[-1])
                pnl = (last_price - pos["entry"]) * pos["qty"]
                self._record_trade(pos, last_price, pnl, "EXPIRY", sorted_dates[-1])

        p("  Simulation complete.")
        return self._compile_results()

    def _compute_regime_series_fast(self, all_data: dict) -> dict:
        """Predict once per 5 days then forward-fill (5x fewer HMM calls)."""
        if not self.use_hmm or self._hmm is None:
            return {}
        try:
            raw = yf.download("^NSEI", period=f"{BACKTEST_YEARS}y",
                              interval="1d", progress=False)
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = [col[0].lower() for col in raw.columns]
            else:
                raw.columns = [c.lower() if isinstance(c, str) else str(c[0]).lower() for c in raw.columns]
            nifty = raw[["open", "high", "low", "close", "volume"]].dropna()
        except Exception:
            nifty = list(all_data.values())[0] if all_data else pd.DataFrame()
        if nifty.empty:
            return {}
        if not self._hmm.is_trained():
            try:
                p("  Training HMM...")
                self._hmm.train(nifty)
                p("  HMM trained.")
            except Exception as exc:
                logger.warning(f"HMM train failed: {exc}"); return {}
        regime_map  = {}
        window = 60; step = 5; last_regime = "Sideways"
        for i in range(window, len(nifty), step):
            chunk = nifty.iloc[i - window: i + 1]
            try: last_regime = self._hmm.predict(chunk)
            except Exception: pass
            for j in range(i, min(i + step, len(nifty))):
                regime_map[nifty.index[j]] = last_regime
        return regime_map

    def _record_trade(self, pos, exit_price, gross_pnl, status, exit_dt):
        self.trades.append({
            "symbol":     pos["symbol"],
            "entry_date": pos["date"].date() if hasattr(pos["date"], "date") else pos["date"],
            "exit_date":  exit_dt.date()     if hasattr(exit_dt, "date") else exit_dt,
            "entry": pos["entry"], "exit": exit_price, "qty": pos["qty"],
            "status": status, "regime": pos["regime"], "score": pos["score"],
            "gross_pnl": round(gross_pnl, 2), "charges": CHARGES_PER,
            "net_pnl": round(gross_pnl - CHARGES_PER, 2),
        })

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
        avg_win  = df_t.loc[df_t["net_pnl"] > 0,  "net_pnl"].mean()
        avg_loss = df_t.loc[df_t["net_pnl"] <= 0, "net_pnl"].mean()
        avg_rr   = abs(avg_win / avg_loss) if avg_loss != 0 else 0
        trading_days  = len(df_d) if len(df_d) > 0 else 1
        avg_gross_day = round(gross_total   / trading_days, 2)
        avg_net_day   = round(net_total     / trading_days, 2)
        avg_charges_d = round(charges_total / trading_days, 2)
        if len(df_d) > 1:
            dr = df_d["net_pnl"].values
            sharpe = round(np.sqrt(252) * dr.mean() / (dr.std() + 1e-9), 2)
        else:
            sharpe = 0.0
        if len(df_d) > 0:
            eq  = df_d["net_pnl"].cumsum()
            max_dd = round(float((eq - eq.cummax()).min()), 2)
        else:
            max_dd = 0.0
        target_days = (df_d["gross_pnl"] >= TARGET_GROSS).sum() if len(df_d) > 0 else 0
        target_rate = round(target_days / trading_days * 100, 1)
        monthly = None
        if len(df_d) > 0:
            df_d["month"] = pd.to_datetime(df_d["date"]).dt.to_period("M")
            monthly = df_d.groupby("month").agg(
                trading_days=("date","count"), gross_pnl=("gross_pnl","sum"),
                charges=("charges","sum"), net_pnl=("net_pnl","sum"),
                num_trades=("num_trades","sum"),
            ).round(2)
        regime_stats = df_t.groupby("regime")["net_pnl"].agg(["count","mean","sum"]).round(2)
        return {
            "total_trades": total_trades, "wins": int(wins), "losses": int(losses),
            "win_rate": win_rate, "avg_rr": round(avg_rr, 2),
            "gross_total": round(gross_total, 2), "charges_total": round(charges_total, 2),
            "net_total": round(net_total, 2), "avg_gross_day": avg_gross_day,
            "avg_net_day": avg_net_day, "avg_charges_day": avg_charges_d,
            "sharpe": sharpe, "max_drawdown": max_dd,
            "trading_days": trading_days, "target_rate": target_rate,
            "monthly": monthly, "regime_stats": regime_stats,
            "df_trades": df_t, "df_daily": df_d,
        }


# -- Reporter ----------------------------------------------------------------
def print_report(r: dict, label: str = "WITH HMM") -> None:
    bar = "=" * 52
    p(f"\n{bar}")
    p(f" BACKTEST RESULTS -- {label}")
    p(f" {BACKTEST_YEARS}-year | {len(SYMBOLS)} symbols | Rs{CAPITAL:,.0f} capital")
    p(bar)
    if "error" in r:
        p(f" ERROR: {r['error']}"); return
    p(f" Total trades:     {r['total_trades']:>6}")
    p(f" Win rate:         {r['win_rate']:>5.1f}%")
    p(f" Avg R:R:          {r['avg_rr']:>5.2f}")
    p(f" Trading days:     {r['trading_days']:>6}")
    p("")
    p(f" GROSS P&L total:  Rs{r['gross_total']:>10,.2f}")
    p(f" Total charges:    Rs{r['charges_total']:>10,.2f}")
    p(f" NET P&L total:    Rs{r['net_total']:>10,.2f}")
    p("")
    p(f" Avg GROSS/day:    Rs{r['avg_gross_day']:>8,.2f}")
    p(f" Avg charges/day:  Rs{r['avg_charges_day']:>8,.2f}")
    p(f" Avg NET/day:      Rs{r['avg_net_day']:>8,.2f}")
    p("")
    p(f" Sharpe ratio:     {r['sharpe']:>6.2f}")
    p(f" Max drawdown:     Rs{r['max_drawdown']:>8,.2f}")
    p(f" Target hit rate:  {r['target_rate']:>5.1f}% (days >= Rs{TARGET_GROSS:,.0f} gross)")
    p(bar)
    if r.get("regime_stats") is not None:
        p("\n By Regime:")
        p(r["regime_stats"].to_string())
    if r.get("monthly") is not None:
        p("\n Monthly Breakdown (last 12 months):")
        for period, row in r["monthly"].tail(12).iterrows():
            p(f"  {period}: Gross=Rs{row['gross_pnl']:>8,.0f}  "
              f"Charges=Rs{row['charges']:>6,.0f}  "
              f"Net=Rs{row['net_pnl']:>8,.0f}  Trades={int(row['num_trades'])}")
    p("")
    if r["avg_net_day"] >= TARGET_GROSS - 500:
        p(f" System meets Rs{TARGET_GROSS-500:,.0f}/day net target")
    else:
        p(f" Net/day Rs{r['avg_net_day']:,.0f} below Rs{TARGET_GROSS-500:,.0f} target")
    p("")


# -- Comparison --------------------------------------------------------------
def run_comparison(all_data: dict) -> None:
    p("\n" + "=" * 52)
    p(" Running WITH-HMM backtest...")
    engine_hmm = BacktestEngine(use_hmm=True)
    r_hmm = engine_hmm.run(all_data)
    print_report(r_hmm, label="WITH HMM REGIME FILTER")

    # Save partial results immediately after first run
    _save_results(r_hmm, {}, partial=True)

    p(" Running WITHOUT-HMM backtest...")
    engine_noh = BacktestEngine(use_hmm=False)
    r_noh = engine_noh.run(all_data)
    print_report(r_noh, label="WITHOUT HMM (all trades)")

    p(" HMM IMPACT:")
    for key, lbl in [("net_total","Net P&L"),("win_rate","Win rate"),("sharpe","Sharpe")]:
        delta = r_hmm.get(key, 0) - r_noh.get(key, 0)
        p(f"  {lbl} change: {delta:+.2f}")
    p("")
    _save_results(r_hmm, r_noh)


def _save_results(r_hmm: dict, r_noh: dict, partial: bool = False) -> None:
    out = {
        "run_date":       datetime.now().isoformat(),
        "backtest_years": BACKTEST_YEARS,
        "symbols":        SYMBOLS,
        "capital":        CAPITAL,
        "partial":        partial,
        "with_hmm":    {k: v for k, v in r_hmm.items()
                        if not isinstance(v, (pd.DataFrame, pd.Series))},
        "without_hmm": {k: v for k, v in r_noh.items()
                        if not isinstance(v, (pd.DataFrame, pd.Series))},
    }
    out_path = config.BASE_DIR / "backtest_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    p(f" Results saved -> {out_path}")


# -- Entry point -------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick",  action="store_true",
                        help="Use 1-year data (finishes in ~60s)")
    parser.add_argument("--no-hmm", action="store_true",
                        help="Skip HMM regime filter")
    args = parser.parse_args()
    years = 1 if args.quick else BACKTEST_YEARS

    p("=" * 52)
    p(f" UPSTOX {years}-YEAR BACKTEST")
    p(f" Symbols : {', '.join(SYMBOLS)}")
    p(f" Capital : Rs{CAPITAL:,.0f}")
    p(f" Charges : Rs{CHARGES_PER:.0f}/trade")
    p(f" Log file: {LOG_FILE}")
    p("=" * 52)
    p("")
    p("NOTE: If running inside Claude Code and it times out,")
    p("the results are still being written to backtest_results.json")
    p("and backtest.log. Run: tail -f backtest.log to monitor.")
    p("")

    p("Step 1: Loading historical data (parallel)...")
    all_data = load_all_data(SYMBOLS, years=years)
    if not all_data:
        p("No data loaded. Check internet connection.")
        sys.exit(1)
    p(f"Loaded {len(all_data)} symbols.")

    if args.no_hmm:
        p("Step 2: Skipping HMM (--no-hmm).")
        engine = BacktestEngine(use_hmm=False)
        r = engine.run(all_data)
        print_report(r, label="NO HMM")
        _save_results(r, {})
    else:
        p("Step 2: Running comparison (with / without HMM)...")
        run_comparison(all_data)

    p(f"DONE. See backtest_results.json and {LOG_FILE}")


if __name__ == "__main__":
    main()
