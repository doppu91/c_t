"""
Intraday 15-minute regime-adaptive backtest with full charge accounting.
Timeframe : 15-minute candles (yfinance interval='15m')
Data range : Up to 1 year via rolling 50-day chunks (yfinance 15m limit = 60 days)
Run options:
    python backtest_runner.py --no-hmm   # skip HMM, faster
    python backtest_runner.py            # with HMM regime filter

HOW TO RUN (avoids Claude Code stream timeout):
    nohup python backtest_runner.py --no-hmm > backtest.log 2>&1 &
    tail -f backtest.log
"""
import warnings
warnings.filterwarnings("ignore")
import sys
import json
import argparse
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, time as dtime
from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf
import config
from models.feature_engine import FeatureEngine
from models.regime_hmm import RegimeHMM
from utils.helpers import setup_logger, est_charges

# ── Logging ──────────────────────────────────────────────────────────────────
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
    log.info(msg)
    sys.stdout.flush()

logger = setup_logger("backtest")

# ── Config ───────────────────────────────────────────────────────────────────
INTERVAL        = "15m"           # intraday candle size
BACKTEST_DAYS   = 365             # ~1 year (yfinance 15m: max ~60 days per call)
CHUNK_DAYS      = 50              # rolling window per download chunk
CAPITAL         = config.CAPITAL
RISK_PCT        = config.RISK_PER_TRADE_PCT
MAX_POS         = config.MAX_POSITIONS
SIGNAL_THRESH   = 0.38            # relaxed for intraday (was 0.55 daily)
ATR_MULT_SL     = 1.0             # tighter SL for intraday
ATR_MULT_TP     = 2.0             # 2:1 R:R
MAX_DAILY_LOSS  = config.MAX_DAILY_LOSS
CHARGES_PER     = config.CHARGES_PER_TRADE
TARGET_GROSS    = config.TARGET_GROSS_DAILY
SYMBOLS         = config.WATCHLIST

# NSE session times (UTC+5:30 → we compare hour/minute in tz-naive index)
SESSION_START   = dtime(9, 15)
SESSION_END     = dtime(15, 15)   # force-close at 15:15 (last entry no later than 14:45)
ENTRY_CUTOFF    = dtime(14, 45)   # no new entries after this

# ── Data loader ──────────────────────────────────────────────────────────────
def _fix_cols(raw: pd.DataFrame) -> pd.DataFrame:
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [col[0].lower() for col in raw.columns]
    else:
        raw.columns = [c.lower() if isinstance(c, str) else str(c[0]).lower() for c in raw.columns]
    return raw

def _download_chunk(symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
    try:
        raw = yf.download(
            f"{symbol}.NS",
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            interval=INTERVAL,
            progress=False,
            auto_adjust=True,
        )
        if raw.empty:
            return pd.DataFrame()
        raw = _fix_cols(raw)
        return raw[["open", "high", "low", "close", "volume"]].dropna()
    except Exception as exc:
        p(f"    chunk {symbol} {start.date()}..{end.date()} FAILED: {exc}")
        return pd.DataFrame()

def _download_one(symbol: str, days: int) -> tuple:
    cache_path = config.DATA_DIR / f"{symbol}_15m_{days}d.parquet"
    if cache_path.exists():
        df = pd.read_parquet(cache_path)
        p(f"  cache {symbol}: {len(df)} bars")
        return symbol, df

    p(f"  downloading {symbol} {days}d @ 15m (rolling chunks)...")
    end = datetime.now()
    start = end - timedelta(days=days)
    chunks = []
    cur = start
    while cur < end:
        chunk_end = min(cur + timedelta(days=CHUNK_DAYS), end)
        df_chunk = _download_chunk(symbol, cur, chunk_end)
        if not df_chunk.empty:
            chunks.append(df_chunk)
        cur = chunk_end + timedelta(days=1)

    if not chunks:
        p(f"  FAILED {symbol}: no data in any chunk")
        return symbol, pd.DataFrame()

    df = pd.concat(chunks)
    df = df[~df.index.duplicated(keep="first")].sort_index()
    df.to_parquet(cache_path)
    p(f"  saved {symbol}: {len(df)} bars")
    return symbol, df

def load_all_data(symbols: list, days: int = BACKTEST_DAYS) -> dict:
    all_data = {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(_download_one, sym, days): sym for sym in symbols}
        for fut in as_completed(futures):
            sym, df = fut.result()
            if not df.empty:
                all_data[sym] = df
    return all_data

# ── Signals ───────────────────────────────────────────────────────────────────
def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    prev_c = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_c).abs(),
        (df["low"] - prev_c).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def compute_signals(df: pd.DataFrame) -> pd.DataFrame:
    fe = FeatureEngine()
    df = fe.compute(df)
    df["atr14"] = _atr(df, period=14)
    return df

def _signal_score(row: pd.Series) -> float:
    score = 0.0
    rsi = row.get("rsi_14", 50)
    if 30 <= rsi < 50:
        score += 0.20
    elif 50 <= rsi <= 65:
        score += 0.14
    elif rsi < 30:
        score += 0.22
    hist = row.get("macd_hist", 0)
    if hist > 0:
        score += 0.20
    elif hist < 0:
        score -= 0.05
    score += 0.15 if row.get("ema_9_20_cross", -1) == 1 else 0
    adx = row.get("adx_14", 20)
    score += 0.15 if adx > 25 else (0.08 if adx > 20 else 0)
    vr = row.get("volume_ratio", 1)
    score += 0.12 if vr > 1.5 else (0.06 if vr > 1.0 else 0)
    bb = row.get("bb_pct_b", 0.5)
    score += 0.10 if bb < 0.30 else (0.05 if bb < 0.45 else 0)
    sk = row.get("stoch_k", 50)
    score += 0.08 if sk < 30 else 0
    return round(min(score, 1.0), 4)

# ── Regime (daily NIFTY) ─────────────────────────────────────────────────────
def _build_regime_map(days: int, use_hmm: bool) -> dict:
    if not use_hmm:
        return {}
    try:
        raw = yf.download("^NSEI", period=f"{max(days//30,1)+1}mo",
                          interval="1d", progress=False, auto_adjust=True)
        raw = _fix_cols(raw)
        nifty = raw[["open", "high", "low", "close", "volume"]].dropna()
    except Exception:
        return {}
    if nifty.empty:
        return {}
    hmm = RegimeHMM()
    try:
        p("  Training HMM on NIFTY daily bars...")
        hmm.train(nifty)
    except Exception as exc:
        p(f"  HMM train failed: {exc}")
        return {}
    regime_map = {}
    window = 60; step = 5; last = "Sideways"
    for i in range(window, len(nifty), step):
        chunk = nifty.iloc[i - window: i + 1]
        try:
            last = hmm.predict(chunk)
        except Exception:
            pass
        for j in range(i, min(i + step, len(nifty))):
            regime_map[nifty.index[j].date()] = last
    return regime_map

# ── Backtest engine ───────────────────────────────────────────────────────────
class BacktestEngine:
    def __init__(self, use_hmm: bool = True) -> None:
        self.use_hmm = use_hmm
        self.trades = []
        self.daily_records = []

    def run(self, all_data: dict, regime_map: dict) -> dict:
        # Pre-compute signals for all symbols
        p("  Computing 15m signals...")
        signals = {}
        for i, (sym, df) in enumerate(all_data.items(), 1):
            if len(df) < 30:
                continue
            try:
                signals[sym] = compute_signals(df.copy())
                if i % 5 == 0:
                    p(f"  signals {i}/{len(all_data)}")
            except Exception as exc:
                p(f"  signal FAIL {sym}: {exc}")

        # Get all unique trading dates
        all_dates = sorted({ts.date() for df in signals.values() for ts in df.index})
        p(f"  {len(all_dates)} trading days x {len(signals)} symbols")
        p("  Running intraday simulation...")

        capital = CAPITAL
        total_days = len(all_dates)

        for day_idx, day in enumerate(all_dates):
            if day_idx % 50 == 0:
                p(f"  day {day_idx}/{total_days} ({day}) capital=Rs{capital:,.0f}")

            regime = regime_map.get(day, "Sideways")
            # Skip bear days
            if regime == "Bear":
                continue

            threshold = 0.45 if regime == "Sideways" else SIGNAL_THRESH

            # Collect all 15m bars for this day across all symbols, sorted by time
            day_bars = {}
            for sym, df in signals.items():
                mask = pd.to_datetime(df.index).date == day
                bars = df[mask].copy()
                # Filter session hours
                bars = bars[bars.index.map(lambda t: SESSION_START <= t.time() <= SESSION_END)]
                if not bars.empty:
                    day_bars[sym] = bars

            if not day_bars:
                continue

            # Build sorted timeline of all bar timestamps this day
            all_ts = sorted({ts for bars in day_bars.values() for ts in bars.index})

            open_positions = {}
            daily_gross = 0.0
            daily_trades = 0
            daily_pnl_net = 0.0

            for ts in all_ts:
                bar_time = ts.time()

                # Force-close all positions at session end
                if bar_time >= SESSION_END:
                    for sym, pos in list(open_positions.items()):
                        bars = day_bars.get(sym)
                        if bars is not None and ts in bars.index:
                            exit_price = float(bars.loc[ts, "close"])
                        else:
                            exit_price = pos["entry"]
                        gross = (exit_price - pos["entry"]) * pos["qty"]
                        daily_gross += gross
                        daily_trades += 1
                        daily_pnl_net += gross - CHARGES_PER
                        self._record_trade(pos, exit_price, gross, "EOD", ts)
                    open_positions.clear()
                    break

                # Check SL/TP for open positions
                to_close = []
                for sym, pos in open_positions.items():
                    bars = day_bars.get(sym)
                    if bars is None or ts not in bars.index:
                        continue
                    row = bars.loc[ts]
                    hi, lo = float(row["high"]), float(row["low"])
                    if lo <= pos["sl"]:
                        gross = (pos["sl"] - pos["entry"]) * pos["qty"]
                        daily_gross += gross; daily_trades += 1
                        daily_pnl_net += gross - CHARGES_PER
                        self._record_trade(pos, pos["sl"], gross, "SL", ts)
                        to_close.append(sym)
                        continue
                    if hi >= pos["target"]:
                        gross = (pos["target"] - pos["entry"]) * pos["qty"]
                        daily_gross += gross; daily_trades += 1
                        daily_pnl_net += gross - CHARGES_PER
                        self._record_trade(pos, pos["target"], gross, "TARGET", ts)
                        to_close.append(sym)
                for sym in to_close:
                    del open_positions[sym]

                # Daily loss limit
                if daily_pnl_net <= -MAX_DAILY_LOSS:
                    break

                # Entry logic (only before cutoff)
                if bar_time >= ENTRY_CUTOFF:
                    continue
                if len(open_positions) >= MAX_POS:
                    continue

                for sym, bars in day_bars.items():
                    if sym in open_positions or len(open_positions) >= MAX_POS:
                        continue
                    if ts not in bars.index:
                        continue
                    row = bars.loc[ts]
                    score = _signal_score(row)
                    if score < threshold:
                        continue
                    atr = float(row.get("atr14", 0))
                    if atr <= 0:
                        continue
                    entry = float(row["close"])
                    sl = entry - atr * ATR_MULT_SL
                    tgt = entry + atr * ATR_MULT_TP
                    stop_dist = entry - sl
                    if stop_dist <= 0:
                        continue
                    qty = int(capital * RISK_PCT / stop_dist)
                    qty = max(1, min(qty, int(capital * 0.15 / entry)))
                    open_positions[sym] = {
                        "symbol": sym, "entry": entry, "sl": sl, "target": tgt,
                        "qty": qty, "date": ts, "score": score, "regime": regime,
                    }

            # Record daily summary
            self.daily_records.append({
                "date": day,
                "gross_pnl": round(daily_gross, 2),
                "charges": daily_trades * CHARGES_PER,
                "net_pnl": round(daily_gross - daily_trades * CHARGES_PER, 2),
                "num_trades": daily_trades,
                "capital": capital,
                "regime": regime,
            })
            capital = max(capital + daily_gross - daily_trades * CHARGES_PER, 0)

        p("  Simulation complete.")
        return self._compile_results()

    def _record_trade(self, pos, exit_price, gross_pnl, status, exit_dt):
        self.trades.append({
            "symbol": pos["symbol"],
            "entry_date": pos["date"].date() if hasattr(pos["date"], "date") else pos["date"],
            "exit_date": exit_dt.date() if hasattr(exit_dt, "date") else exit_dt,
            "entry": pos["entry"],
            "exit": exit_price,
            "qty": pos["qty"],
            "status": status,
            "regime": pos["regime"],
            "score": pos["score"],
            "gross_pnl": round(gross_pnl, 2),
            "charges": CHARGES_PER,
            "net_pnl": round(gross_pnl - CHARGES_PER, 2),
        })

    def _compile_results(self) -> dict:
        if not self.trades:
            return {"error": "No trades generated"}
        df_t = pd.DataFrame(self.trades)
        df_d = pd.DataFrame(self.daily_records)
        total_trades = len(df_t)
        wins   = int((df_t["net_pnl"] > 0).sum())
        losses = int((df_t["net_pnl"] <= 0).sum())
        win_rate = round(wins / total_trades * 100, 1)
        gross_total  = df_t["gross_pnl"].sum()
        charges_total = df_t["charges"].sum()
        net_total    = df_t["net_pnl"].sum()
        avg_win  = df_t.loc[df_t["net_pnl"] > 0, "net_pnl"].mean() if wins > 0 else 0
        avg_loss = df_t.loc[df_t["net_pnl"] <= 0, "net_pnl"].mean() if losses > 0 else -1
        avg_rr   = abs(avg_win / avg_loss) if avg_loss != 0 else 0
        trading_days   = max(len(df_d), 1)
        avg_gross_day  = round(gross_total  / trading_days, 2)
        avg_net_day    = round(net_total    / trading_days, 2)
        avg_charges_d  = round(charges_total/ trading_days, 2)
        if len(df_d) > 1:
            dr = df_d["net_pnl"].values
            sharpe = round(np.sqrt(252) * dr.mean() / (dr.std() + 1e-9), 2)
        else:
            sharpe = 0.0
        if len(df_d) > 0:
            eq = df_d["net_pnl"].cumsum()
            max_dd = round(float((eq - eq.cummax()).min()), 2)
        else:
            max_dd = 0.0
        target_days = int((df_d["gross_pnl"] >= TARGET_GROSS).sum()) if len(df_d) > 0 else 0
        target_rate = round(target_days / trading_days * 100, 1)
        if len(df_d) > 0:
            df_d["month"] = pd.to_datetime(df_d["date"]).dt.to_period("M")
            monthly = df_d.groupby("month").agg(
                trading_days=("date","count"),
                gross_pnl=("gross_pnl","sum"),
                charges=("charges","sum"),
                net_pnl=("net_pnl","sum"),
                num_trades=("num_trades","sum"),
            ).round(2)
        else:
            monthly = None
        regime_stats = df_t.groupby("regime")["net_pnl"].agg(["count","mean","sum"]).round(2)
        return {
            "total_trades": total_trades, "wins": wins, "losses": losses,
            "win_rate": win_rate, "avg_rr": round(avg_rr, 2),
            "gross_total": round(gross_total, 2),
            "charges_total": round(charges_total, 2),
            "net_total": round(net_total, 2),
            "avg_gross_day": avg_gross_day, "avg_net_day": avg_net_day,
            "avg_charges_day": avg_charges_d,
            "sharpe": sharpe, "max_drawdown": max_dd,
            "trading_days": trading_days, "target_rate": target_rate,
            "monthly": monthly, "regime_stats": regime_stats,
            "df_trades": df_t, "df_daily": df_d,
        }

# ── Reporter ─────────────────────────────────────────────────────────────────
def print_report(r: dict, label: str = "RESULT") -> None:
    bar = "=" * 54
    p(f"\n{bar}")
    p(f" INTRADAY 15m BACKTEST -- {label}")
    p(f" {BACKTEST_DAYS}d | {len(SYMBOLS)} symbols | Rs{CAPITAL:,.0f} | {INTERVAL} bars")
    p(bar)
    if "error" in r:
        p(f" ERROR: {r['error']}"); return
    p(f" Total trades    : {r['total_trades']:>6}")
    p(f" Win rate        : {r['win_rate']:>5.1f}%")
    p(f" Avg R:R         : {r['avg_rr']:>5.2f}")
    p(f" Trading days    : {r['trading_days']:>6}")
    p("")
    p(f" GROSS P&L total : Rs{r['gross_total']:>10,.2f}")
    p(f" Total charges   : Rs{r['charges_total']:>10,.2f}")
    p(f" NET P&L total   : Rs{r['net_total']:>10,.2f}")
    p("")
    p(f" Avg GROSS/day   : Rs{r['avg_gross_day']:>8,.2f}")
    p(f" Avg charges/day : Rs{r['avg_charges_day']:>8,.2f}")
    p(f" Avg NET/day     : Rs{r['avg_net_day']:>8,.2f}")
    p("")
    p(f" Sharpe ratio    : {r['sharpe']:>6.2f}")
    p(f" Max drawdown    : Rs{r['max_drawdown']:>8,.2f}")
    p(f" Target hit rate : {r['target_rate']:>5.1f}% (days >= Rs{TARGET_GROSS:,.0f} gross)")
    p(bar)
    if r.get("regime_stats") is not None:
        p("\n By Regime:")
        p(r["regime_stats"].to_string())
    if r.get("monthly") is not None:
        p("\n Monthly Breakdown (last 6 months):")
        for period, row in r["monthly"].tail(6).iterrows():
            p(f"  {period}: Gross=Rs{row['gross_pnl']:>8,.0f}  "
              f"Net=Rs{row['net_pnl']:>8,.0f}  Trades={int(row['num_trades'])}")
    p("")
    if r["avg_net_day"] >= TARGET_GROSS - 500:
        p(f"  System MEETS Rs{TARGET_GROSS-500:,.0f}/day net target ✓")
    else:
        p(f"  Net/day Rs{r['avg_net_day']:,.0f} — below Rs{TARGET_GROSS-500:,.0f} target")
    p("")

# ── Save results ─────────────────────────────────────────────────────────────
def _save_results(r_main: dict, r_alt: dict, partial: bool = False) -> None:
    out = {
        "run_date": datetime.now().isoformat(),
        "interval": INTERVAL,
        "backtest_days": BACKTEST_DAYS,
        "symbols": SYMBOLS,
        "capital": CAPITAL,
        "partial": partial,
        "with_hmm":    {k: v for k, v in r_main.items() if not isinstance(v, (pd.DataFrame, pd.Series))},
        "without_hmm": {k: v for k, v in r_alt.items()  if not isinstance(v, (pd.DataFrame, pd.Series))},
    }
    out_path = config.BASE_DIR / "backtest_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    p(f"  Results saved -> {out_path}")

# ── Main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-hmm", action="store_true", help="Skip HMM regime filter")
    parser.add_argument("--days", type=int, default=BACKTEST_DAYS,
                        help=f"Days of history (default {BACKTEST_DAYS})")
    args = parser.parse_args()
    days = args.days

    p("=" * 54)
    p(f" UPSTOX INTRADAY 15m BACKTEST  ({days} days)")
    p(f" Symbols  : {', '.join(SYMBOLS)}")
    p(f" Capital  : Rs{CAPITAL:,.0f}")
    p(f" Interval : {INTERVAL}")
    p(f" Charges  : Rs{CHARGES_PER:.0f}/trade")
    p(f" Log file : {LOG_FILE}")
    p("=" * 54)
    p("")

    p("Step 1: Loading 15m intraday data (rolling chunks)...")
    all_data = load_all_data(SYMBOLS, days=days)
    if not all_data:
        p("No data loaded. Check internet / yfinance.")
        sys.exit(1)
    p(f"Loaded {len(all_data)} symbols.")

    if args.no_hmm:
        p("Step 2: Skipping HMM (--no-hmm).")
        regime_map = {}
        engine = BacktestEngine(use_hmm=False)
        r = engine.run(all_data, regime_map)
        print_report(r, label="NO HMM")
        _save_results(r, {})
    else:
        p("Step 2: Building daily regime map (HMM on NIFTY)...")
        regime_map = _build_regime_map(days, use_hmm=True)
        p(f"  Regime map: {len(regime_map)} dates")

        p("Step 3: Running WITH-HMM backtest...")
        engine_hmm = BacktestEngine(use_hmm=True)
        r_hmm = engine_hmm.run(all_data, regime_map)
        print_report(r_hmm, label="WITH HMM REGIME FILTER")
        _save_results(r_hmm, {}, partial=True)

        p("Step 4: Running WITHOUT-HMM backtest...")
        engine_noh = BacktestEngine(use_hmm=False)
        r_noh = engine_noh.run(all_data, {})
        print_report(r_noh, label="WITHOUT HMM")

        p(" HMM IMPACT:")
        for key, lbl in [("net_total","Net P&L"),("win_rate","Win rate"),("sharpe","Sharpe")]:
            delta = r_hmm.get(key, 0) - r_noh.get(key, 0)
            p(f"  {lbl} delta: {delta:+.2f}")
        _save_results(r_hmm, r_noh)

    p(f"DONE. See backtest_results.json and {LOG_FILE}")

if __name__ == "__main__":
    main()
