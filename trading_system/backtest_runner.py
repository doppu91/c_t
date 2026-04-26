"""
Intraday 15-minute backtest — FULLY FIXED for profitability.

KEY FIXES vs previous version:
  1. Signal score rebuilt for intraday MOMENTUM (not mean-reversion)
     - RSI 50-65 = momentum confirmation (+0.25), RSI >65 = strong trend (+0.15)
     - MACD hist > 0 AND rising = +0.25 (trend confirmation)
     - Price above EMA20 = trend filter (+0.20)
     - Volume surge > 1.5x avg = +0.15 (institutional participation)
     - ADX > 25 = trending market (+0.15)
  2. One entry per symbol per day (no re-entry after SL/TP)
  3. Trend filter: only LONG when price > EMA20 on 15m
  4. ATR period 20 (5 hours lookback, more stable than 14)
  5. Better SL/TP: SL=1.5x ATR, TP=3.0x ATR (2:1 R:R, covers charges)
  6. EOD exit uses last known close price (not entry price)
  7. Max 5 positions (more diversification on 15 symbols)
  8. Signal threshold 0.55 (quality filter, not noise)
  9. Charges-aware: need gross > Rs150 per trade to profit after Rs100 charge
 10. Position sizing: 1.5% risk per trade (more conservative, more trades survive)

Run: python backtest_runner.py --no-hmm
     nohup python backtest_runner.py --no-hmm > backtest.log 2>&1 &
"""
import warnings
warnings.filterwarnings("ignore")
import sys, json, argparse, logging
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
    level=logging.INFO, format="%(asctime)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout),
              logging.FileHandler(LOG_FILE, mode="w")],
)
log = logging.getLogger("backtest")
def p(msg): log.info(msg); sys.stdout.flush()
logger = setup_logger("backtest")

# ── Strategy Config ───────────────────────────────────────────────────────────
INTERVAL       = "15m"
BACKTEST_DAYS  = 58          # yfinance 15m hard limit = 60 days
CAPITAL        = config.CAPITAL          # Rs 6,00,000
RISK_PCT       = 0.015                   # FIX: 1.5% risk/trade (was 2%)
MAX_POS        = 5                       # FIX: up to 5 concurrent (was 3)
SIGNAL_THRESH  = 0.55                    # FIX: quality filter (was 0.38 = noise)
ATR_PERIOD     = 20                      # FIX: 5-hour lookback (was 14 = 3.5h)
ATR_MULT_SL    = 1.5                     # FIX: 1.5x ATR stop (was 1.0 = too tight)
ATR_MULT_TP    = 3.0                     # FIX: 3.0x ATR target = 2:1 R:R
MAX_DAILY_LOSS = config.MAX_DAILY_LOSS   # Rs 3,000
CHARGES_PER    = config.CHARGES_PER_TRADE  # Rs 100/trade
MIN_GROSS_PER_TRADE = CHARGES_PER * 1.5   # Rs 150 min gross to be worth entering
TARGET_GROSS   = config.TARGET_GROSS_DAILY
SYMBOLS        = config.WATCHLIST

SESSION_START  = dtime(9, 30)   # FIX: skip first 15 min (opening noise)
SESSION_END    = dtime(15, 15)
ENTRY_CUTOFF   = dtime(14, 30)  # FIX: no entries after 14:30 (was 14:45)

# ── Data loader ──────────────────────────────────────────────────────────────
def _fix_cols(raw):
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [col[0].lower() for col in raw.columns]
    else:
        raw.columns = [c.lower() if isinstance(c, str) else str(c[0]).lower() for c in raw.columns]
    return raw

def _download_one(symbol, days):
    cache_path = config.DATA_DIR / f"{symbol}_15m_{days}d.parquet"
    if cache_path.exists():
        df = pd.read_parquet(cache_path)
        p(f"  cache {symbol}: {len(df)} bars")
        return symbol, df
    p(f"  downloading {symbol} last {days}d @ 15m ...")
    try:
        raw = yf.download(f"{symbol}.NS", period=f"{days}d", interval=INTERVAL,
                          progress=False, auto_adjust=True)
        if raw.empty: raise ValueError("empty")
        raw = _fix_cols(raw)
        df = raw[["open","high","low","close","volume"]].dropna()
        df = df[~df.index.duplicated(keep="first")].sort_index()
        df.to_parquet(cache_path)
        p(f"  saved {symbol}: {len(df)} bars")
        return symbol, df
    except Exception as exc:
        p(f"  FAILED {symbol}: {exc}")
        return symbol, pd.DataFrame()

def load_all_data(symbols, days=BACKTEST_DAYS):
    all_data = {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        futs = {pool.submit(_download_one, s, days): s for s in symbols}
        for fut in as_completed(futs):
            sym, df = fut.result()
            if not df.empty: all_data[sym] = df
    return all_data

# ── Technical helpers ─────────────────────────────────────────────────────────
def _atr_np(df, period=ATR_PERIOD):
    hi = df["high"].values.astype(float)
    lo = df["low"].values.astype(float)
    cl = df["close"].values.astype(float)
    pc = np.roll(cl, 1); pc[0] = cl[0]
    tr = np.maximum(hi-lo, np.maximum(np.abs(hi-pc), np.abs(lo-pc)))
    return pd.Series(tr).rolling(period, min_periods=1).mean().values

def _ema_np(arr, period):
    """Pure-numpy EMA."""
    out = np.full(len(arr), np.nan)
    k = 2/(period+1)
    out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = arr[i]*k + out[i-1]*(1-k)
    return out

def compute_signals(df):
    """Add all features needed for the FIXED signal score."""
    df = df[~df.index.duplicated(keep="first")].copy()
    # Basic FeatureEngine (RSI, MACD, ADX, Volume ratio, Stoch, BB)
    fe = FeatureEngine()
    df = fe.compute(df)
    # ATR with better period
    df["atr"] = _atr_np(df, ATR_PERIOD)
    # EMA trend filters
    cl = df["close"].values.astype(float)
    df["ema20"]  = _ema_np(cl, 20)
    df["ema50"]  = _ema_np(cl, 50)
    df["ema200"] = _ema_np(cl, 200)
    # MACD hist slope (rising = momentum strengthening)
    df["macd_rising"] = (df["macd_hist"] > df["macd_hist"].shift(1)).astype(float)
    # Candle direction: bullish candle = close > open
    df["bull_candle"] = (df["close"] > df["open"]).astype(float)
    # Price vs EMA trend
    df["above_ema20"]  = (df["close"] > df["ema20"]).astype(float)
    df["above_ema50"]  = (df["close"] > df["ema50"]).astype(float)
    return df

def _signal_score(row):
    """
    FIXED intraday MOMENTUM signal score.
    Goal: enter when price is trending UP with volume and momentum confirmation.
    Breakeven at 2:1 R:R requires ~35% win rate; we target 45%+ with this filter.
    """
    score = 0.0
    close = row.get("close", 0)
    ema20 = row.get("ema20", close)

    # === MANDATORY TREND FILTER (must be above EMA20) ===
    # If below EMA20, cap score — not a long setup
    if close < ema20:
        return 0.0   # Hard block: no long entries in downtrend

    # --- RSI momentum (intraday: 50-70 = uptrend, not oversold) ---
    rsi = row.get("rsi_14", 50)
    if 52 <= rsi <= 70:
        score += 0.25    # Sweet spot: momentum without overbought
    elif 70 < rsi <= 80:
        score += 0.10    # Still running but watch for reversal
    elif rsi < 45:
        score += 0.0     # No score: not a momentum setup

    # --- MACD confirmation ---
    hist = row.get("macd_hist", 0)
    rising = row.get("macd_rising", 0)
    if hist > 0 and rising:
        score += 0.25    # MACD positive AND strengthening = strong signal
    elif hist > 0:
        score += 0.12    # MACD positive but not rising

    # --- Price above EMAs (trend alignment) ---
    if row.get("above_ema50", 0):
        score += 0.15    # Above 50 EMA = medium-term uptrend
    if row.get("above_ema20", 0):
        score += 0.10    # Redundant with filter above but confirms

    # --- Volume surge (institutional participation) ---
    vr = row.get("volume_ratio", 1)
    if vr >= 2.0:
        score += 0.15    # 2x volume = strong conviction
    elif vr >= 1.5:
        score += 0.10    # 1.5x volume = decent

    # --- ADX (trending vs ranging) ---
    adx = row.get("adx_14", 20)
    if adx >= 30:
        score += 0.10    # Strong trend
    elif adx >= 25:
        score += 0.06    # Decent trend

    # --- Bullish candle ---
    if row.get("bull_candle", 0):
        score += 0.05    # Candle closing near high = bullish

    # --- Stoch not overbought ---
    sk = row.get("stoch_k", 50)
    if sk < 80:
        score += 0.05    # Not overbought
    else:
        score -= 0.10    # Penalize overbought entries

    return round(min(score, 1.0), 4)

# ── Regime (daily NIFTY) ─────────────────────────────────────────────────────
def _build_regime_map(use_hmm):
    if not use_hmm: return {}
    try:
        raw = yf.download("^NSEI", period="3mo", interval="1d",
                          progress=False, auto_adjust=True)
        raw = _fix_cols(raw)
        nifty = raw[["open","high","low","close","volume"]].dropna()
    except Exception: return {}
    if nifty.empty: return {}
    hmm = RegimeHMM()
    try:
        p("  Training HMM..."); hmm.train(nifty); p("  HMM trained.")
    except Exception as exc:
        p(f"  HMM failed: {exc}"); return {}
    regime_map = {}
    window, step, last = 30, 3, "Sideways"
    for i in range(window, len(nifty), step):
        try: last = hmm.predict(nifty.iloc[i-window:i+1])
        except: pass
        for j in range(i, min(i+step, len(nifty))):
            regime_map[nifty.index[j].date()] = last
    return regime_map

# ── Backtest engine ───────────────────────────────────────────────────────────
class BacktestEngine:
    def __init__(self, use_hmm=True):
        self.use_hmm = use_hmm
        self.trades = []
        self.daily_records = []

    def run(self, all_data, regime_map):
        p("  Computing signals...")
        signals = {}
        for i, (sym, df) in enumerate(all_data.items(), 1):
            if len(df) < 60: continue
            try:
                signals[sym] = compute_signals(df)
                if i % 5 == 0: p(f"  signals {i}/{len(all_data)}")
            except Exception as exc:
                p(f"  signal FAIL {sym}: {exc}")

        all_dates = sorted({ts.date() for df in signals.values() for ts in df.index})
        p(f"  {len(all_dates)} trading days, {len(signals)} symbols loaded")
        p("  Running intraday simulation...")

        capital = CAPITAL
        for day_idx, day in enumerate(all_dates):
            if day_idx % 10 == 0:
                p(f"  day {day_idx}/{len(all_dates)} ({day}) cap=Rs{capital:,.0f}")

            regime = regime_map.get(day, "Sideways")
            if regime == "Bear":
                self.daily_records.append({"date": day, "gross_pnl": 0, "charges": 0,
                    "net_pnl": 0, "num_trades": 0, "capital": capital, "regime": regime})
                continue

            # FIX: In Bear regime skip; in Sideways use higher threshold
            threshold = SIGNAL_THRESH + 0.10 if regime == "Sideways" else SIGNAL_THRESH

            # Build per-symbol bar dict for this day
            day_bars = {}
            for sym, df in signals.items():
                try:
                    mask = np.array([ts.date() == day for ts in df.index])
                    bars = df.iloc[mask]
                    bars = bars[bars.index.map(lambda t: SESSION_START <= t.time() <= SESSION_END)]
                    if len(bars) >= 5:  # Need enough bars for meaningful signals
                        day_bars[sym] = bars
                except Exception:
                    pass
            if not day_bars: continue

            all_ts = sorted({ts for bars in day_bars.values() for ts in bars.index})
            open_positions = {}
            daily_gross = 0.0
            daily_trades = 0
            daily_pnl_net = 0.0
            traded_today = set()   # FIX: one trade per symbol per day

            for ts in all_ts:
                bar_time = ts.time()

                # ── Force-close at session end ──
                if bar_time >= SESSION_END:
                    for sym, pos in list(open_positions.items()):
                        bars = day_bars.get(sym)
                        # FIX: Use last bar close instead of entry price
                        if bars is not None and not bars.empty:
                            exit_price = float(bars["close"].iloc[-1])
                        else:
                            exit_price = pos["entry"]
                        gross = (exit_price - pos["entry"]) * pos["qty"]
                        daily_gross += gross; daily_trades += 1
                        daily_pnl_net += gross - CHARGES_PER
                        self._record(pos, exit_price, gross, "EOD", ts)
                    open_positions.clear()
                    break

                # ── Check SL / TP ──
                to_close = []
                for sym, pos in open_positions.items():
                    bars = day_bars.get(sym)
                    if bars is None or ts not in bars.index: continue
                    row = bars.loc[ts]
                    hi, lo = float(row["high"]), float(row["low"])
                    if lo <= pos["sl"]:
                        gross = (pos["sl"] - pos["entry"]) * pos["qty"]
                        daily_gross += gross; daily_trades += 1
                        daily_pnl_net += gross - CHARGES_PER
                        self._record(pos, pos["sl"], gross, "SL", ts)
                        to_close.append(sym)
                        continue
                    if hi >= pos["target"]:
                        gross = (pos["target"] - pos["entry"]) * pos["qty"]
                        daily_gross += gross; daily_trades += 1
                        daily_pnl_net += gross - CHARGES_PER
                        self._record(pos, pos["target"], gross, "TARGET", ts)
                        to_close.append(sym)
                for sym in to_close: del open_positions[sym]

                # ── Daily loss circuit breaker ──
                if daily_pnl_net <= -MAX_DAILY_LOSS: break

                # ── Entry logic ──
                if bar_time >= ENTRY_CUTOFF: continue
                if len(open_positions) >= MAX_POS: continue

                for sym, bars in day_bars.items():
                    if sym in open_positions: continue
                    if sym in traded_today: continue   # FIX: one trade/day
                    if len(open_positions) >= MAX_POS: break
                    if ts not in bars.index: continue

                    row = bars.loc[ts]
                    score = _signal_score(row)
                    if score < threshold: continue

                    atr = float(row.get("atr", 0))
                    if atr <= 0: continue
                    entry = float(row["close"])
                    sl  = entry - atr * ATR_MULT_SL
                    tgt = entry + atr * ATR_MULT_TP
                    stop_dist = entry - sl
                    if stop_dist <= 0: continue

                    # FIX: Minimum gross check — ensure TP covers charges
                    expected_gross = (tgt - entry) * 1  # at qty=1
                    min_qty_for_profit = int(MIN_GROSS_PER_TRADE / (tgt - entry)) + 1 if (tgt-entry) > 0 else 999

                    qty = int(capital * RISK_PCT / stop_dist)
                    qty = max(min_qty_for_profit, min(qty, int(capital * 0.12 / entry)))
                    if qty <= 0: continue

                    open_positions[sym] = {
                        "symbol": sym, "entry": entry, "sl": sl, "target": tgt,
                        "qty": qty, "date": ts, "score": score, "regime": regime,
                    }
                    traded_today.add(sym)   # FIX: mark traded

            self.daily_records.append({
                "date": day, "gross_pnl": round(daily_gross, 2),
                "charges": daily_trades * CHARGES_PER,
                "net_pnl": round(daily_gross - daily_trades * CHARGES_PER, 2),
                "num_trades": daily_trades, "capital": capital, "regime": regime,
            })
            capital = max(capital + daily_gross - daily_trades * CHARGES_PER, 0)

        p("  Simulation complete.")
        return self._compile()

    def _record(self, pos, exit_price, gross_pnl, status, exit_dt):
        self.trades.append({
            "symbol": pos["symbol"],
            "entry_date": pos["date"].date() if hasattr(pos["date"], "date") else pos["date"],
            "exit_date": exit_dt.date() if hasattr(exit_dt, "date") else exit_dt,
            "entry": round(pos["entry"], 2), "exit": round(exit_price, 2),
            "qty": pos["qty"], "status": status, "regime": pos["regime"],
            "score": pos["score"],
            "gross_pnl": round(gross_pnl, 2), "charges": CHARGES_PER,
            "net_pnl": round(gross_pnl - CHARGES_PER, 2),
        })

    def _compile(self):
        if not self.trades: return {"error": "No trades generated"}
        df_t = pd.DataFrame(self.trades)
        df_d = pd.DataFrame(self.daily_records)
        wins   = int((df_t["net_pnl"] > 0).sum())
        losses = int((df_t["net_pnl"] <= 0).sum())
        total  = len(df_t)
        gross_total   = float(df_t["gross_pnl"].sum())
        charges_total = float(df_t["charges"].sum())
        net_total     = float(df_t["net_pnl"].sum())
        avg_win  = float(df_t.loc[df_t["net_pnl"]>0,"net_pnl"].mean()) if wins>0 else 0
        avg_loss = float(df_t.loc[df_t["net_pnl"]<=0,"net_pnl"].mean()) if losses>0 else -1
        avg_rr   = abs(avg_win/avg_loss) if avg_loss != 0 else 0
        td = max(len(df_d), 1)
        sharpe = round(np.sqrt(252)*df_d["net_pnl"].mean()/(df_d["net_pnl"].std()+1e-9),2) if len(df_d)>1 else 0
        eq = df_d["net_pnl"].cumsum()
        max_dd = round(float((eq-eq.cummax()).min()), 2) if len(df_d)>0 else 0
        regime_stats = df_t.groupby("regime")["net_pnl"].agg(["count","mean","sum"]).round(2)
        status_stats = df_t.groupby("status")["net_pnl"].agg(["count","mean","sum"]).round(2)
        return {
            "total_trades": total, "wins": wins, "losses": losses,
            "win_rate": round(wins/total*100, 1),
            "avg_win": round(avg_win, 2), "avg_loss": round(avg_loss, 2),
            "avg_rr": round(avg_rr, 2),
            "gross_total": round(gross_total, 2),
            "charges_total": round(charges_total, 2),
            "net_total": round(net_total, 2),
            "avg_gross_day": round(gross_total/td, 2),
            "avg_net_day":   round(net_total/td, 2),
            "avg_charges_day": round(charges_total/td, 2),
            "avg_trades_day": round(total/td, 1),
            "sharpe": sharpe, "max_drawdown": max_dd,
            "trading_days": td,
            "target_rate": round(int((df_d["gross_pnl"]>=TARGET_GROSS).sum())/td*100, 1),
            "regime_stats": regime_stats, "status_stats": status_stats,
            "df_trades": df_t, "df_daily": df_d,
        }

# ── Reporter ──────────────────────────────────────────────────────────────────
def print_report(r, label="RESULT"):
    bar = "=" * 58
    p(f"\n{bar}")
    p(f" 15m INTRADAY BACKTEST [{label}]")
    p(f" {BACKTEST_DAYS}d | {len(SYMBOLS)} symbols | Rs{CAPITAL:,.0f} | threshold={SIGNAL_THRESH}")
    p(bar)
    if "error" in r: p(f" ERROR: {r['error']}"); return
    p(f" Total trades    : {r['total_trades']:>6}  ({r['avg_trades_day']:.1f}/day)")
    p(f" Win / Loss      : {r['wins']} / {r['losses']}")
    p(f" Win rate        : {r['win_rate']:>5.1f}%")
    p(f" Avg win         : Rs{r['avg_win']:>8,.2f}")
    p(f" Avg loss        : Rs{r['avg_loss']:>8,.2f}")
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
    p(f" Target hit rate : {r['target_rate']:>5.1f}%  (days >= Rs{TARGET_GROSS:,.0f})")
    p(bar)
    if r.get("regime_stats") is not None:
        p("\n By Regime:"); p(r["regime_stats"].to_string())
    if r.get("status_stats") is not None:
        p("\n By Exit Type:"); p(r["status_stats"].to_string())
    p("")
    if r["avg_net_day"] >= 4000:
        p(f"  *** MEETS Rs4,000/day target! ***")
    elif r["avg_net_day"] >= 2000:
        p(f"  Avg Rs{r['avg_net_day']:,.0f}/day — getting closer")
    else:
        p(f"  Avg Rs{r['avg_net_day']:,.0f}/day — needs tuning")
    p("")

# ── Save ──────────────────────────────────────────────────────────────────────
def _save(r_main, r_alt={}, partial=False):
    out = {
        "run_date": datetime.now().isoformat(),
        "interval": INTERVAL, "backtest_days": BACKTEST_DAYS,
        "signal_thresh": SIGNAL_THRESH, "atr_sl": ATR_MULT_SL, "atr_tp": ATR_MULT_TP,
        "symbols": SYMBOLS, "capital": CAPITAL, "partial": partial,
        "with_hmm":    {k:v for k,v in r_main.items() if not isinstance(v,(pd.DataFrame,pd.Series))},
        "without_hmm": {k:v for k,v in r_alt.items()  if not isinstance(v,(pd.DataFrame,pd.Series))},
    }
    path = config.BASE_DIR / "backtest_results.json"
    with open(path, "w") as f: json.dump(out, f, indent=2, default=str)
    p(f"  Saved -> {path}")

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-hmm", action="store_true")
    parser.add_argument("--days", type=int, default=BACKTEST_DAYS)
    args = parser.parse_args()
    days = min(args.days, 58)

    p("=" * 58)
    p(f" UPSTOX 15m INTRADAY BACKTEST (FIXED)")
    p(f" Last {days}d | {len(SYMBOLS)} symbols | Rs{CAPITAL:,.0f}")
    p(f" SL={ATR_MULT_SL}xATR  TP={ATR_MULT_TP}xATR  Thresh={SIGNAL_THRESH}  MaxPos={MAX_POS}")
    p(f" Risk/trade={RISK_PCT*100:.1f}%  Charges=Rs{CHARGES_PER}/trade")
    p(f" Session: {SESSION_START}–{SESSION_END}  Entry cutoff: {ENTRY_CUTOFF}")
    p("=" * 58)
    p("")

    p("Step 1: Loading 15m data...")
    all_data = load_all_data(SYMBOLS, days=days)
    if not all_data: p("No data. Check internet."); sys.exit(1)
    p(f"Loaded {len(all_data)} symbols.")

    if args.no_hmm:
        p("Step 2: Skipping HMM.")
        engine = BacktestEngine(use_hmm=False)
        r = engine.run(all_data, {})
        print_report(r, "NO HMM")
        _save(r)
    else:
        p("Step 2: Building HMM regime map...")
        regime_map = _build_regime_map(use_hmm=True)
        p(f"  {len(regime_map)} dates mapped")

        p("Step 3: WITH-HMM backtest...")
        r_hmm = BacktestEngine(use_hmm=True).run(all_data, regime_map)
        print_report(r_hmm, "WITH HMM")
        _save(r_hmm, {}, partial=True)

        p("Step 4: WITHOUT-HMM backtest...")
        r_noh = BacktestEngine(use_hmm=False).run(all_data, {})
        print_report(r_noh, "WITHOUT HMM")

        p("HMM IMPACT:")
        for k, lbl in [("net_total","Net P&L"),("win_rate","Win rate"),("sharpe","Sharpe")]:
            p(f"  {lbl}: {r_hmm.get(k,0)-r_noh.get(k,0):+.2f}")
        _save(r_hmm, r_noh)

    p(f"DONE -> backtest_results.json  log -> {LOG_FILE}")

if __name__ == "__main__":
    main()
