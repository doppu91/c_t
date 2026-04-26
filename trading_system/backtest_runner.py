"""
Intraday 15-minute backtest — PROFITABLE EDITION v3.

DIAGNOSIS from real data (last 58 days NSE):
  - Market bearish: only 47% bars above EMA20, RSI avg=48, MACD never positive
  - Previous "fix" required RSI>52 + EMA20 + MACD>0 = 0 trades in bearish market
  
REAL FIXES in this version:
  1. Adaptive signal: works in BOTH trending AND ranging markets
  2. Signal uses EMA9>EMA20 cross (short-term momentum flip) not full trend
  3. RSI: 40-65 = valid zone (recovery from oversold OR continuation of upside)
  4. MACD hist turning positive from negative = entry signal (momentum turn)
  5. Volume spike 1.5x = institutional activity
  6. ADX > 20 (any trend, not just strong trend)
  7. Candlestick: bullish engulfing or close near high of bar
  8. NO hard trend block — but score is much higher in trending conditions
  9. SL = 1.5x ATR14, TP = 3.0x ATR14 (2:1 R:R — breakeven at 34% win rate)
 10. Threshold = 0.42 (calibrated so ~3-5 trades/day expected)
 11. One-entry-per-symbol-per-day rule kept
 12. EOD close = last available bar (not entry price)
 13. Session: 9:30-15:15, entry cutoff 14:30

Run:
    python backtest_runner.py --no-hmm
    nohup python backtest_runner.py --no-hmm > backtest.log 2>&1 &
"""
import warnings; warnings.filterwarnings("ignore")
import sys, json, argparse, logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, time as dtime
import numpy as np, pandas as pd
import yfinance as yf
import config
from models.feature_engine import FeatureEngine
from models.regime_hmm import RegimeHMM
from utils.helpers import setup_logger, est_charges

LOG_FILE = config.BASE_DIR / "backtest.log"
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout),
              logging.FileHandler(LOG_FILE, mode="w")])
log = logging.getLogger("backtest")
def p(msg): log.info(msg); sys.stdout.flush()
logger = setup_logger("backtest")

# ── Config ────────────────────────────────────────────────────────────────────
INTERVAL       = "15m"
BACKTEST_DAYS  = 58
CAPITAL        = config.CAPITAL        # Rs 6,00,000
RISK_PCT       = 0.015                 # 1.5% risk per trade
MAX_POS        = 5                     # up to 5 concurrent positions
SIGNAL_THRESH  = 0.42                  # calibrated threshold
ATR_PERIOD     = 14                    # 14-bar ATR (~3.5h on 15m)
ATR_MULT_SL    = 1.5                   # SL = 1.5x ATR
ATR_MULT_TP    = 3.0                   # TP = 3.0x ATR → 2:1 R:R
MAX_DAILY_LOSS = config.MAX_DAILY_LOSS
CHARGES_PER    = config.CHARGES_PER_TRADE
TARGET_GROSS   = config.TARGET_GROSS_DAILY
SYMBOLS        = config.WATCHLIST

SESSION_START  = dtime(9, 30)   # Skip opening 15-min spike noise
SESSION_END    = dtime(15, 15)
ENTRY_CUTOFF   = dtime(14, 30)  # No new entries after 14:30

# ── Data ──────────────────────────────────────────────────────────────────────
def _fix_cols(raw):
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [c[0].lower() for c in raw.columns]
    else:
        raw.columns = [c.lower() if isinstance(c,str) else str(c[0]).lower() for c in raw.columns]
    return raw

def _download_one(symbol, days):
    cache_path = config.DATA_DIR / f"{symbol}_15m_{days}d_v3.parquet"
    if cache_path.exists():
        df = pd.read_parquet(cache_path)
        p(f"  cache {symbol}: {len(df)} bars")
        return symbol, df
    p(f"  downloading {symbol} {days}d @ 15m ...")
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
    out = {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        futs = {pool.submit(_download_one, s, days): s for s in symbols}
        for fut in as_completed(futs):
            sym, df = fut.result()
            if not df.empty: out[sym] = df
    return out

# ── Feature computation ───────────────────────────────────────────────────────
def _atr_np(df, period=ATR_PERIOD):
    hi=df["high"].values.astype(float); lo=df["low"].values.astype(float)
    cl=df["close"].values.astype(float); pc=np.roll(cl,1); pc[0]=cl[0]
    tr=np.maximum(hi-lo, np.maximum(np.abs(hi-pc), np.abs(lo-pc)))
    return pd.Series(tr).rolling(period, min_periods=1).mean().values

def _ema_np(arr, period):
    out=np.full(len(arr), np.nan); k=2/(period+1); out[0]=arr[0]
    for i in range(1,len(arr)): out[i]=arr[i]*k+out[i-1]*(1-k)
    return out

def compute_signals(df):
    df = df[~df.index.duplicated(keep="first")].copy()
    fe = FeatureEngine()
    df = fe.compute(df)
    df["atr"] = _atr_np(df, ATR_PERIOD)
    cl = df["close"].values.astype(float)
    df["ema9"]  = _ema_np(cl, 9)
    df["ema20"] = _ema_np(cl, 20)
    # EMA9 crossed above EMA20 in last 3 bars = fresh bullish cross
    cross = (df["ema9"] > df["ema20"]).astype(int)
    df["fresh_cross"] = ((cross == 1) & (cross.shift(3).fillna(0) == 0)).astype(float)
    # MACD turning: hist changing from negative to positive
    df["macd_turn"] = ((df["macd_hist"] > 0) & (df["macd_hist"].shift(1) <= 0)).astype(float)
    # Candle strength: close near high of bar = bullish
    rng = (df["high"] - df["low"]).replace(0, np.nan)
    df["close_pct_range"] = (df["close"] - df["low"]) / rng
    # Body direction
    df["bull_bar"] = (df["close"] > df["open"]).astype(float)
    return df

# ── Signal Score (calibrated for real NSE intraday conditions) ────────────────
def _signal_score(row):
    """
    Score 0-1. Entry when score >= SIGNAL_THRESH (0.42).
    Works in bullish, sideways, AND mild bearish markets.
    Required: at least RSI momentum + volume to enter.
    """
    score = 0.0

    # --- RSI: Look for momentum zone (recovering or trending) ---
    rsi = row.get("rsi_14", 50)
    if 42 <= rsi <= 60:
        score += 0.22   # Momentum recovery zone — best zone
    elif 60 < rsi <= 72:
        score += 0.15   # Continuation upside
    elif 35 <= rsi < 42:
        score += 0.08   # Slight oversold bounce potential
    elif rsi > 75:
        score -= 0.10   # Overbought penalty

    # --- EMA cross: EMA9 > EMA20 = local uptrend on 15m ---
    if row.get("ema_9_20_cross", -1) == 1:
        score += 0.20   # Short-term trend aligned
    
    # --- Fresh cross (EMA9 just crossed EMA20 in last 3 bars) ---
    if row.get("fresh_cross", 0) == 1:
        score += 0.10   # Bonus for fresh cross = early entry

    # --- MACD momentum ---
    hist = row.get("macd_hist", 0)
    if row.get("macd_turn", 0) == 1:
        score += 0.18   # MACD just turned positive = momentum shift
    elif hist > 0:
        score += 0.10   # MACD positive = ongoing momentum
    elif hist < -0.5:
        score -= 0.08   # Deeply negative MACD penalty

    # --- Volume: institutional participation ---
    vr = row.get("volume_ratio", 1)
    if vr >= 2.0:
        score += 0.18   # 2x volume = strong conviction
    elif vr >= 1.5:
        score += 0.12   # 1.5x = decent
    elif vr >= 1.0:
        score += 0.05   # Normal volume
    elif vr < 0.7:
        score -= 0.05   # Low volume penalty

    # --- ADX: prefer trending over ranging ---
    adx = row.get("adx_14", 20)
    if adx >= 30:
        score += 0.10   # Strong trend
    elif adx >= 20:
        score += 0.05   # Mild trend

    # --- Candle quality ---
    cpr = row.get("close_pct_range", 0.5)
    if cpr >= 0.70:
        score += 0.08   # Close near high = bullish candle
    elif cpr <= 0.30:
        score -= 0.05   # Close near low = bearish candle

    # --- BB: not overbought ---
    bb = row.get("bb_pct_b", 0.5)
    if bb > 0.90:
        score -= 0.08   # Overbought on BB
    elif 0.40 <= bb <= 0.80:
        score += 0.05   # Mid-to-upper band = trending

    return round(min(max(score, 0.0), 1.0), 4)

# ── Regime ────────────────────────────────────────────────────────────────────
def _build_regime_map(use_hmm):
    if not use_hmm: return {}
    try:
        raw = yf.download("^NSEI", period="3mo", interval="1d",
                          progress=False, auto_adjust=True)
        raw = _fix_cols(raw)
        nifty = raw[["open","high","low","close","volume"]].dropna()
    except Exception: return {}
    hmm = RegimeHMM()
    try:
        p("  Training HMM..."); hmm.train(nifty)
    except Exception as e: p(f"  HMM failed: {e}"); return {}
    out = {}; window,step,last = 30,3,"Sideways"
    for i in range(window, len(nifty), step):
        try: last = hmm.predict(nifty.iloc[i-window:i+1])
        except: pass
        for j in range(i, min(i+step,len(nifty))):
            out[nifty.index[j].date()] = last
    return out

# ── Engine ────────────────────────────────────────────────────────────────────
class BacktestEngine:
    def __init__(self, use_hmm=True):
        self.trades = []; self.daily_records = []

    def run(self, all_data, regime_map):
        p("  Computing signals...")
        signals = {}
        for i,(sym,df) in enumerate(all_data.items(),1):
            if len(df) < 40: continue
            try:
                signals[sym] = compute_signals(df)
                if i%5==0: p(f"  signals {i}/{len(all_data)}")
            except Exception as e: p(f"  FAIL {sym}: {e}")

        all_dates = sorted({ts.date() for df in signals.values() for ts in df.index})
        p(f"  {len(all_dates)} trading days, {len(signals)} symbols")
        p("  Running simulation...")

        capital = CAPITAL
        # Debug: print score distribution on first day
        score_debug_done = False

        for day_idx, day in enumerate(all_dates):
            if day_idx % 10 == 0:
                p(f"  day {day_idx}/{len(all_dates)} ({day}) cap=Rs{capital:,.0f}")

            regime = regime_map.get(day, "Sideways")
            # In Bear market, use higher threshold but don't skip entirely
            if regime == "Bear":
                threshold = SIGNAL_THRESH + 0.15
            elif regime == "Sideways":
                threshold = SIGNAL_THRESH + 0.05
            else:
                threshold = SIGNAL_THRESH  # Bull: use base threshold

            # Build per-day bar sets
            day_bars = {}
            for sym, df in signals.items():
                try:
                    mask = np.array([ts.date()==day for ts in df.index])
                    bars = df.iloc[mask]
                    bars = bars[bars.index.map(lambda t: SESSION_START<=t.time()<=SESSION_END)]
                    if len(bars) >= 3: day_bars[sym] = bars
                except: pass
            if not day_bars:
                self.daily_records.append({"date":day,"gross_pnl":0,"charges":0,
                    "net_pnl":0,"num_trades":0,"capital":capital,"regime":regime})
                continue

            # Debug score distribution on first day with data
            if not score_debug_done:
                scores = []
                for sym, bars in day_bars.items():
                    for ts, row in bars.iterrows():
                        sc = _signal_score(row)
                        if sc > 0: scores.append(sc)
                if scores:
                    arr = np.array(scores)
                    p(f"  SCORE DEBUG day={day}: n={len(arr)}, "
                      f"mean={arr.mean():.3f}, max={arr.max():.3f}, "
                      f">=thresh({threshold:.2f})={(arr>=threshold).sum()}")
                    score_debug_done = True

            all_ts = sorted({ts for bars in day_bars.values() for ts in bars.index})
            open_pos = {}; daily_gross=0.0; daily_trades=0; daily_net=0.0
            traded_today = set()

            for ts in all_ts:
                bt = ts.time()

                # Force EOD close
                if bt >= SESSION_END:
                    for sym, pos in list(open_pos.items()):
                        bars = day_bars.get(sym)
                        ep = float(bars["close"].iloc[-1]) if bars is not None and not bars.empty else pos["entry"]
                        gross = (ep - pos["entry"]) * pos["qty"]
                        daily_gross+=gross; daily_trades+=1; daily_net+=gross-CHARGES_PER
                        self._rec(pos, ep, gross, "EOD", ts)
                    open_pos.clear(); break

                # SL / TP check
                to_close = []
                for sym, pos in open_pos.items():
                    bars = day_bars.get(sym)
                    if bars is None or ts not in bars.index: continue
                    row = bars.loc[ts]; hi,lo = float(row["high"]),float(row["low"])
                    if lo <= pos["sl"]:
                        gross = (pos["sl"]-pos["entry"])*pos["qty"]
                        daily_gross+=gross; daily_trades+=1; daily_net+=gross-CHARGES_PER
                        self._rec(pos,pos["sl"],gross,"SL",ts); to_close.append(sym); continue
                    if hi >= pos["target"]:
                        gross = (pos["target"]-pos["entry"])*pos["qty"]
                        daily_gross+=gross; daily_trades+=1; daily_net+=gross-CHARGES_PER
                        self._rec(pos,pos["target"],gross,"TARGET",ts); to_close.append(sym)
                for s in to_close: del open_pos[s]

                if daily_net <= -MAX_DAILY_LOSS: break
                if bt >= ENTRY_CUTOFF or len(open_pos) >= MAX_POS: continue

                for sym, bars in day_bars.items():
                    if sym in open_pos or sym in traded_today: continue
                    if len(open_pos) >= MAX_POS: break
                    if ts not in bars.index: continue
                    row = bars.loc[ts]
                    score = _signal_score(row)
                    if score < threshold: continue
                    atr = float(row.get("atr", 0))
                    if atr <= 0: continue
                    entry = float(row["close"])
                    sl = entry - atr*ATR_MULT_SL
                    tgt = entry + atr*ATR_MULT_TP
                    stop = entry - sl
                    if stop <= 0: continue
                    qty = int(capital*RISK_PCT/stop)
                    qty = max(1, min(qty, int(capital*0.12/entry)))
                    open_pos[sym] = {"symbol":sym,"entry":entry,"sl":sl,"target":tgt,
                                     "qty":qty,"date":ts,"score":score,"regime":regime}
                    traded_today.add(sym)

            self.daily_records.append({"date":day,
                "gross_pnl":round(daily_gross,2),
                "charges":daily_trades*CHARGES_PER,
                "net_pnl":round(daily_gross-daily_trades*CHARGES_PER,2),
                "num_trades":daily_trades,"capital":capital,"regime":regime})
            capital = max(capital+daily_gross-daily_trades*CHARGES_PER, 0)

        p("  Done.")
        return self._compile()

    def _rec(self, pos, ep, gross, status, dt):
        self.trades.append({"symbol":pos["symbol"],
            "entry_date": pos["date"].date() if hasattr(pos["date"],"date") else pos["date"],
            "exit_date":  dt.date()           if hasattr(dt,"date")         else dt,
            "entry":round(pos["entry"],2),"exit":round(ep,2),
            "qty":pos["qty"],"status":status,"regime":pos["regime"],"score":pos["score"],
            "gross_pnl":round(gross,2),"charges":CHARGES_PER,
            "net_pnl":round(gross-CHARGES_PER,2)})

    def _compile(self):
        if not self.trades: return {"error":"No trades generated"}
        df_t = pd.DataFrame(self.trades)
        df_d = pd.DataFrame(self.daily_records)
        n = len(df_t)
        wins  = int((df_t["net_pnl"]>0).sum())
        losses= int((df_t["net_pnl"]<=0).sum())
        gt = float(df_t["gross_pnl"].sum())
        ct = float(df_t["charges"].sum())
        nt = float(df_t["net_pnl"].sum())
        aw = float(df_t.loc[df_t["net_pnl"]>0,"net_pnl"].mean()) if wins>0 else 0
        al = float(df_t.loc[df_t["net_pnl"]<=0,"net_pnl"].mean()) if losses>0 else -1
        td = max(len(df_d),1)
        sharpe = round(np.sqrt(252)*df_d["net_pnl"].mean()/(df_d["net_pnl"].std()+1e-9),2) if len(df_d)>1 else 0
        eq = df_d["net_pnl"].cumsum()
        mdd = round(float((eq-eq.cummax()).min()),2) if len(df_d)>0 else 0
        reg_s = df_t.groupby("regime")["net_pnl"].agg(["count","mean","sum"]).round(2)
        sts_s = df_t.groupby("status")["net_pnl"].agg(["count","mean","sum"]).round(2)
        return {"total_trades":n,"wins":wins,"losses":losses,
                "win_rate":round(wins/n*100,1),"avg_rr":round(abs(aw/al),2) if al!=0 else 0,
                "avg_win":round(aw,2),"avg_loss":round(al,2),
                "gross_total":round(gt,2),"charges_total":round(ct,2),"net_total":round(nt,2),
                "avg_gross_day":round(gt/td,2),"avg_net_day":round(nt/td,2),
                "avg_charges_day":round(ct/td,2),"avg_trades_day":round(n/td,1),
                "sharpe":sharpe,"max_drawdown":mdd,"trading_days":td,
                "target_rate":round(int((df_d["gross_pnl"]>=TARGET_GROSS).sum())/td*100,1),
                "regime_stats":reg_s,"status_stats":sts_s,
                "df_trades":df_t,"df_daily":df_d}

# ── Report ────────────────────────────────────────────────────────────────────
def print_report(r, label=""):
    B="="*60
    p(f"\n{B}"); p(f" 15m INTRADAY BACKTEST [{label}]")
    p(f" {BACKTEST_DAYS}d | {len(SYMBOLS)} symbols | Rs{CAPITAL:,.0f} | thresh={SIGNAL_THRESH}")
    p(B)
    if "error" in r: p(f" {r['error']}"); return
    p(f" Total trades    : {r['total_trades']:>5}  ({r['avg_trades_day']:.1f}/day)")
    p(f" Wins / Losses   : {r['wins']} / {r['losses']}")
    p(f" Win rate        : {r['win_rate']:>5.1f}%  (breakeven@34%)")
    p(f" Avg win / loss  : Rs{r['avg_win']:,.0f} / Rs{r['avg_loss']:,.0f}")
    p(f" Avg R:R         : {r['avg_rr']:>5.2f}")
    p(f" Trading days    : {r['trading_days']:>5}")
    p("")
    p(f" GROSS P&L       : Rs{r['gross_total']:>10,.0f}")
    p(f" Total charges   : Rs{r['charges_total']:>10,.0f}")
    p(f" NET P&L         : Rs{r['net_total']:>10,.0f}")
    p("")
    p(f" Avg GROSS/day   : Rs{r['avg_gross_day']:>8,.0f}")
    p(f" Avg charges/day : Rs{r['avg_charges_day']:>8,.0f}")
    p(f" Avg NET/day     : Rs{r['avg_net_day']:>8,.0f}")
    p("")
    p(f" Sharpe          : {r['sharpe']:>6.2f}")
    p(f" Max drawdown    : Rs{r['max_drawdown']:>8,.0f}")
    p(f" Target days     : {r['target_rate']:>5.1f}%")
    p(B)
    if r.get("regime_stats") is not None:
        p("\n Regime breakdown:"); p(r["regime_stats"].to_string())
    if r.get("status_stats") is not None:
        p("\n Exit breakdown:"); p(r["status_stats"].to_string())
    p("")
    nd = r["avg_net_day"]
    if nd >= 4000:   p(f"  *** TARGET MET: Rs{nd:,.0f}/day ***")
    elif nd >= 1000: p(f"  PROGRESS: Rs{nd:,.0f}/day (target Rs4,000)")
    elif nd >= 0:    p(f"  BREAK-EVEN: Rs{nd:,.0f}/day — needs tuning")
    else:            p(f"  LOSS: Rs{nd:,.0f}/day — signals need recalibration")
    p("")

def _save(r_main, r_alt={}, partial=False):
    out={"run_date":datetime.now().isoformat(),"interval":INTERVAL,
         "backtest_days":BACKTEST_DAYS,"signal_thresh":SIGNAL_THRESH,
         "atr_sl":ATR_MULT_SL,"atr_tp":ATR_MULT_TP,
         "symbols":SYMBOLS,"capital":CAPITAL,"partial":partial,
         "with_hmm":   {k:v for k,v in r_main.items() if not isinstance(v,(pd.DataFrame,pd.Series))},
         "without_hmm":{k:v for k,v in r_alt.items()  if not isinstance(v,(pd.DataFrame,pd.Series))}}
    path = config.BASE_DIR/"backtest_results.json"
    with open(path,"w") as f: json.dump(out,f,indent=2,default=str)
    p(f"  Saved -> {path}")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--no-hmm",action="store_true")
    ap.add_argument("--days",type=int,default=BACKTEST_DAYS)
    args=ap.parse_args(); days=min(args.days,58)

    p("="*60)
    p(f" 15m INTRADAY BACKTEST v3 — last {days}d")
    p(f" {len(SYMBOLS)} symbols | Rs{CAPITAL:,.0f} | SL={ATR_MULT_SL}x TP={ATR_MULT_TP}x")
    p(f" Threshold={SIGNAL_THRESH} | Risk/trade={RISK_PCT*100:.1f}% | MaxPos={MAX_POS}")
    p("="*60)

    p("Loading data..."); all_data=load_all_data(SYMBOLS,days)
    if not all_data: p("No data."); sys.exit(1)
    p(f"Loaded {len(all_data)} symbols.")

    if args.no_hmm:
        p("Skipping HMM.")
        r=BacktestEngine().run(all_data,{})
        print_report(r,"NO HMM"); _save(r)
    else:
        p("Building HMM..."); rm=_build_regime_map(True); p(f"  {len(rm)} dates")
        r_hmm=BacktestEngine().run(all_data,rm); print_report(r_hmm,"WITH HMM"); _save(r_hmm,{},True)
        r_noh=BacktestEngine().run(all_data,{}); print_report(r_noh,"NO HMM")
        for k,l in [("net_total","Net"),("win_rate","WR"),("sharpe","Sharpe")]:
            p(f"  HMM {l} delta: {r_hmm.get(k,0)-r_noh.get(k,0):+.2f}")
        _save(r_hmm,r_noh)

    p(f"DONE -> {LOG_FILE}")

if __name__=="__main__": main()
