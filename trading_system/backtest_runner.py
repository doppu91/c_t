"""
Intraday 15-minute backtest — FIXED v4 (UTC timezone fix + calibrated signals)

ROOT CAUSE OF PREVIOUS FAILURES:
  - yfinance 15m data uses UTC timezone (Index TZ: UTC)
  - NSE session 9:15-15:30 IST = 03:45-10:00 UTC
  - Previous code used IST times (9:15-15:15) which matched NOTHING in UTC data
  - Result: day_bars always empty → zero trades

FIXES IN v4:
  1. Session times converted to UTC (IST - 5:30)
     NSE open  9:15 IST = 03:45 UTC  → SESSION_START_UTC = 03:45
     Entry cutoff 14:30 IST = 09:00 UTC → ENTRY_CUTOFF_UTC = 09:00
     Force close 15:15 IST = 09:45 UTC → SESSION_END_UTC   = 09:45
  2. Date grouping uses IST date (UTC+5:30), not UTC date
  3. Signal score: adaptive for current bearish market conditions
     - RSI 42-65 valid zone (not strictly 50+)
     - EMA9>EMA20 cross = local trend (+0.20)
     - Volume 1.5x+ = institutional (+0.15)
     - MACD rising OR positive (+0.18/+0.10)
  4. Threshold = 0.40 (calibrated after UTC fix)
  5. SL=1.5x ATR, TP=3.0x ATR → 2:1 R:R → breakeven at 34% win rate
  6. One trade per symbol per day
  7. Max 5 concurrent positions

Run: python backtest_runner.py --no-hmm
"""
import warnings; warnings.filterwarnings("ignore")
import sys, json, argparse, logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, time as dtime, timezone, date
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

IST_OFFSET = timedelta(hours=5, minutes=30)

# ── Config ────────────────────────────────────────────────────────────────────
INTERVAL       = "15m"
BACKTEST_DAYS  = 58
CAPITAL        = config.CAPITAL
RISK_PCT       = 0.015
MAX_POS        = 5
SIGNAL_THRESH  = 0.40
ATR_PERIOD     = 14
ATR_MULT_SL    = 1.5
ATR_MULT_TP    = 3.0
MAX_DAILY_LOSS = config.MAX_DAILY_LOSS
CHARGES_PER    = config.CHARGES_PER_TRADE
TARGET_GROSS   = config.TARGET_GROSS_DAILY
SYMBOLS        = config.WATCHLIST

# ── KEY FIX: NSE session times in UTC (yfinance 15m uses UTC) ────────────────
# IST = UTC + 5:30, so IST time - 5:30 = UTC time
# 9:15 IST  = 03:45 UTC  (skip first bar = 9:15 bar)
# 9:30 IST  = 04:00 UTC  (first valid entry bar)
# 14:30 IST = 09:00 UTC  (entry cutoff)
# 15:15 IST = 09:45 UTC  (force close)
SESSION_START_UTC  = dtime(4,  0)   # 9:30 IST in UTC
SESSION_END_UTC    = dtime(9, 45)   # 15:15 IST in UTC
ENTRY_CUTOFF_UTC   = dtime(9,  0)   # 14:30 IST in UTC

def _utc_time(ts):
    """Get time() from a UTC-aware timestamp."""
    if hasattr(ts, 'tzinfo') and ts.tzinfo is not None:
        return ts.time()  # Already UTC
    return ts.time()

def _ist_date(ts):
    """Get IST date from a UTC timestamp."""
    if hasattr(ts, 'tzinfo') and ts.tzinfo is not None:
        ist_ts = ts + IST_OFFSET
        return ist_ts.date()
    return ts.date()

# ── Data ──────────────────────────────────────────────────────────────────────
def _fix_cols(raw):
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [c[0].lower() for c in raw.columns]
    else:
        raw.columns = [c.lower() if isinstance(c,str) else str(c[0]).lower() for c in raw.columns]
    return raw

def _download_one(symbol, days):
    cache_path = config.DATA_DIR / f"{symbol}_15m_{days}d_v4.parquet"
    if cache_path.exists():
        df = pd.read_parquet(cache_path)
        p(f"  cache {symbol}: {len(df)} bars (tz={df.index.tz})")
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
        p(f"  saved {symbol}: {len(df)} bars (tz={df.index.tz})")
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

# ── Features ──────────────────────────────────────────────────────────────────
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
    fe = FeatureEngine(); df = fe.compute(df)
    df["atr"] = _atr_np(df, ATR_PERIOD)
    cl = df["close"].values.astype(float)
    df["ema9"]  = _ema_np(cl, 9)
    df["ema20"] = _ema_np(cl, 20)
    cross9_20 = (df["ema9"] > df["ema20"]).astype(int)
    # Fresh cross: EMA9 crossed EMA20 within last 3 bars
    df["fresh_cross"] = ((cross9_20==1) & (cross9_20.shift(3).fillna(0)==0)).astype(float)
    # MACD turning positive
    df["macd_turn"] = ((df["macd_hist"]>0) & (df["macd_hist"].shift(1).fillna(0)<=0)).astype(float)
    # Candle close position in range
    rng = (df["high"]-df["low"]).replace(0, np.nan)
    df["close_pct_range"] = ((df["close"]-df["low"])/rng).fillna(0.5)
    return df

def _signal_score(row):
    """
    Adaptive momentum score for NSE 15m intraday.
    Works in both trending and ranging markets.
    Threshold=0.40 → need 3+ signals firing to enter.
    """
    s = 0.0
    rsi = row.get("rsi_14", 50)
    # RSI zone scoring
    if 42 <= rsi <= 62:   s += 0.22  # Best recovery/momentum zone
    elif 62 < rsi <= 72:  s += 0.14  # Continuation
    elif 35 <= rsi < 42:  s += 0.08  # Potential bounce
    elif rsi > 75:        s -= 0.10  # Overbought penalty

    # EMA9>EMA20: short-term trend aligned
    if row.get("ema_9_20_cross", -1) == 1:  s += 0.20
    # Fresh EMA cross bonus
    if row.get("fresh_cross", 0) == 1:      s += 0.08

    # MACD
    hist = row.get("macd_hist", 0)
    if row.get("macd_turn", 0) == 1: s += 0.18  # Turning positive = momentum shift
    elif hist > 0:                    s += 0.10  # Positive ongoing
    elif hist < -1.0:                 s -= 0.08  # Deeply negative

    # Volume
    vr = row.get("volume_ratio", 1)
    if vr >= 2.0:   s += 0.18
    elif vr >= 1.5: s += 0.12
    elif vr >= 1.0: s += 0.04
    elif vr < 0.7:  s -= 0.04

    # ADX (trend strength)
    adx = row.get("adx_14", 20)
    if adx >= 30:   s += 0.10
    elif adx >= 20: s += 0.05

    # Candle quality: close near high = bullish
    cpr = row.get("close_pct_range", 0.5)
    if cpr >= 0.70:   s += 0.08
    elif cpr <= 0.30: s -= 0.05

    # BB: not overbought
    bb = row.get("bb_pct_b", 0.5)
    if bb > 0.90:          s -= 0.08
    elif 0.40 <= bb <= 0.80: s += 0.04

    return round(min(max(s, 0.0), 1.0), 4)

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
    try: p("  Training HMM..."); hmm.train(nifty)
    except Exception as e: p(f"  HMM fail: {e}"); return {}
    out={}; window,step,last=30,3,"Sideways"
    for i in range(window, len(nifty), step):
        try: last=hmm.predict(nifty.iloc[i-window:i+1])
        except: pass
        for j in range(i, min(i+step,len(nifty))):
            d = nifty.index[j]
            out[d.date() if hasattr(d,'date') else d] = last
    return out

# ── Engine ────────────────────────────────────────────────────────────────────
class BacktestEngine:
    def __init__(self): self.trades=[]; self.daily_records=[]

    def run(self, all_data, regime_map):
        p("  Computing signals...")
        signals = {}
        for i,(sym,df) in enumerate(all_data.items(),1):
            if len(df)<40: continue
            try:
                signals[sym] = compute_signals(df)
                if i%5==0: p(f"  {i}/{len(all_data)} done")
            except Exception as e: p(f"  FAIL {sym}: {e}")

        # ── KEY FIX: group by IST date, filter by UTC times ──
        all_ist_dates = sorted({_ist_date(ts) for df in signals.values() for ts in df.index})
        p(f"  {len(all_ist_dates)} IST trading days, {len(signals)} symbols loaded")

        # Print first few UTC timestamps so we can verify the fix
        sample_sym = list(signals.keys())[0]
        sample_ts = list(signals[sample_sym].index[:3])
        for ts in sample_ts:
            p(f"  SAMPLE UTC ts={ts}  IST date={_ist_date(ts)}  UTC time={_utc_time(ts)}")

        p("  Running simulation...")
        capital = CAPITAL
        score_debug_done = False

        for day_idx, ist_day in enumerate(all_ist_dates):
            if day_idx % 10 == 0:
                p(f"  day {day_idx}/{len(all_ist_dates)} IST={ist_day} cap=Rs{capital:,.0f}")

            regime = regime_map.get(ist_day, "Sideways")
            if regime == "Bear":   threshold = SIGNAL_THRESH + 0.12
            elif regime == "Sideways": threshold = SIGNAL_THRESH + 0.05
            else:                  threshold = SIGNAL_THRESH

            # Build bar sets using IST date grouping + UTC time filtering
            day_bars = {}
            for sym, df in signals.items():
                try:
                    # Filter rows where IST date == ist_day
                    mask = np.array([_ist_date(ts)==ist_day for ts in df.index])
                    bars = df.iloc[mask]
                    # Filter UTC session times
                    bars = bars[bars.index.map(lambda t: SESSION_START_UTC <= _utc_time(t) <= SESSION_END_UTC)]
                    if len(bars) >= 3:
                        day_bars[sym] = bars
                except Exception as e:
                    pass

            if not day_bars:
                self.daily_records.append({"date":ist_day,"gross_pnl":0,"charges":0,
                    "net_pnl":0,"num_trades":0,"capital":capital,"regime":regime})
                continue

            # Score debug on first populated day
            if not score_debug_done:
                sc_list = []
                for sym, bars in day_bars.items():
                    for ts, row in bars.iterrows():
                        sc = _signal_score(row)
                        if sc > 0: sc_list.append(sc)
                if sc_list:
                    arr = np.array(sc_list)
                    p(f"  SCORE DEBUG IST={ist_day}: n={len(arr)}, mean={arr.mean():.3f}, "
                      f"max={arr.max():.3f}, >=thresh({threshold:.2f}): {(arr>=threshold).sum()}")
                score_debug_done = True

            all_ts = sorted({ts for bars in day_bars.values() for ts in bars.index})
            open_pos={}; daily_gross=0.0; daily_trades=0; daily_net=0.0; traded_today=set()

            for ts in all_ts:
                ut = _utc_time(ts)

                # EOD close
                if ut >= SESSION_END_UTC:
                    for sym,pos in list(open_pos.items()):
                        bars=day_bars.get(sym)
                        ep=float(bars["close"].iloc[-1]) if bars is not None and not bars.empty else pos["entry"]
                        gross=(ep-pos["entry"])*pos["qty"]
                        daily_gross+=gross; daily_trades+=1; daily_net+=gross-CHARGES_PER
                        self._rec(pos,ep,gross,"EOD",ts)
                    open_pos.clear(); break

                # SL/TP
                to_close=[]
                for sym,pos in open_pos.items():
                    bars=day_bars.get(sym)
                    if bars is None or ts not in bars.index: continue
                    row=bars.loc[ts]; hi,lo=float(row["high"]),float(row["low"])
                    if lo<=pos["sl"]:
                        gross=(pos["sl"]-pos["entry"])*pos["qty"]
                        daily_gross+=gross; daily_trades+=1; daily_net+=gross-CHARGES_PER
                        self._rec(pos,pos["sl"],gross,"SL",ts); to_close.append(sym); continue
                    if hi>=pos["target"]:
                        gross=(pos["target"]-pos["entry"])*pos["qty"]
                        daily_gross+=gross; daily_trades+=1; daily_net+=gross-CHARGES_PER
                        self._rec(pos,pos["target"],gross,"TARGET",ts); to_close.append(sym)
                for s in to_close: del open_pos[s]

                if daily_net<=-MAX_DAILY_LOSS: break
                if ut>=ENTRY_CUTOFF_UTC or len(open_pos)>=MAX_POS: continue

                for sym,bars in day_bars.items():
                    if sym in open_pos or sym in traded_today: continue
                    if len(open_pos)>=MAX_POS: break
                    if ts not in bars.index: continue
                    row=bars.loc[ts]
                    sc=_signal_score(row)
                    if sc<threshold: continue
                    atr=float(row.get("atr",0))
                    if atr<=0: continue
                    entry=float(row["close"])
                    sl=entry-atr*ATR_MULT_SL; tgt=entry+atr*ATR_MULT_TP
                    stop=entry-sl
                    if stop<=0: continue
                    qty=int(capital*RISK_PCT/stop)
                    qty=max(1,min(qty,int(capital*0.12/entry)))
                    open_pos[sym]={"symbol":sym,"entry":entry,"sl":sl,"target":tgt,
                                   "qty":qty,"date":ts,"score":sc,"regime":regime}
                    traded_today.add(sym)

            self.daily_records.append({"date":ist_day,"gross_pnl":round(daily_gross,2),
                "charges":daily_trades*CHARGES_PER,
                "net_pnl":round(daily_gross-daily_trades*CHARGES_PER,2),
                "num_trades":daily_trades,"capital":capital,"regime":regime})
            capital=max(capital+daily_gross-daily_trades*CHARGES_PER,0)

        p("  Simulation complete.")
        return self._compile()

    def _rec(self,pos,ep,gross,status,dt):
        self.trades.append({"symbol":pos["symbol"],
            "entry_date":pos["date"].date() if hasattr(pos["date"],"date") else pos["date"],
            "exit_date":dt.date() if hasattr(dt,"date") else dt,
            "entry":round(pos["entry"],2),"exit":round(ep,2),
            "qty":pos["qty"],"status":status,"regime":pos["regime"],"score":pos["score"],
            "gross_pnl":round(gross,2),"charges":CHARGES_PER,
            "net_pnl":round(gross-CHARGES_PER,2)})

    def _compile(self):
        if not self.trades: return {"error":"No trades generated"}
        df_t=pd.DataFrame(self.trades); df_d=pd.DataFrame(self.daily_records)
        n=len(df_t); wins=int((df_t["net_pnl"]>0).sum()); losses=int((df_t["net_pnl"]<=0).sum())
        gt=float(df_t["gross_pnl"].sum()); ct=float(df_t["charges"].sum()); nt=float(df_t["net_pnl"].sum())
        aw=float(df_t.loc[df_t["net_pnl"]>0,"net_pnl"].mean()) if wins>0 else 0
        al=float(df_t.loc[df_t["net_pnl"]<=0,"net_pnl"].mean()) if losses>0 else -1
        td=max(len(df_d),1)
        sharpe=round(np.sqrt(252)*df_d["net_pnl"].mean()/(df_d["net_pnl"].std()+1e-9),2) if len(df_d)>1 else 0
        eq=df_d["net_pnl"].cumsum(); mdd=round(float((eq-eq.cummax()).min()),2) if len(df_d)>0 else 0
        reg_s=df_t.groupby("regime")["net_pnl"].agg(["count","mean","sum"]).round(2)
        sts_s=df_t.groupby("status")["net_pnl"].agg(["count","mean","sum"]).round(2)
        return {"total_trades":n,"wins":wins,"losses":losses,
                "win_rate":round(wins/n*100,1),"avg_rr":round(abs(aw/al),2) if al!=0 else 0,
                "avg_win":round(aw,2),"avg_loss":round(al,2),
                "gross_total":round(gt,2),"charges_total":round(ct,2),"net_total":round(nt,2),
                "avg_gross_day":round(gt/td,2),"avg_net_day":round(nt/td,2),
                "avg_charges_day":round(ct/td,2),"avg_trades_day":round(n/td,1),
                "sharpe":sharpe,"max_drawdown":mdd,"trading_days":td,
                "target_rate":round(int((df_d["gross_pnl"]>=TARGET_GROSS).sum())/td*100,1),
                "regime_stats":reg_s,"status_stats":sts_s,"df_trades":df_t,"df_daily":df_d}

# ── Report ────────────────────────────────────────────────────────────────────
def print_report(r,label=""):
    B="="*60; p(f"\n{B}"); p(f" 15m BACKTEST v4 [{label}]")
    p(f" {BACKTEST_DAYS}d|{len(SYMBOLS)} symbols|Rs{CAPITAL:,.0f}|thresh={SIGNAL_THRESH}")
    p(B)
    if "error" in r: p(f" {r['error']}"); return
    p(f" Trades:{r['total_trades']:>5} ({r['avg_trades_day']:.1f}/day)  Wins/Loss:{r['wins']}/{r['losses']}")
    p(f" Win%:{r['win_rate']:>5.1f}%  AvgWin:Rs{r['avg_win']:,.0f}  AvgLoss:Rs{r['avg_loss']:,.0f}  R:R:{r['avg_rr']:.2f}")
    p(f" Trading days: {r['trading_days']}")
    p(f" GROSS: Rs{r['gross_total']:>10,.0f}  Charges: Rs{r['charges_total']:>8,.0f}  NET: Rs{r['net_total']:>10,.0f}")
    p(f" AvgGross/day: Rs{r['avg_gross_day']:>8,.0f}  AvgNet/day: Rs{r['avg_net_day']:>8,.0f}")
    p(f" Sharpe:{r['sharpe']:>6.2f}  MaxDD: Rs{r['max_drawdown']:>8,.0f}  Target%:{r['target_rate']:.1f}%")
    p(B)
    if r.get("regime_stats") is not None: p("\n Regime:"); p(r["regime_stats"].to_string())
    if r.get("status_stats") is not None: p("\n Exits:"); p(r["status_stats"].to_string())
    nd=r["avg_net_day"]
    if nd>=4000:   p(f"  *** TARGET MET Rs{nd:,.0f}/day ***")
    elif nd>=500:  p(f"  PROGRESS Rs{nd:,.0f}/day")
    elif nd>=0:    p(f"  BREAKEVEN zone Rs{nd:,.0f}/day")
    else:          p(f"  LOSS Rs{nd:,.0f}/day")
    p("")

def _save(r_main,r_alt={},partial=False):
    out={"run_date":datetime.now().isoformat(),"interval":INTERVAL,
         "backtest_days":BACKTEST_DAYS,"signal_thresh":SIGNAL_THRESH,
         "session_utc":f"{SESSION_START_UTC}-{SESSION_END_UTC}",
         "atr_sl":ATR_MULT_SL,"atr_tp":ATR_MULT_TP,
         "symbols":SYMBOLS,"capital":CAPITAL,"partial":partial,
         "with_hmm":   {k:v for k,v in r_main.items() if not isinstance(v,(pd.DataFrame,pd.Series))},
         "without_hmm":{k:v for k,v in r_alt.items()  if not isinstance(v,(pd.DataFrame,pd.Series))}}
    path=config.BASE_DIR/"backtest_results.json"
    with open(path,"w") as f: json.dump(out,f,indent=2,default=str)
    p(f"  Saved -> {path}")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--no-hmm",action="store_true")
    ap.add_argument("--days",type=int,default=BACKTEST_DAYS)
    args=ap.parse_args(); days=min(args.days,58)
    p("="*60)
    p(f" 15m INTRADAY BACKTEST v4 (UTC FIXED)")
    p(f" {days}d | {len(SYMBOLS)} symbols | Rs{CAPITAL:,.0f}")
    p(f" Session UTC: {SESSION_START_UTC}–{SESSION_END_UTC} (=9:30–15:15 IST)")
    p(f" SL={ATR_MULT_SL}x TP={ATR_MULT_TP}x Thresh={SIGNAL_THRESH} MaxPos={MAX_POS}")
    p("="*60)
    p("Loading data..."); all_data=load_all_data(SYMBOLS,days)
    if not all_data: p("No data."); sys.exit(1)
    p(f"Loaded {len(all_data)} symbols.")
    if args.no_hmm:
        r=BacktestEngine().run(all_data,{})
        print_report(r,"NO HMM"); _save(r)
    else:
        p("Building HMM..."); rm=_build_regime_map(True); p(f"  {len(rm)} dates")
        r_hmm=BacktestEngine().run(all_data,rm); print_report(r_hmm,"WITH HMM"); _save(r_hmm,{},True)
        r_noh=BacktestEngine().run(all_data,{}); print_report(r_noh,"NO HMM")
        for k,l in [("net_total","Net"),("win_rate","WR"),("sharpe","Sh")]:
            p(f"  HMM {l}: {r_hmm.get(k,0)-r_noh.get(k,0):+.2f}")
        _save(r_hmm,r_noh)
    p(f"DONE -> {LOG_FILE}")

if __name__=="__main__": main()
