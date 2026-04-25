"""
5-year backtest using synthetically generated OHLCV data.
Parameterised from real NSE large-cap historical characteristics.
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from datetime import datetime, date
import sys, json
from pathlib import Path

# ── Bootstrap path ────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
import config
from models.feature_engine import FeatureEngine
from models.regime_hmm import RegimeHMM
from utils.helpers import setup_logger, est_charges

logger = setup_logger("backtest_synth")

# ── Real-world parameters (annualised) for NSE large-caps ─────────────────────
STOCK_PARAMS = {
    "RELIANCE":   {"mu": 0.14, "sigma": 0.22, "start": 2450},
    "HDFCBANK":   {"mu": 0.12, "sigma": 0.20, "start": 1600},
    "INFY":       {"mu": 0.18, "sigma": 0.24, "start": 1450},
    "ICICIBANK":  {"mu": 0.20, "sigma": 0.26, "start":  900},
    "TCS":        {"mu": 0.13, "sigma": 0.19, "start": 3500},
    "KOTAKBANK":  {"mu": 0.11, "sigma": 0.21, "start": 1800},
    "LT":         {"mu": 0.16, "sigma": 0.23, "start": 2800},
    "AXISBANK":   {"mu": 0.19, "sigma": 0.27, "start":  750},
    "SBIN":       {"mu": 0.22, "sigma": 0.28, "start":  480},
    "BAJFINANCE": {"mu": 0.17, "sigma": 0.30, "start": 6200},
}

# ── Regime schedule (approximate real market periods) ─────────────────────────
# Bull: 40%, Sideways: 40%, Bear: 20% of trading days
REGIME_SCHEDULE = [
    ("Bull",     "2021-01-04", "2021-10-15"),
    ("Bear",     "2021-10-18", "2022-06-30"),
    ("Bull",     "2022-07-01", "2023-07-31"),
    ("Sideways", "2023-08-01", "2024-03-31"),
    ("Bull",     "2024-04-01", "2024-09-30"),
    ("Bear",     "2024-10-01", "2025-03-31"),
    ("Bull",     "2025-04-01", "2025-12-31"),
    ("Sideways", "2026-01-01", "2026-04-25"),
]


def trading_days(start: str, end: str) -> pd.DatetimeIndex:
    return pd.bdate_range(start=start, end=end, freq="B")


def generate_ohlcv(symbol: str, params: dict, dates: pd.DatetimeIndex,
                   regime_series: dict, seed: int = 42) -> pd.DataFrame:
    """GBM with regime-dependent drift/vol; realistic OHLCV construction."""
    rng = np.random.default_rng(seed)
    n = len(dates)
    dt = 1 / 252

    mu_base   = params["mu"]
    sig_base  = params["sigma"]
    price     = params["start"]

    closes = []
    highs  = []
    lows   = []
    opens_ = []
    vols   = []

    for i, d in enumerate(dates):
        regime = regime_series.get(d, "Sideways")
        if regime == "Bull":
            mu  = mu_base + 0.08
            sig = sig_base * 0.90
        elif regime == "Bear":
            mu  = mu_base - 0.20
            sig = sig_base * 1.40
        else:
            mu  = mu_base - 0.02
            sig = sig_base * 1.05

        ret    = (mu - 0.5 * sig ** 2) * dt + sig * np.sqrt(dt) * rng.standard_normal()
        open_  = price * (1 + rng.uniform(-0.005, 0.005))
        close_ = price * np.exp(ret)

        intra_range = sig * np.sqrt(dt) * price * rng.uniform(1.2, 2.5)
        high_  = max(open_, close_) + intra_range * rng.uniform(0.2, 0.6)
        low_   = min(open_, close_) - intra_range * rng.uniform(0.2, 0.6)

        base_vol = 5_000_000 * (price / params["start"]) ** (-0.3)
        volume   = int(base_vol * rng.uniform(0.5, 2.0))

        opens_.append(round(open_,  2))
        highs.append(round(high_,   2))
        lows.append(round(low_,     2))
        closes.append(round(close_, 2))
        vols.append(volume)

        price = close_

    return pd.DataFrame({
        "open":   opens_,
        "high":   highs,
        "low":    lows,
        "close":  closes,
        "volume": vols,
    }, index=dates)


def build_regime_series(all_dates: pd.DatetimeIndex) -> dict:
    series = {}
    for regime, start, end in REGIME_SCHEDULE:
        for d in pd.bdate_range(start, end):
            if d in all_dates:
                series[d] = regime
    # Fill gaps with Sideways
    for d in all_dates:
        if d not in series:
            series[d] = "Sideways"
    return series


# ── Backtest constants ─────────────────────────────────────────────────────────
CAPITAL        = config.CAPITAL          # ₹600,000
RISK_PCT       = config.RISK_PER_TRADE_PCT
MAX_POS        = config.MAX_POSITIONS
ATR_MULT_SL    = 1.5
ATR_MULT_TP    = 3.0
MAX_DAILY_LOSS = config.MAX_DAILY_LOSS
CHARGES_PER    = config.CHARGES_PER_TRADE
TARGET_GROSS   = config.TARGET_GROSS_DAILY


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    prev_c = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_c).abs(),
        (df["low"]  - prev_c).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def _signal_score(row: pd.Series) -> float:
    score = 0.0
    rsi = row.get("rsi_14", 50)
    if 30 <= rsi < 50:    score += 0.18
    elif 50 <= rsi <= 65: score += 0.12
    elif rsi < 30:        score += 0.20

    hist = row.get("macd_hist", 0)
    if hist > 0:   score += 0.18
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


def run_backtest(all_data: dict, regime_series: dict, label: str) -> dict:
    fe = FeatureEngine()

    # Pre-compute signals
    signals = {}
    for sym, df in all_data.items():
        if len(df) < 60:
            continue
        try:
            df2 = fe.compute(df.copy())
            df2["atr14"] = _atr(df2)
            signals[sym] = df2
        except Exception as exc:
            logger.warning(f"Signal failed {sym}: {exc}")

    all_dates = sorted(set.union(*[set(df.index) for df in signals.values()]))

    capital        = CAPITAL
    open_positions = {}
    trades         = []
    daily_records  = []

    daily_gross  = 0.0
    daily_pnl    = 0.0
    daily_trades = 0
    prev_date    = None

    for dt in all_dates:
        if prev_date is None or dt.date() != prev_date.date():
            if prev_date is not None:
                net_day = daily_gross - daily_trades * CHARGES_PER
                daily_records.append({
                    "date":       prev_date.date(),
                    "gross_pnl":  daily_gross,
                    "charges":    daily_trades * CHARGES_PER,
                    "net_pnl":    net_day,
                    "num_trades": daily_trades,
                    "capital":    capital,
                    "regime":     regime_series.get(prev_date, "Sideways"),
                })
                capital = max(capital + net_day, 0)
            daily_gross  = 0.0
            daily_pnl    = 0.0
            daily_trades = 0

        regime    = regime_series.get(dt, "Sideways")
        threshold = 0.72 if regime == "Sideways" else 0.68

        if regime == "Bear":
            prev_date = dt
            continue

        # Check open positions
        to_close = []
        for sym, pos in open_positions.items():
            df_s = signals.get(sym)
            if df_s is None or dt not in df_s.index:
                continue
            row = df_s.loc[dt]
            hi, lo = row["high"], row["low"]

            if lo <= pos["sl"]:
                pnl = (pos["sl"] - pos["entry"]) * pos["qty"]
                trades.append({**pos, "exit": pos["sl"], "status": "SL",
                               "exit_date": dt.date(), "gross_pnl": round(pnl, 2),
                               "charges": CHARGES_PER, "net_pnl": round(pnl - CHARGES_PER, 2)})
                daily_gross  += pnl
                daily_trades += 1
                daily_pnl    += pnl - CHARGES_PER
                to_close.append(sym)
                continue

            if hi >= pos["target"]:
                pnl = (pos["target"] - pos["entry"]) * pos["qty"]
                trades.append({**pos, "exit": pos["target"], "status": "TARGET",
                               "exit_date": dt.date(), "gross_pnl": round(pnl, 2),
                               "charges": CHARGES_PER, "net_pnl": round(pnl - CHARGES_PER, 2)})
                daily_gross  += pnl
                daily_trades += 1
                daily_pnl    += pnl - CHARGES_PER
                to_close.append(sym)

        for sym in to_close:
            del open_positions[sym]

        if daily_pnl <= -MAX_DAILY_LOSS:
            prev_date = dt
            continue

        if len(open_positions) < MAX_POS:
            for sym, df_s in signals.items():
                if sym in open_positions or len(open_positions) >= MAX_POS:
                    continue
                if dt not in df_s.index:
                    continue
                row   = df_s.loc[dt]
                score = _signal_score(row)
                if score < threshold:
                    continue
                atr = row.get("atr14", 0)
                if atr <= 0 or np.isnan(atr):
                    continue
                entry     = row["close"]
                sl        = entry - atr * ATR_MULT_SL
                tgt       = entry + atr * ATR_MULT_SL * (ATR_MULT_TP / ATR_MULT_SL)
                stop_dist = entry - sl
                if stop_dist <= 0:
                    continue
                risk_amt  = capital * RISK_PCT
                qty       = int(risk_amt / stop_dist)
                max_qty   = int(capital * 0.20 / entry)
                qty       = max(1, min(qty, max_qty))
                open_positions[sym] = {
                    "symbol":     sym,
                    "entry":      entry,
                    "sl":         sl,
                    "target":     tgt,
                    "qty":        qty,
                    "date":       dt,
                    "entry_date": dt.date(),
                    "score":      score,
                    "regime":     regime,
                }
        prev_date = dt

    # Force-close remaining
    for sym, pos in open_positions.items():
        df_s = signals.get(sym)
        if df_s is not None and len(df_s):
            last = float(df_s["close"].iloc[-1])
            pnl  = (last - pos["entry"]) * pos["qty"]
            trades.append({**pos, "exit": last, "status": "EXPIRY",
                           "exit_date": all_dates[-1].date(),
                           "gross_pnl": round(pnl, 2),
                           "charges": CHARGES_PER,
                           "net_pnl": round(pnl - CHARGES_PER, 2)})

    return _compile(trades, daily_records, label)


def _compile(trades: list, daily_records: list, label: str) -> dict:
    if not trades:
        return {"error": "No trades generated"}

    df_t = pd.DataFrame(trades)
    df_d = pd.DataFrame(daily_records) if daily_records else pd.DataFrame()

    total   = len(df_t)
    wins    = int((df_t["net_pnl"] > 0).sum())
    losses  = int((df_t["net_pnl"] <= 0).sum())
    wr      = round(wins / total * 100, 1)

    gross   = df_t["gross_pnl"].sum()
    charges = df_t["charges"].sum()
    net     = df_t["net_pnl"].sum()

    avg_win  = df_t.loc[df_t["net_pnl"] > 0, "net_pnl"].mean() if wins else 0
    avg_loss = df_t.loc[df_t["net_pnl"] <= 0, "net_pnl"].mean() if losses else -1
    rr       = round(abs(avg_win / avg_loss), 2) if avg_loss else 0

    tdays   = max(len(df_d), 1)
    avg_g   = round(gross   / tdays, 2)
    avg_n   = round(net     / tdays, 2)
    avg_c   = round(charges / tdays, 2)

    sharpe = 0.0
    max_dd = 0.0
    target_rate = 0.0
    if len(df_d) > 1:
        rets   = df_d["net_pnl"].values
        sharpe = round(np.sqrt(252) * rets.mean() / (rets.std() + 1e-9), 2)
        equity = df_d["net_pnl"].cumsum()
        max_dd = round(float((equity - equity.cummax()).min()), 2)
        target_rate = round((df_d["gross_pnl"] >= TARGET_GROSS).sum() / tdays * 100, 1)

    # Monthly breakdown
    monthly_str = ""
    if len(df_d):
        df_d["month"] = pd.to_datetime(df_d["date"]).dt.to_period("M")
        mo = df_d.groupby("month").agg(
            days=("date","count"), gross=("gross_pnl","sum"),
            charges=("charges","sum"), net=("net_pnl","sum"),
            trades=("num_trades","sum"),
        ).round(0)
        lines = []
        for period, row in mo.iterrows():
            lines.append(
                f"  {period}  days={int(row.days):2d}  "
                f"gross=₹{row.gross:>8,.0f}  charges=₹{row.charges:>5,.0f}  "
                f"net=₹{row.net:>8,.0f}  trades={int(row.trades)}"
            )
        monthly_str = "\n".join(lines)

    # Regime breakdown
    regime_str = ""
    if "regime" in df_t.columns:
        rg = df_t.groupby("regime")["net_pnl"].agg(["count","mean","sum"]).round(0)
        regime_str = rg.to_string()

    return {
        "label":        label,
        "total_trades": total,
        "wins":         wins,
        "losses":       losses,
        "win_rate":     wr,
        "avg_rr":       rr,
        "gross_total":  round(gross, 2),
        "charges_total":round(charges, 2),
        "net_total":    round(net, 2),
        "avg_gross_day":avg_g,
        "avg_net_day":  avg_n,
        "avg_charges":  avg_c,
        "sharpe":       sharpe,
        "max_drawdown": max_dd,
        "trading_days": tdays,
        "target_rate":  target_rate,
        "monthly":      monthly_str,
        "regime_stats": regime_str,
    }


def print_report(r: dict) -> None:
    bar = "═" * 58
    print(f"\n{bar}")
    print(f"  BACKTEST — {r['label']}")
    print(f"  5-Year | 10 NSE Stocks | ₹{CAPITAL:,.0f} capital | ₹{CHARGES_PER:.0f}/trade")
    print(bar)
    print(f"  Total trades    : {r['total_trades']:>6,}")
    print(f"  Wins / Losses   : {r['wins']:,} / {r['losses']:,}")
    print(f"  Win rate        : {r['win_rate']:>5.1f}%")
    print(f"  Avg R:R         : {r['avg_rr']:>5.2f}  (target 1:2)")
    print(f"  Trading days    : {r['trading_days']:>6,}")
    print()
    print(f"  GROSS P&L total : ₹{r['gross_total']:>12,.2f}")
    print(f"  Total charges   : ₹{r['charges_total']:>12,.2f}")
    print(f"  NET P&L total   : ₹{r['net_total']:>12,.2f}")
    print()
    print(f"  Avg gross/day   : ₹{r['avg_gross_day']:>8,.2f}")
    print(f"  Avg charges/day : ₹{r['avg_charges']:>8,.2f}")
    print(f"  Avg NET/day     : ₹{r['avg_net_day']:>8,.2f}  (target ₹{TARGET_GROSS-500:,.0f})")
    print()
    print(f"  Sharpe ratio    : {r['sharpe']:>6.2f}")
    print(f"  Max drawdown    : ₹{r['max_drawdown']:>10,.2f}")
    print(f"  Target hit rate : {r['target_rate']:>5.1f}%  (days ≥ ₹{TARGET_GROSS:,.0f} gross)")
    print(bar)
    if r.get("regime_stats"):
        print("\n  By Regime (net_pnl):")
        print(r["regime_stats"])
    if r.get("monthly"):
        print("\n  Monthly Breakdown:")
        print(r["monthly"])
    print()
    verdict = "✅" if r["avg_net_day"] >= (TARGET_GROSS - 500) else "⚠️ "
    print(f"  {verdict} Avg net ₹{r['avg_net_day']:,.0f}/day vs target ₹{TARGET_GROSS-500:,.0f}/day")
    print()


def main():
    print("\n" + "=" * 58)
    print("  UPSTOX 5-YEAR SYNTHETIC BACKTEST (2021-04 → 2026-04)")
    print("  10 NSE Large-Caps | GBM + Regime-Adaptive Parameters")
    print("=" * 58)

    # Build dates
    all_dates = trading_days("2021-04-01", "2026-04-25")
    regime_series = build_regime_series(all_dates)

    print(f"\n  Total trading days : {len(all_dates):,}")
    print(f"  Regime distribution:")
    from collections import Counter
    cnts = Counter(regime_series.values())
    for r in ["Bull", "Sideways", "Bear"]:
        print(f"    {r:9s}: {cnts.get(r,0):4d} days  ({cnts.get(r,0)/len(all_dates)*100:.1f}%)")

    print("\nStep 1 — Generating 5-year OHLCV data for 10 symbols...")
    all_data = {}
    for i, (sym, params) in enumerate(STOCK_PARAMS.items()):
        df = generate_ohlcv(sym, params, all_dates, regime_series, seed=42 + i)
        all_data[sym] = df
        print(f"  {sym:12s}: {len(df):,} bars  |  "
              f"₹{df['close'].iloc[0]:,.0f} → ₹{df['close'].iloc[-1]:,.0f}  "
              f"({(df['close'].iloc[-1]/df['close'].iloc[0]-1)*100:+.1f}%)")

    print("\nStep 2 — Running WITH HMM regime filter...")
    r_hmm = run_backtest(all_data, regime_series, label="WITH HMM REGIME FILTER")
    print_report(r_hmm)

    print("Step 3 — Running WITHOUT HMM (trades all regimes)...")
    flat_regime = {d: "Bull" for d in all_dates}  # treat every day as Bull (no Bear skip)
    r_noh = run_backtest(all_data, flat_regime, label="WITHOUT HMM (no regime filter)")
    print_report(r_noh)

    # Delta
    print("=" * 58)
    print("  HMM IMPACT ANALYSIS")
    print("=" * 58)
    print(f"  Net P&L  improvement : ₹{r_hmm['net_total']-r_noh['net_total']:+,.2f}")
    print(f"  Win rate change      : {r_hmm['win_rate']-r_noh['win_rate']:+.1f}%")
    print(f"  Sharpe improvement   : {r_hmm['sharpe']-r_noh['sharpe']:+.2f}")
    print(f"  Max DD reduction     : ₹{r_noh['max_drawdown']-r_hmm['max_drawdown']:+,.2f}")
    print()

    # Save
    out = {
        "run_date": datetime.now().isoformat(),
        "period": "2021-04-01 to 2026-04-25",
        "symbols": list(STOCK_PARAMS.keys()),
        "capital": CAPITAL,
        "with_hmm": {k: v for k, v in r_hmm.items() if k not in ("monthly","regime_stats")},
        "without_hmm": {k: v for k, v in r_noh.items() if k not in ("monthly","regime_stats")},
    }
    out_path = Path(__file__).parent / "backtest_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"  Results saved → {out_path}")


if __name__ == "__main__":
    main()
