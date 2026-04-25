"""
Upstox Regime-Adaptive Trading System — main entry point.

Start with:  python main.py
Stop with:   Ctrl+C  OR  Telegram /pause + /close
"""

import signal
import sys
import time
from typing import Optional

import config
from auth.upstox_auth import UpstoxAuth
from data.market_data import MarketData
from data.fundamentals import Fundamentals
from data.global_cues import GlobalCues
from database.trade_logger import TradeLogger
from execution.risk_manager import RiskManager
from execution.upstox_trader import UpstoxTrader
from models.feature_engine import FeatureEngine
from models.lgbm_trainer import LGBMTrainer
from models.regime_hmm import RegimeHMM
from scheduler import TradingScheduler
from signals.fii_flow import FIIFlowSignal
from signals.gex_engine import GEXEngine
from signals.sentiment import SentimentSignal
from signals.signal_combiner import SignalCombiner
from signals.technical import TechnicalSignal
from telegram.bot import TelegramBot, send_message
from telegram import messages as tmsg
from utils.helpers import setup_logger, now_ist, is_market_open

logger = setup_logger("main")

# Global pause flag (toggled by Telegram /pause and /resume commands)
TRADING_PAUSED: bool = False


# ── Component wiring ──────────────────────────────────────────────────────────

def build_system():
    logger.info("Initialising all system components...")

    auth          = UpstoxAuth()
    trade_logger  = TradeLogger()
    market_data   = MarketData(auth)
    fundamentals  = Fundamentals()
    global_cues   = GlobalCues()
    regime_hmm    = RegimeHMM()
    lgbm_trainer  = LGBMTrainer()
    feature_engine= FeatureEngine()
    tech_signal   = TechnicalSignal()
    sentiment     = SentimentSignal(use_finbert=False)  # set True for GPU server
    fii_flow      = FIIFlowSignal()
    gex_engine    = GEXEngine()
    combiner      = SignalCombiner(trade_logger)
    risk_manager  = RiskManager(trade_logger)
    trader        = UpstoxTrader(auth, trade_logger)

    return {
        "auth":           auth,
        "trade_logger":   trade_logger,
        "market_data":    market_data,
        "fundamentals":   fundamentals,
        "global_cues":    global_cues,
        "regime_hmm":     regime_hmm,
        "lgbm_trainer":   lgbm_trainer,
        "feature_engine": feature_engine,
        "tech_signal":    tech_signal,
        "sentiment":      sentiment,
        "fii_flow":       fii_flow,
        "gex_engine":     gex_engine,
        "combiner":       combiner,
        "risk_manager":   risk_manager,
        "trader":         trader,
    }


# ── Core scan loop ────────────────────────────────────────────────────────────

def run_scan(components: dict, regime: str, vix: float) -> None:
    """Called every 5 minutes during market hours."""
    global TRADING_PAUSED

    if TRADING_PAUSED:
        logger.debug("Trading paused — skipping scan")
        return

    can_trade, reason = components["risk_manager"].can_trade()
    if not can_trade:
        logger.info(f"Risk gate blocked: {reason}")
        return

    open_positions = len(components["trade_logger"].get_open_trades())
    cues = components["global_cues"].fetch()

    # Check open positions — monitor for SL/target
    _check_exit_conditions(components, cues.get("vix_value", vix))

    # Scan watchlist for new entries
    if open_positions >= config.MAX_POSITIONS:
        return

    for symbol in config.WATCHLIST:
        try:
            _evaluate_symbol(components, symbol, regime, cues.get("vix_value", vix))
        except Exception as exc:
            logger.error(f"Error evaluating {symbol}: {exc}")


def _evaluate_symbol(components: dict, symbol: str, regime: str, vix: float) -> None:
    mkt = components["market_data"]
    df = mkt.get_candles(symbol, timeframe="5m", bars=100)
    if len(df) < 50:
        return

    # Build signal scores
    tech_score  = components["tech_signal"].score(df)
    sent_score  = components["sentiment"].score(symbol)
    fii_score   = components["fii_flow"].score()
    gex_score   = components["gex_engine"].score()
    global_cues = components["global_cues"].fetch()
    global_score= global_cues.get("score", 50) / 100.0

    fund = components["fundamentals"].get_score(symbol)
    fund_score = fund["score"]

    scores = {
        "technical":  tech_score,
        "sentiment":  sent_score,
        "fii_flow":   fii_score,
        "gex":        gex_score,
        "global_cue": global_score,
    }

    open_positions = len(components["trade_logger"].get_open_trades())

    action, combined, debug = components["combiner"].decide(
        symbol=symbol,
        scores=scores,
        regime=regime,
        fundamental_score=fund_score,
        vix=vix,
        open_positions=open_positions,
    )

    if action != "BUY":
        return

    # Predict with LGBM
    fe = components["feature_engine"]
    df_feat = fe.compute(df)
    feat_matrix = fe.get_feature_matrix(df_feat)
    if len(feat_matrix) == 0:
        return

    lgbm_prob = components["lgbm_trainer"].predict_proba(regime, feat_matrix[-1])
    threshold = components["lgbm_trainer"].get_threshold(regime)

    if lgbm_prob < threshold:
        logger.debug(f"{symbol}: LGBM prob {lgbm_prob:.3f} < threshold {threshold:.3f} — skip")
        return

    # Size position
    atr = components["risk_manager"].get_atr(df)
    entry = mkt.get_live_price(symbol)
    if entry <= 0:
        entry = float(df["close"].iloc[-1])

    pos = components["risk_manager"].compute_position(symbol, entry, atr, "BUY")
    if pos["quantity"] <= 0:
        return

    gross_if_target = (pos["target"] - pos["entry_price"]) * pos["quantity"]

    # Place order
    order_id = components["trader"].place_order(
        symbol=symbol,
        action="BUY",
        qty=pos["quantity"],
        price=pos["entry_price"],
        sl=pos["stop_loss"],
        target=pos["target"],
        signal_score=combined,
        regime=regime,
    )

    if order_id:
        alert = tmsg.trade_alert(
            action="BUY",
            symbol=symbol,
            qty=pos["quantity"],
            entry=pos["entry_price"],
            sl=pos["stop_loss"],
            target=pos["target"],
            signal_score=combined,
            regime=regime,
            gross_if_target=gross_if_target,
        )
        send_message(alert)
        components["risk_manager"].add_trade_charges()


def _check_exit_conditions(components: dict, vix: float) -> None:
    """Monitor open positions and close on target/SL breach."""
    open_trades = components["trade_logger"].get_open_trades()
    mkt = components["market_data"]

    for trade in open_trades:
        symbol = trade["symbol"]
        ltp = mkt.get_live_price(symbol)
        if ltp <= 0:
            continue

        entry = trade["entry_price"]
        sl    = trade["stop_loss"]
        tgt   = trade["target"]
        tid   = trade["id"]
        qty   = trade["quantity"]

        status = None
        if ltp >= tgt:
            status = "TARGET"
        elif ltp <= sl:
            status = "SL"

        if status:
            components["trader"].close_position(symbol)
            gross = (ltp - entry) * qty
            net   = gross - config.CHARGES_PER_TRADE
            daily = components["trade_logger"].get_daily_pnl()

            alert = tmsg.exit_alert(
                symbol=symbol,
                exit_price=ltp,
                entry_price=entry,
                qty=qty,
                status=status,
                gross_pnl=gross,
                charges=config.CHARGES_PER_TRADE,
                net_pnl=net,
                daily_net=daily["net_pnl"],
            )
            send_message(alert)


# ── Morning brief sender ──────────────────────────────────────────────────────

def send_morning_brief(components: dict, regime: str, vix: float) -> None:
    cues = components["global_cues"].fetch()
    fii_data = components["fii_flow"].get_raw()
    expected_ret = components["lgbm_trainer"].get_expected_daily_return(regime)

    brief = tmsg.morning_brief(
        regime=regime,
        global_score=cues.get("score", 50),
        vix=cues.get("vix_value", vix),
        sp500_chg=cues.get("sp500_chg", 0),
        usdinr=cues.get("usdinr", 84),
        crude_chg=cues.get("crude_chg", 0),
        fii_net=fii_data.get("fii_net", 0),
        top_picks=config.WATCHLIST[:5],
        expected_return_pct=expected_ret,
        tradeable=cues.get("tradeable", True),
    )
    send_message(brief)


def send_eod_summary(components: dict, regime: str, vix: float) -> None:
    daily = components["trade_logger"].upsert_daily_summary(regime, vix)
    wins = components["trade_logger"]._count_positive(str(__import__("utils.helpers", fromlist=["today_ist"]).today_ist()))

    summary = tmsg.eod_summary(
        num_trades=daily["num_trades"],
        num_wins=wins,
        num_losses=max(0, daily["num_trades"] - wins),
        gross_pnl=daily["gross_pnl"],
        charges=daily["charges"],
        net_pnl=daily["net_pnl"],
        regime=regime,
        vix=vix,
        target_gross=config.TARGET_GROSS_DAILY,
    )
    send_message(summary)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    logger.info("=" * 60)
    logger.info("  Upstox Regime-Adaptive Trading System v2.0")
    logger.info(f"  Mode: {'PAPER' if config.PAPER_MODE else 'LIVE'}")
    logger.info(f"  Capital: ₹{config.CAPITAL:,.0f}")
    logger.info(f"  Daily target: ₹{config.TARGET_GROSS_DAILY:,.0f} gross → "
                f"₹{config.TARGET_GROSS_DAILY - 500:,.0f} net")
    logger.info(f"  Max loss: ₹{config.MAX_DAILY_LOSS:,.0f} | Max positions: {config.MAX_POSITIONS}")
    logger.info("=" * 60)

    components = build_system()

    # Send startup message
    startup_msg = (
        f"🚀 *Upstox Trading System Online*\n"
        f"Mode: {'PAPER' if config.PAPER_MODE else 'LIVE'} | Capital: ₹{config.CAPITAL:,.0f}\n"
        f"Daily target: ₹{config.TARGET_GROSS_DAILY:,.0f} gross → ₹{config.TARGET_GROSS_DAILY-500:,.0f} net\n"
        f"Max loss: ₹{config.MAX_DAILY_LOSS:,.0f} | Positions: max {config.MAX_POSITIONS}"
    )
    send_message(startup_msg)

    # Start market data stream
    components["market_data"].start_stream()

    # Start Telegram bot
    bot = TelegramBot()
    bot.start()

    # Wire up scheduler
    scheduler = TradingScheduler(
        auth=components["auth"],
        market_data=components["market_data"],
        regime_hmm=components["regime_hmm"],
        lgbm_trainer=components["lgbm_trainer"],
        global_cues=components["global_cues"],
        signal_combiner=components["combiner"],
        trader=components["trader"],
        risk_manager=components["risk_manager"],
        trade_logger=components["trade_logger"],
        telegram_send=send_message,
        morning_brief_fn=lambda r, v: send_morning_brief(components, r, v),
        eod_fn=lambda r, v: send_eod_summary(components, r, v),
        scan_fn=lambda r, v: run_scan(components, r, v),
    )
    scheduler.start()

    # Graceful shutdown
    def _shutdown(sig, frame):
        logger.info("Shutdown signal received — closing all positions...")
        send_message("🛑 System shutting down — closing all positions")
        try:
            components["trader"].close_all_positions()
        except Exception:
            pass
        scheduler.stop()
        components["market_data"].stop_stream()
        sys.exit(0)

    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    logger.info("System running. Press Ctrl+C to stop.")
    while True:
        time.sleep(60)


if __name__ == "__main__":
    main()
