"""Telegram bot — command handler + message sender."""

import asyncio
import threading
from typing import Optional

import requests
from telegram import Update, Bot
from telegram.ext import Application, CommandHandler, ContextTypes

import config
from database.trade_logger import TradeLogger
from signals.signal_combiner import SignalCombiner
from utils.helpers import setup_logger, now_ist

logger = setup_logger("telegram_bot")

_trade_logger = TradeLogger()
_combiner = SignalCombiner(_trade_logger)


# ── Async send helper ──────────────────────────────────────────────────────────

def send_message(text: str, parse_mode: str = "Markdown") -> None:
    """Fire-and-forget Telegram send (thread-safe)."""
    if not config.TELEGRAM_BOT_TOKEN or not config.TELEGRAM_CHAT_ID:
        logger.debug(f"Telegram not configured. Message: {text[:80]}")
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{config.TELEGRAM_BOT_TOKEN}/sendMessage",
            json={
                "chat_id": config.TELEGRAM_CHAT_ID,
                "text": text,
                "parse_mode": parse_mode,
            },
            timeout=15,
        )
    except Exception as exc:
        logger.warning(f"Telegram send failed: {exc}")


# ── Command handlers ───────────────────────────────────────────────────────────

async def cmd_status(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    daily = _trade_logger.get_daily_pnl()
    mode = "PAPER" if config.PAPER_MODE else "LIVE"
    msg = (
        f"*System Status* ({now_ist().strftime('%H:%M IST')})\n"
        f"Mode: {mode} | Capital: ₹{config.CAPITAL:,.0f}\n\n"
        f"Today's P&L:\n"
        f"  Trades: {daily['num_trades']}\n"
        f"  Gross: ₹{daily['gross_pnl']:+,.2f}\n"
        f"  Charges: −₹{daily['charges']:,.2f}\n"
        f"  *Net: ₹{daily['net_pnl']:+,.2f}*"
    )
    await update.message.reply_text(msg, parse_mode="Markdown")


async def cmd_positions(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    open_trades = _trade_logger.get_open_trades()
    if not open_trades:
        await update.message.reply_text("No open positions.")
        return
    lines = ["*Open Positions:*"]
    for t in open_trades:
        lines.append(f"  {t['symbol']}: {t['quantity']} @ ₹{t['entry_price']:.2f} | SL={t['stop_loss']:.2f}")
    await update.message.reply_text("\n".join(lines), parse_mode="Markdown")


async def cmd_close(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "⚠️ *CLOSE ALL* command received.\nClosing all positions...",
        parse_mode="Markdown",
    )
    # Import here to avoid circular imports
    from execution.upstox_trader import UpstoxTrader
    trader = UpstoxTrader()
    closed = trader.close_all_positions()
    if closed:
        msg = f"✅ Closed {len(closed)} position(s)."
    else:
        msg = "No positions to close or close failed."
    await update.message.reply_text(msg)


async def cmd_pause(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    # Set a global pause flag that the main loop checks
    import main
    main.TRADING_PAUSED = True
    await update.message.reply_text("⏸ *Trading PAUSED.* No new orders will be placed.", parse_mode="Markdown")


async def cmd_resume(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    import main
    main.TRADING_PAUSED = False
    await update.message.reply_text("▶️ *Trading RESUMED.*", parse_mode="Markdown")


async def cmd_report(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    args = ctx.args or []
    if "weekly" in args:
        rows = _trade_logger.get_weekly_summary()
        lines = ["*Weekly Summary:*"]
        for r in rows:
            lines.append(
                f"  {r['summary_date']}: "
                f"Trades={r['num_trades']} | "
                f"Net=₹{r['net_pnl']:+,.0f}"
            )
        await update.message.reply_text("\n".join(lines) or "No data.", parse_mode="Markdown")
    else:
        daily = _trade_logger.get_daily_pnl()
        await update.message.reply_text(
            f"*Today's Report:*\n"
            f"Gross: ₹{daily['gross_pnl']:+,.2f}\n"
            f"Charges: −₹{daily['charges']:,.2f}\n"
            f"Net: ₹{daily['net_pnl']:+,.2f}\n"
            f"Trades: {daily['num_trades']}",
            parse_mode="Markdown",
        )


async def cmd_charges(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    daily = _trade_logger.get_daily_pnl()
    n = daily["num_trades"]
    total = daily["charges"]
    await update.message.reply_text(
        f"*Today's Charges:*\n"
        f"  Trades completed: {n}\n"
        f"  Total charges: ₹{total:,.2f}\n"
        f"  Per trade avg: ₹{total/n:.2f}" if n else "  No trades yet today.",
        parse_mode="Markdown",
    )


async def cmd_target(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    prog = _combiner.progress_report()
    achieved = "✅" if prog["gross"] >= prog["target_gross"] else "🔄"
    await update.message.reply_text(
        f"*Daily Target Progress:*\n"
        f"  Gross:   ₹{prog['gross']:+,.2f}\n"
        f"  Charges: −₹{prog['charges']:,.2f}\n"
        f"  Net:     ₹{prog['net']:+,.2f}\n\n"
        f"  {achieved} Progress: {prog['target_pct']}% of ₹{prog['target_gross']:,.0f} gross target",
        parse_mode="Markdown",
    )


async def cmd_help(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    help_text = (
        "*Available Commands:*\n"
        "/status    — System status & today's P&L\n"
        "/positions — Open positions\n"
        "/close     — Close ALL positions (emergency)\n"
        "/pause     — Pause new trades\n"
        "/resume    — Resume trading\n"
        "/charges   — Today's charges breakdown\n"
        "/target    — Progress toward daily target\n"
        "/report         — Today's P&L report\n"
        "/report weekly  — Last 7 days summary\n"
        "/help      — This message"
    )
    await update.message.reply_text(help_text, parse_mode="Markdown")


# ── Bot runner ────────────────────────────────────────────────────────────────

class TelegramBot:
    """Wraps python-telegram-bot Application for background polling."""

    def __init__(self) -> None:
        self._app: Optional[Application] = None

    def start(self) -> None:
        if not config.TELEGRAM_BOT_TOKEN:
            logger.warning("Telegram token not configured — bot disabled")
            return
        t = threading.Thread(target=self._run, daemon=True)
        t.start()
        logger.info("Telegram bot started in background thread")

    def _run(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self._start_async())

    async def _start_async(self) -> None:
        self._app = (
            Application.builder()
            .token(config.TELEGRAM_BOT_TOKEN)
            .build()
        )
        handlers = [
            ("status",    cmd_status),
            ("positions", cmd_positions),
            ("close",     cmd_close),
            ("pause",     cmd_pause),
            ("resume",    cmd_resume),
            ("report",    cmd_report),
            ("charges",   cmd_charges),
            ("target",    cmd_target),
            ("help",      cmd_help),
        ]
        for cmd, handler in handlers:
            self._app.add_handler(CommandHandler(cmd, handler))

        logger.info("Telegram bot polling started")
        await self._app.run_polling(drop_pending_updates=True)
