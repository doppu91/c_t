"""Telegram message templates for all 5 message types."""

from datetime import datetime
from typing import Optional

import config
from utils.helpers import now_ist


def morning_brief(
    regime: str,
    global_score: float,
    vix: float,
    sp500_chg: float,
    usdinr: float,
    crude_chg: float,
    fii_net: float,
    top_picks: list[str],
    expected_return_pct: float,
    tradeable: bool,
) -> str:
    status_icon = "🟢" if regime == "Bull" else "🟡" if regime == "Sideways" else "🔴"
    trade_status = "✅ TRADING ENABLED" if tradeable else "🚫 TRADING BLOCKED (VIX HIGH)"
    mode_tag = "[PAPER]" if config.PAPER_MODE else "[LIVE]"

    picks_text = ", ".join(top_picks) if top_picks else "None meeting threshold"

    return f"""🌅 *MORNING BRIEF — {now_ist().strftime('%d %b %Y')}* {mode_tag}

{status_icon} *Regime:* {regime}
📊 *Global Score:* {global_score:.0f}/100
{trade_status}

━━━━━━━━━━━━━━━━━━━━━━
🌍 *Global Cues:*
  • S&P 500: {sp500_chg:+.2f}%
  • USD/INR: ₹{usdinr:.2f}
  • Crude Oil: {crude_chg:+.2f}%
  • India VIX: {vix:.1f}

💼 *FII Net Flow:* ₹{fii_net:.0f} Cr

📈 *Expected Return ({regime}):* {expected_return_pct:.2f}%/day

🎯 *Today's Watchlist:* {picks_text}

💰 *Targets:*
  Gross: ₹{config.TARGET_GROSS_DAILY:,.0f} → Net: ₹{config.TARGET_GROSS_DAILY - 500:,.0f}
  (after ~₹500 charges for 5 trades)

🛡️ *Max Loss Cap:* ₹{config.MAX_DAILY_LOSS:,.0f}

_Market opens at 09:15 IST_"""


def trade_alert(
    action: str,
    symbol: str,
    qty: int,
    entry: float,
    sl: float,
    target: float,
    signal_score: float,
    regime: str,
    gross_if_target: float,
) -> str:
    r_r = round((target - entry) / (entry - sl), 2) if (entry - sl) != 0 else 0
    net_if_target = round(gross_if_target - config.CHARGES_PER_TRADE, 2)
    icon = "🟢" if action == "BUY" else "🔴"
    mode_tag = "[PAPER] " if config.PAPER_MODE else ""

    return f"""{icon} *{mode_tag}TRADE ALERT — {action}*

📌 *{symbol}* | {now_ist().strftime('%H:%M:%S IST')}
Regime: {regime} | Score: {signal_score:.3f}

━━━━━━━━━━━━━━━━━━━━━━
  Entry:    ₹{entry:,.2f}
  Stop:     ₹{sl:,.2f}
  Target:   ₹{target:,.2f}
  Qty:      {qty} shares
  R:R Ratio: 1:{r_r}

💰 *If Target Hit:*
  Gross P&L: ₹{gross_if_target:,.2f}
  Charges:   −₹{config.CHARGES_PER_TRADE:.0f}
  Est. Net:  ₹{net_if_target:,.2f}"""


def exit_alert(
    symbol: str,
    exit_price: float,
    entry_price: float,
    qty: int,
    status: str,
    gross_pnl: float,
    charges: float,
    net_pnl: float,
    daily_net: float,
) -> str:
    pnl_icon = "✅" if net_pnl >= 0 else "❌"
    status_map = {"TARGET": "🎯 Target Hit", "SL": "🛑 Stop Loss", "CLOSED": "🔒 Manual Close"}
    status_text = status_map.get(status, status)
    mode_tag = "[PAPER] " if config.PAPER_MODE else ""

    return f"""{pnl_icon} *{mode_tag}EXIT — {symbol}*

{status_text} | {now_ist().strftime('%H:%M:%S IST')}

━━━━━━━━━━━━━━━━━━━━━━
  Entry:   ₹{entry_price:,.2f}
  Exit:    ₹{exit_price:,.2f}
  Qty:     {qty} shares

📊 *P&L Breakdown:*
  Gross:    ₹{gross_pnl:+,.2f}
  Charges: −₹{charges:.0f}
  *Net:     ₹{net_pnl:+,.2f}*

🗓️ *Today's Net P&L:* ₹{daily_net:+,.2f}"""


def eod_summary(
    num_trades: int,
    num_wins: int,
    num_losses: int,
    gross_pnl: float,
    charges: float,
    net_pnl: float,
    regime: str,
    vix: float,
    target_gross: float,
) -> str:
    win_rate = round(num_wins / num_trades * 100, 1) if num_trades else 0
    target_pct = round(gross_pnl / target_gross * 100, 1) if target_gross else 0
    achieved = "✅" if gross_pnl >= target_gross else "❌"
    mode_tag = "[PAPER] " if config.PAPER_MODE else ""

    return f"""📊 *{mode_tag}EOD SUMMARY — {now_ist().strftime('%d %b %Y')}*

Regime: {regime} | VIX: {vix:.1f}

━━━━━━━━━━━━━━━━━━━━━━
🔢 *Trades:* {num_trades} ({num_wins}W / {num_losses}L) | Win Rate: {win_rate}%

💰 *P&L Breakdown:*
  📈 Gross P&L:   ₹{gross_pnl:+,.2f}
  📉 Charges:    −₹{charges:,.2f} ({num_trades} trades × ₹{config.CHARGES_PER_TRADE:.0f})
  💰 *NET P&L:    ₹{net_pnl:+,.2f}*

🎯 *Target Progress:*
  {achieved} {target_pct}% of ₹{target_gross:,.0f} gross target

📅 *Monthly on Track:* ₹{net_pnl * 22:,.0f}/month projected

_Set aside 25–30% for taxes if profitable_"""


def signal_skip(symbol: str, reason: str, score: float) -> str:
    return (
        f"⏭ *SKIPPED {symbol}* | Score: {score:.3f}\n"
        f"Reason: {reason}"
    )


def system_alert(message: str, level: str = "INFO") -> str:
    icons = {"INFO": "ℹ️", "WARNING": "⚠️", "ERROR": "🚨", "SUCCESS": "✅"}
    return f"{icons.get(level, 'ℹ️')} *SYSTEM:* {message}\n_{now_ist().strftime('%H:%M IST')}_"
