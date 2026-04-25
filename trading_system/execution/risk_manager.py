"""Position sizing and daily risk management."""

import math
from typing import Optional

import pandas as pd

import config
from database.trade_logger import TradeLogger
from utils.helpers import setup_logger, round_to_tick, calc_quantity

logger = setup_logger("risk_manager")


class RiskManager:
    """Computes position size and enforces capital/risk limits."""

    def __init__(self, trade_logger: Optional[TradeLogger] = None) -> None:
        self._logger = trade_logger or TradeLogger()
        self._daily_charges_estimate: float = 0.0

    # ── Position sizing ────────────────────────────────────────────────────────

    def compute_position(
        self,
        symbol: str,
        entry_price: float,
        atr: float,
        direction: str = "BUY",
    ) -> dict:
        """
        Returns position dict:
            quantity, stop_loss, target, risk_amount, trade_value
        """
        stop_distance = atr * 1.5
        if stop_distance <= 0:
            stop_distance = entry_price * 0.005  # 0.5% fallback

        risk_amount = config.CAPITAL * config.RISK_PER_TRADE_PCT  # ₹9,000
        quantity = calc_quantity(risk_amount, stop_distance, entry_price)

        if direction == "BUY":
            stop_loss = round_to_tick(entry_price - stop_distance)
            target = round_to_tick(entry_price + stop_distance * 2)
        else:
            stop_loss = round_to_tick(entry_price + stop_distance)
            target = round_to_tick(entry_price - stop_distance * 2)

        trade_value = quantity * entry_price

        position = {
            "symbol": symbol,
            "direction": direction,
            "entry_price": entry_price,
            "quantity": quantity,
            "stop_loss": stop_loss,
            "target": target,
            "stop_distance": round(stop_distance, 2),
            "risk_amount": round(risk_amount, 2),
            "trade_value": round(trade_value, 2),
            "est_charges": config.CHARGES_PER_TRADE,
            "min_gross_needed": round(config.CHARGES_PER_TRADE * 1.5, 2),
        }
        logger.info(
            f"Position {symbol}: qty={quantity}, entry={entry_price}, "
            f"SL={stop_loss}, target={target}, risk=₹{risk_amount:.0f}"
        )
        return position

    # ── Daily risk gates ──────────────────────────────────────────────────────

    def can_trade(self) -> tuple[bool, str]:
        """Returns (allowed, reason_if_blocked)."""
        daily = self._logger.get_daily_pnl()
        net = daily["net_pnl"]
        trades = daily["num_trades"]

        if net <= -config.MAX_DAILY_LOSS:
            return False, f"Daily loss cap hit: ₹{net:.0f} <= -₹{config.MAX_DAILY_LOSS:.0f}"

        if trades >= 10:
            return False, f"Max daily trades reached: {trades}"

        return True, "OK"

    def is_daily_target_hit(self) -> bool:
        daily = self._logger.get_daily_pnl()
        return daily["gross_pnl"] >= config.TARGET_GROSS_DAILY

    def get_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Compute ATR from OHLCV DataFrame."""
        if len(df) < period + 1:
            close = df["close"].iloc[-1] if len(df) > 0 else 100
            return close * 0.01  # 1% fallback

        high  = df["high"].values
        low   = df["low"].values
        close = df["close"].values
        prev_close = pd.Series(close).shift(1).fillna(close[0]).values

        tr = pd.Series([
            max(h - l, abs(h - pc), abs(l - pc))
            for h, l, pc in zip(high, low, prev_close)
        ])
        return float(tr.rolling(period).mean().iloc[-1])

    # ── Charges tracker ───────────────────────────────────────────────────────

    def add_trade_charges(self) -> None:
        self._daily_charges_estimate += config.CHARGES_PER_TRADE

    def get_today_charges(self) -> float:
        """Estimated charges from open/closed trades today."""
        daily = self._logger.get_daily_pnl()
        return daily["charges"]

    def get_net_needed_for_target(self) -> float:
        """Remaining gross P&L needed to hit today's target."""
        daily = self._logger.get_daily_pnl()
        remaining = config.TARGET_GROSS_DAILY - daily["gross_pnl"]
        return max(0.0, remaining)

    def daily_summary(self) -> dict:
        daily = self._logger.get_daily_pnl()
        return {
            "gross_pnl": daily["gross_pnl"],
            "charges": daily["charges"],
            "net_pnl": daily["net_pnl"],
            "num_trades": daily["num_trades"],
            "target_gross": config.TARGET_GROSS_DAILY,
            "target_pct": round(daily["gross_pnl"] / config.TARGET_GROSS_DAILY * 100, 1)
                          if config.TARGET_GROSS_DAILY else 0,
            "loss_cap": config.MAX_DAILY_LOSS,
            "loss_used_pct": round(abs(min(0, daily["net_pnl"])) / config.MAX_DAILY_LOSS * 100, 1)
                             if config.MAX_DAILY_LOSS else 0,
        }
