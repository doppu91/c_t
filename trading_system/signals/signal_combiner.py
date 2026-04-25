"""Combines all signal scores into a final trade decision."""

import pandas as pd
from typing import Literal, Optional

import config
from database.trade_logger import TradeLogger
from utils.helpers import setup_logger, net_pnl

logger = setup_logger("signal_combiner")

Action = Literal["BUY", "SELL", "SKIP"]


class SignalCombiner:
    """Weighted signal aggregator with hard gates and charge awareness."""

    def __init__(self, trade_logger: Optional[TradeLogger] = None) -> None:
        self._logger = trade_logger or TradeLogger()

    # ── Main decision ──────────────────────────────────────────────────────────

    def decide(
        self,
        symbol: str,
        scores: dict[str, float],
        regime: str,
        fundamental_score: float,
        vix: float,
        open_positions: int,
    ) -> tuple[Action, float, dict]:
        """
        Returns (action, combined_score, debug_info).
        All hard gates are checked first; any failure → SKIP.
        """
        debug: dict = {
            "symbol": symbol,
            "regime": regime,
            "scores": scores,
            "fundamental_score": fundamental_score,
            "vix": vix,
            "open_positions": open_positions,
        }

        # ── Hard gates ────────────────────────────────────────────────────────
        gate_fail = self._check_hard_gates(
            regime, fundamental_score, vix, open_positions
        )
        if gate_fail:
            debug["gate_blocked"] = gate_fail
            logger.debug(f"{symbol}: BLOCKED by {gate_fail}")
            return "SKIP", 0.0, debug

        # ── Weighted combination ──────────────────────────────────────────────
        combined = self._weighted_score(scores)
        threshold = self._charges_aware_threshold(regime)
        debug["combined"] = combined
        debug["threshold"] = threshold

        # ── Log signal ────────────────────────────────────────────────────────
        action: Action = "BUY" if combined >= threshold else "SKIP"
        self._logger.log_signal(symbol, {**scores, "combined": combined}, regime, action)

        logger.info(
            f"{symbol}: combined={combined:.3f}, threshold={threshold:.3f} → {action}"
        )
        return action, combined, debug

    # ── Helper methods ─────────────────────────────────────────────────────────

    def _check_hard_gates(
        self,
        regime: str,
        fundamental_score: float,
        vix: float,
        open_positions: int,
    ) -> Optional[str]:
        daily_pnl = self._logger.get_daily_pnl()

        if regime == "Bear":
            return "BearRegime"
        if fundamental_score < 50:
            return f"FundamentalScore<50 ({fundamental_score:.0f})"
        if vix > config.INDIA_VIX_MAX:
            return f"VIX>{config.INDIA_VIX_MAX} ({vix:.1f})"
        if daily_pnl["net_pnl"] <= -config.MAX_DAILY_LOSS:
            return f"DailyLossCap (₹{daily_pnl['net_pnl']:.0f})"
        if open_positions >= config.MAX_POSITIONS:
            return f"MaxPositions ({open_positions})"
        return None

    def _weighted_score(self, scores: dict[str, float]) -> float:
        total = 0.0
        weight_sum = 0.0
        for key, weight in config.WEIGHTS.items():
            val = scores.get(key, 0.5)
            total += weight * val
            weight_sum += weight
        return round(total / weight_sum if weight_sum > 0 else 0.5, 4)

    def _charges_aware_threshold(self, regime: str) -> float:
        """Raise threshold slightly when charges are high relative to P&L."""
        base = config.SIGNAL_THRESHOLD  # 0.68
        daily = self._logger.get_daily_pnl()
        num_trades = daily["num_trades"]

        # For Sideways regime always higher threshold
        if regime == "Sideways":
            base = max(base, 0.72)

        # After 4+ trades, we've spent ~₹400 in charges — raise bar
        if num_trades >= 4:
            base = min(base + 0.03, 0.80)
        elif num_trades >= 6:
            base = min(base + 0.05, 0.85)

        return base

    @staticmethod
    def get_estimated_net_pnl(gross_pnl: float, num_trades: int) -> float:
        """Return gross P&L minus estimated charges."""
        return net_pnl(gross_pnl, num_trades)

    def progress_report(self) -> dict:
        """Return current day progress toward daily target."""
        daily = self._logger.get_daily_pnl()
        gross = daily["gross_pnl"]
        charges = daily["charges"]
        net = daily["net_pnl"]
        pct = round(gross / config.TARGET_GROSS_DAILY * 100, 1) if config.TARGET_GROSS_DAILY else 0
        return {
            "gross": gross,
            "charges": charges,
            "net": net,
            "target_gross": config.TARGET_GROSS_DAILY,
            "target_pct": pct,
            "num_trades": daily["num_trades"],
        }
