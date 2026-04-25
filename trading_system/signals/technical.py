"""Technical signal scorer — returns 0.0–1.0."""

import numpy as np
import pandas as pd
from typing import Optional

try:
    import talib
    _TALIB = True
except ImportError:
    _TALIB = False

from models.feature_engine import FeatureEngine
from utils.helpers import setup_logger

logger = setup_logger("technical")


class TechnicalSignal:
    """Multi-indicator technical analysis signal."""

    def __init__(self) -> None:
        self._fe = FeatureEngine()

    def score(self, df: pd.DataFrame) -> float:
        """Return aggregate technical score 0.0–1.0."""
        if len(df) < 50:
            return 0.5

        try:
            df = self._fe.compute(df)
            score = 0.0
            weight_sum = 0.0

            # ── RSI ─────────────────────────────────────────────────────────
            rsi = df["rsi_14"].iloc[-1]
            w = 0.20
            if 40 <= rsi <= 60:
                s = 0.5
            elif 30 <= rsi < 40 or 60 < rsi <= 70:
                s = 0.7
            elif rsi < 30:
                s = 0.9  # oversold → bullish bounce
            elif rsi > 70:
                s = 0.3  # overbought
            else:
                s = 0.5
            score += w * s; weight_sum += w

            # ── MACD ─────────────────────────────────────────────────────────
            hist = df["macd_hist"].iloc[-1]
            prev_hist = df["macd_hist"].iloc[-2] if len(df) >= 2 else hist
            w = 0.20
            if hist > 0 and hist > prev_hist:
                s = 0.85
            elif hist > 0:
                s = 0.65
            elif hist < 0 and hist < prev_hist:
                s = 0.15
            else:
                s = 0.35
            score += w * s; weight_sum += w

            # ── EMA cross ────────────────────────────────────────────────────
            w = 0.15
            cross = df["ema_9_20_cross"].iloc[-1]
            s = 0.8 if cross == 1.0 else 0.2
            score += w * s; weight_sum += w

            # ── Bollinger Band %B ─────────────────────────────────────────────
            bb_pct = df["bb_pct_b"].iloc[-1]
            w = 0.15
            if 0.2 <= bb_pct <= 0.8:
                s = 0.6
            elif bb_pct < 0.2:
                s = 0.8  # near lower band — potential bounce
            else:
                s = 0.3  # near upper band
            score += w * s; weight_sum += w

            # ── ADX trend strength ───────────────────────────────────────────
            adx = df["adx_14"].iloc[-1]
            w = 0.15
            if adx > 30:
                s = 0.8
            elif adx > 20:
                s = 0.6
            else:
                s = 0.4  # weak trend
            score += w * s; weight_sum += w

            # ── Volume ratio ─────────────────────────────────────────────────
            vol_r = df["volume_ratio"].iloc[-1]
            w = 0.10
            if vol_r > 1.5:
                s = 0.8
            elif vol_r > 1.0:
                s = 0.6
            else:
                s = 0.4
            score += w * s; weight_sum += w

            # ── Stochastic %K ────────────────────────────────────────────────
            sk = df["stoch_k"].iloc[-1]
            w = 0.05
            if sk < 20:
                s = 0.85
            elif sk > 80:
                s = 0.2
            else:
                s = 0.55
            score += w * s; weight_sum += w

            final = round(score / weight_sum if weight_sum > 0 else 0.5, 4)
            logger.debug(f"Technical score: {final:.3f}")
            return final

        except Exception as exc:
            logger.error(f"Technical score error: {exc}")
            return 0.5

    def get_direction(self, df: pd.DataFrame) -> str:
        """Return BUY / SELL / NEUTRAL based on technical score."""
        s = self.score(df)
        if s >= 0.65:
            return "BUY"
        elif s <= 0.35:
            return "SELL"
        return "NEUTRAL"
