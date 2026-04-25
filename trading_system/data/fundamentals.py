"""Fundamental scoring via yfinance — cached for 24 hours."""

import time
from typing import Optional

import yfinance as yf

from utils.helpers import setup_logger

logger = setup_logger("fundamentals")

_cache: dict[str, dict] = {}
_TTL = 86_400  # 24 hours


class Fundamentals:
    """Fetch and score stock fundamentals. Score 0–100."""

    def get_score(self, symbol: str) -> dict:
        """Return fundamental dict with score for the given symbol."""
        cached = _cache.get(symbol)
        if cached and (time.time() - cached["_ts"]) < _TTL:
            return cached

        result = self._fetch_and_score(symbol)
        result["_ts"] = time.time()
        _cache[symbol] = result
        return result

    def _fetch_and_score(self, symbol: str) -> dict:
        ns_symbol = symbol + ".NS"
        try:
            ticker = yf.Ticker(ns_symbol)
            info = ticker.info
        except Exception as exc:
            logger.warning(f"yfinance fetch failed for {symbol}: {exc}")
            return self._default_result(symbol)

        pe = info.get("trailingPE") or 0.0
        roe = (info.get("returnOnEquity") or 0.0) * 100  # convert to %
        de = info.get("debtToEquity") or 0.0
        eps_growth = (info.get("earningsGrowth") or 0.0) * 100

        score = 0

        # P/E score
        if pe > 0:
            if pe < 20:
                score += 35
            elif pe < 30:
                score += 25
            elif pe < 40:
                score += 10

        # ROE score
        if roe > 20:
            score += 30
        elif roe > 15:
            score += 25
        elif roe > 10:
            score += 15

        # Debt-to-Equity
        if 0 <= de < 0.5:
            score += 25
        elif de < 1.0:
            score += 15
        elif de < 2.0:
            score += 5

        # EPS growth
        if eps_growth > 20:
            score += 10
        elif eps_growth > 10:
            score += 7
        elif eps_growth > 0:
            score += 3

        score = min(score, 100)

        result = {
            "symbol": symbol,
            "score": score,
            "pe": round(pe, 2),
            "roe": round(roe, 2),
            "de": round(de, 2),
            "eps_growth": round(eps_growth, 2),
            "last_updated": time.strftime("%Y-%m-%d %H:%M"),
        }
        logger.info(f"Fundamentals {symbol}: score={score}, PE={pe:.1f}, ROE={roe:.1f}%")
        return result

    def _default_result(self, symbol: str) -> dict:
        return {
            "symbol": symbol,
            "score": 50,  # neutral when data unavailable
            "pe": 0.0,
            "roe": 0.0,
            "de": 0.0,
            "eps_growth": 0.0,
            "last_updated": time.strftime("%Y-%m-%d %H:%M"),
        }

    def batch_score(self, symbols: list[str]) -> dict[str, dict]:
        return {sym: self.get_score(sym) for sym in symbols}
