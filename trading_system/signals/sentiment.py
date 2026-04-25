"""News sentiment scorer using FinBERT (falls back to rule-based)."""

import re
import time
from typing import Optional

import requests
from bs4 import BeautifulSoup

from utils.helpers import setup_logger

logger = setup_logger("sentiment")

_cache: dict[str, dict] = {}
_TTL = 3600  # 1 hour


class SentimentSignal:
    """Sentiment analysis from financial news headlines."""

    HEADERS = {
        "User-Agent": "Mozilla/5.0 (compatible; TradingBot/1.0)"
    }
    NEWS_SOURCES = [
        "https://economictimes.indiatimes.com/markets/stocks/news",
        "https://www.moneycontrol.com/news/business/markets/",
    ]

    def __init__(self, use_finbert: bool = True) -> None:
        self._pipeline = None
        if use_finbert:
            try:
                from transformers import pipeline
                self._pipeline = pipeline(
                    "text-classification",
                    model="ProsusAI/finbert",
                    device=-1,  # CPU
                    top_k=None,
                )
                logger.info("FinBERT loaded for sentiment analysis")
            except Exception as exc:
                logger.warning(f"FinBERT unavailable ({exc}), using rule-based fallback")

    def score(self, symbol: str) -> float:
        """Return sentiment score 0.0–1.0 for a symbol."""
        cached = _cache.get(symbol)
        if cached and (time.time() - cached["_ts"]) < _TTL:
            return cached["score"]

        headlines = self._fetch_headlines(symbol)
        if not headlines:
            return 0.5

        if self._pipeline:
            s = self._finbert_score(headlines)
        else:
            s = self._rule_based_score(headlines)

        _cache[symbol] = {"score": s, "_ts": time.time()}
        logger.debug(f"Sentiment {symbol}: {s:.3f} ({len(headlines)} headlines)")
        return s

    # ── Internal ──────────────────────────────────────────────────────────────

    def _fetch_headlines(self, symbol: str) -> list[str]:
        headlines: list[str] = []
        for url in self.NEWS_SOURCES:
            try:
                resp = requests.get(url, headers=self.HEADERS, timeout=10)
                soup = BeautifulSoup(resp.text, "html.parser")
                for tag in soup.find_all(["h2", "h3", "a"], limit=50):
                    text = tag.get_text(strip=True)
                    if (symbol.upper() in text.upper() or
                            any(kw in text.lower() for kw in ["nifty", "sensex", "market"])):
                        if len(text) > 20:
                            headlines.append(text)
            except Exception as exc:
                logger.debug(f"News fetch error from {url}: {exc}")
        return list(dict.fromkeys(headlines))[:20]  # deduplicate, cap at 20

    def _finbert_score(self, headlines: list[str]) -> float:
        scores: list[float] = []
        for headline in headlines:
            try:
                result = self._pipeline(headline[:512])  # type: ignore
                probs = {r["label"]: r["score"] for r in result[0]}
                pos = probs.get("positive", 0)
                neg = probs.get("negative", 0)
                # Map to 0–1 (0.5 = neutral)
                scores.append(0.5 + (pos - neg) * 0.5)
            except Exception:
                scores.append(0.5)
        return round(sum(scores) / len(scores), 4) if scores else 0.5

    def _rule_based_score(self, headlines: list[str]) -> float:
        positive_words = {
            "rally", "surge", "gain", "rise", "bullish", "up", "positive",
            "strong", "beat", "outperform", "record", "high", "growth", "profit",
            "buy", "upgrade", "target", "momentum", "breakout",
        }
        negative_words = {
            "fall", "drop", "crash", "loss", "bearish", "down", "negative",
            "weak", "miss", "underperform", "low", "decline", "sell", "downgrade",
            "risk", "concern", "warning", "caution", "volatility",
        }

        total = 0.0
        for h in headlines:
            words = set(re.findall(r"\b\w+\b", h.lower()))
            pos_hits = len(words & positive_words)
            neg_hits = len(words & negative_words)
            total += 0.5 + (pos_hits - neg_hits) * 0.05

        return round(max(0.0, min(1.0, total / len(headlines))), 4) if headlines else 0.5
