"""FII/DII institutional flow signal — scraped from NSE and cached daily."""

import re
import time
from datetime import date

import requests
from bs4 import BeautifulSoup

from utils.helpers import setup_logger

logger = setup_logger("fii_flow")

_cache: dict[str, dict] = {}
_TTL = 3600 * 4  # 4 hours


class FIIFlowSignal:
    """Scores institutional money flow from FII/DII provisional data."""

    NSE_URL = "https://www.nseindia.com/api/fiidiiTradeReact"
    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "application/json",
        "Referer": "https://www.nseindia.com/",
    }

    def score(self) -> float:
        """Return FII/DII flow score 0.0–1.0."""
        key = str(date.today())
        cached = _cache.get(key)
        if cached and (time.time() - cached["_ts"]) < _TTL:
            return cached["score"]

        try:
            data = self._fetch_fii_data()
            s = self._compute_score(data)
        except Exception as exc:
            logger.warning(f"FII flow fetch failed: {exc}. Using neutral 0.5")
            s = 0.5

        _cache[key] = {"score": s, "_ts": time.time(), "raw": _cache.get(key, {}).get("raw", {})}
        return s

    def get_raw(self) -> dict:
        """Return raw FII/DII numbers for display."""
        return _cache.get(str(date.today()), {}).get("raw", {})

    # ── Internal ──────────────────────────────────────────────────────────────

    def _fetch_fii_data(self) -> dict:
        # Try NSE API
        with requests.Session() as sess:
            # Prime the session cookie
            sess.get("https://www.nseindia.com/", headers=self.HEADERS, timeout=10)
            resp = sess.get(self.NSE_URL, headers=self.HEADERS, timeout=15)
            resp.raise_for_status()
            data = resp.json()

        # data is list of records; find equity rows
        fii_buy = fii_sell = dii_buy = dii_sell = 0.0
        for rec in data:
            cat = rec.get("category", "").lower()
            if "fii" in cat or "foreign" in cat:
                fii_buy += float(rec.get("buyValue", 0) or 0)
                fii_sell += float(rec.get("sellValue", 0) or 0)
            elif "dii" in cat or "domestic" in cat:
                dii_buy += float(rec.get("buyValue", 0) or 0)
                dii_sell += float(rec.get("sellValue", 0) or 0)

        result = {
            "fii_net": round(fii_buy - fii_sell, 2),
            "dii_net": round(dii_buy - dii_sell, 2),
            "fii_buy": fii_buy,
            "fii_sell": fii_sell,
        }
        key = str(date.today())
        if key not in _cache:
            _cache[key] = {}
        _cache[key]["raw"] = result
        logger.info(f"FII net: ₹{result['fii_net']:.0f}Cr | DII net: ₹{result['dii_net']:.0f}Cr")
        return result

    def _compute_score(self, data: dict) -> float:
        fii_net = data.get("fii_net", 0)
        dii_net = data.get("dii_net", 0)
        combined_net = fii_net + dii_net * 0.5  # FII weighted higher

        # Score based on net flow in crores
        if combined_net > 3000:
            score = 0.90
        elif combined_net > 1500:
            score = 0.80
        elif combined_net > 500:
            score = 0.70
        elif combined_net > 0:
            score = 0.60
        elif combined_net > -500:
            score = 0.45
        elif combined_net > -1500:
            score = 0.30
        elif combined_net > -3000:
            score = 0.20
        else:
            score = 0.10

        return round(score, 4)
