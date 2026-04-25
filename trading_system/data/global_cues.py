"""Global market cues — fetched at 08:30 IST and cached for the session."""

import time
from typing import Optional

import yfinance as yf

import config
from utils.helpers import setup_logger, pct_change

logger = setup_logger("global_cues")

_session_cache: Optional[dict] = None
_cache_date: str = ""


class GlobalCues:
    """Fetch and score overnight global market data."""

    TICKERS = {
        "nifty":   "^NSEI",
        "sp500":   "^GSPC",
        "vix":     "^INDIAVIX",
        "usdinr":  "USDINR=X",
        "crude":   "CL=F",
        "sgx":     "SGX",
        "dow":     "^DJI",
        "nasdaq":  "^IXIC",
    }

    def fetch(self) -> dict:
        """Return cached result if already fetched today."""
        global _session_cache, _cache_date
        today = time.strftime("%Y-%m-%d")
        if _session_cache and _cache_date == today:
            return _session_cache

        result = self._fetch_and_score()
        _session_cache = result
        _cache_date = today
        return result

    def is_tradeable(self) -> bool:
        """Hard block if India VIX > threshold."""
        cues = self.fetch()
        return cues.get("vix_value", 0.0) <= config.INDIA_VIX_MAX

    def _fetch_and_score(self) -> dict:
        data: dict = {}

        for key, ticker_sym in self.TICKERS.items():
            try:
                tk = yf.Ticker(ticker_sym)
                hist = tk.history(period="5d")
                if hist.empty:
                    continue
                close = hist["Close"]
                if len(close) >= 2:
                    prev = float(close.iloc[-2])
                    last = float(close.iloc[-1])
                    data[key] = {"prev": prev, "last": last, "chg": pct_change(prev, last)}
                else:
                    data[key] = {"prev": 0, "last": float(close.iloc[-1]), "chg": 0}
            except Exception as exc:
                logger.warning(f"Failed to fetch {key} ({ticker_sym}): {exc}")

        score = self._score(data)
        vix_val = data.get("vix", {}).get("last", 0.0)
        tradeable = vix_val <= config.INDIA_VIX_MAX

        result = {
            "score": score,
            "tradeable": tradeable,
            "vix_value": vix_val,
            "sp500_chg": data.get("sp500", {}).get("chg", 0.0),
            "nifty_prev_chg": data.get("nifty", {}).get("chg", 0.0),
            "usdinr": data.get("usdinr", {}).get("last", 0.0),
            "crude_chg": data.get("crude", {}).get("chg", 0.0),
            "dow_chg": data.get("dow", {}).get("chg", 0.0),
            "nasdaq_chg": data.get("nasdaq", {}).get("chg", 0.0),
            "raw": data,
            "fetched_at": time.strftime("%Y-%m-%d %H:%M"),
        }
        logger.info(
            f"Global cues: score={score}, VIX={vix_val:.1f}, "
            f"S&P500={data.get('sp500',{}).get('chg',0):.2f}%, tradeable={tradeable}"
        )
        return result

    def _score(self, data: dict) -> float:
        """Score 0–100 based on overnight global sentiment."""
        score = 50.0  # neutral baseline

        # S&P 500
        sp_chg = data.get("sp500", {}).get("chg", 0)
        if sp_chg > 1:
            score += 15
        elif sp_chg > 0.3:
            score += 8
        elif sp_chg < -1:
            score -= 15
        elif sp_chg < -0.3:
            score -= 8

        # NASDAQ
        nq_chg = data.get("nasdaq", {}).get("chg", 0)
        if nq_chg > 1:
            score += 8
        elif nq_chg < -1:
            score -= 8

        # India VIX (lower = better)
        vix = data.get("vix", {}).get("last", 15)
        if vix < 14:
            score += 10
        elif vix < 17:
            score += 5
        elif vix > 20:
            score -= 20
        elif vix > 17:
            score -= 10

        # USD/INR (rupee weakening = negative for market)
        usdinr_chg = data.get("usdinr", {}).get("chg", 0)
        if usdinr_chg > 0.5:
            score -= 5
        elif usdinr_chg < -0.5:
            score += 5

        # Crude (moderate rise = good, spike = bad for India)
        crude_chg = data.get("crude", {}).get("chg", 0)
        if crude_chg > 3:
            score -= 8
        elif crude_chg < -2:
            score -= 3  # demand worry
        elif 0 < crude_chg < 1:
            score += 2

        return round(max(0.0, min(100.0, score)), 1)
