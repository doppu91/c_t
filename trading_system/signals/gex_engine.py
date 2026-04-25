"""Gamma Exposure (GEX) engine — uses NSE option chain data."""

import time
from typing import Optional

from utils.helpers import setup_logger

logger = setup_logger("gex_engine")

_cache: dict = {}
_TTL = 900  # 15 minutes


class GEXEngine:
    """Computes Gamma Exposure from Nifty option chain to score market bias."""

    def score(self, option_chain: Optional[dict] = None) -> float:
        """Return GEX score 0.0–1.0. Fetches chain if not provided."""
        if not option_chain:
            option_chain = self._fetch_chain()

        if not option_chain:
            return 0.5

        try:
            gex_data = self._compute_gex(option_chain)
            score = self._score_from_gex(gex_data)
            logger.debug(
                f"GEX: net={gex_data['net_gex']:.0f}, "
                f"call_wall={gex_data['call_wall']}, "
                f"put_wall={gex_data['put_wall']}, score={score:.3f}"
            )
            return score
        except Exception as exc:
            logger.warning(f"GEX computation failed: {exc}")
            return 0.5

    def get_levels(self) -> dict:
        """Return key GEX levels for Telegram display."""
        return _cache.get("levels", {})

    # ── Internal ──────────────────────────────────────────────────────────────

    def _fetch_chain(self) -> dict:
        if "chain" in _cache and (time.time() - _cache.get("ts", 0)) < _TTL:
            return _cache["chain"]
        try:
            from nsepython import nse_optionchain_scrapper
            chain = nse_optionchain_scrapper("NIFTY")
            _cache["chain"] = chain
            _cache["ts"] = time.time()
            return chain
        except Exception as exc:
            logger.warning(f"Option chain fetch failed: {exc}")
            return {}

    def _compute_gex(self, chain: dict) -> dict:
        records = chain.get("records", {}).get("data", [])
        spot = float(chain.get("records", {}).get("underlyingValue", 0) or 0)

        call_gex: dict[float, float] = {}
        put_gex: dict[float, float] = {}

        for rec in records:
            strike = float(rec.get("strikePrice", 0))
            expiry_ce = rec.get("CE", {})
            expiry_pe = rec.get("PE", {})

            # Call GEX = gamma × OI × spot² × 0.01 × 50 (lot size)
            if expiry_ce:
                gamma_c = float(expiry_ce.get("gamma", 0) or 0)
                oi_c = float(expiry_ce.get("openInterest", 0) or 0)
                call_gex[strike] = gamma_c * oi_c * (spot ** 2) * 0.01 * 50

            if expiry_pe:
                gamma_p = float(expiry_pe.get("gamma", 0) or 0)
                oi_p = float(expiry_pe.get("openInterest", 0) or 0)
                # Put GEX is negative (dealers short puts = negative gamma)
                put_gex[strike] = -gamma_p * oi_p * (spot ** 2) * 0.01 * 50

        # Net GEX per strike
        all_strikes = set(call_gex) | set(put_gex)
        net_by_strike = {s: call_gex.get(s, 0) + put_gex.get(s, 0) for s in all_strikes}

        net_gex = sum(net_by_strike.values())

        # Zero-gamma level (where net GEX flips sign)
        sorted_strikes = sorted(all_strikes)
        zero_gamma = spot  # default to spot
        for i in range(1, len(sorted_strikes)):
            s1, s2 = sorted_strikes[i - 1], sorted_strikes[i]
            g1, g2 = net_by_strike.get(s1, 0), net_by_strike.get(s2, 0)
            if g1 * g2 < 0:
                # Linear interpolation
                zero_gamma = s1 + (s2 - s1) * abs(g1) / (abs(g1) + abs(g2))
                break

        # Call wall = strike with highest call OI above spot
        call_oi = {s: float(chain.get("records", {}).get("data", [{}])[0].get("CE", {}).get("openInterest", 0) or 0)
                   for s in sorted_strikes if s > spot}
        put_oi = {s: float(0) for s in sorted_strikes if s < spot}

        # Simpler: max call GEX above spot
        above = {s: call_gex.get(s, 0) for s in sorted_strikes if s > spot}
        below = {s: abs(put_gex.get(s, 0)) for s in sorted_strikes if s < spot}
        call_wall = max(above, key=above.get) if above else spot * 1.02
        put_wall = max(below, key=below.get) if below else spot * 0.98

        result = {
            "net_gex": net_gex,
            "zero_gamma": round(zero_gamma, 0),
            "call_wall": call_wall,
            "put_wall": put_wall,
            "spot": spot,
        }
        _cache["levels"] = result
        return result

    def _score_from_gex(self, gex: dict) -> float:
        net = gex["net_gex"]
        spot = gex["spot"]
        call_wall = gex["call_wall"]
        put_wall = gex["put_wall"]

        score = 0.5

        # Positive GEX → market is pinned (mean-reversion), negative → trending
        if net > 0:
            score += 0.10  # dealers are long gamma → stabilizing
        else:
            score -= 0.05  # negative GEX → can accelerate moves

        # Distance from call/put walls
        if spot > 0:
            pct_to_call = (call_wall - spot) / spot * 100
            pct_to_put = (spot - put_wall) / spot * 100

            # More room to call wall = bullish
            if pct_to_call > 2:
                score += 0.15
            elif pct_to_call > 1:
                score += 0.08

            # Closer to put wall = caution
            if pct_to_put < 0.5:
                score -= 0.10
            elif pct_to_put < 1:
                score -= 0.05

        return round(max(0.0, min(1.0, score)), 4)
