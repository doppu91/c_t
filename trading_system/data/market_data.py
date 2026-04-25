"""Live market data via Upstox WebSocket + historical OHLCV via HistoryApi."""

import asyncio
import json
import threading
import time
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import upstox_client

import config
from auth.upstox_auth import UpstoxAuth
from utils.helpers import setup_logger, retry

logger = setup_logger("market_data")


class MarketData:
    """Provides live ticks (WebSocket) and historical OHLCV data."""

    def __init__(self, auth: Optional[UpstoxAuth] = None) -> None:
        self._auth = auth or UpstoxAuth()
        self._candles_1m: dict[str, pd.DataFrame] = {}
        self._live_prices: dict[str, float] = {}
        self._tick_buffer: dict[str, list] = defaultdict(list)
        self._streamer: Optional[upstox_client.MarketDataStreamerV3] = None
        self._running = False
        self._lock = threading.Lock()

    # ── Public API ────────────────────────────────────────────────────────────

    def start_stream(self) -> None:
        """Start WebSocket feed in a background thread."""
        if self._running:
            return
        self._running = True
        t = threading.Thread(target=self._stream_loop, daemon=True)
        t.start()
        logger.info("Market data stream started.")

    def stop_stream(self) -> None:
        self._running = False
        if self._streamer:
            try:
                self._streamer.disconnect()
            except Exception:
                pass

    def get_live_price(self, symbol: str) -> float:
        return self._live_prices.get(symbol, 0.0)

    def get_candles(
        self,
        symbol: str,
        timeframe: str = "1m",
        bars: int = 250,
    ) -> pd.DataFrame:
        """Return last N bars. Falls back to historical if live insufficient."""
        with self._lock:
            df = self._candles_1m.get(symbol, pd.DataFrame())

        if timeframe == "5m":
            df = self._resample(df, "5min")
        elif timeframe == "15m":
            df = self._resample(df, "15min")

        if len(df) < bars:
            df = self._fill_from_history(symbol, timeframe, bars, df)

        return df.tail(bars).copy()

    def build_5min_candles(self, symbol: str) -> pd.DataFrame:
        return self.get_candles(symbol, timeframe="5m")

    @retry(max_attempts=3, delay=2.0)
    def get_historical(
        self,
        symbol: str,
        interval: str = "1minute",
        days: int = 365,
    ) -> pd.DataFrame:
        """Fetch OHLCV from Upstox HistoryApi."""
        cfg = self._auth.configure_upstox_client()
        api = upstox_client.HistoryApi(upstox_client.ApiClient(cfg))
        token = config.INSTRUMENT_MAP.get(symbol, "")
        if not token:
            logger.warning(f"No instrument token for {symbol}")
            return pd.DataFrame()

        to_date = datetime.now()
        from_date = to_date - timedelta(days=days)

        resp = api.get_historical_candle_data(
            instrument_key=token,
            interval=interval,
            to_date=to_date.strftime("%Y-%m-%d"),
            from_date=from_date.strftime("%Y-%m-%d"),
        )
        candles = resp.data.candles if resp and resp.data else []
        if not candles:
            return pd.DataFrame()

        df = pd.DataFrame(
            candles,
            columns=["datetime", "open", "high", "low", "close", "volume", "oi"],
        )
        df["datetime"] = pd.to_datetime(df["datetime"])
        df.set_index("datetime", inplace=True)
        df = df[["open", "high", "low", "close", "volume"]].astype(float)
        df.sort_index(inplace=True)
        return df

    def download_historical(self, years: int = 5) -> None:
        """Pre-download history for all watchlist symbols and cache to disk."""
        days = years * 365
        for symbol in config.WATCHLIST:
            logger.info(f"Downloading {years}y history for {symbol}...")
            try:
                df = self.get_historical(symbol, interval="day", days=days)
                if not df.empty:
                    path = config.DATA_DIR / f"{symbol}_day.parquet"
                    df.to_parquet(path)
                    logger.info(f"  Saved {len(df)} rows → {path}")
            except Exception as exc:
                logger.error(f"  Failed for {symbol}: {exc}")

    @retry(max_attempts=3, delay=2.0)
    def get_option_chain(self, index: str = "NIFTY") -> dict:
        """Fetch option chain via nsepython."""
        try:
            from nsepython import nse_optionchain_scrapper
            return nse_optionchain_scrapper(index)
        except Exception as exc:
            logger.warning(f"Option chain fetch failed: {exc}")
            return {}

    # ── WebSocket streaming ───────────────────────────────────────────────────

    def _stream_loop(self) -> None:
        """Reconnect-loop for the WebSocket feed."""
        backoff = 1
        while self._running:
            try:
                self._connect_stream()
                backoff = 1
            except Exception as exc:
                logger.error(f"Stream error: {exc}. Reconnecting in {backoff}s...")
                time.sleep(backoff)
                backoff = min(backoff * 2, 60)

    def _connect_stream(self) -> None:
        cfg = self._auth.configure_upstox_client()

        tokens = list(config.INSTRUMENT_MAP.values())

        self._streamer = upstox_client.MarketDataStreamerV3(
            upstox_client.ApiClient(cfg),
            tokens,
            "full",
        )

        self._streamer.on("message", self._on_tick)
        self._streamer.on("error", lambda e: logger.error(f"WS error: {e}"))
        self._streamer.on("close", lambda: logger.warning("WS closed"))
        self._streamer.connect()

    def _on_tick(self, message: bytes) -> None:
        """Parse incoming tick and update 1-min candle buffer."""
        try:
            from upstox_client.feeder.full_market_feed_pb2 import FeedResponse
            response = FeedResponse()
            response.ParseFromString(message)

            for token, feed in response.feeds.items():
                symbol = self._token_to_symbol(token)
                if not symbol:
                    continue
                ff = feed.ff
                ltp = ff.market_ff.ltpc.ltp
                ts = datetime.now().replace(second=0, microsecond=0)

                self._live_prices[symbol] = ltp
                self._update_candle(symbol, ts, ltp, ff)
        except Exception as exc:
            logger.debug(f"Tick parse error: {exc}")

    def _update_candle(self, symbol: str, ts: datetime, ltp: float, ff) -> None:
        with self._lock:
            if symbol not in self._candles_1m:
                self._candles_1m[symbol] = pd.DataFrame(
                    columns=["open", "high", "low", "close", "volume"]
                )

            df = self._candles_1m[symbol]
            if ts in df.index:
                row = df.loc[ts]
                df.loc[ts, "high"] = max(row["high"], ltp)
                df.loc[ts, "low"] = min(row["low"], ltp)
                df.loc[ts, "close"] = ltp
                try:
                    vol = ff.market_ff.ltpc.vtt
                    df.loc[ts, "volume"] = vol
                except Exception:
                    pass
            else:
                try:
                    vol = ff.market_ff.ltpc.vtt
                except Exception:
                    vol = 0
                df.loc[ts] = [ltp, ltp, ltp, ltp, vol]

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _token_to_symbol(self, token: str) -> Optional[str]:
        for sym, tok in config.INSTRUMENT_MAP.items():
            if tok == token:
                return sym
        return None

    def _resample(self, df: pd.DataFrame, rule: str) -> pd.DataFrame:
        if df.empty:
            return df
        return df.resample(rule).agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }).dropna()

    def _fill_from_history(
        self,
        symbol: str,
        timeframe: str,
        bars: int,
        existing: pd.DataFrame,
    ) -> pd.DataFrame:
        interval_map = {"1m": "1minute", "5m": "5minute", "15m": "30minute", "1d": "day"}
        interval = interval_map.get(timeframe, "1minute")
        needed_days = max(10, bars // 375 + 5)
        hist = self.get_historical(symbol, interval=interval, days=needed_days)
        if hist.empty:
            return existing
        if not existing.empty:
            hist = pd.concat([hist, existing])
            hist = hist[~hist.index.duplicated(keep="last")]
            hist.sort_index(inplace=True)
        return hist
