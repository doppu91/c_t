"""18-feature engineering for LightGBM models."""

import numpy as np
import pandas as pd

try:
    import talib
    _TALIB = True
except ImportError:
    _TALIB = False

from utils.helpers import setup_logger

logger = setup_logger("feature_engine")


class FeatureEngine:
    """Computes 18 technical features from OHLCV data."""

    FEATURE_NAMES = [
        "rsi_14",
        "macd_hist",
        "macd_signal",
        "ema_9_20_cross",
        "bb_pct_b",
        "atr_14_pct",
        "adx_14",
        "volume_ratio",
        "obv_slope",
        "vwap_dist",
        "hl_range_pct",
        "candle_body_pct",
        "upper_wick_pct",
        "lower_wick_pct",
        "momentum_10",
        "roc_5",
        "stoch_k",
        "close_vs_ema50",
    ]

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return df with 18 feature columns appended.

        Requires columns: open, high, low, close, volume.
        Minimum 60 rows recommended.
        """
        if len(df) < 30:
            logger.warning(f"Insufficient rows ({len(df)}) for feature computation")
            return df

        df = df.copy()
        close = df["close"].values.astype(float)
        high  = df["high"].values.astype(float)
        low   = df["low"].values.astype(float)
        vol   = df["volume"].values.astype(float)
        op    = df["open"].values.astype(float)

        if _TALIB:
            df["rsi_14"]    = talib.RSI(close, timeperiod=14)
            macd, sig, hist = talib.MACD(close)
            df["macd_hist"]   = hist
            df["macd_signal"] = sig
            ema9            = talib.EMA(close, timeperiod=9)
            ema20           = talib.EMA(close, timeperiod=20)
            ema50           = talib.EMA(close, timeperiod=50)
            bb_up, bb_mid, bb_lo = talib.BBANDS(close, timeperiod=20)
            df["atr_14_pct"] = talib.ATR(high, low, close, timeperiod=14) / close * 100
            df["adx_14"]     = talib.ADX(high, low, close, timeperiod=14)
            df["stoch_k"], _ = talib.STOCH(high, low, close)
        else:
            df["rsi_14"]     = self._rsi(close, 14)
            ema9             = self._ema(close, 9)
            ema20            = self._ema(close, 20)
            ema50            = self._ema(close, 50)
            macd_line        = ema9 - ema20
            sig              = self._ema(macd_line, 9)
            df["macd_hist"]   = macd_line - sig
            df["macd_signal"] = sig
            bb_mid           = self._sma(close, 20)
            bb_std           = pd.Series(close).rolling(20).std().values
            bb_up            = bb_mid + 2 * bb_std
            bb_lo            = bb_mid - 2 * bb_std
            atr              = self._atr(high, low, close, 14)
            df["atr_14_pct"] = atr / close * 100
            df["adx_14"]     = self._adx(high, low, close, 14)
            df["stoch_k"]    = self._stoch_k(high, low, close, 14)

        # BB %B
        bb_range = np.where(bb_up - bb_lo == 0, 1, bb_up - bb_lo)
        df["bb_pct_b"]       = (close - bb_lo) / bb_range

        # EMA cross signal
        df["ema_9_20_cross"] = np.where(ema9 > ema20, 1.0, -1.0)
        df["close_vs_ema50"] = (close - ema50) / ema50 * 100

        # Volume features
        vol_ma20             = pd.Series(vol).rolling(20).mean().values
        df["volume_ratio"]   = vol / np.where(vol_ma20 == 0, 1, vol_ma20)
        obv                  = np.cumsum(np.where(
            np.diff(close, prepend=close[0]) >= 0, vol, -vol
        ))
        df["obv_slope"]      = pd.Series(obv).diff(5).fillna(0).values

        # VWAP distance (intraday approximation)
        typical             = (high + low + close) / 3
        cum_vol             = np.cumsum(vol)
        cum_tpv             = np.cumsum(typical * vol)
        vwap                = cum_tpv / np.where(cum_vol == 0, 1, cum_vol)
        df["vwap_dist"]     = (close - vwap) / vwap * 100

        # Candle anatomy
        hl_range             = high - low
        body                 = np.abs(close - op)
        df["hl_range_pct"]   = hl_range / close * 100
        df["candle_body_pct"] = body / np.where(hl_range == 0, 1, hl_range) * 100
        df["upper_wick_pct"] = (high - np.maximum(close, op)) / np.where(hl_range == 0, 1, hl_range) * 100
        df["lower_wick_pct"] = (np.minimum(close, op) - low) / np.where(hl_range == 0, 1, hl_range) * 100

        # Momentum
        df["momentum_10"]    = pd.Series(close).diff(10).fillna(0).values
        df["roc_5"]          = pd.Series(close).pct_change(5).fillna(0).values * 100

        df.replace([np.inf, -np.inf], 0, inplace=True)
        df.fillna(0, inplace=True)
        return df

    def get_feature_matrix(self, df: pd.DataFrame) -> np.ndarray:
        """Return numpy feature matrix for model inference."""
        df = self.compute(df)
        available = [f for f in self.FEATURE_NAMES if f in df.columns]
        return df[available].values

    # ── Pure-Python TA fallbacks ──────────────────────────────────────────────

    @staticmethod
    def _ema(arr: np.ndarray, period: int) -> np.ndarray:
        result = np.full(len(arr), np.nan)
        k = 2 / (period + 1)
        if len(arr) < period:
            return result
        result[period - 1] = arr[:period].mean()
        for i in range(period, len(arr)):
            result[i] = arr[i] * k + result[i - 1] * (1 - k)
        return result

    @staticmethod
    def _sma(arr: np.ndarray, period: int) -> np.ndarray:
        return pd.Series(arr).rolling(period).mean().values

    @staticmethod
    def _rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
        delta = np.diff(close, prepend=close[0])
        gain = np.where(delta > 0, delta, 0.0)
        loss = np.where(delta < 0, -delta, 0.0)
        avg_gain = pd.Series(gain).rolling(period).mean().values
        avg_loss = pd.Series(loss).rolling(period).mean().values
        rs = avg_gain / np.where(avg_loss == 0, 1e-10, avg_loss)
        return 100 - 100 / (1 + rs)

    @staticmethod
    def _atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
        prev_close = np.roll(close, 1)
        prev_close[0] = close[0]
        tr = np.maximum(high - low, np.maximum(abs(high - prev_close), abs(low - prev_close)))
        return pd.Series(tr).rolling(period).mean().values

    @staticmethod
    def _adx(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
        up   = np.diff(high, prepend=high[0])
        down = -np.diff(low, prepend=low[0])
        plus_dm  = np.where((up > down) & (up > 0), up, 0)
        minus_dm = np.where((down > up) & (down > 0), down, 0)
        prev_close = np.roll(close, 1); prev_close[0] = close[0]
        tr = np.maximum(high - low, np.maximum(abs(high - prev_close), abs(low - prev_close)))
        atr      = pd.Series(tr).ewm(span=period, adjust=False).mean().values
        plus_di  = 100 * pd.Series(plus_dm).ewm(span=period, adjust=False).mean().values / np.where(atr == 0, 1, atr)
        minus_di = 100 * pd.Series(minus_dm).ewm(span=period, adjust=False).mean().values / np.where(atr == 0, 1, atr)
        dx = 100 * abs(plus_di - minus_di) / np.where(plus_di + minus_di == 0, 1, plus_di + minus_di)
        return pd.Series(dx).ewm(span=period, adjust=False).mean().values

    @staticmethod
    def _stoch_k(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
        hi_n = pd.Series(high).rolling(period).max().values
        lo_n = pd.Series(low).rolling(period).min().values
        denom = np.where(hi_n - lo_n == 0, 1, hi_n - lo_n)
        return (close - lo_n) / denom * 100
