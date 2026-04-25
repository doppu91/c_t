"""LightGBM classifiers — one per regime (Bull + Sideways)."""

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report
from typing import Literal

import lightgbm as lgb

import config
from models.feature_engine import FeatureEngine
from utils.helpers import setup_logger

logger = setup_logger("lgbm_trainer")

_BULL_PATH = config.MODEL_DIR / "lgbm_bull.pkl"
_SIDE_PATH = config.MODEL_DIR / "lgbm_sideways.pkl"

THRESHOLDS = {"Bull": 0.65, "Sideways": 0.72}


class LGBMTrainer:
    """Train and serve LightGBM signal classifiers."""

    def __init__(self) -> None:
        self._models: dict[str, lgb.LGBMClassifier] = {}
        self._feature_engine = FeatureEngine()
        self._load_if_exists()

    # ── Public interface ──────────────────────────────────────────────────────

    def train_all(self) -> None:
        for regime in ("Bull", "Sideways"):
            self.train_regime(regime)

    def train_regime(self, regime: Literal["Bull", "Sideways"]) -> None:
        """Train classifier for a given regime using 5yr historical data."""
        df = self._load_training_data(regime)
        if df.empty:
            logger.warning(f"No training data for {regime} regime — skipping")
            return

        X, y = self._prepare_xy(df)
        if len(np.unique(y)) < 2:
            logger.warning(f"Single-class training data for {regime} — skipping")
            return

        model = lgb.LGBMClassifier(
            n_estimators=500,
            learning_rate=0.03,
            max_depth=6,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=20,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )

        tscv = TimeSeriesSplit(n_splits=5)
        cv_scores = []
        for train_idx, val_idx in tscv.split(X):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
            )
            score = model.score(X_val, y_val)
            cv_scores.append(score)

        logger.info(f"{regime} model CV accuracy: {np.mean(cv_scores):.3f} ± {np.std(cv_scores):.3f}")

        # Final fit on all data
        model.fit(X, y)
        self._models[regime] = model
        path = _BULL_PATH if regime == "Bull" else _SIDE_PATH
        joblib.dump(model, path)
        logger.info(f"{regime} model saved → {path}")

    def predict_proba(self, regime: str, features: np.ndarray) -> float:
        """Return probability of BUY signal for the given regime."""
        model = self._models.get(regime)
        if model is None:
            logger.warning(f"No model for regime {regime}; returning 0.5")
            return 0.5
        if features.ndim == 1:
            features = features.reshape(1, -1)
        proba = model.predict_proba(features)[0]
        return float(proba[1]) if len(proba) > 1 else float(proba[0])

    def get_threshold(self, regime: str) -> float:
        return THRESHOLDS.get(regime, 0.68)

    def is_trained(self, regime: str) -> bool:
        return regime in self._models

    def get_expected_daily_return(self, regime: str) -> float:
        """Historical avg daily return by regime (from backtest metadata)."""
        defaults = {"Bull": 0.85, "Sideways": 0.42, "Bear": 0.0}
        return defaults.get(regime, 0.0)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _load_training_data(self, regime: str) -> pd.DataFrame:
        """Load 5yr OHLCV from cached parquet and label for regime."""
        frames = []
        for symbol in config.WATCHLIST:
            path = config.DATA_DIR / f"{symbol}_day.parquet"
            if not path.exists():
                path = config.DATA_DIR / f"{symbol}_1minute.parquet"
            if not path.exists():
                continue
            try:
                df = pd.read_parquet(path)
                df["symbol"] = symbol
                frames.append(df)
            except Exception as exc:
                logger.warning(f"Failed to load {symbol}: {exc}")

        if not frames:
            logger.warning("No cached parquet files found — downloading via yfinance")
            return self._download_and_prepare(regime)

        combined = pd.concat(frames)
        combined.sort_index(inplace=True)
        return combined

    def _download_and_prepare(self, regime: str) -> pd.DataFrame:
        import yfinance as yf
        dfs = []
        for sym in config.WATCHLIST[:3]:  # limit to avoid rate limits
            try:
                df = yf.download(f"{sym}.NS", period="5y", interval="1d", progress=False)
                df.columns = [c.lower() for c in df.columns]
                dfs.append(df[["open", "high", "low", "close", "volume"]].dropna())
            except Exception:
                pass
        if not dfs:
            return pd.DataFrame()
        return pd.concat(dfs)

    def _prepare_xy(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        df = self._feature_engine.compute(df)

        # Label: 1 if next-bar close > current close by at least 0.3%
        df["target"] = (df["close"].shift(-1) > df["close"] * 1.003).astype(int)
        df.dropna(inplace=True)

        feature_cols = [f for f in FeatureEngine.FEATURE_NAMES if f in df.columns]
        X = df[feature_cols].values
        y = df["target"].values
        return X, y

    def _load_if_exists(self) -> None:
        for regime, path in [("Bull", _BULL_PATH), ("Sideways", _SIDE_PATH)]:
            if path.exists():
                try:
                    self._models[regime] = joblib.load(path)
                    logger.info(f"{regime} LightGBM model loaded from {path}")
                except Exception as exc:
                    logger.warning(f"Could not load {regime} model: {exc}")
