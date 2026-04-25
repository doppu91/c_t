"""Hidden Markov Model for market regime detection (Bull / Sideways / Bear)."""

import joblib
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from pathlib import Path
from typing import Literal

import config
from utils.helpers import setup_logger

logger = setup_logger("regime_hmm")

RegimeLabel = Literal["Bull", "Sideways", "Bear"]

_MODEL_PATH = config.MODEL_DIR / "regime_hmm.pkl"


class RegimeHMM:
    """3-state Gaussian HMM: Bull / Sideways / Bear."""

    N_COMPONENTS = 3
    N_ITER = 200

    def __init__(self) -> None:
        self._model: GaussianHMM | None = None
        self._regime_map: dict[int, RegimeLabel] = {}
        self._load_if_exists()

    # ── Public interface ──────────────────────────────────────────────────────

    def train(self, df: pd.DataFrame | None = None) -> None:
        """Train HMM on log-return + volatility features of Nifty data."""
        if df is None:
            df = self._load_nifty_data()

        feats = self._build_hmm_features(df)
        logger.info(f"Training HMM on {len(feats)} observations...")

        model = GaussianHMM(
            n_components=self.N_COMPONENTS,
            covariance_type="full",
            n_iter=self.N_ITER,
            random_state=42,
        )
        model.fit(feats)

        # Label regimes by mean return (highest = Bull, lowest = Bear)
        means = model.means_[:, 0]  # first feature is log-return
        order = np.argsort(means)[::-1]  # descending
        label_map: dict[int, RegimeLabel] = {}
        labels: list[RegimeLabel] = ["Bull", "Sideways", "Bear"]
        for rank, state in enumerate(order):
            label_map[int(state)] = labels[rank]

        self._model = model
        self._regime_map = label_map

        joblib.dump({"model": model, "regime_map": label_map}, _MODEL_PATH)
        logger.info(f"HMM saved → {_MODEL_PATH}. Regime map: {label_map}")

    def predict(self, df: pd.DataFrame) -> RegimeLabel:
        """Return current regime label for the given OHLCV DataFrame."""
        if self._model is None:
            logger.warning("HMM not trained — defaulting to Sideways")
            return "Sideways"

        feats = self._build_hmm_features(df)
        if len(feats) < 5:
            return "Sideways"

        state_seq = self._model.predict(feats)
        current_state = int(state_seq[-1])
        regime = self._regime_map.get(current_state, "Sideways")
        logger.debug(f"HMM state={current_state} → {regime}")
        return regime

    def predict_proba(self, df: pd.DataFrame) -> dict[str, float]:
        """Return probability distribution over regimes."""
        if self._model is None:
            return {"Bull": 0.33, "Sideways": 0.34, "Bear": 0.33}

        feats = self._build_hmm_features(df)
        if len(feats) < 5:
            return {"Bull": 0.33, "Sideways": 0.34, "Bear": 0.33}

        log_proba = self._model.predict_proba(feats)[-1]
        result: dict[str, float] = {}
        for state_idx, prob in enumerate(log_proba):
            label = self._regime_map.get(state_idx, "Sideways")
            result[label] = result.get(label, 0) + float(prob)
        return result

    def is_trained(self) -> bool:
        return self._model is not None

    # ── Internal ──────────────────────────────────────────────────────────────

    def _build_hmm_features(self, df: pd.DataFrame) -> np.ndarray:
        close = df["close"].values.astype(float)
        log_ret = np.diff(np.log(close), prepend=np.log(close[0]))
        vol = pd.Series(log_ret).rolling(5).std().fillna(0).values
        hl_ratio = ((df["high"] - df["low"]) / df["close"]).values.astype(float)
        feats = np.column_stack([log_ret, vol, hl_ratio])
        feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
        return feats

    def _load_nifty_data(self) -> pd.DataFrame:
        """Load cached Nifty daily OHLCV or fallback to yfinance."""
        path = config.DATA_DIR / "NIFTY_day.parquet"
        if path.exists():
            return pd.read_parquet(path)

        logger.info("Downloading Nifty 5yr daily data via yfinance for HMM training...")
        import yfinance as yf
        df = yf.download("^NSEI", period="5y", interval="1d", progress=False)
        df.columns = [c.lower() for c in df.columns]
        df = df[["open", "high", "low", "close", "volume"]].dropna()
        df.to_parquet(path)
        return df

    def _load_if_exists(self) -> None:
        if _MODEL_PATH.exists():
            try:
                data = joblib.load(_MODEL_PATH)
                self._model = data["model"]
                self._regime_map = data["regime_map"]
                logger.info(f"HMM loaded from {_MODEL_PATH}")
            except Exception as exc:
                logger.warning(f"Could not load HMM: {exc}")
