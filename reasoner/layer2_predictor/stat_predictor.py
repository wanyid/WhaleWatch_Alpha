"""Reasoner Layer 2 — Statistical baseline predictor.

Two models trained independently on resolved SignalEvent history:
  1. Win-rate classifier: LogisticRegression → P(outcome == WIN)
  2. Holding-period regressor: Ridge regression → holding_period_minutes

Both use the same feature vector from features.py.
Models are persisted to disk and reloaded automatically on startup.

Upgrade path: swap this for nn_predictor.py once ≥ 200 labeled events exist.
"""

import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from models.signal_event import SignalEvent
from reasoner.layer2_predictor.base_predictor import BasePredictor
from reasoner.layer2_predictor.features import (
    FEATURE_NAMES,
    N_FEATURES,
    build_feature_vector,
    events_to_dataframe,
)

logger = logging.getLogger(__name__)

# Default model save path (relative to project root)
DEFAULT_MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "models" / "saved"

# Holding period bounds (minutes)
MIN_HOLD = 1
MAX_HOLD = 4320  # 3 days

# Fallback values used before any training data is available
_FALLBACK_CONFIDENCE = 0.50
_FALLBACK_HOLDING_MIN = 60


class StatPredictor(BasePredictor):
    """Logistic regression win-rate classifier + Ridge holding-period regressor.

    Before sufficient training data exists (< 10 labeled events), the predictor
    returns conservative fallback values: 50% confidence, 60-minute hold.
    This ensures the confidence threshold (≥ 0.60) in the risk manager will
    block all trades until the model has learned something meaningful.

    Args:
        model_dir: Directory to save/load fitted model files.
        min_train_samples: Minimum labeled events required before fitting.
    """

    def __init__(
        self,
        model_dir: Path = DEFAULT_MODEL_DIR,
        min_train_samples: int = 10,
    ) -> None:
        self._model_dir = Path(model_dir)
        self._min_train_samples = min_train_samples
        self._clf: Optional[Pipeline] = None   # win-rate classifier
        self._reg: Optional[Pipeline] = None   # holding-period regressor
        self._trained = False

        # Try loading persisted models on startup
        self._load_if_exists()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def predict(self, event: SignalEvent) -> tuple[float, int]:
        """Return (confidence, holding_period_minutes)."""
        if not self._trained:
            logger.debug("Layer 2 not trained yet — returning fallback values.")
            return _FALLBACK_CONFIDENCE, _FALLBACK_HOLDING_MIN

        x = build_feature_vector(event).reshape(1, -1)

        # Win-rate probability
        confidence = float(self._clf.predict_proba(x)[0, 1])  # P(class=1 / WIN)
        confidence = float(np.clip(confidence, 0.0, 1.0))

        # Holding period (clamp to valid range)
        hold_raw = float(self._reg.predict(x)[0])
        holding = int(np.clip(round(hold_raw), MIN_HOLD, MAX_HOLD))

        logger.debug("Layer 2 predict: confidence=%.3f holding=%dm", confidence, holding)
        return confidence, holding

    def train(self, history: pd.DataFrame) -> None:
        """Fit classifier and regressor on resolved SignalEvent history.

        Args:
            history: DataFrame from events_to_dataframe() with columns
                     matching FEATURE_NAMES plus 'won' and 'holding_minutes'.
        """
        if len(history) < self._min_train_samples:
            logger.warning(
                "Layer 2: only %d labeled events (need ≥ %d) — skipping fit.",
                len(history), self._min_train_samples,
            )
            return

        X = history[FEATURE_NAMES].values.astype(np.float32)
        y_win = history["won"].values.astype(int)
        y_hold = history["holding_minutes"].values.astype(np.float32)

        logger.info(
            "Layer 2 training on %d events | win rate=%.1f%%",
            len(history), y_win.mean() * 100,
        )

        # Win-rate classifier
        self._clf = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                C=1.0,
                class_weight="balanced",    # handles imbalanced win/loss ratio
                max_iter=1000,
                random_state=42,
            )),
        ])
        self._clf.fit(X, y_win)

        # Holding-period regressor
        self._reg = Pipeline([
            ("scaler", StandardScaler()),
            ("reg", Ridge(alpha=1.0)),
        ])
        self._reg.fit(X, y_hold)

        self._trained = True
        self._save()

        # Log training metrics
        win_pred = self._clf.predict(X)
        acc = (win_pred == y_win).mean()
        hold_pred = self._reg.predict(X)
        hold_mae = float(np.abs(hold_pred - y_hold).mean())
        logger.info(
            "Layer 2 fit complete | win accuracy=%.1f%% | hold MAE=%.0f min",
            acc * 100, hold_mae,
        )

    def train_from_events(self, events: list[SignalEvent]) -> None:
        """Convenience wrapper: convert events to DataFrame then train."""
        df = events_to_dataframe(events)
        self.train(df)

    def is_trained(self) -> bool:
        return self._trained

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _model_path(self, name: str) -> Path:
        return self._model_dir / f"stat_predictor_{name}.pkl"

    def _save(self) -> None:
        self._model_dir.mkdir(parents=True, exist_ok=True)
        with open(self._model_path("clf"), "wb") as f:
            pickle.dump(self._clf, f)
        with open(self._model_path("reg"), "wb") as f:
            pickle.dump(self._reg, f)
        logger.info("Layer 2 models saved to %s", self._model_dir)

    def _load_if_exists(self) -> None:
        clf_path = self._model_path("clf")
        reg_path = self._model_path("reg")
        if clf_path.exists() and reg_path.exists():
            try:
                with open(clf_path, "rb") as f:
                    self._clf = pickle.load(f)
                with open(reg_path, "rb") as f:
                    self._reg = pickle.load(f)
                self._trained = True
                logger.info("Layer 2 models loaded from %s", self._model_dir)
            except Exception as exc:
                logger.warning("Failed to load Layer 2 models: %s", exc)
                self._trained = False
