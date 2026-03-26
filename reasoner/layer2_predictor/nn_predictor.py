"""Reasoner Layer 2 — Neural Network predictor.

Drop-in replacement for stat_predictor.py using sklearn MLPClassifier /
MLPRegressor. Activate when ≥ 200 labeled events are available.

Switch via settings.yaml:
    reasoner:
      layer2:
        predictor: nn       # stat (default) | nn

Same public interface as StatPredictor:
  predict(event) → (confidence, holding_period_minutes)
  train(history_df)
  is_trained() → bool
"""

import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from models.signal_event import SignalEvent
from reasoner.layer2_predictor.base_predictor import BasePredictor
from reasoner.layer2_predictor.features import (
    FEATURE_NAMES,
    build_feature_vector,
    events_to_dataframe,
)

logger = logging.getLogger(__name__)

DEFAULT_MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "models" / "saved"

MIN_HOLD = 1
MAX_HOLD = 4320  # 3 days in minutes

_FALLBACK_CONFIDENCE = 0.50
_FALLBACK_HOLDING_MIN = 60

# Recommended minimum before the NN is meaningful
MIN_TRAIN_SAMPLES = 200


class NNPredictor(BasePredictor):
    """Feedforward neural network win-rate classifier + holding-period regressor.

    Architecture:
      Classifier  — two hidden layers (64, 32), ReLU, Adam, early stopping
      Regressor   — same topology; trained on log-transformed holding minutes
                    to handle the wide range [1, 4320]

    The model is persisted to models/saved/nn_predictor_{clf,reg}.pkl and
    reloaded automatically on startup. Until trained, returns fallback values
    (same as StatPredictor) so the confidence gate blocks all trades.

    Args:
        model_dir:         directory for pickle files
        min_train_samples: minimum events before fitting (default 200)
    """

    def __init__(
        self,
        model_dir: Path = DEFAULT_MODEL_DIR,
        min_train_samples: int = MIN_TRAIN_SAMPLES,
    ) -> None:
        self._model_dir = Path(model_dir)
        self._min_train_samples = min_train_samples
        self._clf: Optional[Pipeline] = None
        self._reg: Optional[Pipeline] = None
        self._trained = False
        self._load_if_exists()

    # ------------------------------------------------------------------
    # Public interface (mirrors StatPredictor)
    # ------------------------------------------------------------------

    def predict(self, event: SignalEvent) -> tuple[float, int]:
        """Return (confidence, holding_period_minutes)."""
        if not self._trained:
            logger.debug("NNPredictor not trained yet — returning fallback values.")
            event.confidence = _FALLBACK_CONFIDENCE
            event.holding_period_minutes = _FALLBACK_HOLDING_MIN
            return _FALLBACK_CONFIDENCE, _FALLBACK_HOLDING_MIN

        x = build_feature_vector(event).reshape(1, -1)

        confidence = float(np.clip(self._clf.predict_proba(x)[0, 1], 0.0, 1.0))

        # Regressor trained on log(minutes); inverse-transform and clamp
        log_hold = float(self._reg.predict(x)[0])
        hold_raw = float(np.expm1(max(log_hold, 0.0)))
        holding = int(np.clip(round(hold_raw), MIN_HOLD, MAX_HOLD))

        event.confidence = confidence
        event.holding_period_minutes = holding

        logger.debug("NNPredictor predict: confidence=%.3f holding=%dm", confidence, holding)
        return confidence, holding

    def train(self, history: pd.DataFrame) -> None:
        """Fit the neural network on resolved SignalEvent history.

        Args:
            history: DataFrame from events_to_dataframe() with columns
                     matching FEATURE_NAMES plus 'won' and 'holding_minutes'.
        """
        if len(history) < self._min_train_samples:
            logger.warning(
                "NNPredictor: only %d labeled events (need ≥ %d) — skipping fit.",
                len(history), self._min_train_samples,
            )
            return

        # Ensure all feature columns present
        for f in FEATURE_NAMES:
            if f not in history.columns:
                history[f] = 0.0

        X = history[FEATURE_NAMES].values.astype(np.float32)
        y_win = history["won"].values.astype(int)
        # Log-transform holding minutes for better regression
        y_hold_log = np.log1p(
            history["holding_minutes"].values.clip(MIN_HOLD, MAX_HOLD).astype(np.float32)
        )

        logger.info(
            "NNPredictor training on %d events | win rate=%.1f%%",
            len(history), y_win.mean() * 100,
        )

        self._clf = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", MLPClassifier(
                hidden_layer_sizes=(64, 32),
                activation="relu",
                solver="adam",
                max_iter=500,
                early_stopping=True,
                validation_fraction=0.1,
                random_state=42,
                verbose=False,
            )),
        ])
        self._clf.fit(X, y_win)

        self._reg = Pipeline([
            ("scaler", StandardScaler()),
            ("reg", MLPRegressor(
                hidden_layer_sizes=(64, 32),
                activation="relu",
                solver="adam",
                max_iter=500,
                early_stopping=True,
                validation_fraction=0.1,
                random_state=42,
                verbose=False,
            )),
        ])
        self._reg.fit(X, y_hold_log)

        self._trained = True
        self._save()

        # In-sample metrics
        win_acc = (self._clf.predict(X) == y_win).mean()
        hold_pred_log = self._reg.predict(X)
        hold_pred = np.expm1(hold_pred_log.clip(0))
        hold_mae = float(np.abs(hold_pred - history["holding_minutes"].values.clip(MIN_HOLD, MAX_HOLD)).mean())
        logger.info(
            "NNPredictor fit complete | win accuracy=%.1f%% | hold MAE=%.0f min",
            win_acc * 100, hold_mae,
        )

    def train_from_events(self, events: list[SignalEvent]) -> None:
        df = events_to_dataframe(events)
        self.train(df)

    def is_trained(self) -> bool:
        return self._trained

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _model_path(self, name: str) -> Path:
        return self._model_dir / f"nn_predictor_{name}.pkl"

    def _save(self) -> None:
        self._model_dir.mkdir(parents=True, exist_ok=True)
        with open(self._model_path("clf"), "wb") as f:
            pickle.dump(self._clf, f)
        with open(self._model_path("reg"), "wb") as f:
            pickle.dump(self._reg, f)
        logger.info("NNPredictor models saved to %s", self._model_dir)

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
                logger.info("NNPredictor models loaded from %s", self._model_dir)
            except Exception as exc:
                logger.warning("Failed to load NNPredictor models: %s", exc)
                self._trained = False
