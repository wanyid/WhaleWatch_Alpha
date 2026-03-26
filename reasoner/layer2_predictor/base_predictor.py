from abc import ABC, abstractmethod

import pandas as pd

from models.signal_event import L1Signal, SignalEvent


class BasePredictor(ABC):
    """Abstract interface for Reasoner Layer 2 predictors.

    Layer 2 takes a Layer 1 signal plus a feature vector derived from the
    raw event context, and outputs:
      - confidence: estimated win rate (0.0–1.0)
      - holding_period_minutes: how long to hold the position (1–4320)

    Swap implementations via settings.yaml → reasoner.layer2.predictor.
    """

    @abstractmethod
    def predict(self, event: SignalEvent) -> tuple[float, int]:
        """Return (confidence, holding_period_minutes) for a signal event.

        Args:
            event: A SignalEvent with L1 fields already populated.

        Returns:
            Tuple of (confidence, holding_period_minutes).
            confidence is the estimated win rate in [0.0, 1.0].
            holding_period_minutes is in [1, 4320].
        """
        ...

    @abstractmethod
    def train(self, history: pd.DataFrame) -> None:
        """Train the predictor on a DataFrame of resolved SignalEvents.

        Args:
            history: DataFrame where each row is a resolved SignalEvent.
                     Must contain feature columns and an 'outcome' column
                     with values WIN / LOSS / STOP_OUT.
        """
        ...

    @abstractmethod
    def is_trained(self) -> bool:
        """Return True if the model has been fitted and is ready to predict."""
        ...
