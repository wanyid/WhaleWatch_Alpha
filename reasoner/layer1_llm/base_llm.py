from abc import ABC, abstractmethod
from typing import Union

from models.raw_events import PolymarketRawEvent, TruthSocialRawEvent
from models.signal_event import L1Signal


class BaseLLM(ABC):
    """Abstract interface for Reasoner Layer 1 LLM providers.

    Swap implementations by changing settings.yaml → reasoner.layer1.provider.
    Each implementation must return an L1Signal with direction and ticker;
    no free-text reasoning is stored.
    """

    @abstractmethod
    def get_signal(
        self,
        event: Union[TruthSocialRawEvent, PolymarketRawEvent],
    ) -> L1Signal:
        """Analyse a raw scanner event and return a tradeable signal.

        Args:
            event: A TruthSocialRawEvent or PolymarketRawEvent from a scanner.

        Returns:
            L1Signal with direction (BUY/SHORT/HOLD), ticker (SPY/QQQ/VIX),
            model ID, and source event type.
        """
        ...

    @abstractmethod
    def model_id(self) -> str:
        """Return the model identifier string (e.g. 'claude-opus-4-6')."""
        ...
