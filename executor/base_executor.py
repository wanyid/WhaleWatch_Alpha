"""Abstract executor interface.

All executor implementations (paper, live broker) must implement this.
Swap the concrete class via settings.yaml → executor.provider.
"""

from abc import ABC, abstractmethod
from typing import Optional

from models.signal_event import SignalEvent


class BaseExecutor(ABC):
    @abstractmethod
    def submit_signal(self, event: SignalEvent) -> str:
        """Submit an approved signal. Returns an order_id string."""

    @abstractmethod
    def close_position(self, order_id: str, reason: str = "MANUAL") -> Optional[float]:
        """Close an open position by order_id. Returns realized P&L or None."""

    @abstractmethod
    def close_expired_positions(self) -> int:
        """Sweep open positions and close any that have exceeded their holding period.
        Returns count of positions closed."""
