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
    def close_expired_positions(self) -> list:
        """Sweep open positions and close any that have exceeded their holding period.
        Returns list of realized P&L values for every position closed."""

    @abstractmethod
    def open_positions(self) -> list:
        """Return all currently open positions as a list of dicts."""

    @abstractmethod
    def check_true_news_stop(self, market_id: str, current_prob: float) -> list:
        """Apply True News Stop for open positions tied to a Polymarket market.
        Returns list of order_ids closed."""

    @abstractmethod
    def session_summary(self) -> dict:
        """Return today's P&L summary as a dict."""
