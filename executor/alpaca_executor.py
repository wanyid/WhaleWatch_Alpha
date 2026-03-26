"""Alpaca executor stub — live broker integration (future phase).

Swap in by setting settings.yaml → executor.provider: "alpaca"
and populating ALPACA_API_KEY + ALPACA_SECRET_KEY in .env.

This stub raises NotImplementedError on all methods until implemented.
"""

from typing import Optional

from executor.base_executor import BaseExecutor
from models.signal_event import SignalEvent


class AlpacaExecutor(BaseExecutor):
    """Live broker executor via Alpaca Markets API.

    Implementation checklist (when switching from paper to live):
      1. pip install alpaca-py
      2. Set ALPACA_API_KEY + ALPACA_SECRET_KEY in .env
      3. Set ALPACA_BASE_URL=https://paper-api.alpaca.markets for paper first
      4. Implement submit_signal: place market order, return Alpaca order_id
      5. Implement close_position: place opposing market order
      6. Implement close_expired_positions: query open orders, close elapsed ones
      7. Mirror P&L tracking to SQLite alongside paper executor for comparison
    """

    def __init__(self) -> None:
        raise NotImplementedError(
            "AlpacaExecutor is not yet implemented. "
            "Use PaperExecutor until live trading phase."
        )

    def submit_signal(self, event: SignalEvent) -> str:
        raise NotImplementedError

    def close_position(self, order_id: str, reason: str = "MANUAL") -> Optional[float]:
        raise NotImplementedError

    def close_expired_positions(self) -> int:
        raise NotImplementedError
