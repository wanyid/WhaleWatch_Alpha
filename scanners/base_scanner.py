from abc import ABC, abstractmethod
from typing import Generator, Any


class BaseScanner(ABC):
    """Abstract base for all signal scanners.

    Each scanner polls an external source and yields raw event dicts.
    Scanners are independent — either scanner alone is sufficient to
    trigger the Reasoner pipeline.
    """

    @abstractmethod
    def scan(self) -> Generator[Any, None, None]:
        """Continuously poll the source and yield raw event objects.

        Implementations should:
        - Run indefinitely (while True loop with sleep)
        - Track state (e.g. last_id) to avoid re-emitting the same event
        - Handle transient errors with retry logic and backoff
        - Yield one raw event object per new item found
        """
        ...

    @abstractmethod
    def name(self) -> str:
        """Human-readable scanner name for logging."""
        ...
