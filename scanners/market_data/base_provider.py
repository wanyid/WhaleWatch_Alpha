from abc import ABC, abstractmethod

import pandas as pd


class BaseMarketDataProvider(ABC):
    """Abstract interface for equity/index market data.

    Implementations must provide OHLCV data for SPY, QQQ, and VIX.
    Swap the provider by changing settings.yaml → data.market_data_provider
    and updating the factory in market_data/__init__.py.
    """

    @abstractmethod
    def get_ohlcv(
        self,
        ticker: str,
        start: str,
        end: str,
        interval: str = "1h",
    ) -> pd.DataFrame:
        """Fetch OHLCV bars for a single ticker.

        Args:
            ticker: e.g. "SPY", "QQQ", "^VIX"
            start:  ISO date string, e.g. "2024-01-01"
            end:    ISO date string, e.g. "2025-12-31"
            interval: bar size — "1m", "5m", "1h", "1d"

        Returns:
            DataFrame with DatetimeIndex (UTC) and columns:
            open, high, low, close, volume
        """
        ...

    @abstractmethod
    def get_latest_price(self, ticker: str) -> float:
        """Return the most recent closing price for a ticker."""
        ...
