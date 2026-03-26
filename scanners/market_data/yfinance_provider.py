"""yfinance implementation of BaseMarketDataProvider.

Free, no API key required. Suitable for backtesting and historical pulls.
For live trading swap to polygon_provider.py by changing settings.yaml.
"""

import logging

import pandas as pd

from scanners.market_data.base_provider import BaseMarketDataProvider

logger = logging.getLogger(__name__)

# yfinance uses "^VIX" but we accept "VIX" for convenience
_TICKER_MAP = {"VIX": "^VIX"}


class YFinanceProvider(BaseMarketDataProvider):

    def get_ohlcv(
        self,
        ticker: str,
        start: str,
        end: str,
        interval: str = "1h",
    ) -> pd.DataFrame:
        try:
            import yfinance as yf
        except ImportError as exc:
            raise ImportError("yfinance not installed. Run: pip install yfinance") from exc

        yf_ticker = _TICKER_MAP.get(ticker.upper(), ticker.upper())
        logger.info("Fetching %s %s bars from %s to %s", interval, yf_ticker, start, end)

        df = yf.download(
            yf_ticker,
            start=start,
            end=end,
            interval=interval,
            auto_adjust=True,
            progress=False,
        )

        if df.empty:
            logger.warning("No data returned for %s (%s to %s)", yf_ticker, start, end)
            return df

        # Normalize column names to lowercase
        df.columns = [c.lower() if isinstance(c, str) else c[0].lower() for c in df.columns]
        df.index = pd.to_datetime(df.index, utc=True)
        df = df[["open", "high", "low", "close", "volume"]].dropna()
        return df

    def get_latest_price(self, ticker: str) -> float:
        try:
            import yfinance as yf
        except ImportError as exc:
            raise ImportError("yfinance not installed. Run: pip install yfinance") from exc

        yf_ticker = _TICKER_MAP.get(ticker.upper(), ticker.upper())
        data = yf.Ticker(yf_ticker).fast_info
        return float(data.last_price)
