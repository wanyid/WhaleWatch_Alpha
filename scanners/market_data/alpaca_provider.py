"""Alpaca Markets data provider — free historical 1-minute bars.

Provides 1-minute OHLCV going back to 2015 for US equities (SPY, QQQ).
VIX is an index and not available via Alpaca — use YFinanceProvider for VIX.

Requires a free Alpaca account: https://alpaca.markets
Set ALPACA_API_KEY and ALPACA_SECRET_KEY in .env (paper or live keys both work).
"""

import logging
import os
from datetime import datetime, timezone

import pandas as pd

from scanners.market_data.base_provider import BaseMarketDataProvider

logger = logging.getLogger(__name__)

# Alpaca interval string mapping
_INTERVAL_MAP = {
    "1m":  "1Min",
    "5m":  "5Min",
    "15m": "15Min",
    "30m": "30Min",
    "1h":  "1Hour",
    "1d":  "1Day",
}


class AlpacaProvider(BaseMarketDataProvider):
    """Historical market data from Alpaca (free tier, IEX feed).

    Supports SPY, QQQ, and most US equities at 1-minute resolution.
    Does NOT support VIX — use YFinanceProvider for index data.
    """

    def __init__(self) -> None:
        self._client = self._build_client()

    def get_ohlcv(
        self,
        ticker: str,
        start: str,
        end: str,
        interval: str = "1m",
    ) -> pd.DataFrame:
        timeframe = _INTERVAL_MAP.get(interval)
        if timeframe is None:
            raise ValueError(f"Unsupported interval '{interval}'. Use: {list(_INTERVAL_MAP)}")

        logger.info("Alpaca: fetching %s %s bars %s → %s", interval, ticker, start, end)

        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

        # Parse timeframe
        tf_map = {
            "1Min": TimeFrame(1, TimeFrameUnit.Minute),
            "5Min": TimeFrame(5, TimeFrameUnit.Minute),
            "15Min": TimeFrame(15, TimeFrameUnit.Minute),
            "30Min": TimeFrame(30, TimeFrameUnit.Minute),
            "1Hour": TimeFrame(1, TimeFrameUnit.Hour),
            "1Day": TimeFrame(1, TimeFrameUnit.Day),
        }
        tf = tf_map[timeframe]

        request = StockBarsRequest(
            symbol_or_symbols=ticker.upper(),
            timeframe=tf,
            start=start,
            end=end,
            feed="iex",         # free tier feed
        )

        bars = self._client.get_stock_bars(request)
        df = bars.df

        if df.empty:
            logger.warning("No data returned for %s (%s to %s)", ticker, start, end)
            return df

        # bars.df has a MultiIndex (symbol, timestamp) when one symbol is passed
        if isinstance(df.index, pd.MultiIndex):
            df = df.droplevel(0)

        df.index = pd.to_datetime(df.index, utc=True)
        df.index.name = "timestamp"

        # Normalize to standard column names
        df = df.rename(columns={
            "open": "open", "high": "high", "low": "low",
            "close": "close", "volume": "volume",
        })
        keep = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
        return df[keep].sort_index()

    def get_latest_price(self, ticker: str) -> float:
        from alpaca.data.requests import StockLatestQuoteRequest

        req = StockLatestQuoteRequest(symbol_or_symbols=ticker.upper(), feed="iex")
        quote = self._client.get_stock_latest_quote(req)
        return float(quote[ticker.upper()].ask_price)

    def _build_client(self):
        try:
            from alpaca.data import StockHistoricalDataClient
        except ImportError as exc:
            raise ImportError(
                "alpaca-py not installed. Run: pip install alpaca-py"
            ) from exc

        api_key = os.environ.get("ALPACA_API_KEY")
        secret_key = os.environ.get("ALPACA_SECRET_KEY")
        if not api_key or not secret_key:
            raise EnvironmentError(
                "ALPACA_API_KEY and ALPACA_SECRET_KEY must be set in .env\n"
                "Sign up free at https://alpaca.markets to get keys."
            )
        return StockHistoricalDataClient(api_key=api_key, secret_key=secret_key)
