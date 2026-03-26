"""yfinance implementation of BaseMarketDataProvider.

Free, no API key required. Suitable for backtesting and historical pulls.
For live trading swap to polygon_provider.py by changing settings.yaml.

Intraday interval limits (per Yahoo Finance API):
  1m  → last 7 days only
  2m  → 60-day window per request
  5m  → 60-day window per request   ← default for historical pulls
  1h  → 730-day window per request
  1d  → unlimited

Use get_ohlcv_chunked() to download multi-year 5m history by stitching
multiple 50-day windows automatically.
"""

import logging
from datetime import datetime, timedelta, timezone

import pandas as pd

from scanners.market_data.base_provider import BaseMarketDataProvider

logger = logging.getLogger(__name__)

# yfinance uses "^VIX" but we accept plain "VIX" for convenience
_TICKER_MAP = {"VIX": "^VIX"}

# Safe chunk size well within the 60-day per-request limit for 5m/2m data
_CHUNK_DAYS = 50


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names and index to UTC datetime."""
    if df.empty:
        return df
    # yfinance sometimes returns MultiIndex columns (ticker, field)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0].lower() for col in df.columns]
    else:
        df.columns = [c.lower() for c in df.columns]
    df.index = pd.to_datetime(df.index, utc=True)
    df.index.name = "timestamp"
    keep = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
    return df[keep].sort_index().dropna()


class YFinanceProvider(BaseMarketDataProvider):
    """yfinance-backed market data — free, no API key needed.

    For intervals ≤ 5m use get_ohlcv_chunked() to pull long date ranges.
    For 1h / 1d use get_ohlcv() directly (single request handles 2 years).
    """

    def get_ohlcv(
        self,
        ticker: str,
        start: str,
        end: str,
        interval: str = "5m",
    ) -> pd.DataFrame:
        """Single-request OHLCV fetch.

        Works reliably for:
          - interval="1h" or "1d" across any date range ≤ 730 / unlimited days
          - interval="5m" for spans ≤ 50 days

        For multi-year 5m pulls use get_ohlcv_chunked() instead.
        """
        try:
            import yfinance as yf
        except ImportError as exc:
            raise ImportError("yfinance not installed. Run: pip install yfinance") from exc

        yf_ticker = _TICKER_MAP.get(ticker.upper(), ticker.upper())
        logger.info("yfinance: %s %s bars %s → %s", interval, yf_ticker, start, end)

        df = yf.download(
            yf_ticker,
            start=start,
            end=end,
            interval=interval,
            auto_adjust=True,
            progress=False,
        )
        return _normalize(df)

    def get_ohlcv_chunked(
        self,
        ticker: str,
        start: str,
        end: str,
        interval: str = "5m",
        chunk_days: int = _CHUNK_DAYS,
    ) -> pd.DataFrame:
        """Download long-range intraday data by stitching 50-day chunks.

        Use this for 5m data spanning more than 60 days (e.g. full 2024–present).
        Each chunk is fetched with a small overlap to avoid gaps at boundaries.

        Args:
            ticker:     e.g. "SPY", "QQQ", "VIX"
            start:      ISO date string "YYYY-MM-DD"
            end:        ISO date string "YYYY-MM-DD"
            interval:   "5m" (default) or "2m"
            chunk_days: window size per request (default 50, safe under 60-day limit)
        """
        try:
            import yfinance as yf
        except ImportError as exc:
            raise ImportError("yfinance not installed. Run: pip install yfinance") from exc

        yf_ticker = _TICKER_MAP.get(ticker.upper(), ticker.upper())
        start_dt = datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        end_dt = datetime.strptime(end, "%Y-%m-%d").replace(tzinfo=timezone.utc)

        chunks: list[pd.DataFrame] = []
        current = start_dt
        total_days = (end_dt - start_dt).days
        fetched_days = 0

        logger.info(
            "yfinance chunked: %s %s bars %s → %s (%d days, chunk=%d)",
            interval, yf_ticker, start, end, total_days, chunk_days,
        )

        while current < end_dt:
            chunk_end = min(current + timedelta(days=chunk_days), end_dt)
            chunk_start_str = current.strftime("%Y-%m-%d")
            chunk_end_str = chunk_end.strftime("%Y-%m-%d")

            try:
                df = yf.download(
                    yf_ticker,
                    start=chunk_start_str,
                    end=chunk_end_str,
                    interval=interval,
                    auto_adjust=True,
                    progress=False,
                )
                df = _normalize(df)
                if not df.empty:
                    chunks.append(df)
                    fetched_days += chunk_days
                    pct = min(fetched_days / total_days * 100, 100)
                    logger.info(
                        "  chunk %s → %s: %d bars (%.0f%% done)",
                        chunk_start_str, chunk_end_str, len(df), pct,
                    )
                else:
                    logger.debug("  chunk %s → %s: empty", chunk_start_str, chunk_end_str)
            except Exception as exc:
                logger.warning("  chunk %s → %s failed: %s", chunk_start_str, chunk_end_str, exc)

            current = chunk_end

        if not chunks:
            logger.warning("No data returned for %s (%s to %s)", ticker, start, end)
            return pd.DataFrame()

        combined = pd.concat(chunks)
        combined = combined[~combined.index.duplicated(keep="first")].sort_index()
        logger.info(
            "yfinance chunked complete: %s total %d bars (%.1f MB in memory)",
            ticker, len(combined), combined.memory_usage(deep=True).sum() / 1e6,
        )
        return combined

    def get_latest_price(self, ticker: str) -> float:
        try:
            import yfinance as yf
        except ImportError as exc:
            raise ImportError("yfinance not installed. Run: pip install yfinance") from exc

        yf_ticker = _TICKER_MAP.get(ticker.upper(), ticker.upper())
        return float(yf.Ticker(yf_ticker).fast_info.last_price)
