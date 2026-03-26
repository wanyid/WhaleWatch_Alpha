"""pull_2m_equity.py — Download 2-minute OHLCV bars for SPY, QQQ, VIX.

Why 2m?
  - 5m bars are coarse for short holding periods (15–60 min signals)
  - 2m bars improve entry/exit price accuracy in backtesting
  - Yahoo Finance allows 2m bars up to 60 days per request (same window as 5m)
  - 2m bars are stitched in 50-day chunks for full history since 2025-01-20

Output:
  D:/WhaleWatch_Data/equity/SPY_2m.parquet
  D:/WhaleWatch_Data/equity/QQQ_2m.parquet
  D:/WhaleWatch_Data/equity/VIX_2m.parquet

Usage:
    python scripts/pull_2m_equity.py
    python scripts/pull_2m_equity.py --tickers SPY QQQ   # specific tickers only
    python scripts/pull_2m_equity.py --start 2025-01-20  # custom start date

Re-running is incremental: if a parquet already exists, new bars are appended
and the file is deduplicated by timestamp.
"""

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scanners.market_data.yfinance_provider import YFinanceProvider

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("pull_2m_equity")

DATA_ROOT = Path("D:/WhaleWatch_Data")
EQUITY_DIR = DATA_ROOT / "equity"
TICKERS = ["SPY", "QQQ", "VIX"]

# Training data floor — match label_events.py and clean_data.py
DEFAULT_START = "2025-01-20"


def _merge_incremental(existing: pd.DataFrame, new_bars: pd.DataFrame) -> pd.DataFrame:
    """Append new bars to existing, dedup by timestamp index."""
    if existing.empty:
        return new_bars
    combined = pd.concat([existing, new_bars])
    return combined[~combined.index.duplicated(keep="last")].sort_index()


def pull_2m(tickers: list[str], start: str, end: str) -> None:
    EQUITY_DIR.mkdir(parents=True, exist_ok=True)
    provider = YFinanceProvider()

    for ticker in tickers:
        out_path = EQUITY_DIR / f"{ticker}_2m.parquet"

        # Load existing data to determine incremental start
        existing = pd.DataFrame()
        if out_path.exists():
            try:
                existing = pd.read_parquet(out_path)
                if not existing.empty:
                    last_ts = existing.index.max()
                    # Resume from last stored bar's date
                    incremental_start = last_ts.strftime("%Y-%m-%d")
                    logger.info(
                        "%s: existing parquet has %d bars through %s — pulling from %s",
                        ticker, len(existing), last_ts.date(), incremental_start,
                    )
                    start = incremental_start
            except Exception as exc:
                logger.warning("%s: could not read existing parquet (%s) — full pull", ticker, exc)
                existing = pd.DataFrame()

        logger.info("Pulling %s 2m bars: %s → %s", ticker, start, end)
        try:
            new_bars = provider.get_ohlcv_chunked(
                ticker=ticker,
                start=start,
                end=end,
                interval="2m",
                chunk_days=50,  # safe under Yahoo's 60-day per-request limit
            )
        except Exception as exc:
            logger.error("%s: pull failed — %s", ticker, exc)
            continue

        if new_bars.empty:
            logger.warning("%s: no new bars returned", ticker)
            continue

        merged = _merge_incremental(existing, new_bars)
        merged.to_parquet(out_path)

        size_mb = out_path.stat().st_size / 1e6
        logger.info(
            "%s: saved %d total bars to %s (%.1f MB)",
            ticker, len(merged), out_path.name, size_mb,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Pull 2-minute equity bars for training")
    parser.add_argument("--tickers", nargs="+", default=TICKERS,
                        help="Tickers to download (default: SPY QQQ VIX)")
    parser.add_argument("--start", default=DEFAULT_START,
                        help="Start date YYYY-MM-DD (default: 2025-01-20)")
    parser.add_argument("--end", default=None,
                        help="End date YYYY-MM-DD (default: today)")
    args = parser.parse_args()

    end = args.end or datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")

    logger.info("=" * 60)
    logger.info("WhaleWatch_Alpha — 2-Minute Equity Data Pull")
    logger.info("Tickers: %s", ", ".join(args.tickers))
    logger.info("Range  : %s → %s", args.start, end)
    logger.info("Output : %s", EQUITY_DIR)
    logger.info("=" * 60)

    pull_2m(tickers=args.tickers, start=args.start, end=end)

    logger.info("\nDone.")
    for p in sorted(EQUITY_DIR.glob("*_2m.parquet")):
        logger.info("  %-50s  %.1f MB", p.name, p.stat().st_size / 1e6)


if __name__ == "__main__":
    main()
