"""pull_intraday_history.py — Pull 1h intraday history via yfinance (chunked).

yfinance limits 1h data to ~700 days per request window, but chunking 89-day
slices allows pulling the full available range (~Apr 2024 → present).

Saves to D:/WhaleWatch_Data/equity/{TICKER}_1h.parquet
Resumes from existing data — only fetches new bars.

Usage:
    python scripts/pull_intraday_history.py
    python scripts/pull_intraday_history.py --tickers SPY QQQ
"""

import argparse
import logging
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import yfinance as yf

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("pull_intraday_history")

EQUITY_DIR   = Path("D:/WhaleWatch_Data/equity")
# yfinance uses ^VIX for the index, but we store it as "VIX" locally
DEFAULT_TICKERS = ["SPY", "QQQ", "VIX"]
YF_TICKER_MAP   = {"VIX": "^VIX"}   # maps local name → yfinance symbol
CHUNK_DAYS   = 89     # safe chunk size for yfinance 1h
MAX_DAYS     = 700    # yfinance 1h hard limit from today
SLEEP_SEC    = 1.5    # polite pause between chunks


def _fetch_1h_chunked(ticker: str, start: datetime, end: datetime) -> pd.DataFrame:
    """Fetch 1h bars for ticker from start→end in CHUNK_DAYS slices."""
    chunks = []
    current_end = end

    while current_end > start:
        current_start = max(current_end - timedelta(days=CHUNK_DAYS), start)
        try:
            df = yf.download(
                ticker,
                start=current_start.strftime("%Y-%m-%d"),
                end=current_end.strftime("%Y-%m-%d"),
                interval="1h",
                progress=False,
                auto_adjust=True,
            )
        except Exception as exc:
            logger.warning("Chunk %s → %s failed: %s", current_start.date(), current_end.date(), exc)
            current_end = current_start
            time.sleep(SLEEP_SEC * 2)
            continue

        if not df.empty:
            # Flatten MultiIndex columns if present
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0].lower() for c in df.columns]
            else:
                df.columns = [c.lower() for c in df.columns]

            # Ensure UTC timezone
            if df.index.tz is None:
                df.index = df.index.tz_localize("UTC")
            else:
                df.index = df.index.tz_convert("UTC")

            chunks.append(df)
            logger.info(
                "  %s  %s → %s : %d bars",
                ticker, current_start.date(), current_end.date(), len(df),
            )
        else:
            logger.warning("  %s  %s → %s : empty", ticker, current_start.date(), current_end.date())

        current_end = current_start
        time.sleep(SLEEP_SEC)

    if not chunks:
        return pd.DataFrame()

    combined = (
        pd.concat(chunks)
        .sort_index()
        .drop_duplicates()
    )
    return combined


def pull_ticker(ticker: str, force_full: bool = False) -> None:
    EQUITY_DIR.mkdir(parents=True, exist_ok=True)
    yf_symbol = YF_TICKER_MAP.get(ticker, ticker)
    out_path  = EQUITY_DIR / f"{ticker}_1h.parquet"

    now = datetime.now(tz=timezone.utc)
    hard_start = now - timedelta(days=MAX_DAYS)

    # Resume: find last bar in existing file
    existing_end = None
    existing_df  = pd.DataFrame()
    if out_path.exists() and not force_full:
        try:
            existing_df = pd.read_parquet(out_path)
            if existing_df.index.tz is None:
                existing_df.index = existing_df.index.tz_localize("UTC")
            last_ts = existing_df.index.max()
            # Start fetch from day after last bar
            existing_end = last_ts.to_pydatetime().replace(tzinfo=timezone.utc)
            logger.info("%s: existing data ends %s — fetching incremental update", ticker, last_ts.date())
        except Exception:
            existing_df = pd.DataFrame()

    fetch_start = existing_end if existing_end else hard_start
    fetch_end   = now

    if existing_end and (fetch_end - fetch_start).days < 1:
        logger.info("%s: already up to date", ticker)
        return

    logger.info("%s: fetching %s → %s", ticker, fetch_start.date(), fetch_end.date())
    new_df = _fetch_1h_chunked(yf_symbol, fetch_start, fetch_end)

    if new_df.empty:
        logger.warning("%s: no new data retrieved", ticker)
        return

    # Merge with existing
    if not existing_df.empty:
        combined = pd.concat([existing_df, new_df]).sort_index().drop_duplicates()
    else:
        combined = new_df

    # Keep only close column (plus open/high/low/volume if present) for storage
    keep_cols = [c for c in ["open", "high", "low", "close", "volume"] if c in combined.columns]
    combined  = combined[keep_cols]

    combined.to_parquet(out_path)
    logger.info(
        "%s: saved %d bars (%s → %s) → %s",
        ticker, len(combined),
        combined.index[0].date(), combined.index[-1].date(),
        out_path,
    )


def run(tickers: list[str] = DEFAULT_TICKERS, force_full: bool = False) -> None:
    for ticker in tickers:
        pull_ticker(ticker, force_full=force_full)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pull 1h intraday history via yfinance")
    parser.add_argument(
        "--tickers", nargs="+", default=DEFAULT_TICKERS,
        help=f"Tickers to pull (default: {DEFAULT_TICKERS})",
    )
    parser.add_argument(
        "--force-full", action="store_true",
        help="Re-fetch full history even if local file exists",
    )
    args = parser.parse_args()
    run(tickers=args.tickers, force_full=args.force_full)
