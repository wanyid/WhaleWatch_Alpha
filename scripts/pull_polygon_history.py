"""pull_polygon_history.py — Pull intraday history from Polygon.io REST API.

Pulls 1h and 5m OHLCV bars for SPY, QQQ, VIX (index), and VIXY.
Saves to D:/WhaleWatch_Data/equity/{TICKER}_{interval}_poly.parquet

Free tier limits: 5 API calls/minute — script sleeps 13s between calls.
Pagination: Polygon returns max 50,000 bars per call; script follows next_url
cursors automatically so long date ranges are handled in one invocation.

Requires POLYGON_API_KEY in .env

Usage:
    python scripts/pull_polygon_history.py
    python scripts/pull_polygon_history.py --start 2025-01-20  # default
    python scripts/pull_polygon_history.py --intervals 1h       # single interval
    python scripts/pull_polygon_history.py --tickers SPY QQQ    # subset
"""

import argparse
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("pull_polygon")

EQUITY_DIR   = Path("D:/WhaleWatch_Data/equity")
POLY_BASE    = "https://api.polygon.io"
DEFAULT_START = "2025-01-20"

# Local name → Polygon ticker symbol
# Indices use the "I:" prefix on Polygon
POLYGON_TICKER_MAP = {
    "SPY":  "SPY",
    "QQQ":  "QQQ",
    "VIX":  "I:VIX",    # CBOE Volatility Index
    "VIXY": "VIXY",     # ProShares VIX Short-Term Futures ETF
}

DEFAULT_TICKERS   = list(POLYGON_TICKER_MAP.keys())
DEFAULT_INTERVALS = ["1h", "5m"]

# Map human interval → (multiplier, timespan) for Polygon API
INTERVAL_PARAMS = {
    "1m":  (1,  "minute"),
    "5m":  (5,  "minute"),
    "15m": (15, "minute"),
    "30m": (30, "minute"),
    "1h":  (1,  "hour"),
    "1d":  (1,  "day"),
}

# Free tier: 5 calls/minute → sleep 13s between calls to stay safe
CALL_SLEEP_SEC = 13


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def _get(url: str, params: dict, api_key: str) -> dict | None:
    params = {**params, "apiKey": api_key}
    try:
        resp = requests.get(url, params=params, timeout=30)
    except Exception as exc:
        logger.warning("Request failed: %s", exc)
        return None

    if resp.status_code == 200:
        return resp.json()
    elif resp.status_code == 429:
        logger.warning("Rate limited (429) — sleeping 60s")
        time.sleep(60)
        return None
    elif resp.status_code == 403:
        logger.error("403 Forbidden — check API key and plan tier")
        return None
    else:
        logger.warning("HTTP %d: %s", resp.status_code, resp.text[:200])
        return None


def _fetch_aggs(
    poly_ticker: str,
    multiplier: int,
    timespan: str,
    from_date: str,
    to_date: str,
    api_key: str,
) -> pd.DataFrame:
    """Fetch all aggregate bars with automatic pagination via next_url cursor."""
    url = f"{POLY_BASE}/v2/aggs/ticker/{poly_ticker}/range/{multiplier}/{timespan}/{from_date}/{to_date}"
    params = {
        "adjusted": "true",
        "sort": "asc",
        "limit": 50000,
    }

    all_results = []
    page = 1

    while url:
        logger.debug("Page %d  url=%s", page, url)
        data = _get(url, params if page == 1 else {}, api_key)
        time.sleep(CALL_SLEEP_SEC)

        if data is None:
            break

        status = data.get("status", "")
        if status not in ("OK", "DELAYED"):
            logger.warning("Unexpected status '%s' for %s", status, poly_ticker)
            break

        results = data.get("results", [])
        if results:
            all_results.extend(results)
            logger.info(
                "  %s  page %d: +%d bars  (total so far: %d)",
                poly_ticker, page, len(results), len(all_results),
            )

        # Polygon returns a next_url when there are more pages
        url = data.get("next_url")
        params = {}  # next_url already contains all query params except apiKey
        page += 1

    if not all_results:
        return pd.DataFrame()

    df = pd.DataFrame(all_results)

    # Polygon timestamps are Unix milliseconds UTC
    df["datetime"] = pd.to_datetime(df["t"], unit="ms", utc=True)
    df = df.set_index("datetime").sort_index()

    # Rename columns to standard OHLCV names
    col_map = {"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume", "vw": "vwap", "n": "n_trades"}
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    keep = [c for c in ["open", "high", "low", "close", "volume", "vwap"] if c in df.columns]
    return df[keep]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def pull_ticker_interval(
    local_name: str,
    interval: str,
    start: str,
    end: str,
    api_key: str,
    force_full: bool = False,
) -> None:
    EQUITY_DIR.mkdir(parents=True, exist_ok=True)
    poly_ticker = POLYGON_TICKER_MAP.get(local_name, local_name)
    out_path    = EQUITY_DIR / f"{local_name}_{interval}_poly.parquet"

    multiplier, timespan = INTERVAL_PARAMS[interval]

    # Resume from last bar if file exists
    fetch_start = start
    existing_df = pd.DataFrame()

    if out_path.exists() and not force_full:
        try:
            existing_df = pd.read_parquet(out_path)
            if existing_df.index.tz is None:
                existing_df.index = existing_df.index.tz_localize("UTC")
            last_ts = existing_df.index.max()
            # Resume from the day after the last bar
            fetch_start = (last_ts + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            logger.info(
                "%s %s: resuming from %s (existing: %d bars → %s)",
                local_name, interval, fetch_start, len(existing_df), last_ts.date(),
            )
        except Exception as exc:
            logger.warning("Could not read existing file: %s — full pull", exc)
            existing_df = pd.DataFrame()

    if fetch_start >= end:
        logger.info("%s %s: already up to date", local_name, interval)
        return

    logger.info(
        "%s %s: fetching %s → %s  (polygon ticker: %s)",
        local_name, interval, fetch_start, end, poly_ticker,
    )

    new_df = _fetch_aggs(poly_ticker, multiplier, timespan, fetch_start, end, api_key)

    if new_df.empty:
        logger.warning("%s %s: no data returned", local_name, interval)
        return

    # Merge with existing
    if not existing_df.empty:
        combined = pd.concat([existing_df, new_df]).sort_index()
        combined = combined[~combined.index.duplicated(keep="last")]
    else:
        combined = new_df

    combined.to_parquet(out_path)
    logger.info(
        "%s %s: saved %d bars (%s → %s) → %s",
        local_name, interval, len(combined),
        combined.index[0].date(), combined.index[-1].date(),
        out_path,
    )


def run(
    tickers: list[str] = DEFAULT_TICKERS,
    intervals: list[str] = DEFAULT_INTERVALS,
    start: str = DEFAULT_START,
    force_full: bool = False,
) -> None:
    api_key = os.environ.get("POLYGON_API_KEY", "")
    if not api_key:
        logger.error(
            "POLYGON_API_KEY not set in .env\n"
            "  1. Sign up free at https://polygon.io\n"
            "  2. Copy your API key\n"
            "  3. Add to .env:  POLYGON_API_KEY=your_key_here"
        )
        sys.exit(1)

    end = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")

    logger.info("Polygon pull: tickers=%s  intervals=%s  %s → %s", tickers, intervals, start, end)
    logger.info("Rate limit: %ds between calls (free tier = 5 calls/min)", CALL_SLEEP_SEC)

    # Estimate time
    n_calls = len(tickers) * len(intervals)
    est_min = n_calls * CALL_SLEEP_SEC / 60
    logger.info("Estimated time: ~%.0f minutes for %d ticker×interval combos", est_min, n_calls)

    for local_name in tickers:
        if local_name not in POLYGON_TICKER_MAP:
            logger.warning("Unknown ticker '%s' — skipping", local_name)
            continue
        for interval in intervals:
            if interval not in INTERVAL_PARAMS:
                logger.warning("Unknown interval '%s' — skipping", interval)
                continue
            pull_ticker_interval(local_name, interval, start, end, api_key, force_full=force_full)

    logger.info("Polygon pull complete.")
    logger.info("Files saved to %s", EQUITY_DIR)
    logger.info("Next step: run build_post_market_data.py --full to rebuild the dataset")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pull intraday history from Polygon.io")
    parser.add_argument(
        "--tickers", nargs="+", default=DEFAULT_TICKERS,
        help=f"Tickers to pull (default: {DEFAULT_TICKERS})",
    )
    parser.add_argument(
        "--intervals", nargs="+", default=DEFAULT_INTERVALS,
        choices=list(INTERVAL_PARAMS.keys()),
        help=f"Intervals to pull (default: {DEFAULT_INTERVALS})",
    )
    parser.add_argument(
        "--start", default=DEFAULT_START,
        help=f"Start date YYYY-MM-DD (default: {DEFAULT_START})",
    )
    parser.add_argument(
        "--force-full", action="store_true",
        help="Re-fetch full history even if local file exists",
    )
    args = parser.parse_args()

    run(
        tickers=args.tickers,
        intervals=args.intervals,
        start=args.start,
        force_full=args.force_full,
    )
