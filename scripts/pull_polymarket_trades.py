"""pull_polymarket_trades.py — Pull trade history from Polymarket CLOB.

For each market in market_meta.parquet, fetches historical trades and
aggregates them into hourly USDC volume buckets. This gives us a "whale
bet size" signal: a large price move accompanied by a volume spike is a
much stronger indicator than a price move alone.

Output:
  D:/WhaleWatch_Data/polymarket/volume/<safe_id>.parquet
  Index: datetime_utc (1-hour buckets)
  Column: volume_usd (total USDC traded in that hour)

Primary endpoint:  GET https://clob.polymarket.com/trades?market_id=<condition_id>
Fallback endpoint: GET https://data-api.polymarket.com/activity?market=<condition_id>

No authentication required — both endpoints are public read-only.

Note: The CLOB may only retain 30–90 days of trade history for closed markets.
Where history is unavailable, the price-only anomaly signal is still used.

Usage:
    python scripts/pull_polymarket_trades.py
    python scripts/pull_polymarket_trades.py --start 2024-01-01
    python scripts/pull_polymarket_trades.py --force-full
"""

import argparse
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
import sys

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("pull_poly_trades")

CLOB_BASE     = "https://clob.polymarket.com"
DATA_API_BASE = "https://data-api.polymarket.com"

POLY_DIR   = Path("D:/WhaleWatch_Data/polymarket")
VOLUME_DIR = POLY_DIR / "volume"

DEFAULT_START = "2024-01-01"
CALL_SLEEP    = 0.3   # seconds between requests


# ---------------------------------------------------------------------------
# CLOB trades endpoint (primary)
# ---------------------------------------------------------------------------

def _fetch_clob_trades(
    session: requests.Session,
    condition_id: str,
    after_ts: int,
) -> list[dict]:
    """Fetch all trades from CLOB /trades endpoint with cursor pagination.

    Returns list of raw trade dicts with keys: price, size, timestamp (ms).
    Returns empty list if endpoint returns 401/403 or no data.
    """
    trades   = []
    cursor   = ""
    page     = 0
    max_pages = 200   # safety cap

    while page < max_pages:
        params: dict = {"market_id": condition_id, "limit": 500}
        if cursor:
            params["next_cursor"] = cursor
        elif after_ts:
            params["after"] = after_ts * 1000   # CLOB uses ms

        try:
            resp = session.get(f"{CLOB_BASE}/trades", params=params, timeout=10)
        except requests.RequestException as exc:
            logger.debug("CLOB /trades request error: %s", exc)
            break

        if resp.status_code in (401, 403):
            logger.debug("CLOB /trades: auth required — will try fallback")
            return []
        if resp.status_code == 404:
            return []
        if resp.status_code != 200:
            logger.debug("CLOB /trades HTTP %d", resp.status_code)
            break

        data  = resp.json()
        items = data if isinstance(data, list) else data.get("data", [])

        if not items:
            break

        trades.extend(items)
        cursor = data.get("next_cursor", "") if isinstance(data, dict) else ""
        if not cursor:
            break

        page += 1
        time.sleep(CALL_SLEEP)

    return trades


# ---------------------------------------------------------------------------
# Data API fallback
# ---------------------------------------------------------------------------

def _fetch_data_api_trades(
    session: requests.Session,
    condition_id: str,
    after_ts: int,
) -> list[dict]:
    """Fetch trades from data-api.polymarket.com/activity (fallback).

    Returns list of trade dicts normalised to {price, size, timestamp_ms}.
    """
    trades  = []
    offset  = 0
    limit   = 500

    while True:
        params = {
            "market": condition_id,
            "type":   "trade",
            "limit":  limit,
            "offset": offset,
        }
        try:
            resp = session.get(f"{DATA_API_BASE}/activity", params=params, timeout=10)
        except requests.RequestException as exc:
            logger.debug("data-api /activity error: %s", exc)
            break

        if resp.status_code in (401, 403, 404):
            break
        if resp.status_code != 200:
            break

        items = resp.json()
        if not items:
            break
        if not isinstance(items, list):
            items = items.get("data", [])
        if not items:
            break

        # Normalise to common schema
        for item in items:
            ts_raw = item.get("timestamp") or item.get("created_at") or item.get("t")
            size   = item.get("size") or item.get("usdcSize") or item.get("amount", 0)
            price  = item.get("price", 0)
            if ts_raw and size:
                trades.append({
                    "price":        float(price),
                    "size":         float(size),
                    "timestamp_ms": int(float(ts_raw)) if float(ts_raw) > 1e10 else int(float(ts_raw) * 1000),
                })

        # Stop if we've gone past after_ts
        if trades:
            oldest_ms = min(t["timestamp_ms"] for t in trades[-len(items):])
            if oldest_ms < after_ts * 1000:
                break

        if len(items) < limit:
            break
        offset += limit
        time.sleep(CALL_SLEEP)

    return trades


# ---------------------------------------------------------------------------
# Parse + aggregate into hourly volume
# ---------------------------------------------------------------------------

def _normalise_clob_trades(raw: list[dict]) -> list[dict]:
    """Convert CLOB trade records to {price, size, timestamp_ms}."""
    out = []
    for t in raw:
        ts = t.get("timestamp") or t.get("created_at") or t.get("t") or 0
        try:
            ts_ms = int(float(ts))
            if ts_ms < 2e12:          # seconds → ms
                ts_ms = ts_ms * 1000
        except (ValueError, TypeError):
            continue
        size  = t.get("size") or t.get("usdcSize") or t.get("amount") or 0
        price = t.get("price", 0)
        try:
            out.append({"price": float(price), "size": float(size), "timestamp_ms": ts_ms})
        except (ValueError, TypeError):
            continue
    return out


def _to_hourly_volume(trades: list[dict]) -> pd.DataFrame:
    """Aggregate trade list → hourly USDC volume DataFrame."""
    if not trades:
        return pd.DataFrame()

    df = pd.DataFrame(trades)
    df["datetime"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)
    df = df.set_index("datetime").sort_index()

    # size is USDC traded per fill
    hourly = df["size"].resample("1h").sum().rename("volume_usd")
    hourly = hourly[hourly > 0]
    return hourly.to_frame()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def fetch_market_volume(
    session: requests.Session,
    condition_id: str,
    after_ts: int,
) -> pd.DataFrame:
    """Try CLOB primary, fall back to data-api. Return hourly volume DataFrame."""
    # Primary: CLOB /trades
    raw = _fetch_clob_trades(session, condition_id, after_ts)
    if raw:
        trades = _normalise_clob_trades(raw)
        if trades:
            return _to_hourly_volume(trades)

    # Fallback: data-api /activity
    time.sleep(CALL_SLEEP)
    trades = _fetch_data_api_trades(session, condition_id, after_ts)
    if trades:
        return _to_hourly_volume(trades)

    return pd.DataFrame()


def run(start: str, force_full: bool) -> None:
    VOLUME_DIR.mkdir(parents=True, exist_ok=True)

    meta_path = POLY_DIR / "market_meta.parquet"
    if not meta_path.exists():
        logger.error("market_meta.parquet not found. Run pull_polymarket_history.py first.")
        return

    meta     = pd.read_parquet(meta_path)
    after_ts = int(datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())

    session = requests.Session()
    session.headers.update({"Accept": "application/json"})

    n_ok = n_skip = n_fail = 0
    total = len(meta)

    for idx, (_, row) in enumerate(meta.iterrows(), 1):
        cid     = row["condition_id"]
        safe_id = cid.replace("0x", "")[:24]
        out_path = VOLUME_DIR / f"{safe_id}.parquet"

        # Resume: only fetch newer bars
        fetch_after = after_ts
        existing    = pd.DataFrame()
        if out_path.exists() and not force_full:
            try:
                existing = pd.read_parquet(out_path)
                if not existing.empty:
                    if existing.index.tz is None:
                        existing.index = existing.index.tz_localize("UTC")
                    last_ts = int(existing.index.max().timestamp())
                    if last_ts >= int(datetime.now(tz=timezone.utc).timestamp()) - 3600:
                        n_skip += 1
                        continue
                    fetch_after = last_ts
            except Exception:
                existing = pd.DataFrame()

        new_df = fetch_market_volume(session, cid, fetch_after)

        if new_df.empty:
            n_fail += 1
            logger.debug("No trade data: %s", row.get("slug", cid[:12]))
            continue

        if not existing.empty:
            combined = pd.concat([existing, new_df]).sort_index()
            combined = combined[~combined.index.duplicated(keep="last")]
        else:
            combined = new_df

        combined.to_parquet(out_path)
        n_ok += 1

        if idx % 25 == 0 or idx == total:
            logger.info(
                "[%d/%d]  pulled=%d  skipped=%d  no_data=%d",
                idx, total, n_ok, n_skip, n_fail,
            )

    logger.info(
        "Done.  Pulled: %d  Skipped: %d  No data (history unavailable): %d",
        n_ok, n_skip, n_fail,
    )
    logger.info(
        "Note: closed markets older than ~90 days may have no trade history "
        "in the CLOB. Price-based anomaly detection is used as fallback."
    )
    logger.info("Next: python scripts/build_poly_market_data.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pull Polymarket trade history → hourly volume")
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--force-full", action="store_true",
                        help="Re-fetch all history even if local file exists")
    args = parser.parse_args()
    run(start=args.start, force_full=args.force_full)
