"""pull_polymarket_history.py — Pull historical YES-price data from Polymarket.

Discovers politically-relevant markets (active + closed since --start)
via the Gamma API, then fetches hourly YES-price history from the CLOB
prices-history endpoint.

Output:
  D:/WhaleWatch_Data/polymarket/market_meta.parquet   — market metadata
  D:/WhaleWatch_Data/polymarket/prices/<id>.parquet   — hourly prices per market

No authentication required — both APIs are public read-only.

Usage:
    python scripts/pull_polymarket_history.py
    python scripts/pull_polymarket_history.py --start 2024-01-01
    python scripts/pull_polymarket_history.py --min-volume 50000
    python scripts/pull_polymarket_history.py --force-full
"""

import argparse
import logging
import json
import socket
import time
# ThreadPoolExecutor removed — accumulating thread objects caused main-thread
# freezes after ~700 markets.  Direct calls + per-market sessions are simpler
# and fully reliable; the external watchdog handles rare socket-level hangs.
from datetime import datetime, timezone
from pathlib import Path
import sys

import pandas as pd
import requests

# Hard socket-level timeout — prevents zombie TCP connections from stalling
# indefinitely on Windows even when requests timeout=(connect, read) is set.
socket.setdefaulttimeout(30)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("pull_poly_history")

GAMMA_BASE = "https://gamma-api.polymarket.com"
CLOB_BASE  = "https://clob.polymarket.com"

POLY_DIR   = Path("D:/WhaleWatch_Data/polymarket")
PRICES_DIR = POLY_DIR / "prices"

DEFAULT_START   = "2024-01-01"
DEFAULT_MIN_VOL = 50_000     # USD total volume — filters out thin markets
CALL_SLEEP_SEC  = 0.5        # generous; Polymarket has no stated rate limit

# Polymarket CLOB prices-history silently caps responses to ~28 days of bars
# regardless of start_ts when fidelity=60. Chunked requests work around this.
CHUNK_DAYS = 25              # request window size — kept under the 28-day cap

# ---------------------------------------------------------------------------
# Topic bucket classification
# ---------------------------------------------------------------------------

TOPIC_BUCKETS: dict[str, list[str]] = {
    "tariff": [
        "tariff", "tariffs", "trade war", "trade deal", "import duty",
        "customs duty", "section 232", "section 301", "trade deficit",
    ],
    "geopolitical": [
        "ukraine", "russia", "china", "iran", "israel", "nato", "north korea",
        "ceasefire", "invasion", "sanctions", "war", "military strike",
        "peace deal", "middle east", "taiwan",
    ],
    "fed": [
        "federal reserve", "fed rate", "interest rate", "rate cut", "rate hike",
        "inflation", "recession", "gdp", "debt ceiling", "fomc", "powell",
        "basis points", "quantitative",
    ],
    "energy": [
        "oil price", "opec", "lng", "crude oil", "gas price", "energy price",
        "petroleum", "natural gas", "oil production",
    ],
    "executive": [
        "trump", "executive order", "cabinet", "fired", "resign", "appointed",
        "doge", "department of government efficiency", "white house",
        "president", "administration", "congress", "senate",
    ],
}

ALL_KEYWORDS: list[str] = [kw for kws in TOPIC_BUCKETS.values() for kw in kws]


def classify_topic(question: str) -> str:
    """Return the best-matching topic bucket for a market question."""
    q = question.lower()
    # Return the first bucket that matches (ordering = priority)
    for bucket, keywords in TOPIC_BUCKETS.items():
        if any(kw in q for kw in keywords):
            return bucket
    return "other"


def is_relevant(question: str) -> bool:
    q = question.lower()
    return any(kw in q for kw in ALL_KEYWORDS)


# ---------------------------------------------------------------------------
# Gamma API helpers
# ---------------------------------------------------------------------------

_session = requests.Session()
_session.headers.update({"Accept": "application/json"})


def _gamma_get_all(params: dict) -> list[dict]:
    """Page through all results from Gamma /markets."""
    results = []
    limit  = 100
    offset = 0
    while True:
        p = {**params, "limit": limit, "offset": offset}
        try:
            resp = _session.get(f"{GAMMA_BASE}/markets", params=p, timeout=10)
            resp.raise_for_status()
            data  = resp.json()
            items = data if isinstance(data, list) else data.get("markets", [])
            if not items:
                break
            results.extend(items)
            if len(items) < limit:
                break
            offset += limit
            time.sleep(CALL_SLEEP_SEC)
        except requests.RequestException as exc:
            logger.warning("Gamma API error at offset %d: %s", offset, exc)
            break
    return results


def discover_markets(start_date: str, min_volume: float) -> list[dict]:
    """Return relevant markets: active ones + closed markets active since start_date."""
    markets: dict[str, dict] = {}

    logger.info("Discovering active markets (volume >= $%.0f)...", min_volume)
    for m in _gamma_get_all({"active": "true", "closed": "false", "volume_num_min": min_volume}):
        q = m.get("question", "")
        if is_relevant(q):
            markets[m.get("conditionId", "")] = m
    logger.info("  Active: %d relevant markets", len(markets))

    logger.info("Discovering closed markets since %s...", start_date)
    n_before = len(markets)
    for m in _gamma_get_all({
        "active": "false",
        "closed": "true",
        "end_date_min": start_date,
        "volume_num_min": min_volume,
    }):
        cid = m.get("conditionId", "")
        if cid and is_relevant(m.get("question", "")):
            markets[cid] = m
    logger.info("  Closed: %d relevant markets", len(markets) - n_before)
    logger.info("Total: %d markets to pull", len(markets))
    return list(markets.values())


def extract_yes_token(market: dict) -> str | None:
    """Extract the YES outcome token_id from a market dict.

    The Gamma API returns clobTokenIds as a JSON string (e.g. '["id1","id2"]')
    where index 0 is the YES token and index 1 is the NO token.
    The older 'tokens' list shape is also handled as a fallback.
    """
    for token in market.get("tokens", []):
        if str(token.get("outcome", "")).upper() == "YES":
            return str(token.get("token_id", ""))
    # Primary path: clobTokenIds is a JSON-encoded string of [yes_id, no_id]
    raw = market.get("clobTokenIds", "")
    if raw:
        try:
            clob_ids = json.loads(raw) if isinstance(raw, str) else raw
            if clob_ids:
                return str(clob_ids[0])
        except (json.JSONDecodeError, ValueError, IndexError):
            pass
    return None


# ---------------------------------------------------------------------------
# CLOB price history
# ---------------------------------------------------------------------------

def fetch_price_history(
    token_id: str,
    start_ts: int,
    end_ts: int,
    fidelity: int = 60,   # minutes per bucket (1 | 60 | 1440)
    session: requests.Session | None = None,
) -> pd.DataFrame:
    """Fetch YES price history for a single time window from CLOB prices-history.

    Uses the provided session if given, otherwise falls back to the module-level
    _session (used only during Gamma discovery).
    """
    sess = session if session is not None else _session
    try:
        resp = sess.get(
            f"{CLOB_BASE}/prices-history",
            params={
                "market":   token_id,
                "fidelity": fidelity,
                "startTs":  start_ts,
                "endTs":    end_ts,
            },
            timeout=(5, 12),  # (connect, read) — prevents server body-stall hangs
        )
        resp.raise_for_status()
        history = resp.json().get("history", [])
        if not history:
            return pd.DataFrame()

        df = pd.DataFrame(history)
        df["datetime"] = pd.to_datetime(df["t"], unit="s", utc=True)
        df = df.set_index("datetime").rename(columns={"p": "price"})[["price"]]
        return df.sort_index()

    except requests.exceptions.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else "?"
        logger.debug("CLOB %s for token %s", status, token_id[:16])
        return pd.DataFrame()
    except Exception as exc:
        logger.debug("CLOB history error for token %s: %s", token_id[:16], exc)
        return pd.DataFrame()


def fetch_price_history_chunked(
    token_id: str,
    start_ts: int,
    end_ts: int,
    fidelity: int = 60,
    chunk_days: int = CHUNK_DAYS,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    """Fetch full history by issuing multiple requests in chunk_days windows.

    The CLOB prices-history endpoint silently caps responses to ~28 days of
    hourly bars regardless of the requested start_ts. Chunking works around
    this: we slide a window from start_ts to end_ts in chunk_days increments
    and stitch the results together.

    Each market gets its own isolated session (passed in) so there is no shared
    TCP connection pool that can exhaust and silently hang after ~650 markets.
    """
    chunk_secs = chunk_days * 24 * 3600
    all_dfs: list[pd.DataFrame] = []
    t = start_ts

    while t < end_ts:
        t_end = min(t + chunk_secs, end_ts)
        chunk = fetch_price_history(token_id, t, t_end, fidelity=fidelity, session=session)
        if not chunk.empty:
            all_dfs.append(chunk)
        t = t_end
        time.sleep(CALL_SLEEP_SEC)

    if not all_dfs:
        return pd.DataFrame()

    combined = pd.concat(all_dfs).sort_index()
    combined = combined[~combined.index.duplicated(keep="last")]
    return combined


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(start: str, min_volume: float, force_full: bool, fidelity: int = 60,
        use_cached_catalog: bool = False) -> None:
    PRICES_DIR.mkdir(parents=True, exist_ok=True)

    start_ts = int(datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())
    end_ts   = int(datetime.now(tz=timezone.utc).timestamp())

    meta_path = POLY_DIR / "market_meta.parquet"

    # --use-cached-catalog: skip the 12-min Gamma discovery and load the saved catalog.
    # Safe to use on restarts since the catalog was already built on the first run.
    if use_cached_catalog and meta_path.exists():
        logger.info("Using cached market catalog: %s", meta_path)
        meta_df = pd.read_parquet(meta_path)
        logger.info("Loaded %d markets from catalog", len(meta_df))
        for bucket, count in meta_df["topic_bucket"].value_counts().items():
            logger.info("  %-14s  %d markets", bucket, count)
    else:
        markets = discover_markets(start_date=start, min_volume=min_volume)

        # Build metadata table
        meta_rows = []
        for m in markets:
            yes_token = extract_yes_token(m)
            if not yes_token:
                continue
            cid = m.get("conditionId", "")
            meta_rows.append({
                "condition_id": cid,
                "question":     m.get("question", ""),
                "slug":         m.get("market_slug", m.get("slug", "")),
                "topic_bucket": classify_topic(m.get("question", "")),
                "yes_token_id": yes_token,
                "start_date":   m.get("startDate", m.get("created_at", "")),
                "end_date":     m.get("endDate", m.get("end_date_iso", "")),
                "volume":       float(m.get("volumeNum", m.get("volume", 0)) or 0),
            })

        meta_df = pd.DataFrame(meta_rows).drop_duplicates("condition_id")
        meta_path = POLY_DIR / "market_meta.parquet"
    meta_df.to_parquet(meta_path)
    logger.info("Saved %d markets metadata → %s", len(meta_df), meta_path)

    # Topic breakdown
    for bucket, count in meta_df["topic_bucket"].value_counts().items():
        logger.info("  %-14s  %d markets", bucket, count)

    # Pull price history for each market.
    # Design: each market gets its own isolated requests.Session.
    # This eliminates TCP connection-pool exhaustion — the root cause of silent
    # hangs that previously occurred around market #650+.
    #
    # On Windows, requests.get(timeout=...) occasionally hangs on specific markets
    # (zombie TCP state). The external watchdog kills and restarts the process.
    # To avoid re-hanging on the SAME market, we write a progress marker before
    # each fetch. On --use-cached-catalog restarts, markets up to the marker are
    # skipped (they either already have files or were the one that caused the hang).
    n_ok = n_skip = n_fail = 0
    total = len(meta_df)
    LOG_EVERY = 10

    # Progress marker: records the last market idx that was ATTEMPTED.
    # On restart after a watchdog kill, skip to marker+1 so we don't re-hang
    # on the same problematic market.
    progress_path = POLY_DIR / ".pull_progress"
    resume_from = 0
    if use_cached_catalog and progress_path.exists():
        try:
            resume_from = int(progress_path.read_text().strip())
            logger.info("Resuming from market %d (skipping previously attempted markets)", resume_from + 1)
        except Exception:
            resume_from = 0

    for idx, (_, row) in enumerate(meta_df.iterrows(), 1):
        # Skip markets already attempted in a previous run (before watchdog kill)
        if idx <= resume_from:
            cid = row["condition_id"]
            safe_id = cid.replace("0x", "")[:24]
            out_path = PRICES_DIR / f"{safe_id}.parquet"
            if out_path.exists():
                n_skip += 1
            else:
                n_fail += 1  # was attempted but never saved → skip it
            continue

        cid       = row["condition_id"]
        safe_id   = cid.replace("0x", "")[:24]
        out_path  = PRICES_DIR / f"{safe_id}.parquet"

        # Resume: only fetch bars newer than last saved bar
        fetch_start = start_ts
        existing    = pd.DataFrame()

        # Clamp fetch_start to the market's known start_date to skip pre-creation chunks.
        market_start_str = row.get("start_date", "")
        if market_start_str:
            try:
                mstart = int(
                    datetime.strptime(str(market_start_str)[:10], "%Y-%m-%d")
                    .replace(tzinfo=timezone.utc)
                    .timestamp()
                )
                fetch_start = max(fetch_start, mstart)
            except Exception:
                pass

        if out_path.exists() and not force_full:
            try:
                existing = pd.read_parquet(out_path)
                if not existing.empty:
                    if existing.index.tz is None:
                        existing.index = existing.index.tz_localize("UTC")
                    last_ts = int(existing.index.max().timestamp())
                    if last_ts >= end_ts - 3600:
                        n_skip += 1
                        continue
                    fetch_start = last_ts
            except Exception:
                existing = pd.DataFrame()

        # Write progress marker BEFORE fetch — if we hang here, the watchdog
        # kills us and on restart we skip past this market.
        progress_path.write_text(str(idx))

        # Each market gets a dedicated session — isolated TCP pool, no shared state.
        mkt_session = requests.Session()
        mkt_session.headers.update({"Accept": "application/json"})
        try:
            new_df = fetch_price_history_chunked(
                row["yes_token_id"], fetch_start, end_ts,
                fidelity, CHUNK_DAYS, mkt_session,
            )
        except Exception as exc:
            logger.warning("Error on market %s: %s — skipping", cid[:16], exc)
            n_fail += 1
            continue
        finally:
            mkt_session.close()

        if new_df.empty:
            n_fail += 1
            continue

        if not existing.empty:
            combined = pd.concat([existing, new_df]).sort_index()
            combined = combined[~combined.index.duplicated(keep="last")]
        else:
            combined = new_df

        combined.to_parquet(out_path)
        n_ok += 1

        if idx % LOG_EVERY == 0 or idx == total:
            logger.info(
                "[%d/%d]  pulled=%d  skipped=%d  failed=%d",
                idx, total, n_ok, n_skip, n_fail,
            )

    # Clean up progress marker on successful completion
    try:
        progress_path.unlink(missing_ok=True)
    except Exception:
        pass

    logger.info("Done.  Pulled: %d  Skipped (up to date): %d  Failed: %d", n_ok, n_skip, n_fail)
    logger.info("Price files → %s  (fidelity=%dm)", PRICES_DIR, fidelity)
    logger.info("Next: python scripts/build_poly_market_data.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pull Polymarket historical price data")
    parser.add_argument("--start", default=DEFAULT_START,
                        help="Start date YYYY-MM-DD (default: 2024-01-01)")
    parser.add_argument("--min-volume", type=float, default=DEFAULT_MIN_VOL,
                        help="Minimum total USD volume to include a market (default: 50000)")
    parser.add_argument("--force-full", action="store_true",
                        help="Re-fetch all history even if local file exists")
    parser.add_argument("--fidelity", type=int, default=60,
                        help="Bar size in minutes: 60=hourly (default), 1440=daily")
    parser.add_argument("--use-cached-catalog", action="store_true",
                        help="Skip Gamma discovery and use existing market_meta.parquet")
    args = parser.parse_args()
    run(start=args.start, min_volume=args.min_volume,
        force_full=args.force_full, fidelity=args.fidelity,
        use_cached_catalog=args.use_cached_catalog)
