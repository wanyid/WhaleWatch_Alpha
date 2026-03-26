"""pull_truthsocial_history.py — Download historical @realDonaldTrump posts.

Pulls all Truth Social posts since the training data floor (2025-01-20) and
saves them to a parquet file for use in:
  - label_events.py (Truth Social signal labeling)
  - clean_data.py (ts_keyword_count, ts_engagement features)
  - Offline backtesting of Signal B

Output:
  D:/WhaleWatch_Data/truth_social/posts.parquet

  Schema:
    post_id          : str   — unique Truth Social status ID
    posted_at        : datetime (UTC)
    pulled_at        : datetime (UTC)
    content          : str   — plain text (HTML stripped)
    keywords         : list[str]  — market-relevant keywords detected
    keyword_count    : int
    engagement       : float — log1p(replies + reblogs + favourites)
    replies_count    : int
    reblogs_count    : int
    favourites_count : int
    is_repost        : bool
    language         : str | None

Incremental: re-running appends only new posts (keyed on post_id).

Usage:
    python scripts/pull_truthsocial_history.py
    python scripts/pull_truthsocial_history.py --start 2025-06-01
    python scripts/pull_truthsocial_history.py --include-reposts
    python scripts/pull_truthsocial_history.py --dry-run    # print first 5, don't save
"""

import argparse
import html
import logging
import math
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load credentials from .env
load_dotenv(PROJECT_ROOT / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("pull_ts_history")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_ROOT = Path("D:/WhaleWatch_Data")
OUT_DIR = DATA_ROOT / "truth_social"
OUT_PATH = OUT_DIR / "posts.parquet"

TRUTHSOCIAL_USERNAME = "realDonaldTrump"
DEFAULT_START = datetime(2025, 1, 20, tzinfo=timezone.utc)

# Rate-limit guard: pause between API retries
RETRY_SLEEP_SEC = 10
MAX_RETRIES = 3


# ---------------------------------------------------------------------------
# Keyword extraction (mirrors truthsocial_scanner.py)
# ---------------------------------------------------------------------------
KEYWORD_GROUPS: dict[str, list[str]] = {
    "trade": [
        "tariff", "tariffs", "trade war", "trade deal", "import", "export",
        "china", "usmca", "wto", "trade deficit", "trade surplus",
    ],
    "economy": [
        "inflation", "jobs", "gdp", "economy", "economic", "recession",
        "growth", "unemployment", "wage", "wages", "manufacturing",
    ],
    "markets": [
        "stock", "stocks", "market", "wall street", "nasdaq", "s&p", "dow",
        "fed", "federal reserve", "interest rate", "bond", "bonds", "dollar",
        "crypto", "bitcoin",
    ],
    "geopolitical": [
        "nato", "ukraine", "russia", "iran", "china", "israel", "sanctions",
        "military", "war", "ceasefire", "deal", "agreement", "treaty",
        "north korea",
    ],
    "policy": [
        "executive order", "tariff", "tax", "taxes", "cut", "spending",
        "budget", "deficit", "debt", "deregulation", "regulation",
        "department of", "doge", "fired", "resign", "appointed", "nominee",
    ],
    "energy": [
        "oil", "gas", "energy", "lng", "pipeline", "drill", "drilling",
        "opec", "gasoline",
    ],
}
_ALL_KEYWORDS: set[str] = {kw for group in KEYWORD_GROUPS.values() for kw in group}


def _strip_html(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def _extract_keywords(text: str) -> list[str]:
    lower = text.lower()
    return sorted({kw for kw in _ALL_KEYWORDS if kw in lower})


def _parse_dt(value) -> datetime:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    value = str(value).rstrip("Z")
    dt = datetime.fromisoformat(value)
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def _build_api():
    try:
        from truthbrush.api import Api
    except ImportError as e:
        raise ImportError("truthbrush not installed. Run: pip install truthbrush") from e

    username = os.environ.get("TRUTHSOCIAL_USERNAME")
    password = os.environ.get("TRUTHSOCIAL_PASSWORD")

    if not username or not password:
        raise EnvironmentError(
            "TRUTHSOCIAL_USERNAME and TRUTHSOCIAL_PASSWORD must be set in .env"
        )

    return Api(username, password)


def _post_to_row(raw: dict, include_reposts: bool) -> Optional[dict]:
    """Convert a raw truthbrush status dict to a clean row dict. Returns None to skip."""
    is_repost = raw.get("reblog") is not None

    if is_repost and not include_reposts:
        return None

    content_raw = raw.get("content", "")
    if is_repost and raw.get("reblog"):
        content_raw = raw["reblog"].get("content", content_raw)

    content = _strip_html(content_raw)
    if not content:
        return None

    keywords = _extract_keywords(content)
    replies = int(raw.get("replies_count", 0))
    reblogs = int(raw.get("reblogs_count", 0))
    favourites = int(raw.get("favourites_count", 0))
    engagement = float(np.log1p(replies + reblogs + favourites))

    return {
        "post_id": str(raw["id"]),
        "posted_at": _parse_dt(raw["created_at"]),
        "pulled_at": datetime.now(tz=timezone.utc),
        "content": content,
        "keywords": keywords,
        "keyword_count": len(keywords),
        "engagement": round(engagement, 4),
        "replies_count": replies,
        "reblogs_count": reblogs,
        "favourites_count": favourites,
        "is_repost": is_repost,
        "language": raw.get("language"),
    }


def _pull_with_retry(api, username: str, created_after: datetime, since_id: Optional[str]) -> list[dict]:
    """Call pull_statuses with retry on transient failures. Materialises the generator."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            kwargs = {"replies": False, "created_after": created_after}
            if since_id:
                kwargs["since_id"] = since_id
            result = api.pull_statuses(username, **kwargs)
            # pull_statuses may return a generator or a list — materialise either way
            return list(result)
        except Exception as exc:
            logger.warning("API call failed (attempt %d/%d): %s", attempt, MAX_RETRIES, exc)
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_SLEEP_SEC * attempt)
    return []


# ---------------------------------------------------------------------------
# Main pull
# ---------------------------------------------------------------------------

def pull(
    start: datetime = DEFAULT_START,
    include_reposts: bool = False,
    dry_run: bool = False,
) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load existing posts for incremental update
    existing_ids: set[str] = set()
    existing_df = pd.DataFrame()
    since_id: Optional[str] = None

    if OUT_PATH.exists():
        try:
            existing_df = pd.read_parquet(OUT_PATH)
            existing_ids = set(existing_df["post_id"].astype(str))
            if not existing_df.empty:
                # Use the newest post_id as since_id for incremental pull
                # (truthbrush returns newest-first, so max post_id = newest)
                since_id = str(existing_df["post_id"].astype(str).max())
                latest_date = existing_df["posted_at"].max()
                logger.info(
                    "Existing posts: %d  latest=%s  since_id=%s",
                    len(existing_df), latest_date, since_id,
                )
        except Exception as exc:
            logger.warning("Could not read existing parquet: %s — full pull", exc)
            existing_df = pd.DataFrame()

    logger.info("Building API client ...")
    api = _build_api()

    logger.info(
        "Pulling @%s posts since %s%s",
        TRUTHSOCIAL_USERNAME,
        start.date(),
        f" (incremental from post {since_id})" if since_id else "",
    )

    raw_posts = _pull_with_retry(api, TRUTHSOCIAL_USERNAME, created_after=start, since_id=since_id)

    if not raw_posts:
        logger.warning("No posts returned — check credentials and rate limits")
        return

    logger.info("API returned %d raw posts", len(raw_posts))

    rows = []
    skipped_repost = 0
    skipped_empty = 0
    skipped_duplicate = 0

    for raw in raw_posts:
        post_id = str(raw.get("id", ""))

        if post_id in existing_ids:
            skipped_duplicate += 1
            continue

        row = _post_to_row(raw, include_reposts)
        if row is None:
            if raw.get("reblog") is not None:
                skipped_repost += 1
            else:
                skipped_empty += 1
            continue

        rows.append(row)

    logger.info(
        "Parsed: %d new rows  (skipped: reposts=%d, empty=%d, duplicate=%d)",
        len(rows), skipped_repost, skipped_empty, skipped_duplicate,
    )

    if not rows:
        logger.info("No new posts to add.")
        return

    if dry_run:
        logger.info("Dry run — sample posts:")
        for r in rows[:5]:
            logger.info(
                "  [%s] %s  keywords=%s  engagement=%.2f",
                r["posted_at"].strftime("%Y-%m-%d %H:%M"),
                r["content"][:80],
                r["keywords"],
                r["engagement"],
            )
        return

    # Merge and save
    new_df = pd.DataFrame(rows)
    combined = pd.concat([existing_df, new_df], ignore_index=True) if not existing_df.empty else new_df
    combined = combined.drop_duplicates(subset=["post_id"]).sort_values("posted_at").reset_index(drop=True)

    combined.to_parquet(OUT_PATH, index=False)
    size_mb = OUT_PATH.stat().st_size / 1e6

    logger.info(
        "Saved %d total posts → %s  (%.2f MB)",
        len(combined), OUT_PATH, size_mb,
    )

    # Summary
    kw_posts = (combined["keyword_count"] > 0).sum()
    print("\n" + "=" * 60)
    print("TRUTH SOCIAL PULL SUMMARY")
    print("=" * 60)
    print(f"  Total posts stored    : {len(combined)}")
    print(f"  New posts added       : {len(new_df)}")
    print(f"  Date range            : {combined['posted_at'].min().date()} → {combined['posted_at'].max().date()}")
    print(f"  Posts with keywords   : {kw_posts}  ({kw_posts/len(combined)*100:.1f}%)")
    print(f"  Avg engagement        : {combined['engagement'].mean():.2f}")
    print()
    print("  Top keywords:")
    from collections import Counter
    all_kws = [kw for kws in combined["keywords"] for kw in kws]
    for kw, cnt in Counter(all_kws).most_common(10):
        print(f"    {kw:25s}: {cnt}")
    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Convert existing JSONL → parquet
# ---------------------------------------------------------------------------

def import_from_jsonl(jsonl_path: Path, include_reposts: bool = False) -> None:
    """Parse an existing trump_posts.jsonl (raw truthbrush format) and merge into posts.parquet.

    The JSONL was produced by pull_historical_data.py; each line is a raw status dict.
    """
    import json

    if not jsonl_path.exists():
        logger.error("JSONL not found: %s", jsonl_path)
        return

    # Load existing parquet for dedup
    existing_ids: set[str] = set()
    existing_df = pd.DataFrame()
    if OUT_PATH.exists():
        try:
            existing_df = pd.read_parquet(OUT_PATH)
            existing_ids = set(existing_df["post_id"].astype(str))
            logger.info("Existing posts.parquet: %d posts", len(existing_df))
        except Exception as exc:
            logger.warning("Could not read existing parquet: %s", exc)

    raw_posts = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    raw_posts.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    logger.info("Loaded %d raw posts from %s", len(raw_posts), jsonl_path)

    rows = []
    skipped_repost = skipped_empty = skipped_dup = 0

    for raw in raw_posts:
        pid = str(raw.get("id", ""))
        if pid in existing_ids:
            skipped_dup += 1
            continue
        row = _post_to_row(raw, include_reposts)
        if row is None:
            if raw.get("reblog") is not None:
                skipped_repost += 1
            else:
                skipped_empty += 1
            continue
        rows.append(row)

    logger.info(
        "New rows: %d  (skipped: reposts=%d, empty=%d, dup=%d)",
        len(rows), skipped_repost, skipped_empty, skipped_dup,
    )

    if not rows:
        logger.info("No new posts to add.")
        return

    new_df = pd.DataFrame(rows)
    combined = pd.concat([existing_df, new_df], ignore_index=True) if not existing_df.empty else new_df
    combined = combined.drop_duplicates(subset=["post_id"]).sort_values("posted_at").reset_index(drop=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(OUT_PATH, index=False)
    size_mb = OUT_PATH.stat().st_size / 1e6
    logger.info("Saved %d total posts → %s  (%.2f MB)", len(combined), OUT_PATH, size_mb)

    kw_posts = (combined["keyword_count"] > 0).sum()
    from collections import Counter
    all_kws = [kw for kws in combined["keywords"] for kw in kws]
    print("\n" + "=" * 60)
    print("TRUTH SOCIAL IMPORT SUMMARY")
    print("=" * 60)
    print(f"  Total posts stored    : {len(combined)}")
    print(f"  New posts added       : {len(new_df)}")
    print(f"  Date range            : {combined['posted_at'].min().date()} → {combined['posted_at'].max().date()}")
    print(f"  Posts with keywords   : {kw_posts}  ({kw_posts/len(combined)*100:.1f}%)")
    print(f"  Avg engagement        : {combined['engagement'].mean():.2f}")
    print()
    print("  Top keywords:")
    for kw, cnt in Counter(all_kws).most_common(10):
        print(f"    {kw:25s}: {cnt}")
    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pull historical @realDonaldTrump Truth Social posts")
    parser.add_argument(
        "--start", default=DEFAULT_START.strftime("%Y-%m-%d"),
        help="Earliest post date to pull (default: 2025-01-20)",
    )
    parser.add_argument(
        "--include-reposts", action="store_true",
        help="Include reposts/reblogs (default: original posts only)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Fetch and print first 5 posts, don't save",
    )
    parser.add_argument(
        "--from-jsonl", metavar="PATH", default=None,
        help="Convert an existing trump_posts.jsonl to posts.parquet instead of hitting the API",
    )
    args = parser.parse_args()

    if args.from_jsonl:
        import_from_jsonl(Path(args.from_jsonl), include_reposts=args.include_reposts)
    else:
        start_dt = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        pull(
            start=start_dt,
            include_reposts=args.include_reposts,
            dry_run=args.dry_run,
        )
