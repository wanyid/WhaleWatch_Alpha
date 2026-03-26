"""pull_ts_slow.py — Slow, Cloudflare-safe Truth Social backfill scraper.

Bypasses the Cloudflare 1015 rate-limit by:
  1. Using the stored TRUTHSOCIAL_TOKEN directly (no login POST that triggers CF)
  2. Calling the API with curl_cffi impersonating Chrome 136 (TLS fingerprint match)
  3. Pausing 6 seconds between pages (10 req/min — well under CF's soft limit)
  4. Random jitter ±2s to avoid bot-pattern detection
  5. Resuming from a checkpoint file so progress is never lost

Output appends to D:/WhaleWatch_Data/truth_social/trump_posts.jsonl (same format
as the earlier historical pull) and then auto-converts to posts.parquet.

Usage:
    python scripts/pull_ts_slow.py
    python scripts/pull_ts_slow.py --start 2025-01-20   # default
    python scripts/pull_ts_slow.py --delay 10            # seconds between pages
    python scripts/pull_ts_slow.py --limit 50            # posts per page (max 40 for TS)
"""

import argparse
import json
import logging
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("pull_ts_slow")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
API_BASE = "https://truthsocial.com/api"
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 12_2_1) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36"
)

# Trump's Truth Social account ID (stable — won't change)
TRUMP_ACCOUNT_ID = "107780257626128497"

DATA_ROOT = Path("D:/WhaleWatch_Data")
JSONL_PATH = DATA_ROOT / "truth_social" / "trump_posts.jsonl"
CHECKPOINT_PATH = DATA_ROOT / "truth_social" / "pull_ts_checkpoint.json"

DEFAULT_START = "2025-01-20"
DEFAULT_DELAY = 6       # seconds between pages
PAGE_SIZE = 40          # Truth Social max


# ---------------------------------------------------------------------------
# HTTP client — curl_cffi with Chrome fingerprint
# ---------------------------------------------------------------------------

def _get_session(token: str):
    from curl_cffi import requests as cf
    s = cf.Session(impersonate="chrome136")
    s.headers.update({
        "Authorization": f"Bearer {token}",
        "User-Agent": USER_AGENT,
    })
    return s


def _api_get(session, path: str, params: dict = None) -> dict | list | None:
    """GET from Truth Social API. Returns parsed JSON or None on error."""
    url = API_BASE + path
    try:
        resp = session.get(url, params=params, timeout=30)
    except Exception as exc:
        logger.warning("Request failed: %s", exc)
        return None

    if resp.status_code == 200:
        try:
            return resp.json()
        except Exception:
            logger.warning("Could not parse JSON from %s", url)
            return None
    elif resp.status_code == 429 or "rate limited" in resp.text.lower():
        logger.warning("Rate-limited (429) — sleeping 60s then continuing")
        time.sleep(60)
        return None
    elif resp.status_code == 401:
        logger.error("Token expired or invalid (401) — re-login needed")
        return None
    elif "1015" in resp.text or "rate limit" in resp.text.lower():
        logger.warning("Cloudflare 1015 — sleeping 90s")
        time.sleep(90)
        return None
    else:
        logger.warning("HTTP %d for %s", resp.status_code, url)
        return None


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _load_checkpoint() -> dict:
    if CHECKPOINT_PATH.exists():
        try:
            return json.loads(CHECKPOINT_PATH.read_text())
        except Exception:
            pass
    return {}


def _save_checkpoint(max_id: str | None, total_saved: int) -> None:
    CHECKPOINT_PATH.write_text(json.dumps({
        "max_id": max_id,
        "total_saved": total_saved,
        "updated_at": datetime.now(tz=timezone.utc).isoformat(),
    }, indent=2))


# ---------------------------------------------------------------------------
# Existing posts dedup index
# ---------------------------------------------------------------------------

def _load_existing_ids() -> set[str]:
    ids: set[str] = set()
    if JSONL_PATH.exists():
        with open(JSONL_PATH, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        ids.add(str(json.loads(line)["id"]))
                    except Exception:
                        pass
    return ids


# ---------------------------------------------------------------------------
# Main pull
# ---------------------------------------------------------------------------

def pull(start: str = DEFAULT_START, delay: float = DEFAULT_DELAY) -> None:
    token = os.environ.get("TRUTHSOCIAL_TOKEN") or os.environ.get("TRUTHSOCIAL_PASSWORD")
    if not token:
        logger.error("TRUTHSOCIAL_TOKEN not set in .env — cannot proceed")
        return

    # If it's a password not a token, we need to log in first
    if "@" in token or len(token) < 20:
        logger.error(
            "TRUTHSOCIAL_TOKEN looks like a password, not a bearer token. "
            "Run pull_truthsocial_history.py first to obtain the token."
        )
        return

    start_dt = datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    # Load existing JSONL post IDs for dedup
    existing_ids = _load_existing_ids()
    logger.info("Existing JSONL posts: %d", len(existing_ids))

    # Resume from checkpoint if available
    ckpt = _load_checkpoint()
    max_id: str | None = ckpt.get("max_id")
    total_saved = ckpt.get("total_saved", 0)
    if max_id:
        logger.info("Resuming from checkpoint: max_id=%s  total_saved=%d", max_id, total_saved)
    else:
        logger.info("Starting fresh from %s", start)

    session = _get_session(token)

    JSONL_PATH.parent.mkdir(parents=True, exist_ok=True)
    jsonl_file = open(JSONL_PATH, "a", encoding="utf-8")

    page = 0
    stop = False

    try:
        while not stop:
            params = {"exclude_replies": "true", "limit": PAGE_SIZE}
            if max_id:
                params["max_id"] = max_id

            logger.info("Page %d  max_id=%s", page + 1, max_id or "latest")
            result = _api_get(session, f"/v1/accounts/{TRUMP_ACCOUNT_ID}/statuses", params)

            if result is None:
                logger.warning("Empty/error response — retrying after 30s")
                time.sleep(30)
                continue

            if not isinstance(result, list) or len(result) == 0:
                logger.info("No more posts returned — done.")
                break

            # Sort oldest-first within page to find the floor
            posts_sorted = sorted(result, key=lambda p: p["id"])

            new_on_page = 0
            for post in reversed(posts_sorted):  # newest→oldest for max_id logic
                pid = str(post["id"])

                # Check date floor
                try:
                    from dateutil import parser as dp
                    post_dt = dp.parse(post["created_at"]).replace(tzinfo=timezone.utc)
                    if post_dt <= start_dt:
                        logger.info("Reached start date %s — stopping", start)
                        stop = True
                        break
                except Exception:
                    pass

                if pid not in existing_ids:
                    jsonl_file.write(json.dumps(post) + "\n")
                    existing_ids.add(pid)
                    new_on_page += 1
                    total_saved += 1

            # oldest post ID on this page becomes the next max_id
            max_id = posts_sorted[0]["id"]

            logger.info(
                "Page %d: %d posts, %d new  total_saved=%d  oldest=%s",
                page + 1,
                len(result),
                new_on_page,
                total_saved,
                posts_sorted[0].get("created_at", "?")[:10],
            )

            jsonl_file.flush()
            _save_checkpoint(max_id, total_saved)
            page += 1

            if not stop:
                jitter = random.uniform(-2, 2)
                sleep_s = max(2.0, delay + jitter)
                logger.debug("Sleeping %.1fs", sleep_s)
                time.sleep(sleep_s)

    except KeyboardInterrupt:
        logger.info("Interrupted — checkpoint saved. Re-run to continue.")
    finally:
        jsonl_file.close()

    logger.info("Pull complete. %d new posts saved to %s", total_saved, JSONL_PATH)

    # Auto-convert to parquet
    if total_saved > 0:
        logger.info("Converting JSONL → posts.parquet ...")
        try:
            from scripts.pull_truthsocial_history import import_from_jsonl
            import_from_jsonl(JSONL_PATH)
        except Exception as exc:
            logger.warning("Parquet conversion failed: %s — run manually with --from-jsonl", exc)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Slow, CF-safe Truth Social backfill (uses stored token)"
    )
    parser.add_argument(
        "--start", default=DEFAULT_START,
        help=f"Earliest date to pull back to (default: {DEFAULT_START})",
    )
    parser.add_argument(
        "--delay", type=float, default=DEFAULT_DELAY,
        help=f"Seconds between page requests (default: {DEFAULT_DELAY})",
    )
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("WhaleWatch_Alpha — Truth Social Slow Backfill")
    logger.info("Target  : @realDonaldTrump (ID: %s)", TRUMP_ACCOUNT_ID)
    logger.info("From    : %s", args.start)
    logger.info("Delay   : %.1fs/page  (~%d pages/min)", args.delay, 60 / args.delay)
    logger.info("Output  : %s", JSONL_PATH)
    logger.info("=" * 60)

    pull(start=args.start, delay=args.delay)
