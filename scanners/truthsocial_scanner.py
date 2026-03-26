"""Truth Social scanner — Signal B.

Polls @realDonaldTrump for new posts using the truthbrush library and
emits TruthSocialRawEvent objects. Runs as an independent signal source;
the Reasoner will consume these events whether or not Polymarket data is present.

Rate-limit note: Truth Social / Cloudflare blocks after ~40-50 rapid requests.
Default poll interval is 60 seconds to stay well within safe limits.
"""

import html
import logging
import os
import re
import time
from datetime import datetime, timezone
from typing import Generator, Optional

from models.raw_events import TruthSocialRawEvent
from scanners.base_scanner import BaseScanner

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Market-relevant keyword taxonomy
# Topics that historically precede equity / vol moves when posted by Trump
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

# Flat lookup set for fast membership test
_ALL_KEYWORDS: set[str] = {kw for group in KEYWORD_GROUPS.values() for kw in group}


def _strip_html(text: str) -> str:
    """Remove HTML tags and decode HTML entities."""
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def _extract_keywords(text: str) -> list[str]:
    """Return market-relevant keywords found in the post text (lowercased)."""
    lower = text.lower()
    return sorted({kw for kw in _ALL_KEYWORDS if kw in lower})


def _parse_dt(value: str | datetime) -> datetime:
    """Normalize a datetime value to UTC-aware datetime."""
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value
    # truthbrush returns ISO 8601 strings like "2025-01-20T12:00:00.000Z"
    value = value.rstrip("Z")
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


class TruthSocialScanner(BaseScanner):
    """Polls @realDonaldTrump on Truth Social for new posts.

    Args:
        username: Truth Social handle to monitor (default: realDonaldTrump).
        poll_interval: Seconds between polls (default: 60).
        start_since_id: If provided, only fetch posts newer than this ID on
            the first poll. Pass None to fetch the latest batch on startup
            and use that as the baseline (no backfill).
        include_reposts: Whether to yield reposts / reblogs (default: False).
    """

    def __init__(
        self,
        username: str = "realDonaldTrump",
        poll_interval: int = 60,
        start_since_id: Optional[str] = None,
        include_reposts: bool = False,
    ) -> None:
        self._username = username
        self._poll_interval = poll_interval
        self._since_id: Optional[str] = start_since_id
        self._include_reposts = include_reposts
        self._api = self._build_api()

    def name(self) -> str:
        return f"TruthSocialScanner({self._username})"

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def scan(self) -> Generator[TruthSocialRawEvent, None, None]:
        """Yield new TruthSocialRawEvent objects as they appear.

        On the very first call, if no start_since_id was provided, we fetch
        the latest post, record its ID as baseline, and do NOT yield it —
        this prevents flooding the pipeline with historical posts on startup.
        """
        logger.info("%s starting poll loop (interval=%ds)", self.name(), self._poll_interval)

        first_run = self._since_id is None

        while True:
            try:
                posts = self._fetch_new_posts()

                if first_run and posts:
                    # Establish baseline — don't process existing posts
                    self._since_id = posts[0]["id"]
                    logger.info(
                        "%s baseline set to post_id=%s, skipping %d existing posts",
                        self.name(), self._since_id, len(posts),
                    )
                    first_run = False
                else:
                    for raw in posts:
                        event = self._to_event(raw)
                        if event is not None:
                            logger.info(
                                "%s new post id=%s keywords=%s",
                                self.name(), event.post_id, event.keywords,
                            )
                            yield event

            except Exception as exc:
                logger.warning("%s poll error: %s — retrying in %ds", self.name(), exc, self._poll_interval)

            time.sleep(self._poll_interval)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_api(self):
        """Instantiate the truthbrush Api client from environment credentials."""
        try:
            from truthbrush.api import Api  # type: ignore
        except ImportError as e:
            raise ImportError(
                "truthbrush is not installed. Run: pip install truthbrush"
            ) from e

        username = os.environ.get("TRUTHSOCIAL_USERNAME")
        password = os.environ.get("TRUTHSOCIAL_PASSWORD")

        if not username or not password:
            raise EnvironmentError(
                "TRUTHSOCIAL_USERNAME and TRUTHSOCIAL_PASSWORD must be set in environment. "
                "Copy .env.example to .env and fill in your credentials."
            )

        return Api(username, password)

    def _fetch_new_posts(self) -> list[dict]:
        """Fetch posts newer than self._since_id, newest-first."""
        posts: list[dict] = []

        try:
            for post in self._api.pull_statuses(
                self._username,
                replies=False,
                since_id=self._since_id,
            ):
                posts.append(post)
        except StopIteration:
            pass

        if posts:
            # Update cursor to the newest post ID (posts come newest-first)
            self._since_id = posts[0]["id"]

        return posts

    def _to_event(self, raw: dict) -> Optional[TruthSocialRawEvent]:
        """Convert a raw truthbrush post dict to a TruthSocialRawEvent.

        Returns None if the post should be filtered out (e.g. repost when
        include_reposts=False, or empty content).
        """
        is_repost = raw.get("reblog") is not None

        if is_repost and not self._include_reposts:
            return None

        # If this is a repost, use the original post's content
        content_raw = raw.get("content", "")
        if is_repost and raw.get("reblog"):
            content_raw = raw["reblog"].get("content", content_raw)

        content = _strip_html(content_raw)
        if not content:
            return None

        keywords = _extract_keywords(content)
        pulled_at = datetime.now(tz=timezone.utc)

        return TruthSocialRawEvent(
            post_id=str(raw["id"]),
            content=content,
            posted_at=_parse_dt(raw["created_at"]),
            pulled_at=pulled_at,
            replies_count=int(raw.get("replies_count", 0)),
            reblogs_count=int(raw.get("reblogs_count", 0)),
            favourites_count=int(raw.get("favourites_count", 0)),
            keywords=keywords,
            is_repost=is_repost,
            language=raw.get("language"),
        )
