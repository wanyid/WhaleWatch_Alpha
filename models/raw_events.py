from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional


@dataclass
class TruthSocialRawEvent:
    """Raw event emitted by the Truth Social scanner.

    Represents a single post from @realDonaldTrump. Fields map directly
    to the truthbrush API response; no interpretation is applied here.
    """
    post_id: str
    content: str                        # raw HTML-stripped text
    posted_at: datetime                 # UTC
    pulled_at: datetime                 # when the scanner fetched this
    replies_count: int
    reblogs_count: int
    favourites_count: int
    keywords: List[str]                 # pre-LLM market-relevant keyword tags
    is_repost: bool                     # True if this is a reblog of another post
    language: Optional[str] = None


@dataclass
class PolymarketRawEvent:
    """Raw event emitted by the Polymarket scanner (stub for future use)."""
    market_id: str
    market_slug: str
    market_question: str
    outcome_token: str
    price_before: float
    price_after: float
    price_delta: float
    volume_24h: float
    volume_spike_pct: float
    detected_at: datetime
