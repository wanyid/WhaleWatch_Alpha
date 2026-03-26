"""Polymarket scanner — Signal A.

Monitors the Polymarket CLOB for anomalous price/volume moves in politically-
relevant prediction markets (tariffs, cabinet picks, geopolitical events, etc.)
and emits PolymarketRawEvent objects when thresholds are exceeded.

No authentication is required — all endpoints used here are public read-only.
"""

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Deque, Dict, Generator, List, Optional

import requests

from models.raw_events import PolymarketRawEvent
from scanners.base_scanner import BaseScanner

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Gamma API — market discovery (no auth required)
# ---------------------------------------------------------------------------
GAMMA_BASE = "https://gamma-api.polymarket.com"

# ---------------------------------------------------------------------------
# Keyword filter: only watch markets whose question contains these terms.
# Keeps the watchlist focused on events likely to move SPY/QQQ/VIX.
# ---------------------------------------------------------------------------
WATCHLIST_KEYWORDS: list[str] = [
    # Trade / tariffs
    "tariff", "tariffs", "trade war", "trade deal", "import duty",
    # Geopolitical
    "ukraine", "russia", "china", "iran", "israel", "nato", "north korea",
    "ceasefire", "invasion", "sanctions",
    # US executive / policy
    "trump", "executive order", "cabinet", "fired", "resign", "appointed",
    "department of government efficiency", "doge",
    # Macro / markets
    "federal reserve", "fed rate", "interest rate", "inflation",
    "recession", "gdp", "debt ceiling",
    # Energy
    "oil price", "opec", "lng",
]


# ---------------------------------------------------------------------------
# Internal state per tracked market
# ---------------------------------------------------------------------------
@dataclass
class _MarketState:
    """Rolling price and volume observations for one market."""
    condition_id: str
    question: str
    slug: str
    yes_token_id: str
    # Rolling window of (timestamp, yes_price) tuples
    price_history: Deque[tuple[datetime, float]] = field(
        default_factory=lambda: deque(maxlen=20)
    )
    # Rolling window of volume_24h snapshots
    volume_history: Deque[float] = field(
        default_factory=lambda: deque(maxlen=20)
    )


class PolymarketScanner(BaseScanner):
    """Polls Polymarket for anomalous moves in politically-relevant markets.

    Anomaly triggers (either condition is sufficient to emit an event):
      - |YES price delta| >= min_price_delta  (vs oldest reading in window)
      - volume_24h spike >= volume_spike_threshold_pct  (vs rolling average)

    Args:
        poll_interval: Seconds between market price polls (default: 30).
        volume_spike_threshold_pct: % above rolling avg volume to trigger
            (default: 50.0 → 50% spike).
        min_price_delta: Absolute YES-price change to trigger
            (default: 0.05 → 5 percentage-point move).
        watchlist_refresh_interval: How often (seconds) to re-discover
            relevant markets from Gamma (default: 3600 → 1 hour).
    """

    def __init__(
        self,
        poll_interval: int = 30,
        volume_spike_threshold_pct: float = 50.0,
        min_price_delta: float = 0.05,
        watchlist_refresh_interval: int = 3600,
    ) -> None:
        self._poll_interval = poll_interval
        self._volume_spike_threshold = volume_spike_threshold_pct
        self._min_price_delta = min_price_delta
        self._watchlist_refresh_interval = watchlist_refresh_interval

        self._markets: Dict[str, _MarketState] = {}   # condition_id → state
        self._last_watchlist_refresh: float = 0.0
        self._session = requests.Session()
        self._session.headers.update({"Accept": "application/json"})

    def name(self) -> str:
        return "PolymarketScanner"

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def scan(self) -> Generator[PolymarketRawEvent, None, None]:
        """Continuously poll Polymarket and yield anomaly events."""
        logger.info("%s starting poll loop (interval=%ds)", self.name(), self._poll_interval)

        while True:
            try:
                self._maybe_refresh_watchlist()
                yield from self._poll_and_detect()
            except Exception as exc:
                logger.warning("%s poll error: %s — retrying in %ds", self.name(), exc, self._poll_interval)

            time.sleep(self._poll_interval)

    # ------------------------------------------------------------------
    # Watchlist discovery (Gamma API)
    # ------------------------------------------------------------------

    def _maybe_refresh_watchlist(self) -> None:
        """Refresh the watchlist if the refresh interval has elapsed."""
        now = time.monotonic()
        if now - self._last_watchlist_refresh < self._watchlist_refresh_interval:
            return

        logger.info("%s refreshing market watchlist...", self.name())
        markets = self._discover_relevant_markets()

        # Add newly discovered markets; preserve existing rolling state
        added = 0
        for m in markets:
            cid = m["condition_id"]
            if cid not in self._markets:
                yes_token = self._extract_yes_token_id(m)
                if yes_token:
                    self._markets[cid] = _MarketState(
                        condition_id=cid,
                        question=m.get("question", ""),
                        slug=m.get("market_slug", ""),
                        yes_token_id=yes_token,
                    )
                    added += 1

        self._last_watchlist_refresh = now
        logger.info(
            "%s watchlist: %d total markets (+%d new)",
            self.name(), len(self._markets), added,
        )

    def _discover_relevant_markets(self) -> list[dict]:
        """Query Gamma API for active markets matching watchlist keywords."""
        relevant: list[dict] = []
        offset = 0
        limit = 100

        while True:
            try:
                resp = self._session.get(
                    f"{GAMMA_BASE}/markets",
                    params={
                        "closed": "false",
                        "active": "true",
                        "limit": limit,
                        "offset": offset,
                    },
                    timeout=10,
                )
                resp.raise_for_status()
            except requests.RequestException as exc:
                logger.warning("%s Gamma API error: %s", self.name(), exc)
                break

            data = resp.json()
            # Gamma returns a list or a dict with a "markets" key depending on version
            items: list[dict] = data if isinstance(data, list) else data.get("markets", [])

            if not items:
                break

            for market in items:
                question = market.get("question", "").lower()
                if self._is_relevant(question):
                    relevant.append(market)

            if len(items) < limit:
                break  # Last page
            offset += limit

        logger.info("%s discovered %d relevant markets", self.name(), len(relevant))
        return relevant

    def _is_relevant(self, question_lower: str) -> bool:
        """Return True if the market question contains any watchlist keyword."""
        return any(kw in question_lower for kw in WATCHLIST_KEYWORDS)

    def _extract_yes_token_id(self, market: dict) -> Optional[str]:
        """Extract the YES outcome token_id from a market dict."""
        for token in market.get("tokens", []):
            if str(token.get("outcome", "")).upper() == "YES":
                return str(token.get("token_id", ""))
        # Fallback: some market shapes use clobTokenIds list
        clob_ids = market.get("clobTokenIds", [])
        if clob_ids:
            return str(clob_ids[0])
        return None

    # ------------------------------------------------------------------
    # Price polling + anomaly detection (CLOB API)
    # ------------------------------------------------------------------

    def _poll_and_detect(self) -> Generator[PolymarketRawEvent, None, None]:
        """Fetch current prices for all watched markets and emit anomalies."""
        if not self._markets:
            return

        now = datetime.now(tz=timezone.utc)

        for state in list(self._markets.values()):
            try:
                price, volume = self._fetch_price_and_volume(state)
            except Exception as exc:
                logger.debug("%s price fetch failed for %s: %s", self.name(), state.slug, exc)
                continue

            if price is None:
                continue

            # Record observations
            state.price_history.append((now, price))
            if volume is not None:
                state.volume_history.append(volume)

            # Need at least 2 price readings to compute delta
            if len(state.price_history) < 2:
                continue

            # Compute anomaly metrics
            oldest_price = state.price_history[0][1]
            price_delta = price - oldest_price

            volume_spike_pct = 0.0
            if len(state.volume_history) >= 3 and volume is not None:
                avg_volume = sum(list(state.volume_history)[:-1]) / (len(state.volume_history) - 1)
                if avg_volume > 0:
                    volume_spike_pct = ((volume - avg_volume) / avg_volume) * 100.0

            is_price_anomaly = abs(price_delta) >= self._min_price_delta
            is_volume_anomaly = volume_spike_pct >= self._volume_spike_threshold

            if not (is_price_anomaly or is_volume_anomaly):
                continue

            outcome_token = "YES" if price_delta >= 0 else "NO"
            logger.info(
                "%s ANOMALY: %s | price_delta=%.3f volume_spike=%.1f%%",
                self.name(), state.question[:60], price_delta, volume_spike_pct,
            )

            yield PolymarketRawEvent(
                market_id=state.condition_id,
                market_slug=state.slug,
                market_question=state.question,
                outcome_token=outcome_token,
                price_before=oldest_price,
                price_after=price,
                price_delta=price_delta,
                volume_24h=volume or 0.0,
                volume_spike_pct=volume_spike_pct,
                detected_at=now,
            )

    def _fetch_price_and_volume(self, state: _MarketState) -> tuple[Optional[float], Optional[float]]:
        """Fetch current YES midpoint and 24h volume for a market.

        Uses the CLOB /midpoint endpoint for price and the Gamma /markets
        endpoint for volume (CLOB doesn't expose volume directly per market).
        """
        # YES midpoint price from CLOB
        price: Optional[float] = None
        try:
            resp = self._session.get(
                "https://clob.polymarket.com/midpoint",
                params={"token_id": state.yes_token_id},
                timeout=5,
            )
            resp.raise_for_status()
            data = resp.json()
            mid = data.get("mid")
            if mid is not None:
                price = float(mid)
        except Exception as exc:
            logger.debug("CLOB midpoint fetch failed: %s", exc)

        # 24h volume from Gamma (lightweight)
        volume: Optional[float] = None
        try:
            resp = self._session.get(
                f"{GAMMA_BASE}/markets/{state.condition_id}",
                timeout=5,
            )
            resp.raise_for_status()
            data = resp.json()
            v = data.get("volume24hr") or data.get("volume_24h") or data.get("volumeNum")
            if v is not None:
                volume = float(v)
        except Exception as exc:
            logger.debug("Gamma volume fetch failed: %s", exc)

        return price, volume
