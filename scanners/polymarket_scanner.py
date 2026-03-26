"""Polymarket scanner — Signal A.

Monitors the Polymarket CLOB for anomalous price/volume moves in politically-
relevant prediction markets (tariffs, cabinet picks, geopolitical events, etc.)

Two-stage pipeline:
  1. Raw anomaly detection — emits PolymarketRawEvent when a YES-price or
     volume spike threshold is crossed (one event per market per observation).
  2. Session aggregation — SessionManager groups nearby events into signal
     sessions; at session close it builds the feature vector, runs the L2
     model, and yields a scored PolymarketSessionEvent if confidence >= min.

No authentication required — all endpoints used here are public read-only.
"""

import logging
import pickle
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Deque, Dict, Generator, List, Optional

import numpy as np
import pandas as pd
import requests

from models.raw_events import PolymarketRawEvent
from scanners.base_scanner import BaseScanner

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR   = PROJECT_ROOT / "models" / "saved"

# ---------------------------------------------------------------------------
# Gamma / CLOB endpoints
# ---------------------------------------------------------------------------
GAMMA_BASE = "https://gamma-api.polymarket.com"
CLOB_BASE  = "https://clob.polymarket.com"

# ---------------------------------------------------------------------------
# Topic bucket classification (mirrors build_poly_market_data.py)
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

WATCHLIST_KEYWORDS = ALL_KEYWORDS   # backward compatibility


def _classify_topic(question: str) -> str:
    q = question.lower()
    for bucket, keywords in TOPIC_BUCKETS.items():
        if any(kw in q for kw in keywords):
            return bucket
    return "other"


def _is_relevant(question: str) -> bool:
    q = question.lower()
    return any(kw in q for kw in ALL_KEYWORDS)


# ---------------------------------------------------------------------------
# Per-market rolling state
# ---------------------------------------------------------------------------
@dataclass
class _MarketState:
    condition_id: str
    question: str
    slug: str
    yes_token_id: str
    topic_bucket: str
    price_history:  Deque[tuple[datetime, float]] = field(
        default_factory=lambda: deque(maxlen=20)
    )
    volume_history: Deque[float] = field(
        default_factory=lambda: deque(maxlen=20)
    )


# ---------------------------------------------------------------------------
# Scored session event
# ---------------------------------------------------------------------------
@dataclass
class PolymarketSessionEvent:
    """Emitted when a session closes and the L2 model has scored it."""
    session_start:         datetime
    session_end:           datetime
    dominant_topic:        str
    n_events:              int
    n_markets:             int
    n_corroborating:       int
    n_opposing:            int
    max_price_delta:       float
    cumulative_delta:      float
    # L2 model output
    signal_direction:      str    # "BUY" | "SHORT" | "HOLD"
    confidence:            float  # estimated win rate 0.0–1.0
    holding_period_minutes: int
    # Contributing raw events (for upstream context / LLM enrichment)
    raw_events:            List[PolymarketRawEvent] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Session manager
# ---------------------------------------------------------------------------
class SessionManager:
    """Aggregates PolymarketRawEvents into scored sessions.

    A session opens when the first anomaly event arrives and closes
    when no new event is received for session_timeout_min minutes.
    At close, it builds the feature vector, runs the L2 model, and
    returns a PolymarketSessionEvent if confidence >= min_confidence.

    Args:
        session_timeout_min: Idle minutes before a session is closed.
        min_confidence:      Minimum L2 win probability to emit a signal.
        vix_level:           Current VIX level (injected from outside; caller
                             should refresh this periodically).
        vix_percentile:      Current VIX percentile (0–1).
        model_period:        Which holding period model to use (e.g. "1h").
    """

    def __init__(
        self,
        session_timeout_min: int = 60,
        min_confidence: float = 0.60,
        vix_level: float = 20.0,
        vix_percentile: float = 0.5,
        vixy_level: float = 0.0,
        model_period: str = "1h",
    ) -> None:
        self._timeout_min    = session_timeout_min
        self._min_confidence = min_confidence
        self._vix_level      = vix_level
        self._vix_percentile = vix_percentile
        self._vixy_level     = vixy_level
        self._model_period   = model_period

        self._events:     List[PolymarketRawEvent]          = []
        self._session_open: Optional[datetime]              = None
        self._last_event_time: Optional[datetime]           = None
        self._model:      Optional[object]                  = None
        self._features:   Optional[List[str]]               = None

        self._load_model(model_period)

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self, period: str) -> None:
        """Load the best available L2 model for the given period."""
        # Prefer VIX-regime model if we know current regime
        suffixes = ["_high_vix", "_low_vix", ""] if self._vix_level >= 20 else ["_low_vix", ""]
        for suffix in suffixes:
            path = MODELS_DIR / f"poly_direction_{period}{suffix}.pkl"
            if path.exists():
                try:
                    with open(path, "rb") as f:
                        payload = pickle.load(f)
                    self._model    = payload["model"]
                    self._features = payload["features"]
                    logger.info("SessionManager: loaded %s (AUC=%.3f)", path.name,
                                payload.get("cv_metrics", {}).get("auc", 0))
                    return
                except Exception as exc:
                    logger.warning("SessionManager: failed to load %s: %s", path.name, exc)
        logger.warning(
            "SessionManager: no poly_direction_%s*.pkl found in %s — "
            "sessions will emit with confidence=0 (HOLD)",
            period, MODELS_DIR,
        )

    def update_vix(self, vix_level: float, vix_percentile: float, vixy_level: float = 0.0) -> None:
        """Update market regime; reload model if regime boundary crossed."""
        old_regime           = "high" if self._vix_level >= 20 else "low"
        new_regime           = "high" if vix_level >= 20 else "low"
        self._vix_level      = vix_level
        self._vix_percentile = vix_percentile
        self._vixy_level     = vixy_level
        if old_regime != new_regime:
            logger.info("VIX regime change (%s → %s); reloading model", old_regime, new_regime)
            self._load_model(self._model_period)

    # ------------------------------------------------------------------
    # Event ingestion
    # ------------------------------------------------------------------

    def add_event(self, event: PolymarketRawEvent) -> None:
        """Add a raw anomaly event to the current session."""
        now = event.detected_at
        if self._session_open is None:
            self._session_open = now
            logger.debug("SessionManager: new session opened at %s", now)
        self._events.append(event)
        self._last_event_time = now

    def check_expiry(self) -> Optional[PolymarketSessionEvent]:
        """Return a scored session event if the session has timed out, else None."""
        if not self._events or self._last_event_time is None:
            return None
        now     = datetime.now(tz=timezone.utc)
        elapsed = (now - self._last_event_time).total_seconds() / 60
        if elapsed >= self._timeout_min:
            return self._close_session()
        return None

    # ------------------------------------------------------------------
    # Session scoring
    # ------------------------------------------------------------------

    def _close_session(self) -> Optional[PolymarketSessionEvent]:
        events        = self._events
        session_start = self._session_open
        session_end   = self._last_event_time

        # Reset state immediately
        self._events          = []
        self._session_open    = None
        self._last_event_time = None

        if not events:
            return None

        features = self._build_features(events, session_start)
        direction, confidence, holding = self._score(features)

        n_events = len(events)
        n_markets = len({e.market_id for e in events})
        cum_delta = sum(e.price_delta for e in events)
        dom_sign  = 1 if cum_delta >= 0 else -1
        n_corr    = sum(1 for e in events if e.price_delta * dom_sign > 0)
        n_opp     = sum(1 for e in events if e.price_delta * dom_sign < 0)

        topics    = [_classify_topic(e.market_question) for e in events]
        dom_topic = max(set(topics), key=topics.count) if topics else "other"

        scored = PolymarketSessionEvent(
            session_start         = session_start,
            session_end           = session_end,
            dominant_topic        = dom_topic,
            n_events              = n_events,
            n_markets             = n_markets,
            n_corroborating       = n_corr,
            n_opposing            = n_opp,
            max_price_delta       = max(abs(e.price_delta) for e in events),
            cumulative_delta      = cum_delta,
            signal_direction      = direction,
            confidence            = confidence,
            holding_period_minutes = holding,
            raw_events            = events,
        )

        logger.info(
            "Session closed: topic=%s  events=%d  markets=%d  delta=%.3f  "
            "→ %s  conf=%.2f  hold=%dm",
            dom_topic, n_events, n_markets, cum_delta,
            direction, confidence, holding,
        )
        return scored if direction != "HOLD" else None

    def _build_features(
        self,
        events: List[PolymarketRawEvent],
        session_start: datetime,
    ) -> dict:
        """Build the feature vector that the L2 model expects."""
        cum_delta  = sum(e.price_delta for e in events)
        dom_sign   = 1 if cum_delta >= 0 else -1
        n_corr     = sum(1 for e in events if e.price_delta * dom_sign > 0)
        duration   = (
            (events[-1].detected_at - session_start).total_seconds() / 60
            if len(events) > 1 else 0.0
        )
        topics     = [_classify_topic(e.market_question) for e in events]

        local_ts   = session_start.astimezone(
            __import__("zoneinfo").ZoneInfo("America/New_York")
        )

        return {
            "max_price_delta":      max(abs(e.price_delta) for e in events),
            "cumulative_delta":     cum_delta,
            "net_delta_abs":        abs(cum_delta),
            "dominant_direction":   1 if cum_delta >= 0 else -1,
            "n_events":             len(events),
            "n_markets":            len({e.market_id for e in events}),
            "n_corroborating":      n_corr,
            "n_opposing":           len(events) - n_corr,
            "corroboration_ratio":  n_corr / max(len(events), 1),
            "session_duration_min": duration,
            "has_tariff":           int("tariff" in topics),
            "has_geopolitical":     int("geopolitical" in topics),
            "has_fed":              int("fed" in topics),
            "has_energy":           int("energy" in topics),
            "has_executive":        int("executive" in topics),
            "vix_level":            self._vix_level,
            "vix_percentile":       self._vix_percentile,
            "vixy_level":           self._vixy_level,
            "is_market_hours":      int(9 <= local_ts.hour < 16 and local_ts.weekday() < 5),
            "hour_of_day":          local_ts.hour,
            "day_of_week":          local_ts.weekday(),
        }

    def _score(self, feature_dict: dict) -> tuple[str, float, int]:
        """Run L2 model → (direction, confidence, holding_period_minutes)."""
        if self._model is None or self._features is None:
            return "HOLD", 0.0, 0

        try:
            row    = pd.DataFrame([{f: feature_dict.get(f, 0) for f in self._features}])
            prob   = float(self._model.predict_proba(row)[0, 1])
            conf   = max(prob, 1 - prob)   # distance from 0.5, regardless of direction
            direction = "BUY" if prob >= 0.5 else "SHORT"

            if conf < self._min_confidence:
                return "HOLD", conf, 0

            # Holding period: map model period to minutes
            period_map = {"5m": 5, "30m": 30, "1h": 60, "2h": 120, "4h": 240, "1d": 1440}
            holding    = period_map.get(self._model_period, 60)
            return direction, conf, holding

        except Exception as exc:
            logger.warning("SessionManager scoring error: %s", exc)
            return "HOLD", 0.0, 0


# ---------------------------------------------------------------------------
# Main scanner
# ---------------------------------------------------------------------------

class PolymarketScanner(BaseScanner):
    """Polls Polymarket for anomalous moves in politically-relevant markets.

    Raw anomaly triggers (either is sufficient to emit a PolymarketRawEvent):
      - |YES price delta| >= min_price_delta  (vs oldest reading in window)
      - volume_24h spike >= volume_spike_threshold_pct  (vs rolling average)

    Raw events are fed into a SessionManager that groups them into sessions
    and scores each session with the L2 model.

    Args:
        poll_interval:              Seconds between price polls (default: 30).
        volume_spike_threshold_pct: % above rolling avg to trigger (default: 50).
        min_price_delta:            Absolute YES-price change to trigger (default: 0.05).
        watchlist_refresh_interval: Seconds between Gamma market discovery (default: 3600).
        session_timeout_min:        Idle minutes before session closes (default: 60).
        min_confidence:             Min L2 win probability to emit signal (default: 0.60).
        model_period:               L2 model holding period to load (default: "1h").
    """

    def __init__(
        self,
        poll_interval: int = 30,
        volume_spike_threshold_pct: float = 50.0,
        min_price_delta: float = 0.05,
        watchlist_refresh_interval: int = 3600,
        session_timeout_min: int = 60,
        min_confidence: float = 0.60,
        model_period: str = "1h",
    ) -> None:
        self._poll_interval          = poll_interval
        self._volume_spike_threshold = volume_spike_threshold_pct
        self._min_price_delta        = min_price_delta
        self._watchlist_refresh_interval = watchlist_refresh_interval

        self._markets: Dict[str, _MarketState] = {}
        self._last_watchlist_refresh: float    = 0.0
        self._session_mgr = SessionManager(
            session_timeout_min = session_timeout_min,
            min_confidence      = min_confidence,
            model_period        = model_period,
        )

        self._http = requests.Session()
        self._http.headers.update({"Accept": "application/json"})

    def name(self) -> str:
        return "PolymarketScanner"

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def scan(self) -> Generator[PolymarketSessionEvent, None, None]:
        """Continuously poll Polymarket; yield scored PolymarketSessionEvents."""
        logger.info("%s starting (poll=%ds)", self.name(), self._poll_interval)

        while True:
            try:
                self._maybe_refresh_watchlist()
                raw_events = list(self._poll_and_detect())

                for ev in raw_events:
                    self._session_mgr.add_event(ev)

                # Check if any open session has timed out
                scored = self._session_mgr.check_expiry()
                if scored is not None:
                    yield scored

            except Exception as exc:
                logger.warning("%s poll error: %s — retrying in %ds",
                               self.name(), exc, self._poll_interval)

            time.sleep(self._poll_interval)

    def update_vix(self, vix_level: float, vix_percentile: float, vixy_level: float = 0.0) -> None:
        """Inject current VIX/VIXY into the session manager (call periodically)."""
        self._session_mgr.update_vix(vix_level, vix_percentile, vixy_level)

    # ------------------------------------------------------------------
    # Watchlist discovery
    # ------------------------------------------------------------------

    def _maybe_refresh_watchlist(self) -> None:
        now = time.monotonic()
        if now - self._last_watchlist_refresh < self._watchlist_refresh_interval:
            return
        logger.info("%s refreshing market watchlist...", self.name())
        markets = self._discover_relevant_markets()
        added   = 0
        for m in markets:
            cid = m.get("conditionId", "")
            if cid and cid not in self._markets:
                yes_token = self._extract_yes_token_id(m)
                if yes_token:
                    q = m.get("question", "")
                    self._markets[cid] = _MarketState(
                        condition_id = cid,
                        question     = q,
                        slug         = m.get("market_slug", ""),
                        yes_token_id = yes_token,
                        topic_bucket = _classify_topic(q),
                    )
                    added += 1
        self._last_watchlist_refresh = now
        logger.info("%s watchlist: %d total (+%d new)", self.name(), len(self._markets), added)

    def _discover_relevant_markets(self) -> list[dict]:
        relevant: list[dict] = []
        offset = 0
        limit  = 100
        while True:
            try:
                resp = self._http.get(
                    f"{GAMMA_BASE}/markets",
                    params={"closed": "false", "active": "true",
                            "limit": limit, "offset": offset},
                    timeout=10,
                )
                resp.raise_for_status()
                data  = resp.json()
                items = data if isinstance(data, list) else data.get("markets", [])
                if not items:
                    break
                for m in items:
                    if _is_relevant(m.get("question", "")):
                        relevant.append(m)
                if len(items) < limit:
                    break
                offset += limit
            except requests.RequestException as exc:
                logger.warning("%s Gamma API error: %s", self.name(), exc)
                break
        logger.info("%s discovered %d relevant markets", self.name(), len(relevant))
        return relevant

    def _extract_yes_token_id(self, market: dict) -> Optional[str]:
        for token in market.get("tokens", []):
            if str(token.get("outcome", "")).upper() == "YES":
                return str(token.get("token_id", ""))
        clob_ids = market.get("clobTokenIds", [])
        return str(clob_ids[0]) if clob_ids else None

    # ------------------------------------------------------------------
    # Price polling + raw anomaly detection
    # ------------------------------------------------------------------

    def _poll_and_detect(self) -> Generator[PolymarketRawEvent, None, None]:
        if not self._markets:
            return
        now = datetime.now(tz=timezone.utc)

        for state in list(self._markets.values()):
            try:
                price, volume = self._fetch_price_and_volume(state)
            except Exception as exc:
                logger.debug("%s price fetch failed for %s: %s",
                             self.name(), state.slug, exc)
                continue

            if price is None:
                continue

            state.price_history.append((now, price))
            if volume is not None:
                state.volume_history.append(volume)

            if len(state.price_history) < 2:
                continue

            oldest_price = state.price_history[0][1]
            price_delta  = price - oldest_price

            volume_spike_pct = 0.0
            if len(state.volume_history) >= 3 and volume is not None:
                avg_vol = sum(list(state.volume_history)[:-1]) / (len(state.volume_history) - 1)
                if avg_vol > 0:
                    volume_spike_pct = ((volume - avg_vol) / avg_vol) * 100.0

            is_price_anomaly  = abs(price_delta) >= self._min_price_delta
            is_volume_anomaly = volume_spike_pct >= self._volume_spike_threshold

            if not (is_price_anomaly or is_volume_anomaly):
                continue

            outcome_token = "YES" if price_delta >= 0 else "NO"
            logger.info(
                "%s ANOMALY: [%s] %s | Δprice=%.3f  vol_spike=%.1f%%",
                self.name(), state.topic_bucket, state.question[:55],
                price_delta, volume_spike_pct,
            )

            yield PolymarketRawEvent(
                market_id         = state.condition_id,
                market_slug       = state.slug,
                market_question   = state.question,
                outcome_token     = outcome_token,
                price_before      = oldest_price,
                price_after       = price,
                price_delta       = price_delta,
                volume_24h        = volume or 0.0,
                volume_spike_pct  = volume_spike_pct,
                detected_at       = now,
            )

    def _fetch_price_and_volume(
        self, state: _MarketState
    ) -> tuple[Optional[float], Optional[float]]:
        price: Optional[float]  = None
        volume: Optional[float] = None

        try:
            resp = self._http.get(
                f"{CLOB_BASE}/midpoint",
                params={"token_id": state.yes_token_id},
                timeout=5,
            )
            resp.raise_for_status()
            mid = resp.json().get("mid")
            if mid is not None:
                price = float(mid)
        except Exception as exc:
            logger.debug("CLOB midpoint error %s: %s", state.slug, exc)

        try:
            resp = self._http.get(
                f"{GAMMA_BASE}/markets/{state.condition_id}",
                timeout=5,
            )
            resp.raise_for_status()
            data = resp.json()
            v    = data.get("volume24hr") or data.get("volume_24h") or data.get("volumeNum")
            if v is not None:
                volume = float(v)
        except Exception as exc:
            logger.debug("Gamma volume error %s: %s", state.slug, exc)

        return price, volume
