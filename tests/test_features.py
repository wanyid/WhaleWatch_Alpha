"""Tests for shared feature modules and feature-set consistency.

Covers:
  - post_features / poly_features export the expected groups and totals
  - ALL_DIRECTIONAL_FEATURES and ALL_FADE_FEATURES are deduplicated and ordered
  - Saved spy_direction_*.pkl models carry a 'features' list that matches
    ALL_DIRECTIONAL_FEATURES (ensures training/inference alignment)
  - PolymarketScanner._build_features() returns all keys in ALL_DIRECTIONAL_FEATURES
  - close_expired_positions() returns a list of floats (for circuit breaker wiring)
"""

import pickle
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reasoner.layer2_predictor.post_features import (
    ALL_DIRECTIONAL_FEATURES as POST_DIR_FEATURES,
    ALL_FADE_FEATURES as POST_FADE_FEATURES,
    KEYWORD_FEATURES,
    POST_FEATURES,
    TEMPORAL_FEATURES,
    MARKET_FEATURES,
    FADE_FEATURES as POST_FADE_ONLY,
)
from reasoner.layer2_predictor.poly_features import (
    ALL_DIRECTIONAL_FEATURES as POLY_DIR_FEATURES,
    ALL_FADE_FEATURES as POLY_FADE_FEATURES,
    STRENGTH_FEATURES,
    TOPIC_FEATURES,
    REGIME_FEATURES,
    FADE_FEATURES as POLY_FADE_ONLY,
)

MODELS_DIR = Path(__file__).resolve().parent.parent / "models" / "saved"


# ---------------------------------------------------------------------------
# post_features
# ---------------------------------------------------------------------------

class TestPostFeatures:
    def test_directional_total(self):
        assert len(POST_DIR_FEATURES) == 20

    def test_directional_groups(self):
        assert len(KEYWORD_FEATURES) == 7
        assert len(POST_FEATURES) == 6
        assert len(TEMPORAL_FEATURES) == 4
        assert len(MARKET_FEATURES) == 3

    def test_directional_is_concat(self):
        assert POST_DIR_FEATURES == (
            KEYWORD_FEATURES + POST_FEATURES + TEMPORAL_FEATURES + MARKET_FEATURES
        )

    def test_no_duplicates_directional(self):
        assert len(POST_DIR_FEATURES) == len(set(POST_DIR_FEATURES))

    def test_fade_total(self):
        assert len(POST_FADE_FEATURES) == 23

    def test_fade_is_directional_plus_fade(self):
        assert POST_FADE_FEATURES == POST_DIR_FEATURES + POST_FADE_ONLY

    def test_no_duplicates_fade(self):
        assert len(POST_FADE_FEATURES) == len(set(POST_FADE_FEATURES))

    def test_fade_feature_names(self):
        expected = ["spy_initial_ret", "spy_initial_direction", "spy_abs_initial_ret"]
        assert POST_FADE_ONLY == expected


# ---------------------------------------------------------------------------
# poly_features
# ---------------------------------------------------------------------------

class TestPolyFeatures:
    def test_directional_total(self):
        assert len(POLY_DIR_FEATURES) == 24

    def test_directional_groups(self):
        assert len(STRENGTH_FEATURES) == 13
        assert len(TOPIC_FEATURES) == 5
        assert len(REGIME_FEATURES) == 6

    def test_directional_is_concat(self):
        assert POLY_DIR_FEATURES == (
            STRENGTH_FEATURES + TOPIC_FEATURES + REGIME_FEATURES
        )

    def test_no_duplicates_directional(self):
        assert len(POLY_DIR_FEATURES) == len(set(POLY_DIR_FEATURES))

    def test_fade_total(self):
        assert len(POLY_FADE_FEATURES) == 27

    def test_fade_is_directional_plus_fade(self):
        assert POLY_FADE_FEATURES == POLY_DIR_FEATURES + POLY_FADE_ONLY

    def test_no_duplicates_fade(self):
        assert len(POLY_FADE_FEATURES) == len(set(POLY_FADE_FEATURES))

    def test_fade_feature_names(self):
        expected = ["initial_ret", "initial_direction", "abs_initial_ret"]
        assert POLY_FADE_ONLY == expected


# ---------------------------------------------------------------------------
# Saved model / features alignment
# ---------------------------------------------------------------------------

class TestSavedModelFeatures:
    """Verify that already-trained spy_direction_*.pkl models store a 'features'
    key whose contents are all present in POST_DIR_FEATURES.

    If this test fails after a model re-train, it means the training script
    drifted from the shared feature module.
    """

    @pytest.mark.parametrize(
        "pkl_path",
        sorted(MODELS_DIR.glob("spy_direction_*.pkl")) if MODELS_DIR.exists() else [],
    )
    def test_model_features_subset_of_shared(self, pkl_path):
        with open(pkl_path, "rb") as f:
            payload = pickle.load(f)

        assert "features" in payload, f"{pkl_path.name} missing 'features' key"
        stored = payload["features"]
        assert isinstance(stored, list), f"{pkl_path.name}: 'features' should be list"

        unknown = [feat for feat in stored if feat not in POST_DIR_FEATURES]
        assert unknown == [], (
            f"{pkl_path.name}: stored features not in ALL_DIRECTIONAL_FEATURES: {unknown}"
        )

    def test_skipped_gracefully_when_no_models(self):
        """Non-parametric sentinel: always passes — documents intent."""
        pass


# ---------------------------------------------------------------------------
# PolymarketScanner._build_features() key coverage
# ---------------------------------------------------------------------------

class TestPolymarketScannerFeatures:
    def test_build_features_covers_all_directional(self):
        """Scanner's live feature builder must produce every key that the
        trained model expects.  Uses a minimal mock of scanner internals."""
        from scanners.polymarket_scanner import SessionManager
        from models.raw_events import PolymarketRawEvent

        mgr = SessionManager.__new__(SessionManager)
        mgr._vix_level      = 18.0
        mgr._vix_percentile = 0.4
        mgr._vixy_level     = 17.5
        mgr._timeout_min    = 60
        mgr._min_confidence = 0.60
        mgr._model          = None
        mgr._features       = None

        now = datetime(2026, 3, 10, 14, 30, tzinfo=timezone.utc)
        events = [
            PolymarketRawEvent(
                market_id="mkt-001",
                market_slug="tariff-25",
                market_question="Will tariffs exceed 25%?",
                outcome_token="YES",
                price_before=0.55,
                price_after=0.70,
                price_delta=0.15,
                volume_24h=2_000_000,
                volume_spike_pct=80.0,
                detected_at=now,
            )
        ]

        feat = mgr._build_features(events, session_start=now)

        missing = [k for k in POLY_DIR_FEATURES if k not in feat]
        assert missing == [], f"Scanner missing features: {missing}"


# ---------------------------------------------------------------------------
# close_expired_positions return type (circuit breaker wiring)
# ---------------------------------------------------------------------------

class TestCloseExpiredReturnType:
    def test_returns_list_of_floats(self, tmp_path):
        from executor.paper_executor import PaperExecutor
        from models.signal_event import SignalEvent
        import sqlite3
        from datetime import timedelta

        db = str(tmp_path / "test.db")
        provider = MagicMock()
        provider.get_latest_price.return_value = 500.0

        executor = PaperExecutor.__new__(PaperExecutor)
        executor._provider = provider
        executor._db_path = db
        executor._init_db()

        ev = SignalEvent(
            event_id=str(uuid.uuid4()),
            created_at=datetime.now(tz=timezone.utc),
        )
        ev.signal_direction = "BUY"
        ev.signal_ticker = "SPY"
        ev.llm_model = "test"
        ev.confidence = 0.70
        ev.holding_period_minutes = 1
        ev.stop_loss_pct = 0.02
        ev.take_profit_pct = 0.04
        ev.dual_signal = False

        order_id = executor.submit_signal(ev)

        # Back-date the position so it expires immediately
        past = (datetime.now(tz=timezone.utc) - timedelta(minutes=5)).isoformat()
        with sqlite3.connect(db) as conn:
            conn.execute(
                "UPDATE positions SET created_at=? WHERE order_id=?", (past, order_id)
            )

        result = executor.close_expired_positions()
        assert isinstance(result, list), "close_expired_positions should return a list"
        assert len(result) == 1
        assert all(isinstance(v, float) for v in result)
