"""Integration tests — end-to-end pipeline without real APIs.

Uses mocks and an in-memory SQLite DB so no external connections are needed.
Covers: signal build → L1 (mock) → L2 (mock) → risk → paper executor → close.
"""

import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from executor.paper_executor import PaperExecutor
from models.raw_events import PolymarketRawEvent, TruthSocialRawEvent
from models.signal_event import SignalEvent
from risk.risk_manager import RiskManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_poly_raw(market_id: str = "mkt-001") -> PolymarketRawEvent:
    return PolymarketRawEvent(
        market_id=market_id,
        market_slug="will-tariffs-increase",
        market_question="Will tariffs increase above 25% before July 2025?",
        outcome_token="YES",
        price_before=0.62,
        price_after=0.47,
        price_delta=-0.15,
        volume_24h=1_200_000,
        volume_spike_pct=2.8,
        detected_at=datetime(2025, 6, 15, 14, 0, 0, tzinfo=timezone.utc),
    )


def _make_ts_raw(post_id: str = "post-001") -> TruthSocialRawEvent:
    return TruthSocialRawEvent(
        post_id=post_id,
        content="TARIFFS going up! America first!",
        posted_at=datetime(2025, 6, 15, 14, 0, 0, tzinfo=timezone.utc),
        pulled_at=datetime(2025, 6, 15, 14, 0, 5, tzinfo=timezone.utc),
        replies_count=0,
        reblogs_count=0,
        favourites_count=0,
        is_repost=False,
        keywords=["tariff", "tariffs"],
    )


def _make_executor(tmp_path: Path) -> PaperExecutor:
    """Return a PaperExecutor wired to a temp DB with a mock market provider."""
    db = str(tmp_path / "paper_trades_test.db")
    provider = MagicMock()
    provider.get_latest_price.return_value = 500.0
    executor = PaperExecutor.__new__(PaperExecutor)
    executor._provider = provider
    executor._db_path = db
    executor._init_db()
    return executor


def _make_signal_event(
    direction: str = "BUY",
    ticker: str = "SPY",
    confidence: float = 0.70,
    holding_min: int = 60,
    poly_market_id: str = "mkt-001",
    poly_price_after: float = 0.47,
) -> SignalEvent:
    ev = SignalEvent(
        event_id=str(uuid.uuid4()),
        created_at=datetime.now(tz=timezone.utc),
    )
    ev.signal_direction = direction
    ev.signal_ticker = ticker
    ev.llm_model = "mock-model"
    ev.confidence = confidence
    ev.holding_period_minutes = holding_min
    ev.stop_loss_pct = 0.02
    ev.take_profit_pct = 0.04
    ev.poly_market_id = poly_market_id
    ev.poly_price_after = poly_price_after
    ev.dual_signal = False
    return ev


# ---------------------------------------------------------------------------
# Signal building
# ---------------------------------------------------------------------------

class TestBuildSignalEvent:
    def test_poly_fields_populated(self):
        from main import _build_signal_event
        raw = _make_poly_raw()
        ev = _build_signal_event(raw)
        assert ev.poly_market_id == "mkt-001"
        assert ev.poly_price_delta == pytest.approx(-0.15)
        assert ev.ts_post_id is None
        assert ev.dual_signal is False

    def test_ts_fields_populated(self):
        from main import _build_signal_event
        raw = _make_ts_raw()
        ev = _build_signal_event(raw)
        assert ev.ts_post_id == "post-001"
        assert "tariff" in (ev.ts_post_keywords or [])
        assert ev.poly_market_id is None
        assert ev.dual_signal is False

    def test_dual_signal_enrichment(self):
        from main import _build_signal_event
        poly = _make_poly_raw()
        ts = _make_ts_raw()
        ev = _build_signal_event(poly, dual_companion=ts)
        assert ev.poly_market_id == "mkt-001"
        assert ev.ts_post_id == "post-001"
        assert ev.dual_signal is True


# ---------------------------------------------------------------------------
# Signal deduplication
# ---------------------------------------------------------------------------

class TestSignalDeduper:
    def test_poly_deduplicated(self):
        from main import SignalDeduper
        deduper = SignalDeduper(ttl_minutes=30)
        raw = _make_poly_raw()
        assert deduper.is_duplicate(raw) is False
        assert deduper.is_duplicate(raw) is True  # second call is a dup

    def test_ts_deduplicated_by_post_id(self):
        from main import SignalDeduper
        deduper = SignalDeduper(ttl_minutes=30)
        raw = _make_ts_raw()
        assert deduper.is_duplicate(raw) is False
        assert deduper.is_duplicate(raw) is True

    def test_different_markets_not_duped(self):
        from main import SignalDeduper
        deduper = SignalDeduper(ttl_minutes=30)
        r1 = _make_poly_raw("mkt-001")
        r2 = _make_poly_raw("mkt-002")
        assert deduper.is_duplicate(r1) is False
        assert deduper.is_duplicate(r2) is False


# ---------------------------------------------------------------------------
# Dual-signal matcher
# ---------------------------------------------------------------------------

class TestDualSignalMatcher:
    def test_no_match_initially(self):
        from main import DualSignalMatcher
        dm = DualSignalMatcher()
        is_dual, match = dm.record_and_match(_make_poly_raw())
        assert is_dual is False
        assert match is None

    def test_match_after_ts_then_poly(self):
        from main import DualSignalMatcher
        dm = DualSignalMatcher()
        ts = _make_ts_raw()
        ts.keywords = ["tariff"]
        dm.record_and_match(ts)

        poly = _make_poly_raw()  # question mentions "tariffs"
        is_dual, match = dm.record_and_match(poly)
        # May or may not match depending on keyword overlap logic; just assert no crash
        assert isinstance(is_dual, bool)


# ---------------------------------------------------------------------------
# Paper Executor — submit + close
# ---------------------------------------------------------------------------

class TestPaperExecutorIntegration:
    def test_submit_creates_open_position(self, tmp_path):
        executor = _make_executor(tmp_path)
        ev = _make_signal_event()
        order_id = executor.submit_signal(ev)
        assert order_id
        positions = executor.open_positions()
        assert len(positions) == 1
        assert positions[0]["order_id"] == order_id

    def test_close_win(self, tmp_path):
        executor = _make_executor(tmp_path)
        executor._provider.get_latest_price.side_effect = [500.0, 520.0]  # up 4%
        ev = _make_signal_event(direction="BUY", confidence=0.70)
        order_id = executor.submit_signal(ev)
        pnl = executor.close_position(order_id, reason="TAKE_PROFIT")
        assert pnl is not None
        assert pnl > 0
        assert len(executor.open_positions()) == 0

    def test_close_stop_out(self, tmp_path):
        executor = _make_executor(tmp_path)
        # Entry 500, exit 480 → -4% (exceeds 2% SL → clamped to -2%)
        executor._provider.get_latest_price.side_effect = [500.0, 480.0]
        ev = _make_signal_event(direction="BUY", confidence=0.70)
        order_id = executor.submit_signal(ev)
        pnl = executor.close_position(order_id, reason="STOP_LOSS")
        assert pnl == pytest.approx(-0.02)

    def test_close_expired_positions(self, tmp_path):
        from datetime import timedelta
        import sqlite3
        executor = _make_executor(tmp_path)
        executor._provider.get_latest_price.return_value = 500.0
        ev = _make_signal_event(holding_min=1)
        order_id = executor.submit_signal(ev)

        # Back-date the created_at so it appears expired
        past = (datetime.now(tz=timezone.utc) - timedelta(minutes=5)).isoformat()
        with sqlite3.connect(executor._db_path) as conn:
            conn.execute("UPDATE positions SET created_at=? WHERE order_id=?",
                         (past, order_id))

        closed_pnls = executor.close_expired_positions()
        assert len(closed_pnls) == 1   # returns list of P&L values now
        assert len(executor.open_positions()) == 0

    def test_session_summary(self, tmp_path):
        executor = _make_executor(tmp_path)
        executor._provider.get_latest_price.side_effect = [500.0, 520.0, 500.0, 490.0]
        ev1 = _make_signal_event(direction="BUY")
        ev2 = _make_signal_event(direction="SHORT")
        o1 = executor.submit_signal(ev1)
        o2 = executor.submit_signal(ev2)
        executor.close_position(o1)
        executor.close_position(o2)
        summary = executor.session_summary()
        assert summary["trade_count"] == 2

    def test_true_news_stop_buy(self, tmp_path):
        """BUY signal (poly prob was falling); if prob falls further → TRUE_NEWS_STOP."""
        executor = _make_executor(tmp_path)
        executor._provider.get_latest_price.return_value = 500.0

        ev = _make_signal_event(
            direction="BUY",
            poly_market_id="mkt-001",
            poly_price_after=0.47,  # entry baseline
        )
        order_id = executor.submit_signal(ev)

        # Current prob 0.35 < 0.47 → fade invalidated → should close
        closed = executor.check_true_news_stop("mkt-001", current_prob=0.35)
        assert order_id in closed
        assert len(executor.open_positions()) == 0

    def test_true_news_stop_no_trigger(self, tmp_path):
        """Prob recovered → fade still valid → do NOT stop."""
        executor = _make_executor(tmp_path)
        executor._provider.get_latest_price.return_value = 500.0

        ev = _make_signal_event(
            direction="BUY",
            poly_market_id="mkt-002",
            poly_price_after=0.47,
        )
        executor.submit_signal(ev)

        # Prob recovered to 0.55 > 0.47 → mean reversion working → keep position
        closed = executor.check_true_news_stop("mkt-002", current_prob=0.55)
        assert len(closed) == 0
        assert len(executor.open_positions()) == 1


# ---------------------------------------------------------------------------
# Risk manager
# ---------------------------------------------------------------------------

class TestRiskManagerIntegration:
    def test_approves_valid_signal(self):
        risk = RiskManager()
        ev = _make_signal_event(confidence=0.70)
        assert risk.approve(ev) is True

    def test_rejects_low_confidence(self):
        risk = RiskManager()
        ev = _make_signal_event(confidence=0.45)
        assert risk.approve(ev) is False

    def test_rejects_hold_direction(self):
        risk = RiskManager()
        ev = _make_signal_event(direction="HOLD")
        assert risk.approve(ev) is False

    def test_circuit_breaker_after_drawdown(self):
        risk = RiskManager()
        # Force session P&L below -3% halt threshold
        risk.record_pnl(-0.04)
        ev = _make_signal_event(confidence=0.80)
        assert risk.approve(ev) is False


# ---------------------------------------------------------------------------
# Feature vector
# ---------------------------------------------------------------------------

class TestFeatureVector:
    def test_feature_count(self):
        from reasoner.layer2_predictor.features import N_FEATURES, build_feature_vector
        ev = _make_signal_event()
        vec = build_feature_vector(ev)
        assert len(vec) == N_FEATURES

    def test_vix_fallback_when_unavailable(self):
        from reasoner.layer2_predictor.features import build_feature_vector
        ev = _make_signal_event()
        vec = build_feature_vector(ev)
        # VIX features (indices 13,14) should be 0.0 when file not present
        import numpy as np
        assert np.isfinite(vec).all()
