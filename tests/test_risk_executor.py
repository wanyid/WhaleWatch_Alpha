"""Smoke tests for RiskManager + PaperExecutor (no live API calls needed)."""

import sys
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from executor.paper_executor import PaperExecutor
from models.signal_event import SignalEvent
from risk.risk_manager import RiskManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_event(direction="BUY", ticker="SPY", confidence=0.75, hold_min=60) -> SignalEvent:
    return SignalEvent(
        event_id=str(uuid.uuid4()),
        created_at=datetime.now(tz=timezone.utc),
        signal_direction=direction,
        signal_ticker=ticker,
        confidence=confidence,
        holding_period_minutes=hold_min,
    )


def _make_executor(price: float = 500.0) -> tuple[PaperExecutor, str]:
    """Return a PaperExecutor backed by a temp SQLite DB."""
    tmp = tempfile.mktemp(suffix=".db")
    provider = MagicMock()
    provider.get_latest_price.return_value = price
    executor = PaperExecutor.__new__(PaperExecutor)
    executor._provider = provider
    executor._db_path = tmp
    executor._init_db()
    return executor, tmp


# ---------------------------------------------------------------------------
# RiskManager tests
# ---------------------------------------------------------------------------

class TestRiskManager:
    def test_approve_valid_signal(self):
        rm = RiskManager()
        ev = _make_event(confidence=0.75)
        assert rm.approve(ev) is True
        assert ev.stop_loss_pct == rm.stop_loss_pct
        assert ev.take_profit_pct == rm.take_profit_pct

    def test_reject_low_confidence(self):
        rm = RiskManager()
        ev = _make_event(confidence=0.50)
        assert rm.approve(ev) is False

    def test_reject_hold_direction(self):
        rm = RiskManager()
        ev = _make_event(direction="HOLD")
        assert rm.approve(ev) is False

    def test_reject_none_direction(self):
        rm = RiskManager()
        ev = _make_event(direction=None)
        assert rm.approve(ev) is False

    def test_circuit_breaker_triggers(self):
        rm = RiskManager()
        rm.record_pnl(-0.04)   # -4% > daily_drawdown_halt_pct (3%)
        assert rm.is_halted is True
        ev = _make_event(confidence=0.90)
        assert rm.approve(ev) is False   # rejected by circuit breaker

    def test_holding_period_clipped(self):
        rm = RiskManager()
        ev = _make_event(hold_min=99999)
        rm.approve(ev)
        assert ev.holding_period_minutes == rm.max_holding_minutes

    def test_holding_period_floor(self):
        rm = RiskManager()
        ev = _make_event(hold_min=0)
        rm.approve(ev)
        assert ev.holding_period_minutes == rm.min_holding_minutes


# ---------------------------------------------------------------------------
# PaperExecutor tests
# ---------------------------------------------------------------------------

class TestPaperExecutor:
    def test_submit_and_open_position(self):
        executor, _ = _make_executor(price=500.0)
        ev = _make_event()
        ev.stop_loss_pct = 0.02
        ev.take_profit_pct = 0.04
        order_id = executor.submit_signal(ev)
        assert order_id
        positions = executor.open_positions()
        assert len(positions) == 1
        assert positions[0]["order_id"] == order_id

    def test_close_winning_position(self):
        executor, _ = _make_executor(price=510.0)   # exit higher → WIN for BUY
        # Set up open position at entry=500
        ev = _make_event(direction="BUY")
        ev.stop_loss_pct = 0.02
        ev.take_profit_pct = 0.04
        executor._provider.get_latest_price.return_value = 500.0
        order_id = executor.submit_signal(ev)

        # Close at 510 (2% gain)
        executor._provider.get_latest_price.return_value = 510.0
        pnl = executor.close_position(order_id, reason="TEST")
        assert pnl > 0

    def test_close_losing_position(self):
        executor, _ = _make_executor(price=490.0)   # exit lower → LOSS for BUY
        ev = _make_event(direction="BUY")
        ev.stop_loss_pct = 0.02
        ev.take_profit_pct = 0.04
        executor._provider.get_latest_price.return_value = 500.0
        order_id = executor.submit_signal(ev)

        executor._provider.get_latest_price.return_value = 490.0
        pnl = executor.close_position(order_id, reason="TEST")
        assert pnl < 0

    def test_stop_loss_clamps_pnl(self):
        executor, _ = _make_executor()
        ev = _make_event(direction="BUY")
        ev.stop_loss_pct = 0.02
        ev.take_profit_pct = 0.04
        executor._provider.get_latest_price.return_value = 500.0
        order_id = executor.submit_signal(ev)

        # Exit at catastrophic loss (-10%) → clamped to -2%
        executor._provider.get_latest_price.return_value = 450.0
        pnl = executor.close_position(order_id, reason="STOP")
        assert pnl == pytest.approx(-0.02, abs=1e-5)

    def test_session_summary(self):
        executor, _ = _make_executor()
        ev = _make_event()
        ev.stop_loss_pct = 0.02
        ev.take_profit_pct = 0.04
        executor._provider.get_latest_price.return_value = 500.0
        order_id = executor.submit_signal(ev)
        executor._provider.get_latest_price.return_value = 510.0
        executor.close_position(order_id)
        summary = executor.session_summary()
        assert summary["trade_count"] == 1
        assert summary["win_count"] == 1
