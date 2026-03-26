"""RiskManager — applies guardrails between Layer 2 output and Executor.

Responsibilities:
  1. Reject signals below the minimum confidence threshold.
  2. Set stop_loss_pct and take_profit_pct on the SignalEvent.
  3. Enforce the daily drawdown circuit breaker — no new trades once the
     portfolio is down more than `daily_drawdown_halt_pct` on the session.
  4. Clip holding_period_minutes to [1, 4320].

Usage:
    rm = RiskManager()
    approved = rm.approve(event)   # returns True/False
    if approved:
        executor.submit_signal(event)
"""

import logging
from datetime import date, datetime, timezone

import yaml

from models.signal_event import SignalEvent

logger = logging.getLogger(__name__)

_SETTINGS_PATH = "config/settings.yaml"


def _load_risk_cfg() -> dict:
    with open(_SETTINGS_PATH, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg.get("risk", {})


def _load_l2_cfg() -> dict:
    with open(_SETTINGS_PATH, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg.get("reasoner", {}).get("layer2", {})


class RiskManager:
    """Stateful guardrail layer; tracks intra-session P&L for circuit breaker."""

    def __init__(self) -> None:
        risk_cfg = _load_risk_cfg()
        l2_cfg = _load_l2_cfg()

        self.stop_loss_pct: float = risk_cfg.get("stop_loss_pct", 0.02)
        self.take_profit_pct: float = risk_cfg.get("take_profit_pct", 0.04)
        self.daily_drawdown_halt_pct: float = risk_cfg.get("daily_drawdown_halt_pct", 0.03)
        self.min_confidence: float = l2_cfg.get("min_confidence", 0.60)
        self.min_holding_minutes: int = l2_cfg.get("min_holding_minutes", 1)
        self.max_holding_minutes: int = l2_cfg.get("max_holding_minutes", 4320)

        # Session state
        self._session_date: date = datetime.now(tz=timezone.utc).date()
        self._session_pnl: float = 0.0   # sum of realized_pnl for today
        self._halted: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def approve(self, event: SignalEvent) -> bool:
        """Return True if the signal passes all risk checks and is ready to execute.

        Side effects on approval:
          - Sets event.stop_loss_pct and event.take_profit_pct.
          - Clips event.holding_period_minutes to [min, max].
        """
        self._roll_session_if_new_day()

        # 1. Circuit breaker
        if self._halted:
            logger.warning(
                "RISK HALT — daily drawdown exceeded %.1f%%. Signal %s rejected.",
                self.daily_drawdown_halt_pct * 100,
                event.event_id,
            )
            return False

        # 2. Confidence threshold
        confidence = event.confidence or 0.0
        if confidence < self.min_confidence:
            logger.info(
                "Signal %s rejected — confidence %.3f < threshold %.2f",
                event.event_id,
                confidence,
                self.min_confidence,
            )
            return False

        # 3. Must have direction and ticker from L1
        if event.signal_direction in (None, "HOLD"):
            logger.info("Signal %s rejected — direction is HOLD or None", event.event_id)
            return False

        if not event.signal_ticker:
            logger.info("Signal %s rejected — no ticker from L1", event.event_id)
            return False

        # 4. Stamp risk parameters onto the event
        event.stop_loss_pct = self.stop_loss_pct
        event.take_profit_pct = self.take_profit_pct

        # 5. Clip holding period
        if event.holding_period_minutes is not None:
            event.holding_period_minutes = max(
                self.min_holding_minutes,
                min(self.max_holding_minutes, event.holding_period_minutes),
            )

        logger.info(
            "Signal %s APPROVED — %s %s  confidence=%.3f  hold=%dm  sl=%.1f%%  tp=%.1f%%",
            event.event_id,
            event.signal_direction,
            event.signal_ticker,
            confidence,
            event.holding_period_minutes or 0,
            self.stop_loss_pct * 100,
            self.take_profit_pct * 100,
        )
        return True

    def record_pnl(self, pnl: float) -> None:
        """Call this when a position closes so the circuit breaker stays current."""
        self._roll_session_if_new_day()
        self._session_pnl += pnl

        if self._session_pnl < -abs(self.daily_drawdown_halt_pct):
            self._halted = True
            logger.warning(
                "CIRCUIT BREAKER TRIGGERED — session P&L %.4f < -%.1f%%  "
                "No new signals until next session.",
                self._session_pnl,
                self.daily_drawdown_halt_pct * 100,
            )

    @property
    def session_pnl(self) -> float:
        return self._session_pnl

    @property
    def is_halted(self) -> bool:
        return self._halted

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _roll_session_if_new_day(self) -> None:
        today = datetime.now(tz=timezone.utc).date()
        if today != self._session_date:
            logger.info(
                "New trading day %s — resetting session P&L (was %.4f) and circuit breaker.",
                today,
                self._session_pnl,
            )
            self._session_date = today
            self._session_pnl = 0.0
            self._halted = False
