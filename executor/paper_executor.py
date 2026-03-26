"""PaperExecutor — simulates order execution and tracks P&L in SQLite.

Each approved SignalEvent becomes a "position" row. When the holding period
expires (or stop-loss / take-profit is hit), the position is closed and
realized_pnl is computed from the entry/exit prices fetched from the market
data provider.

Database: D:/WhaleWatch_Data/paper_trades.db  (path from settings.yaml or env)

Tables:
  positions  — one row per open/closed trade
  daily_pnl  — aggregated session P&L for monitoring

Usage (called by main.py):
    executor = PaperExecutor(market_provider)
    order_id = executor.submit_signal(event)
    ...
    executor.close_position(order_id, reason="TIMEOUT")
    executor.close_expired_positions()   # call periodically to sweep timeouts
"""

import logging
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import yaml

from executor.base_executor import BaseExecutor
from models.signal_event import SignalEvent
from scanners.market_data.base_provider import BaseMarketDataProvider

logger = logging.getLogger(__name__)

_SETTINGS_PATH = "config/settings.yaml"
_DEFAULT_DB = "D:/WhaleWatch_Data/paper_trades.db"

_DDL = """
CREATE TABLE IF NOT EXISTS positions (
    order_id            TEXT PRIMARY KEY,
    event_id            TEXT NOT NULL,
    created_at          TEXT NOT NULL,
    signal_direction    TEXT NOT NULL,
    signal_ticker       TEXT NOT NULL,
    confidence          REAL,
    holding_period_min  INTEGER,
    stop_loss_pct       REAL,
    take_profit_pct     REAL,
    entry_price         REAL,
    exit_price          REAL,
    realized_pnl        REAL,
    outcome             TEXT,           -- WIN | LOSS | STOP_OUT | OPEN
    closed_at           TEXT,
    close_reason        TEXT
);

CREATE TABLE IF NOT EXISTS daily_pnl (
    session_date        TEXT PRIMARY KEY,
    total_pnl           REAL DEFAULT 0.0,
    trade_count         INTEGER DEFAULT 0,
    win_count           INTEGER DEFAULT 0
);
"""


class PaperExecutor(BaseExecutor):
    def __init__(self, market_provider: BaseMarketDataProvider) -> None:
        self._provider = market_provider
        self._db_path = self._resolve_db_path()
        self._init_db()

    # ------------------------------------------------------------------
    # BaseExecutor interface
    # ------------------------------------------------------------------

    def submit_signal(self, event: SignalEvent) -> str:
        """Record an approved signal as an open position. Returns order_id."""
        order_id = str(uuid.uuid4())
        entry_price = self._provider.get_latest_price(event.signal_ticker)

        event.market_price_at_signal = entry_price

        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO positions
                  (order_id, event_id, created_at, signal_direction, signal_ticker,
                   confidence, holding_period_min, stop_loss_pct, take_profit_pct,
                   entry_price, outcome)
                VALUES (?,?,?,?,?,?,?,?,?,?,'OPEN')
                """,
                (
                    order_id,
                    event.event_id,
                    event.created_at.isoformat(),
                    event.signal_direction,
                    event.signal_ticker,
                    event.confidence,
                    event.holding_period_minutes,
                    event.stop_loss_pct,
                    event.take_profit_pct,
                    entry_price,
                ),
            )

        logger.info(
            "PAPER OPEN  order=%s  %s %s  entry=%.4f  hold=%dm",
            order_id[:8],
            event.signal_direction,
            event.signal_ticker,
            entry_price or 0,
            event.holding_period_minutes or 0,
        )
        return order_id

    def close_position(self, order_id: str, reason: str = "MANUAL") -> Optional[float]:
        """Close an open position. Returns realized P&L or None if not found."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT signal_direction, signal_ticker, entry_price, stop_loss_pct, take_profit_pct "
                "FROM positions WHERE order_id=? AND outcome='OPEN'",
                (order_id,),
            ).fetchone()

        if not row:
            logger.warning("close_position: order %s not found or already closed", order_id)
            return None

        direction, ticker, entry_price, sl_pct, tp_pct = row
        exit_price = self._provider.get_latest_price(ticker)

        pnl = self._compute_pnl(direction, entry_price, exit_price, sl_pct, tp_pct)
        outcome = self._classify_outcome(pnl, direction, entry_price, exit_price, sl_pct, tp_pct)

        now = datetime.now(tz=timezone.utc).isoformat()
        with self._conn() as conn:
            conn.execute(
                """
                UPDATE positions SET
                  exit_price=?, realized_pnl=?, outcome=?, closed_at=?, close_reason=?
                WHERE order_id=?
                """,
                (exit_price, pnl, outcome, now, reason, order_id),
            )
            self._upsert_daily_pnl(conn, pnl, outcome)

        logger.info(
            "PAPER CLOSE order=%s  %s %s  entry=%.4f  exit=%.4f  pnl=%.4f  %s  reason=%s",
            order_id[:8],
            direction,
            ticker,
            entry_price or 0,
            exit_price or 0,
            pnl,
            outcome,
            reason,
        )
        return pnl

    def close_expired_positions(self) -> int:
        """Close any OPEN positions whose holding period has elapsed. Returns count closed."""
        now = datetime.now(tz=timezone.utc)
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT order_id, created_at, holding_period_min FROM positions WHERE outcome='OPEN'"
            ).fetchall()

        closed = 0
        for order_id, created_at_str, hold_min in rows:
            created_at = datetime.fromisoformat(created_at_str)
            if created_at.tzinfo is None:
                created_at = created_at.replace(tzinfo=timezone.utc)
            elapsed = (now - created_at).total_seconds() / 60
            if hold_min and elapsed >= hold_min:
                self.close_position(order_id, reason="TIMEOUT")
                closed += 1

        return closed

    def open_positions(self) -> list[dict]:
        """Return all currently open positions."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT order_id, event_id, signal_direction, signal_ticker, "
                "entry_price, holding_period_min, created_at "
                "FROM positions WHERE outcome='OPEN'"
            ).fetchall()
        cols = ["order_id", "event_id", "direction", "ticker", "entry_price", "hold_min", "created_at"]
        return [dict(zip(cols, r)) for r in rows]

    def session_summary(self) -> dict:
        """Return today's P&L summary."""
        today = datetime.now(tz=timezone.utc).date().isoformat()
        with self._conn() as conn:
            row = conn.execute(
                "SELECT total_pnl, trade_count, win_count FROM daily_pnl WHERE session_date=?",
                (today,),
            ).fetchone()
        if row:
            total_pnl, trade_count, win_count = row
            win_rate = win_count / trade_count if trade_count else 0.0
            return {"date": today, "total_pnl": total_pnl, "trade_count": trade_count,
                    "win_count": win_count, "win_rate": win_rate}
        return {"date": today, "total_pnl": 0.0, "trade_count": 0, "win_count": 0, "win_rate": 0.0}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_pnl(
        self,
        direction: str,
        entry: Optional[float],
        exit_: Optional[float],
        sl_pct: Optional[float],
        tp_pct: Optional[float],
    ) -> float:
        if entry is None or exit_ is None:
            return 0.0
        raw = (exit_ - entry) / entry if direction == "BUY" else (entry - exit_) / entry
        # Clamp to stop-loss / take-profit bounds
        if sl_pct and raw < -sl_pct:
            raw = -sl_pct
        if tp_pct and raw > tp_pct:
            raw = tp_pct
        return round(raw, 6)

    def _classify_outcome(
        self,
        pnl: float,
        direction: str,
        entry: Optional[float],
        exit_: Optional[float],
        sl_pct: Optional[float],
        tp_pct: Optional[float],
    ) -> str:
        if entry is None or exit_ is None:
            return "OPEN"
        raw = (exit_ - entry) / entry if direction == "BUY" else (entry - exit_) / entry
        if sl_pct and raw <= -sl_pct:
            return "STOP_OUT"
        if pnl > 0:
            return "WIN"
        return "LOSS"

    def _upsert_daily_pnl(self, conn: sqlite3.Connection, pnl: float, outcome: str) -> None:
        today = datetime.now(tz=timezone.utc).date().isoformat()
        win = 1 if outcome == "WIN" else 0
        conn.execute(
            """
            INSERT INTO daily_pnl (session_date, total_pnl, trade_count, win_count)
            VALUES (?, ?, 1, ?)
            ON CONFLICT(session_date) DO UPDATE SET
              total_pnl   = total_pnl + excluded.total_pnl,
              trade_count = trade_count + 1,
              win_count   = win_count + excluded.win_count
            """,
            (today, pnl, win),
        )

    def _resolve_db_path(self) -> str:
        try:
            with open(_SETTINGS_PATH, "r") as f:
                cfg = yaml.safe_load(f)
            return cfg.get("executor", {}).get("paper_db_path", _DEFAULT_DB)
        except Exception:
            return _DEFAULT_DB

    def _init_db(self) -> None:
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as conn:
            conn.executescript(_DDL)

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self._db_path)
