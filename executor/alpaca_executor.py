"""AlpacaExecutor — live/paper broker integration via Alpaca Markets API.

Swap in by setting settings.yaml → executor.provider: "alpaca"
and populating ALPACA_API_KEY + ALPACA_SECRET_KEY in .env.

Paper vs live is controlled by settings.yaml → executor.alpaca_paper (default: true).
Always run paper mode first and verify P&L against PaperExecutor before going live.

Position tracking is mirrored to a local SQLite database so that holding-period
timeouts, True News Stops, and session summaries work exactly like PaperExecutor.
The Alpaca order_id is stored alongside our internal UUID for reconciliation.

VIX note:
  VIX is not directly tradeable on Alpaca. It is mapped to a proxy ETF configured
  in settings.yaml → executor.vix_proxy_ticker (default: UVXY).
  BUY VIX  → buy UVXY  (volatility long)
  SHORT VIX → short UVXY (requires margin on a live account)

Usage:
  # In .env:
  ALPACA_API_KEY=your_key
  ALPACA_SECRET_KEY=your_secret

  # In settings.yaml:
  executor:
    provider: "alpaca"
    alpaca_paper: true
    alpaca_notional_per_trade: 1000
    vix_proxy_ticker: "UVXY"
"""

import logging
import sqlite3
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import yaml

from executor.base_executor import BaseExecutor
from models.signal_event import SignalEvent

logger = logging.getLogger(__name__)

_SETTINGS_PATH = "config/settings.yaml"
_DEFAULT_DB     = "D:/WhaleWatch_Data/alpaca_trades.db"
_FILL_POLL_SEC  = 0.5   # interval between fill-status polls
_FILL_TIMEOUT   = 10    # max seconds to wait for a fill before logging a warning

_DDL = """
CREATE TABLE IF NOT EXISTS positions (
    order_id            TEXT PRIMARY KEY,   -- our internal UUID
    alpaca_order_id     TEXT,               -- Alpaca-assigned order ID
    event_id            TEXT NOT NULL,
    created_at          TEXT NOT NULL,
    signal_direction    TEXT NOT NULL,
    signal_ticker       TEXT NOT NULL,      -- original (SPY | QQQ | VIX)
    alpaca_ticker       TEXT NOT NULL,      -- actual symbol submitted to Alpaca
    notional            REAL,               -- USD notional of the order
    filled_qty          REAL,               -- shares filled (for closing)
    confidence          REAL,
    holding_period_min  INTEGER,
    stop_loss_pct       REAL,
    take_profit_pct     REAL,
    entry_price         REAL,               -- filled avg price on open
    exit_price          REAL,
    realized_pnl        REAL,
    outcome             TEXT,               -- WIN | LOSS | STOP_OUT | OPEN
    closed_at           TEXT,
    close_reason        TEXT,
    poly_market_id      TEXT,
    poly_entry_prob     REAL
);

CREATE TABLE IF NOT EXISTS daily_pnl (
    session_date        TEXT PRIMARY KEY,
    total_pnl           REAL DEFAULT 0.0,
    trade_count         INTEGER DEFAULT 0,
    win_count           INTEGER DEFAULT 0
);
"""


def _load_settings() -> dict:
    try:
        with open(_SETTINGS_PATH) as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


class AlpacaExecutor(BaseExecutor):
    """Broker executor via Alpaca Markets API (paper or live).

    Requires alpaca-py>=0.31.0  (pip install alpaca-py).
    Credentials: ALPACA_API_KEY + ALPACA_SECRET_KEY in .env.
    """

    def __init__(self) -> None:
        # Late import so the rest of the app doesn't hard-depend on alpaca-py
        try:
            from alpaca.trading.client import TradingClient
            from alpaca.trading.enums import OrderSide, TimeInForce, QueryOrderStatus
            from alpaca.trading.requests import MarketOrderRequest, GetOrdersRequest
        except ImportError as exc:
            raise ImportError(
                "alpaca-py is required for AlpacaExecutor. "
                "Run: pip install alpaca-py>=0.31.0"
            ) from exc

        import os
        api_key    = os.getenv("ALPACA_API_KEY", "")
        secret_key = os.getenv("ALPACA_SECRET_KEY", "")
        if not api_key or not secret_key:
            raise EnvironmentError(
                "ALPACA_API_KEY and ALPACA_SECRET_KEY must be set in .env "
                "to use AlpacaExecutor."
            )

        cfg = _load_settings().get("executor", {})
        paper_mode     = cfg.get("alpaca_paper", True)
        self._notional = float(cfg.get("alpaca_notional_per_trade", 1000))
        self._vix_proxy = cfg.get("vix_proxy_ticker", "UVXY")
        self._db_path   = cfg.get("alpaca_db_path", _DEFAULT_DB)

        self._client         = TradingClient(api_key, secret_key, paper=paper_mode)
        self._OrderSide      = OrderSide
        self._TimeInForce    = TimeInForce
        self._QueryOrderStatus = QueryOrderStatus
        self._MarketOrderRequest  = MarketOrderRequest
        self._GetOrdersRequest    = GetOrdersRequest

        mode_label = "PAPER" if paper_mode else "LIVE"
        logger.info("AlpacaExecutor initialised (%s mode, notional=$%.0f)", mode_label, self._notional)

        self._init_db()

    # ------------------------------------------------------------------
    # BaseExecutor interface
    # ------------------------------------------------------------------

    def submit_signal(self, event: SignalEvent) -> str:
        """Place a market order on Alpaca and record the position locally."""
        order_id    = str(uuid.uuid4())
        alpaca_sym  = self._map_ticker(event.signal_ticker)
        side        = (self._OrderSide.BUY
                       if event.signal_direction == "BUY"
                       else self._OrderSide.SELL)

        req = self._MarketOrderRequest(
            symbol=alpaca_sym,
            notional=self._notional,
            side=side,
            time_in_force=self._TimeInForce.DAY,
        )

        try:
            order = self._client.submit_order(req)
            alpaca_order_id = str(order.id)
        except Exception as exc:
            logger.error("Alpaca submit_order failed: %s", exc)
            raise

        # Poll for fill to get filled_avg_price and filled_qty
        entry_price, filled_qty = self._await_fill(alpaca_order_id)

        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO positions
                  (order_id, alpaca_order_id, event_id, created_at,
                   signal_direction, signal_ticker, alpaca_ticker,
                   notional, filled_qty, confidence, holding_period_min,
                   stop_loss_pct, take_profit_pct, entry_price, outcome,
                   poly_market_id, poly_entry_prob)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,'OPEN',?,?)
                """,
                (
                    order_id, alpaca_order_id, event_id := event.event_id,
                    event.created_at.isoformat(),
                    event.signal_direction, event.signal_ticker, alpaca_sym,
                    self._notional, filled_qty,
                    event.confidence, event.holding_period_minutes,
                    event.stop_loss_pct, event.take_profit_pct,
                    entry_price,
                    event.poly_market_id, event.poly_price_after,
                ),
            )

        event.market_price_at_signal = entry_price

        logger.info(
            "ALPACA OPEN  order=%s  alpaca=%s  %s %s  entry=%.4f  qty=%.4f  hold=%dm",
            order_id[:8], alpaca_order_id[:8],
            event.signal_direction, alpaca_sym,
            entry_price or 0, filled_qty or 0,
            event.holding_period_minutes or 0,
        )
        return order_id

    def close_position(self, order_id: str, reason: str = "MANUAL") -> Optional[float]:
        """Close an open position by placing the opposing market order."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT signal_direction, alpaca_ticker, entry_price, "
                "filled_qty, stop_loss_pct, take_profit_pct "
                "FROM positions WHERE order_id=? AND outcome='OPEN'",
                (order_id,),
            ).fetchone()

        if not row:
            logger.warning("close_position: order %s not found or already closed", order_id)
            return None

        direction, alpaca_sym, entry_price, filled_qty, sl_pct, tp_pct = row

        # Opposing side to close
        close_side = (self._OrderSide.SELL
                      if direction == "BUY"
                      else self._OrderSide.BUY)

        if filled_qty and filled_qty > 0:
            # Use exact qty for the closing leg
            req = self._MarketOrderRequest(
                symbol=alpaca_sym,
                qty=round(filled_qty, 9),
                side=close_side,
                time_in_force=self._TimeInForce.DAY,
            )
        else:
            # Fallback to notional close if qty wasn't captured
            req = self._MarketOrderRequest(
                symbol=alpaca_sym,
                notional=self._notional,
                side=close_side,
                time_in_force=self._TimeInForce.DAY,
            )

        try:
            close_order = self._client.submit_order(req)
            close_alpaca_id = str(close_order.id)
        except Exception as exc:
            logger.error("Alpaca close_order failed for %s: %s", order_id[:8], exc)
            return None

        exit_price, _ = self._await_fill(close_alpaca_id)

        pnl     = self._compute_pnl(direction, entry_price, exit_price, sl_pct, tp_pct)
        outcome = self._classify_outcome(pnl, direction, entry_price, exit_price, sl_pct, tp_pct)
        now     = datetime.now(tz=timezone.utc).isoformat()

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
            "ALPACA CLOSE order=%s  %s %s  entry=%.4f  exit=%.4f  pnl=%.4f  %s  reason=%s",
            order_id[:8], direction, alpaca_sym,
            entry_price or 0, exit_price or 0, pnl, outcome, reason,
        )
        return pnl

    def close_expired_positions(self) -> list:
        """Close any OPEN positions whose holding period has elapsed."""
        now = datetime.now(tz=timezone.utc)
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT order_id, created_at, holding_period_min "
                "FROM positions WHERE outcome='OPEN'"
            ).fetchall()

        pnls: list[float] = []
        for order_id, created_at_str, hold_min in rows:
            created_at = datetime.fromisoformat(created_at_str)
            if created_at.tzinfo is None:
                created_at = created_at.replace(tzinfo=timezone.utc)
            elapsed = (now - created_at).total_seconds() / 60
            if hold_min and elapsed >= hold_min:
                pnl = self.close_position(order_id, reason="TIMEOUT")
                if pnl is not None:
                    pnls.append(pnl)

        return pnls

    def open_positions(self) -> list:
        """Return all currently open positions."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT order_id, event_id, signal_direction, signal_ticker, "
                "entry_price, holding_period_min, created_at "
                "FROM positions WHERE outcome='OPEN'"
            ).fetchall()
        cols = ["order_id", "event_id", "direction", "ticker", "entry_price", "hold_min", "created_at"]
        return [dict(zip(cols, r)) for r in rows]

    def check_true_news_stop(self, market_id: str, current_prob: float) -> list:
        """Apply True News Stop for Polymarket-linked open positions."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT order_id, signal_direction, poly_entry_prob "
                "FROM positions "
                "WHERE outcome='OPEN' AND poly_market_id=?",
                (market_id,),
            ).fetchall()

        closed_ids: list[str] = []
        for order_id, direction, entry_prob in rows:
            if entry_prob is None:
                continue
            triggered = (
                direction == "BUY"   and current_prob < entry_prob
                or direction == "SHORT" and current_prob > entry_prob
            )
            if triggered:
                self.close_position(order_id, reason="TRUE_NEWS_STOP")
                closed_ids.append(order_id)
                logger.info(
                    "TRUE_NEWS_STOP order=%s  direction=%s  entry_prob=%.3f  current=%.3f",
                    order_id[:8], direction, entry_prob, current_prob,
                )

        return closed_ids

    def session_summary(self) -> dict:
        """Return today's P&L summary."""
        today = datetime.now(tz=timezone.utc).date().isoformat()
        with self._conn() as conn:
            row = conn.execute(
                "SELECT total_pnl, trade_count, win_count "
                "FROM daily_pnl WHERE session_date=?",
                (today,),
            ).fetchone()
        if row:
            total_pnl, trade_count, win_count = row
            win_rate = win_count / trade_count if trade_count else 0.0
            return {
                "date": today, "total_pnl": total_pnl,
                "trade_count": trade_count, "win_count": win_count,
                "win_rate": win_rate,
            }
        return {"date": today, "total_pnl": 0.0, "trade_count": 0, "win_count": 0, "win_rate": 0.0}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _map_ticker(self, signal_ticker: str) -> str:
        """Map signal ticker to a tradeable Alpaca symbol."""
        if signal_ticker.upper() == "VIX":
            return self._vix_proxy
        return signal_ticker.upper()

    def _await_fill(self, alpaca_order_id: str) -> tuple[Optional[float], Optional[float]]:
        """Poll until the order is filled or timeout. Returns (filled_avg_price, filled_qty)."""
        deadline = time.monotonic() + _FILL_TIMEOUT
        while time.monotonic() < deadline:
            try:
                order = self._client.get_order_by_id(alpaca_order_id)
                status = str(order.status).lower()
                if status in ("filled", "partially_filled"):
                    price = float(order.filled_avg_price) if order.filled_avg_price else None
                    qty   = float(order.filled_qty)       if order.filled_qty       else None
                    return price, qty
                if status in ("canceled", "expired", "rejected"):
                    logger.warning("Alpaca order %s ended with status: %s", alpaca_order_id[:8], status)
                    return None, None
            except Exception as exc:
                logger.debug("Poll fill error for %s: %s", alpaca_order_id[:8], exc)
            time.sleep(_FILL_POLL_SEC)

        logger.warning(
            "Alpaca order %s did not fill within %ds — entry price unknown",
            alpaca_order_id[:8], _FILL_TIMEOUT,
        )
        return None, None

    def _compute_pnl(
        self,
        direction: str,
        entry: Optional[float],
        exit_: Optional[float],
        sl_pct: Optional[float],
        tp_pct: Optional[float],
    ) -> float:
        if entry is None or exit_ is None or entry == 0:
            return 0.0
        raw = (exit_ - entry) / entry if direction == "BUY" else (entry - exit_) / entry
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
        win   = 1 if outcome == "WIN" else 0
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

    def _init_db(self) -> None:
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript(_DDL)

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self._db_path)
