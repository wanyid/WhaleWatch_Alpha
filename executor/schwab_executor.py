"""SchwabExecutor — live broker integration via Charles Schwab API (schwab-py).

Use PaperExecutor for all simulation and strategy validation.
Switch to this executor only when ready to trade with real money.

Swap in by setting settings.yaml → executor.provider: "schwab"
and populating the required credentials in .env (see below).

---- First-time setup ----
1. Register a developer app at https://developer.schwab.com/
2. Set the callback URL to https://127.0.0.1:8182 in the app settings
3. Add credentials to .env:
     SCHWAB_API_KEY=your_app_key
     SCHWAB_APP_SECRET=your_app_secret
     SCHWAB_ACCOUNT_HASH=your_account_hash   (from get_account_numbers())
     SCHWAB_TOKEN_PATH=D:/WhaleWatch_Data/schwab_token.json
4. Run the one-time auth helper to create the token file:
     python scripts/setup_schwab_auth.py
5. Re-run the auth helper every 7 days — Schwab's refresh token expires weekly.

---- Behaviour ----
- Orders are whole-share market orders. Qty = floor(notional / last_price).
  Minimum 1 share per signal.
- BUY direction  → equity_buy_market / close with equity_sell_market
- SHORT direction → equity_sell_short_market / close with equity_buy_to_cover_market
- VIX is not directly tradeable — mapped to proxy ETF via settings.yaml
  executor.vix_proxy_ticker (default: UVXY)
- Position tracking mirrored to local SQLite so holding-period timeouts,
  True News Stops, and session summaries work identically to PaperExecutor.
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

_SETTINGS_PATH  = "config/settings.yaml"
_DEFAULT_DB     = "D:/WhaleWatch_Data/schwab_trades.db"
_FILL_POLL_SEC  = 0.5
_FILL_TIMEOUT   = 15    # Schwab fills can be slightly slower than Alpaca

_DDL = """
CREATE TABLE IF NOT EXISTS positions (
    order_id            TEXT PRIMARY KEY,   -- our internal UUID
    schwab_order_id     TEXT,               -- Schwab-assigned order ID
    event_id            TEXT NOT NULL,
    created_at          TEXT NOT NULL,
    signal_direction    TEXT NOT NULL,
    signal_ticker       TEXT NOT NULL,      -- original (SPY | QQQ | VIX)
    schwab_ticker       TEXT NOT NULL,      -- actual symbol submitted to Schwab
    notional            REAL,               -- USD notional target
    filled_qty          REAL,               -- whole shares filled (for closing)
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


class SchwabExecutor(BaseExecutor):
    """Broker executor via Charles Schwab API (schwab-py).

    Requires schwab-py>=0.4.0  (pip install schwab-py).
    Credentials: SCHWAB_API_KEY, SCHWAB_APP_SECRET, SCHWAB_ACCOUNT_HASH,
                 SCHWAB_TOKEN_PATH  — all in .env.
    Run scripts/setup_schwab_auth.py once to create the token file.
    Re-run every 7 days when Schwab's refresh token expires.
    """

    def __init__(self) -> None:
        try:
            import schwab
            from schwab import auth
            from schwab.orders.equities import (
                equity_buy_market,
                equity_sell_market,
                equity_sell_short_market,
                equity_buy_to_cover_market,
            )
            from schwab.utils import Utils
        except ImportError as exc:
            raise ImportError(
                "schwab-py is required for SchwabExecutor. "
                "Run: pip install schwab-py>=0.4.0"
            ) from exc

        import os
        api_key      = os.getenv("SCHWAB_API_KEY", "")
        app_secret   = os.getenv("SCHWAB_APP_SECRET", "")
        account_hash = os.getenv("SCHWAB_ACCOUNT_HASH", "")
        token_path   = os.getenv("SCHWAB_TOKEN_PATH", "D:/WhaleWatch_Data/schwab_token.json")

        missing = [k for k, v in {
            "SCHWAB_API_KEY": api_key,
            "SCHWAB_APP_SECRET": app_secret,
            "SCHWAB_ACCOUNT_HASH": account_hash,
        }.items() if not v]
        if missing:
            raise EnvironmentError(
                f"Missing required env vars for SchwabExecutor: {', '.join(missing)}. "
                "Add them to .env and run scripts/setup_schwab_auth.py for first-time token setup."
            )

        if not Path(token_path).exists():
            raise FileNotFoundError(
                f"Schwab token file not found: {token_path}\n"
                "Run: python scripts/setup_schwab_auth.py"
            )

        self._client       = auth.client_from_token_file(token_path, api_key, app_secret)
        self._account_hash = account_hash
        self._Utils        = Utils

        # Store order builder functions for use in methods
        self._equity_buy_market          = equity_buy_market
        self._equity_sell_market         = equity_sell_market
        self._equity_sell_short_market   = equity_sell_short_market
        self._equity_buy_to_cover_market = equity_buy_to_cover_market

        cfg = _load_settings().get("executor", {})
        self._notional  = float(cfg.get("schwab_notional_per_trade", 1000))
        self._vix_proxy = cfg.get("vix_proxy_ticker", "UVXY")
        self._db_path   = cfg.get("schwab_db_path", _DEFAULT_DB)

        logger.info(
            "SchwabExecutor initialised (LIVE, notional=$%.0f, vix_proxy=%s)",
            self._notional, self._vix_proxy,
        )
        self._init_db()

    # ------------------------------------------------------------------
    # BaseExecutor interface
    # ------------------------------------------------------------------

    def submit_signal(self, event: SignalEvent) -> str:
        """Place a live market order on Schwab and record the position locally."""
        order_id   = str(uuid.uuid4())
        schwab_sym = self._map_ticker(event.signal_ticker)

        # Get current price to compute whole-share quantity
        current_price = self._get_last_price(schwab_sym)
        if not current_price:
            raise RuntimeError(f"Could not get quote for {schwab_sym} — aborting signal")

        qty = max(1, int(self._notional / current_price))

        if event.signal_direction == "BUY":
            order_req = self._equity_buy_market(schwab_sym, qty)
        else:  # SHORT
            order_req = self._equity_sell_short_market(schwab_sym, qty)

        try:
            resp = self._client.place_order(self._account_hash, order_req)
            schwab_order_id = str(self._Utils.extract_order_id(resp))
        except Exception as exc:
            logger.error("Schwab place_order failed: %s", exc)
            raise

        entry_price, filled_qty = self._await_fill(schwab_order_id)
        # Fall back to the pre-order quote if fill price not captured in time
        if entry_price is None:
            entry_price = current_price
        if filled_qty is None:
            filled_qty = float(qty)

        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO positions
                  (order_id, schwab_order_id, event_id, created_at,
                   signal_direction, signal_ticker, schwab_ticker,
                   notional, filled_qty, confidence, holding_period_min,
                   stop_loss_pct, take_profit_pct, entry_price, outcome,
                   poly_market_id, poly_entry_prob)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,'OPEN',?,?)
                """,
                (
                    order_id, schwab_order_id, event.event_id,
                    event.created_at.isoformat(),
                    event.signal_direction, event.signal_ticker, schwab_sym,
                    self._notional, filled_qty,
                    event.confidence, event.holding_period_minutes,
                    event.stop_loss_pct, event.take_profit_pct,
                    entry_price,
                    event.poly_market_id, event.poly_price_after,
                ),
            )

        event.market_price_at_signal = entry_price

        logger.info(
            "SCHWAB OPEN  order=%s  schwab=%s  %s %s  qty=%d  entry=%.4f  hold=%dm",
            order_id[:8], schwab_order_id,
            event.signal_direction, schwab_sym,
            int(filled_qty), entry_price or 0,
            event.holding_period_minutes or 0,
        )
        return order_id

    def close_position(self, order_id: str, reason: str = "MANUAL") -> Optional[float]:
        """Close an open position with the appropriate closing order type."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT signal_direction, schwab_ticker, entry_price, "
                "filled_qty, stop_loss_pct, take_profit_pct "
                "FROM positions WHERE order_id=? AND outcome='OPEN'",
                (order_id,),
            ).fetchone()

        if not row:
            logger.warning("close_position: order %s not found or already closed", order_id)
            return None

        direction, schwab_sym, entry_price, filled_qty, sl_pct, tp_pct = row
        qty = max(1, int(filled_qty or 1))

        # Correct closing order type per direction
        if direction == "BUY":
            close_req = self._equity_sell_market(schwab_sym, qty)
        else:  # SHORT was opened with sell_short → close with buy_to_cover
            close_req = self._equity_buy_to_cover_market(schwab_sym, qty)

        try:
            resp = self._client.place_order(self._account_hash, close_req)
            close_schwab_id = str(self._Utils.extract_order_id(resp))
        except Exception as exc:
            logger.error("Schwab close_order failed for %s: %s", order_id[:8], exc)
            return None

        exit_price, _ = self._await_fill(close_schwab_id)
        if exit_price is None:
            exit_price = self._get_last_price(schwab_sym)

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
            "SCHWAB CLOSE order=%s  %s %s  entry=%.4f  exit=%.4f  pnl=%.4f  %s  reason=%s",
            order_id[:8], direction, schwab_sym,
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
        if signal_ticker.upper() == "VIX":
            return self._vix_proxy
        return signal_ticker.upper()

    def _get_last_price(self, symbol: str) -> Optional[float]:
        """Fetch the last trade price for a symbol via Schwab quotes API."""
        try:
            resp = self._client.get_quotes([symbol])
            data = resp.json()
            return float(data[symbol]["quote"]["lastPrice"])
        except Exception as exc:
            logger.warning("Could not fetch quote for %s: %s", symbol, exc)
            return None

    def _await_fill(self, schwab_order_id: str) -> tuple[Optional[float], Optional[float]]:
        """Poll until the order is filled or timeout. Returns (avg_price, filled_qty)."""
        deadline = time.monotonic() + _FILL_TIMEOUT
        while time.monotonic() < deadline:
            try:
                resp  = self._client.get_order(schwab_order_id, self._account_hash)
                order = resp.json()
                status = str(order.get("status", "")).upper()
                if status == "FILLED":
                    avg_price  = order.get("averagePrice")
                    filled_qty = order.get("filledQuantity")
                    return (
                        float(avg_price)  if avg_price  is not None else None,
                        float(filled_qty) if filled_qty is not None else None,
                    )
                if status in ("CANCELED", "EXPIRED", "REJECTED"):
                    logger.warning("Schwab order %s ended with status: %s", schwab_order_id, status)
                    return None, None
            except Exception as exc:
                logger.debug("Poll fill error for %s: %s", schwab_order_id, exc)
            time.sleep(_FILL_POLL_SEC)

        logger.warning(
            "Schwab order %s did not fill within %ds — using pre-order quote as fallback",
            schwab_order_id, _FILL_TIMEOUT,
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
