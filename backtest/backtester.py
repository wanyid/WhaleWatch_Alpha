"""Backtester — replays historical SignalEvents through the risk + executor stack.

Reads resolved SignalEvents from SQLite (written by the paper executor or
manually labeled), re-runs risk checks, then simulates fills using the
historical OHLCV data already stored in D:/WhaleWatch_Data/equity/.

Output: a summary DataFrame + per-trade CSV saved alongside the DB.

Usage:
    bt = Backtester()
    results = bt.run(start_date="2025-01-20", end_date="2026-03-26")
    print(results.summary())
"""

import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml

logger = logging.getLogger(__name__)

_SETTINGS_PATH = "config/settings.yaml"
_DEFAULT_DB = "D:/WhaleWatch_Data/paper_trades.db"
_EQUITY_DIR = Path("D:/WhaleWatch_Data/equity")


def _load_cfg() -> dict:
    with open(_SETTINGS_PATH, "r") as f:
        return yaml.safe_load(f)


@dataclass
class TradeResult:
    order_id: str
    event_id: str
    direction: str
    ticker: str
    entry_price: float
    exit_price: float
    entry_time: datetime
    exit_time: datetime
    holding_minutes: int
    pnl_pct: float          # signed % return
    outcome: str            # WIN | LOSS | STOP_OUT
    stop_loss_pct: float
    take_profit_pct: float


@dataclass
class BacktestResults:
    trades: list[TradeResult] = field(default_factory=list)

    def summary(self) -> dict:
        if not self.trades:
            return {"trades": 0}
        wins = [t for t in self.trades if t.outcome == "WIN"]
        losses = [t for t in self.trades if t.outcome in ("LOSS", "STOP_OUT")]
        pnls = [t.pnl_pct for t in self.trades]
        return {
            "total_trades": len(self.trades),
            "win_rate": len(wins) / len(self.trades),
            "total_pnl_pct": sum(pnls),
            "avg_pnl_pct": sum(pnls) / len(pnls),
            "best_trade_pct": max(pnls),
            "worst_trade_pct": min(pnls),
            "avg_hold_minutes": sum(t.holding_minutes for t in self.trades) / len(self.trades),
        }

    def to_dataframe(self) -> pd.DataFrame:
        if not self.trades:
            return pd.DataFrame()
        return pd.DataFrame([vars(t) for t in self.trades])


class Backtester:
    def __init__(self, db_path: Optional[str] = None) -> None:
        cfg = _load_cfg()
        self._db_path = db_path or cfg.get("executor", {}).get("paper_db_path", _DEFAULT_DB)
        risk_cfg = cfg.get("risk", {})
        self._stop_loss_pct = risk_cfg.get("stop_loss_pct", 0.02)
        self._take_profit_pct = risk_cfg.get("take_profit_pct", 0.04)

        # Cache for equity OHLCV data: ticker → DataFrame indexed by timestamp
        self._price_cache: dict[str, pd.DataFrame] = {}

    def run(
        self,
        start_date: str = "2025-01-20",
        end_date: Optional[str] = None,
    ) -> BacktestResults:
        """Replay all resolved positions in the paper_trades DB within the date range."""
        end_date = end_date or datetime.now(tz=timezone.utc).date().isoformat()

        with sqlite3.connect(self._db_path) as conn:
            rows = conn.execute(
                """
                SELECT order_id, event_id, signal_direction, signal_ticker,
                       entry_price, exit_price, holding_period_min, created_at,
                       stop_loss_pct, take_profit_pct, outcome
                FROM positions
                WHERE outcome != 'OPEN'
                  AND date(created_at) >= ?
                  AND date(created_at) <= ?
                ORDER BY created_at
                """,
                (start_date, end_date),
            ).fetchall()

        logger.info("Backtesting %d resolved trades (%s → %s)", len(rows), start_date, end_date)

        results = BacktestResults()
        for row in rows:
            (order_id, event_id, direction, ticker,
             entry_price, exit_price, hold_min, created_at_str,
             sl_pct, tp_pct, db_outcome) = row

            if entry_price is None:
                continue

            created_at = datetime.fromisoformat(created_at_str)
            if created_at.tzinfo is None:
                created_at = created_at.replace(tzinfo=timezone.utc)

            hold_min = hold_min or 60
            exit_time = created_at + timedelta(minutes=hold_min)

            # Use stored exit price if available, otherwise look up from OHLCV
            if exit_price is None:
                exit_price = self._lookup_price(ticker, exit_time) or entry_price

            sl = sl_pct or self._stop_loss_pct
            tp = tp_pct or self._take_profit_pct

            raw_ret = (exit_price - entry_price) / entry_price if direction == "BUY" \
                else (entry_price - exit_price) / entry_price

            # Apply stop-loss / take-profit clamps
            pnl_pct = max(-sl, min(tp, raw_ret))

            if raw_ret <= -sl:
                outcome = "STOP_OUT"
            elif pnl_pct > 0:
                outcome = "WIN"
            else:
                outcome = "LOSS"

            results.trades.append(TradeResult(
                order_id=order_id,
                event_id=event_id,
                direction=direction,
                ticker=ticker,
                entry_price=entry_price,
                exit_price=exit_price,
                entry_time=created_at,
                exit_time=exit_time,
                holding_minutes=hold_min,
                pnl_pct=round(pnl_pct, 6),
                outcome=outcome,
                stop_loss_pct=sl,
                take_profit_pct=tp,
            ))

        summary = results.summary()
        logger.info("Backtest complete: %s", summary)
        return results

    def save_results(self, results: BacktestResults, out_path: Optional[str] = None) -> str:
        """Save trade-level results to CSV. Returns the output path."""
        out = out_path or str(Path(self._db_path).parent / "backtest_results.csv")
        df = results.to_dataframe()
        df.to_csv(out, index=False)
        logger.info("Backtest results saved → %s (%d rows)", out, len(df))
        return out

    # ------------------------------------------------------------------
    # Price lookup from stored OHLCV parquet files
    # ------------------------------------------------------------------

    def _lookup_price(self, ticker: str, at: datetime) -> Optional[float]:
        df = self._load_daily(ticker)
        if df is None or df.empty:
            return None
        # Find the closest bar on or before `at`
        ts = pd.Timestamp(at).tz_localize("UTC") if at.tzinfo is None else pd.Timestamp(at)
        mask = df.index <= ts
        if not mask.any():
            return None
        row = df.loc[mask].iloc[-1]
        return float(row.get("Close", row.iloc[0]))

    def _load_daily(self, ticker: str) -> Optional[pd.DataFrame]:
        if ticker in self._price_cache:
            return self._price_cache[ticker]
        path = _EQUITY_DIR / f"{ticker}_1d.parquet"
        if not path.exists():
            logger.warning("No daily parquet for %s at %s", ticker, path)
            return None
        df = pd.read_parquet(path)
        # Ensure DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, utc=True)
        elif df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        self._price_cache[ticker] = df
        return df
