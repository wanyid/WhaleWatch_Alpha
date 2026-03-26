"""monitor.py — Live terminal dashboard for WhaleWatch_Alpha paper trading.

Shows:
  - Open positions (order_id, ticker, direction, entry price, age, unrealised P&L estimate)
  - Today's session P&L summary
  - Circuit breaker state (from risk_manager settings)
  - Last 5 closed trades

Refreshes every N seconds (default 15).

Usage:
    python scripts/monitor.py
    python scripts/monitor.py --interval 30   # slower refresh
    python scripts/monitor.py --once          # print once and exit
"""

import argparse
import os
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

DB_PATH = "D:/WhaleWatch_Data/paper_trades.db"
SETTINGS_PATH = PROJECT_ROOT / "config" / "settings.yaml"


def _load_cfg() -> dict:
    try:
        with open(SETTINGS_PATH) as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def _conn(db_path: str) -> sqlite3.Connection:
    return sqlite3.connect(db_path)


def _get_open_positions(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute(
        """
        SELECT order_id, signal_ticker, signal_direction,
               entry_price, holding_period_min, created_at, confidence
        FROM positions
        WHERE outcome = 'OPEN'
        ORDER BY created_at DESC
        """
    ).fetchall()
    now = datetime.now(tz=timezone.utc)
    result = []
    for r in rows:
        order_id, ticker, direction, entry, hold_min, created_str, conf = r
        try:
            created = datetime.fromisoformat(created_str)
            if created.tzinfo is None:
                created = created.replace(tzinfo=timezone.utc)
            age_min = (now - created).total_seconds() / 60
            remaining = max(0, (hold_min or 60) - age_min)
        except Exception:
            age_min = 0
            remaining = 0
        result.append({
            "order_id": (order_id or "")[:8],
            "ticker": ticker,
            "direction": direction,
            "entry": entry or 0.0,
            "age_min": round(age_min, 1),
            "remaining_min": round(remaining, 1),
            "conf": conf or 0.0,
        })
    return result


def _get_session_summary(conn: sqlite3.Connection) -> dict:
    today = datetime.now(tz=timezone.utc).date().isoformat()
    row = conn.execute(
        "SELECT total_pnl, trade_count, win_count FROM daily_pnl WHERE session_date=?",
        (today,),
    ).fetchone()
    if row:
        total_pnl, trade_count, win_count = row
        win_rate = win_count / trade_count if trade_count else 0.0
        return {
            "date": today,
            "total_pnl": total_pnl or 0.0,
            "trade_count": trade_count or 0,
            "win_count": win_count or 0,
            "win_rate": win_rate,
        }
    return {"date": today, "total_pnl": 0.0, "trade_count": 0, "win_count": 0, "win_rate": 0.0}


def _get_recent_closed(conn: sqlite3.Connection, n: int = 5) -> list[dict]:
    rows = conn.execute(
        """
        SELECT order_id, signal_ticker, signal_direction,
               entry_price, exit_price, realized_pnl, outcome, closed_at, close_reason
        FROM positions
        WHERE outcome != 'OPEN'
        ORDER BY closed_at DESC
        LIMIT ?
        """,
        (n,),
    ).fetchall()
    result = []
    for r in rows:
        order_id, ticker, direction, entry, exit_, pnl, outcome, closed_at, reason = r
        result.append({
            "order_id": (order_id or "")[:8],
            "ticker": ticker,
            "direction": direction,
            "entry": entry or 0.0,
            "exit": exit_ or 0.0,
            "pnl": pnl or 0.0,
            "outcome": outcome,
            "closed_at": (closed_at or "")[:16],
            "reason": reason or "",
        })
    return result


def _circuit_breaker_state(cfg: dict, session_pnl: float) -> tuple[bool, float]:
    """Returns (halted, threshold)."""
    threshold = cfg.get("risk", {}).get("daily_drawdown_halt", -0.03)
    halted = session_pnl <= threshold
    return halted, threshold


def _clear_screen() -> None:
    os.system("cls" if os.name == "nt" else "clear")


def _render(db_path: str, cfg: dict) -> None:
    if not Path(db_path).exists():
        print(f"[monitor] DB not found: {db_path}")
        print("  Run label_events.py or main.py first to create the database.")
        return

    with _conn(db_path) as conn:
        open_pos = _get_open_positions(conn)
        summary = _get_session_summary(conn)
        recent = _get_recent_closed(conn)

    halted, halt_thresh = _circuit_breaker_state(cfg, summary["total_pnl"])
    now_str = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    print("=" * 72)
    print(f"  WhaleWatch_Alpha  |  Paper Trading Monitor  |  {now_str}")
    print("=" * 72)

    # Session summary
    pnl_sign = "+" if summary["total_pnl"] >= 0 else ""
    print(f"\n  Session P&L : {pnl_sign}{summary['total_pnl']:.4f}"
          f"  |  Trades: {summary['trade_count']}"
          f"  |  Wins: {summary['win_count']}"
          f"  |  Win Rate: {summary['win_rate']:.1%}")

    cb_status = "HALTED ⛔" if halted else "OK ✓"
    print(f"  Circuit breaker: {cb_status}  (halt threshold: {halt_thresh:.1%})")

    # Open positions
    print(f"\n  Open Positions ({len(open_pos)}):")
    if open_pos:
        header = f"  {'ID':8}  {'Ticker':6}  {'Dir':6}  {'Entry':>9}  {'Age':>6}  {'Rem':>6}  {'Conf':>5}"
        print(header)
        print("  " + "-" * 62)
        for p in open_pos:
            print(
                f"  {p['order_id']:8}  {p['ticker']:6}  {p['direction']:6}"
                f"  {p['entry']:9.4f}  {p['age_min']:>5.0f}m  {p['remaining_min']:>5.0f}m"
                f"  {p['conf']:5.2f}"
            )
    else:
        print("  (none)")

    # Recent closed trades
    print(f"\n  Recent Closed Trades (last {len(recent)}):")
    if recent:
        header2 = f"  {'ID':8}  {'Ticker':6}  {'Dir':6}  {'PnL':>8}  {'Outcome':8}  {'Reason':14}  {'Closed':16}"
        print(header2)
        print("  " + "-" * 75)
        for t in recent:
            pnl_str = f"{t['pnl']:+.4f}"
            print(
                f"  {t['order_id']:8}  {t['ticker']:6}  {t['direction']:6}"
                f"  {pnl_str:>8}  {t['outcome']:8}  {t['reason']:14}  {t['closed_at']:16}"
            )
    else:
        print("  (none)")

    print("\n" + "=" * 72)


def main() -> None:
    parser = argparse.ArgumentParser(description="WhaleWatch_Alpha live monitor")
    parser.add_argument("--interval", type=int, default=15,
                        help="Refresh interval in seconds (default 15)")
    parser.add_argument("--once", action="store_true",
                        help="Print once and exit (no loop)")
    parser.add_argument("--db", default=DB_PATH,
                        help="Path to paper_trades.db")
    args = parser.parse_args()

    cfg = _load_cfg()

    if args.once:
        _render(args.db, cfg)
        return

    try:
        while True:
            _clear_screen()
            _render(args.db, cfg)
            print(f"\n  Refreshing every {args.interval}s  |  Ctrl-C to exit\n")
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nMonitor stopped.")


if __name__ == "__main__":
    main()
