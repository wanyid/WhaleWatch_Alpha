"""run_backtest.py — Full backtest with performance report and optimisation sweeps.

Reads resolved trades from paper_trades.db, runs the backtester, then:
  1. Prints the full performance report (win rate, Sharpe, Sortino, Kelly, etc.)
  2. Sweeps stop-loss thresholds to find the optimal SL by Sharpe ratio
  3. Sweeps min-confidence thresholds to find the optimal confidence gate
  4. Shows bet-size scaling table (quarter / half / full Kelly)
  5. Saves all results to D:/WhaleWatch_Data/backtest_YYYYMMDD.csv

Usage:
    python scripts/run_backtest.py
    python scripts/run_backtest.py --start 2025-06-01
    python scripts/run_backtest.py --no-optimize   # skip parameter sweeps
"""

import argparse
import logging
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backtest.backtester import Backtester
from backtest.performance import (
    compute_metrics,
    optimize_bet_size,
    optimize_confidence,
    optimize_stop_loss,
    print_report,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("run_backtest")

DB_PATH = "D:/WhaleWatch_Data/paper_trades.db"
OUT_DIR = Path("D:/WhaleWatch_Data")


def _load_trades(db_path: str, start: str, end: str) -> pd.DataFrame:
    """Load resolved trades directly from DB."""
    if not Path(db_path).exists():
        logger.error("DB not found: %s — run label_events.py first", db_path)
        sys.exit(1)

    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query(
            """
            SELECT order_id, event_id, created_at, signal_direction, signal_ticker,
                   confidence, holding_period_min, stop_loss_pct, take_profit_pct,
                   entry_price, exit_price, realized_pnl, outcome, close_reason
            FROM positions
            WHERE outcome IN ('WIN', 'LOSS', 'STOP_OUT')
              AND date(created_at) >= ?
              AND date(created_at) <= ?
            ORDER BY created_at
            """,
            conn,
            params=(start, end),
        )

    logger.info("Loaded %d resolved trades (%s → %s)", len(df), start, end)
    return df


def run(
    start: str = "2025-01-20",
    end: str | None = None,
    optimize: bool = True,
    db_path: str = DB_PATH,
) -> None:
    end = end or datetime.now(tz=timezone.utc).date().isoformat()

    df = _load_trades(db_path, start, end)

    if df.empty:
        logger.warning("No trades in date range — run label_events.py first")
        return

    # --- Full performance report ---
    m = compute_metrics(df)
    print_report(m)

    if not optimize:
        return

    # --- Stop-loss optimisation ---
    print("\n" + "=" * 60)
    print("STOP-LOSS OPTIMISATION (by Sharpe ratio)")
    print("=" * 60)
    sl_sweep = optimize_stop_loss(df)
    print(sl_sweep.to_string(index=False))
    best_sl = sl_sweep.iloc[0]
    print(f"\n  ► Optimal SL: {best_sl['stop_loss_pct']:.1%}"
          f"  (Sharpe={best_sl['sharpe']:.3f}"
          f"  win_rate={best_sl['win_rate']:.1%}"
          f"  pnl={best_sl['total_pnl']:+.4f})")

    # --- Confidence threshold optimisation ---
    if "confidence" in df.columns and df["confidence"].notna().any():
        print("\n" + "=" * 60)
        print("CONFIDENCE THRESHOLD OPTIMISATION (by Sharpe ratio)")
        print("=" * 60)
        conf_sweep = optimize_confidence(df)
        print(conf_sweep.to_string(index=False))
        best_conf = conf_sweep.iloc[0]
        print(f"\n  ► Optimal min-confidence: {best_conf['min_confidence']:.2f}"
              f"  (Sharpe={best_conf['sharpe']:.3f}"
              f"  trades={int(best_conf['trades'])}"
              f"  win_rate={best_conf['win_rate']:.1%})")
    else:
        logger.info("No confidence column in data — skipping confidence sweep")
        conf_sweep = pd.DataFrame()

    # --- Bet size / Kelly ---
    print("\n" + "=" * 60)
    print(f"BET-SIZE SCALING TABLE  (Kelly f*={m.kelly_f:.1%}  Half-Kelly={m.half_kelly_f:.1%})")
    print("=" * 60)
    kelly_sweep = optimize_bet_size(df, m.kelly_f)
    print(kelly_sweep.to_string(index=False))
    print(f"\n  ► Recommendation: start at Half-Kelly ({m.half_kelly_f:.1%} per trade)")
    print(f"    Scale up toward Full-Kelly ({m.kelly_f:.1%}) once 100+ live trades confirm the edge")

    # --- Save all results ---
    date_tag = datetime.now(tz=timezone.utc).strftime("%Y%m%d")
    trades_path = OUT_DIR / f"backtest_{date_tag}.csv"
    df.to_csv(trades_path, index=False)

    sl_path = OUT_DIR / f"sl_sweep_{date_tag}.csv"
    sl_sweep.to_csv(sl_path, index=False)

    if not conf_sweep.empty:
        conf_path = OUT_DIR / f"conf_sweep_{date_tag}.csv"
        conf_sweep.to_csv(conf_path, index=False)

    kelly_path = OUT_DIR / f"kelly_sweep_{date_tag}.csv"
    kelly_sweep.to_csv(kelly_path, index=False)

    logger.info("Results saved to %s/backtest_%s*.csv", OUT_DIR, date_tag)
    print(f"\nAll results saved to {OUT_DIR}/backtest_{date_tag}*.csv\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run backtest with full performance analysis")
    parser.add_argument("--start", default="2025-01-20",
                        help="Backtest start date (YYYY-MM-DD)")
    parser.add_argument("--end", default=None,
                        help="Backtest end date (YYYY-MM-DD), default=today")
    parser.add_argument("--no-optimize", action="store_true",
                        help="Skip parameter sweep optimisations")
    parser.add_argument("--db", default=DB_PATH,
                        help="Path to paper_trades.db")
    args = parser.parse_args()

    run(
        start=args.start,
        end=args.end,
        optimize=not args.no_optimize,
        db_path=args.db,
    )
