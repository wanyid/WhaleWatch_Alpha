"""label_events.py — Generate Layer 2 training labels from historical data.

For each resolved Polymarket market (condition_id in markets_catalog.parquet):
  1. Load the market's price history (YES token probability over time).
  2. Detect "whale moves" — price delta >= threshold within a rolling window.
  3. For each whale move, look up SPY/QQQ/VIX price at T and T+hold for
     several holding periods (15m, 30m, 1h, 4h, 1d).
  4. Determine which ticker moved most and whether the move was directional.
  5. Write one labeled SignalEvent row per detected whale move to SQLite
     (paper_trades.db positions table) with outcome WIN/LOSS/STOP_OUT.

This populates the training corpus for stat_predictor.py without requiring
the live pipeline to have run first.

Usage:
    python scripts/label_events.py
    python scripts/label_events.py --min-delta 0.08 --min-volume 100000
    python scripts/label_events.py --dry-run   # print rows, don't write
"""

import argparse
import logging
import sqlite3
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("label_events")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_ROOT = Path("D:/WhaleWatch_Data")
EQUITY_DIR = DATA_ROOT / "equity"
POLY_PRICES_DIR = DATA_ROOT / "polymarket" / "prices"
CATALOG_PATH = DATA_ROOT / "polymarket" / "markets_catalog.parquet"
DB_PATH = DATA_ROOT / "paper_trades.db"

TRAINING_START = "2025-01-20"  # Trump inauguration = our regime floor

# Holding periods to evaluate (minutes)
HOLD_PERIODS_MIN = [15, 30, 60, 240, 1440]

# Equity tickers and their 5m parquet files for intraday resolution
TICKERS = ["SPY", "QQQ", "VIX"]


def _load_settings() -> dict:
    with open("config/settings.yaml") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Price lookup helpers
# ---------------------------------------------------------------------------

_equity_cache: dict[str, pd.DataFrame] = {}


def _load_equity(ticker: str, resolution: str = "1d") -> Optional[pd.DataFrame]:
    key = f"{ticker}_{resolution}"
    if key in _equity_cache:
        return _equity_cache[key]

    suffix = "1d" if resolution == "1d" else "5m_recent"
    path = EQUITY_DIR / f"{ticker}_{suffix}.parquet"
    if not path.exists():
        logger.warning("Missing equity file: %s", path)
        return None

    df = pd.read_parquet(path)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
    elif df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    _equity_cache[key] = df
    return df


def _price_at(ticker: str, ts: datetime) -> Optional[float]:
    """Return the Close price of `ticker` closest to `ts` using 5m then daily."""
    for res in ("5m", "1d"):
        df = _load_equity(ticker, res)
        if df is None:
            continue
        target = pd.Timestamp(ts).tz_localize("UTC") if ts.tzinfo is None else pd.Timestamp(ts)
        # Find bar that starts at or before ts
        mask = df.index <= target
        if not mask.any():
            continue
        row = df.loc[mask].iloc[-1]
        close_col = "Close" if "Close" in row.index else row.index[3]  # OHLCV column 3
        return float(row[close_col])
    return None


# ---------------------------------------------------------------------------
# Whale move detection on Polymarket price series
# ---------------------------------------------------------------------------

def _detect_whale_moves(
    prices_df: pd.DataFrame,
    min_delta: float = 0.05,
    window_minutes: int = 60,
) -> list[dict]:
    """Return list of {ts, price_before, price_after, price_delta} dicts."""
    if prices_df.empty:
        return []

    moves = []
    prices_df = prices_df.sort_values("t")
    ts_arr = prices_df["t"].values
    p_arr = prices_df["p"].values

    for i in range(1, len(ts_arr)):
        dt = (ts_arr[i] - ts_arr[i - 1])  # seconds
        if dt > window_minutes * 60:
            continue
        delta = p_arr[i] - p_arr[i - 1]
        if abs(delta) >= min_delta:
            moves.append({
                "ts": datetime.fromtimestamp(int(ts_arr[i]), tz=timezone.utc),
                "price_before": float(p_arr[i - 1]),
                "price_after": float(p_arr[i]),
                "price_delta": float(delta),
            })
    return moves


# ---------------------------------------------------------------------------
# Labeling logic
# ---------------------------------------------------------------------------

def _label_move(
    move: dict,
    market_row: pd.Series,
    hold_minutes: int,
    stop_loss_pct: float,
    take_profit_pct: float,
) -> Optional[dict]:
    """Map one whale move → one labeled SignalEvent row."""
    signal_ts = move["ts"]

    # Only label events within the training window
    if signal_ts.date().isoformat() < TRAINING_START:
        return None

    # Determine expected direction from poly price delta
    # Rising probability = news is bullish for the outcome
    # We map: big UP move in Poly → uncertainty resolved → SPY reaction
    delta = move["price_delta"]
    # Large upward delta = outcome becoming more certain → often risk-off → SHORT SPY
    # Large downward delta = uncertainty rising → often risk-off → SHORT SPY or HOLD
    # (Simplified heuristic; L1 LLM replaces this in live mode)
    direction = "BUY" if delta < -0.05 else "SHORT" if delta > 0.05 else None
    if direction is None:
        return None

    # Most impactful ticker heuristic: use SPY as default, upgrade to VIX if
    # the market question mentions geopolitical/war terms
    question = str(market_row.get("question", "")).lower()
    geo_terms = ["war", "ceasefire", "invasion", "military", "nato", "nuclear"]
    ticker = "VIX" if any(t in question for t in geo_terms) else "SPY"

    # Look up equity prices
    entry_price = _price_at(ticker, signal_ts)
    exit_ts = signal_ts + timedelta(minutes=hold_minutes)
    exit_price = _price_at(ticker, exit_ts)

    if entry_price is None or exit_price is None:
        return None

    # Compute P&L
    raw_ret = (exit_price - entry_price) / entry_price if direction == "BUY" \
              else (entry_price - exit_price) / entry_price
    pnl = max(-stop_loss_pct, min(take_profit_pct, raw_ret))

    if raw_ret <= -stop_loss_pct:
        outcome = "STOP_OUT"
    elif pnl > 0:
        outcome = "WIN"
    else:
        outcome = "LOSS"

    return {
        "order_id": str(uuid.uuid4()),
        "event_id": str(uuid.uuid4()),
        "created_at": signal_ts.isoformat(),
        "signal_direction": direction,
        "signal_ticker": ticker,
        "confidence": 0.55,          # placeholder — L2 hasn't trained yet
        "holding_period_min": hold_minutes,
        "stop_loss_pct": stop_loss_pct,
        "take_profit_pct": take_profit_pct,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "realized_pnl": round(pnl, 6),
        "outcome": outcome,
        "closed_at": exit_ts.isoformat(),
        "close_reason": "LABELED_HISTORICAL",
    }


# ---------------------------------------------------------------------------
# DB write
# ---------------------------------------------------------------------------

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
    outcome             TEXT,
    closed_at           TEXT,
    close_reason        TEXT
);
"""


def _write_to_db(rows: list[dict], db_path: str) -> int:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.executescript(_DDL)
        existing = {r[0] for r in conn.execute("SELECT order_id FROM positions").fetchall()}
        new_rows = [r for r in rows if r["order_id"] not in existing]
        conn.executemany(
            """INSERT OR IGNORE INTO positions
               (order_id,event_id,created_at,signal_direction,signal_ticker,
                confidence,holding_period_min,stop_loss_pct,take_profit_pct,
                entry_price,exit_price,realized_pnl,outcome,closed_at,close_reason)
               VALUES (:order_id,:event_id,:created_at,:signal_direction,:signal_ticker,
                       :confidence,:holding_period_min,:stop_loss_pct,:take_profit_pct,
                       :entry_price,:exit_price,:realized_pnl,:outcome,:closed_at,:close_reason)
            """,
            new_rows,
        )
        return len(new_rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(
    min_delta: float = 0.05,
    min_volume: float = 10000,
    dry_run: bool = False,
    max_markets: Optional[int] = None,
) -> None:
    cfg = _load_settings()
    risk_cfg = cfg.get("risk", {})
    stop_loss_pct = risk_cfg.get("stop_loss_pct", 0.02)
    take_profit_pct = risk_cfg.get("take_profit_pct", 0.04)

    if not CATALOG_PATH.exists():
        logger.error("Catalog not found: %s — run pull_historical_data.py first", CATALOG_PATH)
        return

    catalog = pd.read_parquet(CATALOG_PATH)
    logger.info("Catalog: %d markets total", len(catalog))

    # Filter to markets with enough volume
    if "volume_24h" in catalog.columns:
        catalog = catalog[catalog["volume_24h"] >= min_volume]
    logger.info("After volume filter (>= %s): %d markets", min_volume, len(catalog))

    if max_markets:
        catalog = catalog.head(max_markets)

    all_rows: list[dict] = []
    skipped_no_file = 0
    skipped_no_moves = 0
    skipped_no_price = 0

    for i, (_, mrow) in enumerate(catalog.iterrows(), 1):
        cid = mrow.get("condition_id", "")
        if not cid:
            skipped_no_file += 1
            continue

        price_path = POLY_PRICES_DIR / f"{cid}_YES.parquet"
        if not price_path.exists():
            skipped_no_file += 1
            continue

        try:
            prices_df = pd.read_parquet(price_path)
        except Exception as exc:
            logger.debug("Could not read %s: %s", price_path, exc)
            skipped_no_file += 1
            continue

        moves = _detect_whale_moves(prices_df, min_delta=min_delta)
        if not moves:
            skipped_no_moves += 1
            continue

        for move in moves:
            # Use the median holding period from settings as the label horizon
            for hold_min in HOLD_PERIODS_MIN:
                row = _label_move(move, mrow, hold_min, stop_loss_pct, take_profit_pct)
                if row is None:
                    skipped_no_price += 1
                    continue
                all_rows.append(row)

        if i % 100 == 0:
            logger.info("[%d/%d] processed  %d labeled rows so far", i, len(catalog), len(all_rows))

    logger.info(
        "Labeling complete: %d rows  (skipped: no_file=%d, no_moves=%d, no_equity_price=%d)",
        len(all_rows), skipped_no_file, skipped_no_moves, skipped_no_price,
    )

    if dry_run:
        logger.info("Dry run — not writing to DB. Sample rows:")
        for r in all_rows[:5]:
            logger.info("  %s", r)
        return

    if not all_rows:
        logger.warning("No labeled rows generated — check data pull and thresholds")
        return

    written = _write_to_db(all_rows, str(DB_PATH))
    logger.info("Wrote %d new rows to %s", written, DB_PATH)

    # Quick summary
    df = pd.DataFrame(all_rows)
    wins = (df["outcome"] == "WIN").sum()
    total = len(df)
    logger.info(
        "Label summary: %d total rows  win_rate=%.1f%%  tickers=%s",
        total,
        wins / total * 100 if total else 0,
        df["signal_ticker"].value_counts().to_dict(),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Label historical Polymarket whale moves")
    parser.add_argument("--min-delta", type=float, default=0.05,
                        help="Min price delta to count as whale move (default 0.05)")
    parser.add_argument("--min-volume", type=float, default=10000,
                        help="Min 24h volume to include a market (default $10k)")
    parser.add_argument("--max-markets", type=int, default=None,
                        help="Limit to first N markets (for testing)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print rows but don't write to DB")
    args = parser.parse_args()

    run(
        min_delta=args.min_delta,
        min_volume=args.min_volume,
        dry_run=args.dry_run,
        max_markets=args.max_markets,
    )
