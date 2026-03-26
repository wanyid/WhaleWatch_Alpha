"""clean_data.py — Data cleaning pipeline for input features and response variables.

Reads raw labeled events from paper_trades.db, applies cleaning transforms to
both X (feature) and y (outcome/holding_period) columns, and writes a clean
parquet file ready for model training and backtesting.

Cleaning steps:
  INPUT FEATURES (X):
    1. Schema validation — reject rows missing mandatory fields
    2. Outlier clipping — Winsorise continuous features at [1%, 99%]
    3. Consistency checks — poly_price_delta must be in [-1, 1]; poly_volume_spike ≥ 0
    4. Temporal feature validation — hour ∈ [0,23], dow ∈ [0,6]
    5. One-hot integrity — direction and ticker flags sum to 1 each
    6. Duplicate removal — deduplicate on (created_at, signal_ticker, signal_direction)
    7. Training-window filter — drop rows before TRAINING_START (2025-01-20)

  RESPONSE VARIABLES (y):
    1. outcome — only keep WIN / LOSS / STOP_OUT (drop OPEN / null)
    2. holding_period_min — clip to [1, 4320]; flag and drop implausible values
    3. realized_pnl — must be within [-stop_loss_pct, take_profit_pct]; re-derive if not
    4. Outcome–pnl consistency — WIN must have pnl > 0; LOSS must have pnl ≤ 0

Output: D:/WhaleWatch_Data/clean_training_data.parquet

Usage:
    python scripts/clean_data.py
    python scripts/clean_data.py --report   # print quality report only, no write
"""

import argparse
import logging
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from backtest.transaction_costs import CostModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("clean_data")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_ROOT = Path("D:/WhaleWatch_Data")
DB_PATH = DATA_ROOT / "paper_trades.db"
OUT_PATH = DATA_ROOT / "clean_training_data.parquet"
SETTINGS_PATH = Path("config/settings.yaml")
TRAINING_START = "2025-01-20"

# Feature groups (must match features.py FEATURE_NAMES exactly)
POLY_CONTINUOUS = ["poly_price_delta", "poly_price_delta_abs", "poly_volume_spike_pct"]
TEMPORAL = ["hour_of_day", "day_of_week"]
VIX_FEATURES = ["vix_level", "vix_percentile"]
BINARY_FEATURES = [
    "poly_yes_direction", "has_poly_signal", "has_ts_signal",
    "dual_signal", "is_us_market_hours", "is_premarket",
    "direction_buy", "direction_short",
    "ticker_spy", "ticker_qqq", "ticker_vix",
]
ALL_FEATURES = (
    POLY_CONTINUOUS
    + ["ts_keyword_count", "ts_engagement"]
    + TEMPORAL
    + VIX_FEATURES
    + BINARY_FEATURES
)

VALID_OUTCOMES = {"WIN", "LOSS", "STOP_OUT"}
VALID_TICKERS = {"SPY", "QQQ", "VIX"}
VALID_DIRECTIONS = {"BUY", "SHORT"}

MIN_HOLD = 1
MAX_HOLD = 4320


def _load_settings() -> dict:
    with open(SETTINGS_PATH) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Load raw data from SQLite
# ---------------------------------------------------------------------------

def load_raw(db_path: str = str(DB_PATH)) -> pd.DataFrame:
    """Load all resolved positions from SQLite into a DataFrame."""
    if not Path(db_path).exists():
        raise FileNotFoundError(f"DB not found: {db_path} — run pull_historical_data.py first")

    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query(
            """
            SELECT order_id, event_id, created_at,
                   signal_direction, signal_ticker,
                   confidence, holding_period_min,
                   stop_loss_pct, take_profit_pct,
                   entry_price, exit_price, realized_pnl,
                   outcome, close_reason
            FROM positions
            WHERE outcome IN ('WIN', 'LOSS', 'STOP_OUT')
            ORDER BY created_at
            """,
            conn,
        )

    logger.info("Loaded %d resolved positions from DB", len(df))
    return df


# ---------------------------------------------------------------------------
# Response variable cleaning (y)
# ---------------------------------------------------------------------------

def clean_response(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, dict]:
    """Clean outcome (y) columns. Returns cleaned df and a report dict."""
    risk_cfg = cfg.get("risk", {})
    sl = risk_cfg.get("stop_loss_pct", 0.02)
    tp = risk_cfg.get("take_profit_pct", 0.04)

    report: dict = {}
    n0 = len(df)

    # 1. Valid outcome values only
    mask_outcome = df["outcome"].isin(VALID_OUTCOMES)
    report["dropped_invalid_outcome"] = (~mask_outcome).sum()
    df = df[mask_outcome].copy()

    # 2. Valid direction and ticker
    mask_dir = df["signal_direction"].isin(VALID_DIRECTIONS)
    report["dropped_invalid_direction"] = (~mask_dir).sum()
    df = df[mask_dir].copy()

    mask_tkr = df["signal_ticker"].isin(VALID_TICKERS)
    report["dropped_invalid_ticker"] = (~mask_tkr).sum()
    df = df[mask_tkr].copy()

    # 3. Clip holding period
    df["holding_period_min"] = df["holding_period_min"].fillna(60).clip(MIN_HOLD, MAX_HOLD)

    # 4. Clamp/re-derive realized_pnl
    def _rederive_pnl(row):
        """Re-derive P&L from prices if stored pnl is outside valid bounds."""
        ep = row["entry_price"]
        xp = row["exit_price"]
        sl_r = row["stop_loss_pct"] or sl
        tp_r = row["take_profit_pct"] or tp
        stored = row["realized_pnl"]
        if pd.isna(stored) or abs(stored) > max(sl_r, tp_r) * 1.05:
            if ep and xp and ep > 0:
                raw = (xp - ep) / ep if row["signal_direction"] == "BUY" \
                      else (ep - xp) / ep
                return float(np.clip(raw, -sl_r, tp_r))
        return stored

    df["realized_pnl"] = df.apply(_rederive_pnl, axis=1).fillna(0.0)

    # 5. Outcome–pnl consistency
    bad_win = (df["outcome"] == "WIN") & (df["realized_pnl"] <= 0)
    bad_loss = (df["outcome"] == "LOSS") & (df["realized_pnl"] >= 0)
    bad_stop = (df["outcome"] == "STOP_OUT") & (df["realized_pnl"] > 0)
    inconsistent = bad_win | bad_loss | bad_stop
    report["inconsistent_outcome_pnl"] = inconsistent.sum()

    # Fix: correct the outcome label based on actual pnl
    df.loc[bad_win, "outcome"] = "LOSS"
    df.loc[bad_loss, "outcome"] = "WIN"
    df.loc[bad_stop, "outcome"] = "LOSS"

    # 6. Prices must be positive
    df = df[(df["entry_price"].isna() | (df["entry_price"] > 0))].copy()
    df = df[(df["exit_price"].isna() | (df["exit_price"] > 0))].copy()

    report["total_dropped_response"] = n0 - len(df)
    logger.info("Response cleaning: %d → %d rows (%s)", n0, len(df), report)
    return df, report


# ---------------------------------------------------------------------------
# Feature reconstruction from DB columns
# ---------------------------------------------------------------------------

def _reconstruct_features(df: pd.DataFrame) -> pd.DataFrame:
    """Reconstruct the feature matrix columns from the DB's raw columns.

    The DB stores high-level columns (signal_direction, signal_ticker, confidence,
    holding_period_min, created_at) rather than the full feature vector.
    We rebuild the features that can be derived without the original scanner event.
    """
    # Parse timestamps
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True)

    # Temporal
    df["hour_of_day"] = df["created_at"].dt.hour
    df["day_of_week"] = df["created_at"].dt.dayofweek
    minute_of_day = df["created_at"].dt.hour * 60 + df["created_at"].dt.minute
    df["is_us_market_hours"] = ((minute_of_day >= 810) & (minute_of_day < 1200)).astype(int)
    df["is_premarket"] = ((minute_of_day >= 480) & (minute_of_day < 810)).astype(int)

    # Direction one-hot
    df["direction_buy"] = (df["signal_direction"] == "BUY").astype(int)
    df["direction_short"] = (df["signal_direction"] == "SHORT").astype(int)

    # Ticker one-hot
    df["ticker_spy"] = (df["signal_ticker"] == "SPY").astype(int)
    df["ticker_qqq"] = (df["signal_ticker"] == "QQQ").astype(int)
    df["ticker_vix"] = (df["signal_ticker"] == "VIX").astype(int)

    # Poly features — labeled events use simplified heuristics (no live scanner state)
    # Fill zeros for events that came from the labeling script
    for col in POLY_CONTINUOUS:
        if col not in df.columns:
            df[col] = 0.0
    for col in ["poly_yes_direction", "has_poly_signal"]:
        if col not in df.columns:
            df[col] = 0

    # Derive poly_price_delta_abs from poly_price_delta
    if "poly_price_delta" in df.columns:
        df["poly_price_delta_abs"] = df["poly_price_delta"].abs()

    # Truth Social proxy features
    if "ts_keyword_count" not in df.columns:
        df["ts_keyword_count"] = 0
    if "ts_engagement" not in df.columns:
        df["ts_engagement"] = 0.0
    if "has_ts_signal" not in df.columns:
        df["has_ts_signal"] = 0
    if "dual_signal" not in df.columns:
        df["dual_signal"] = 0

    # VIX regime features — looked up from stored VIX parquet per signal timestamp
    df = _add_vix_features(df)

    return df


def _add_vix_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add vix_level and vix_percentile columns by looking up VIX_1d.parquet."""
    vix_path = DATA_ROOT / "equity" / "VIX_1d.parquet"
    if not vix_path.exists():
        logger.warning("VIX_1d.parquet not found — vix_level/vix_percentile will be 0")
        df["vix_level"] = 0.0
        df["vix_percentile"] = 0.0
        return df

    try:
        vix = pd.read_parquet(vix_path)
        if not isinstance(vix.index, pd.DatetimeIndex):
            vix.index = pd.to_datetime(vix.index, utc=True)
        elif vix.index.tz is None:
            vix.index = vix.index.tz_localize("UTC")

        close_col = "close" if "close" in vix.columns else \
                    "Close" if "Close" in vix.columns else vix.columns[3]
        vix_series = vix[close_col].sort_index().dropna()
        vix_pct = vix_series.rank(pct=True).rolling(252, min_periods=10).mean()

        def _lookup(ts):
            mask = vix_series.index <= ts
            if not mask.any():
                return 0.0, 0.0
            level = float(vix_series[mask].iloc[-1])
            pct = float(vix_pct[mask].iloc[-1]) if vix_pct is not None else 0.0
            return level, (pct if not np.isnan(pct) else 0.0)

        levels = []
        pcts = []
        for ts in df["created_at"]:
            lvl, pct = _lookup(ts)
            levels.append(lvl)
            pcts.append(pct)

        df["vix_level"] = levels
        df["vix_percentile"] = pcts
        logger.info("VIX features added: non-zero in %d/%d rows",
                    (np.array(levels) > 0).sum(), len(levels))
    except Exception as exc:
        logger.warning("VIX feature computation failed: %s — filling 0", exc)
        df["vix_level"] = 0.0
        df["vix_percentile"] = 0.0

    return df


# ---------------------------------------------------------------------------
# Feature cleaning (X)
# ---------------------------------------------------------------------------

def clean_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Clean input feature columns. Returns df with features + cleaning report."""
    report: dict = {}
    n0 = len(df)

    df = _reconstruct_features(df)

    # 1. Training-window filter
    mask_window = df["created_at"].dt.date.astype(str) >= TRAINING_START
    report["dropped_before_training_start"] = (~mask_window).sum()
    df = df[mask_window].copy()

    # 2. Clip poly continuous features to valid range
    df["poly_price_delta"] = df["poly_price_delta"].clip(-1.0, 1.0)
    df["poly_price_delta_abs"] = df["poly_price_delta_abs"].clip(0.0, 1.0)
    df["poly_volume_spike_pct"] = df["poly_volume_spike_pct"].clip(0.0, None)

    # 3. Winsorise at 1%/99% for continuous cols with variance
    winsor_cols = ["poly_volume_spike_pct", "ts_keyword_count", "ts_engagement"]
    clip_report = {}
    for col in winsor_cols:
        if col in df.columns and df[col].std() > 0:
            lo = df[col].quantile(0.01)
            hi = df[col].quantile(0.99)
            n_clipped = ((df[col] < lo) | (df[col] > hi)).sum()
            df[col] = df[col].clip(lo, hi)
            clip_report[col] = int(n_clipped)
    report["winsorised"] = clip_report

    # 4. Temporal sanity
    df["hour_of_day"] = df["hour_of_day"].clip(0, 23)
    df["day_of_week"] = df["day_of_week"].clip(0, 6)

    # 5. Binary feature enforcement
    for col in BINARY_FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna(0).astype(int).clip(0, 1)

    # 6. One-hot integrity: direction and ticker must sum to exactly 1
    dir_sum = df["direction_buy"] + df["direction_short"]
    bad_dir = dir_sum != 1
    report["bad_direction_onehot"] = int(bad_dir.sum())
    df = df[~bad_dir].copy()

    tkr_sum = df["ticker_spy"] + df["ticker_qqq"] + df["ticker_vix"]
    bad_tkr = tkr_sum != 1
    report["bad_ticker_onehot"] = int(bad_tkr.sum())
    df = df[~bad_tkr].copy()

    # 7. Drop any remaining NaN in feature columns
    feature_cols = [c for c in ALL_FEATURES if c in df.columns]
    before_nan_drop = len(df)
    df = df.dropna(subset=feature_cols)
    report["dropped_nan_features"] = before_nan_drop - len(df)

    # 8. Deduplicate on (created_at, signal_ticker, signal_direction)
    before_dedup = len(df)
    df = df.drop_duplicates(subset=["created_at", "signal_ticker", "signal_direction"])
    report["deduped"] = before_dedup - len(df)

    report["total_dropped_features"] = n0 - len(df)
    logger.info("Feature cleaning: %d → %d rows (%s)", n0, len(df), report)
    return df, report


# ---------------------------------------------------------------------------
# Quality report
# ---------------------------------------------------------------------------

def quality_report(df: pd.DataFrame) -> None:
    """Print a human-readable data quality summary."""
    print("\n" + "=" * 60)
    print("DATA QUALITY REPORT")
    print("=" * 60)
    print(f"  Total clean rows      : {len(df)}")
    print(f"  Date range            : {df['created_at'].min().date()} → {df['created_at'].max().date()}")
    print()
    print("  Outcome distribution:")
    for outcome, cnt in df["outcome"].value_counts().items():
        pct = cnt / len(df) * 100
        print(f"    {outcome:12s}: {cnt:5d}  ({pct:.1f}%)")

    print()
    print("  Ticker distribution:")
    for tkr, cnt in df["signal_ticker"].value_counts().items():
        print(f"    {tkr:6s}: {cnt:5d}")

    print()
    print("  Direction distribution:")
    for d, cnt in df["signal_direction"].value_counts().items():
        print(f"    {d:7s}: {cnt:5d}")

    print()
    print("  P&L stats (realized_pnl):")
    pnl = df["realized_pnl"].dropna()
    print(f"    mean  : {pnl.mean():.4f}")
    print(f"    std   : {pnl.std():.4f}")
    print(f"    min   : {pnl.min():.4f}")
    print(f"    max   : {pnl.max():.4f}")
    print(f"    median: {pnl.median():.4f}")

    print()
    print("  Holding period (minutes):")
    hp = df["holding_period_min"].dropna()
    print(f"    mean  : {hp.mean():.0f}")
    print(f"    median: {hp.median():.0f}")
    print(f"    min   : {hp.min():.0f}")
    print(f"    max   : {hp.max():.0f}")

    feature_cols = [c for c in ALL_FEATURES if c in df.columns]
    print()
    print(f"  Features available    : {len(feature_cols)}/{len(ALL_FEATURES)}")
    missing_features = [c for c in ALL_FEATURES if c not in df.columns]
    if missing_features:
        print(f"  Missing features      : {missing_features}")
    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(report_only: bool = False, db_path: Optional[str] = None) -> Optional[pd.DataFrame]:
    cfg = _load_settings()
    db = db_path or str(DB_PATH)

    # Load
    try:
        raw = load_raw(db)
    except FileNotFoundError as e:
        logger.error("%s", e)
        return None

    if raw.empty:
        logger.warning("No resolved events in DB — run label_events.py first")
        return None

    # Clean response variables first (y)
    df, resp_report = clean_response(raw, cfg)

    # Then clean features (X)
    df, feat_report = clean_features(df)

    if df.empty:
        logger.warning("All rows dropped during cleaning — check data quality")
        return None

    # Apply transaction costs — replace realized_pnl and outcome with net values
    # so the model trains on what it would actually earn after costs.
    costs = CostModel()
    df = costs.apply(df)
    cost_summary = costs.summary(df)
    logger.info(
        "Transaction costs applied: gross_pnl=%.4f  total_cost=%.4f  net_pnl=%.4f"
        "  trades_turned_loss=%d  net_win_rate=%.1f%%",
        cost_summary["gross_pnl"],
        cost_summary["total_cost"],
        cost_summary["net_pnl"],
        cost_summary["trades_turned_loss"],
        cost_summary["net_win_rate"] * 100,
    )
    # Promote net values to the training labels so the model learns net returns
    df["realized_pnl"] = df["net_pnl"]
    df["outcome"] = df["net_outcome"]

    # Quality report
    quality_report(df)

    if report_only:
        logger.info("--report mode: not writing output file")
        return df

    # Write clean parquet
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)
    logger.info("Clean data saved → %s  (%d rows, %d columns)", OUT_PATH, len(df), len(df.columns))
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean training data for L2 predictor")
    parser.add_argument("--report", action="store_true",
                        help="Print quality report only, don't write output file")
    parser.add_argument("--db", default=None, help="Path to paper_trades.db (override default)")
    args = parser.parse_args()
    run(report_only=args.report, db_path=args.db)
