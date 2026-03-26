"""generate_ts_training_data.py — Build real L2 training data from Trump posts + LLM.

Problem: The existing clean_training_data.parquet was generated synthetically with random
directions. Signal features (has_ts_signal, ts_keyword_count, etc.) are all zero.
The model cannot learn signal-direction-outcome relationships from it.

Fix: For each qualifying Trump Truth Social post:
  1. Build a TruthSocialRawEvent from posts.parquet
  2. Call Layer 1 LLM (Haiku — cheap) → direction + ticker
  3. Skip HOLD decisions
  4. Find entry price on the post's date (daily close), exit 1 trading day later
  5. Compute realized_pnl and outcome (WIN / LOSS / STOP_OUT)
  6. Build the full feature vector (all FEATURE_NAMES populated from real signal data)
  7. Save as ts_training_data.parquet

This gives ~1,000–2,000 real signal events where direction was LLM-inferred from
actual post content — exactly what Layer 2 needs to learn from.

The script checkpoints every CHECKPOINT_EVERY posts so you can safely interrupt and
resume. Already-processed posts are skipped based on the checkpoint file.

Estimated cost: ~2,598 posts × ~300 tokens @ Haiku pricing ≈ $0.80 total.

Usage:
    python scripts/generate_ts_training_data.py
    python scripts/generate_ts_training_data.py --min-keywords 2   # stricter signal filter
    python scripts/generate_ts_training_data.py --model claude-haiku-4-5-20251001
    python scripts/generate_ts_training_data.py --merge            # merge into clean_training_data.parquet
    python scripts/generate_ts_training_data.py --dry-run 20       # process only first 20 posts

Output:
    D:/WhaleWatch_Data/ts_training_data.parquet   — new real training examples
    D:/WhaleWatch_Data/ts_training_checkpoint.json — checkpoint for resume

Schema: same columns as clean_training_data.parquet, compatible with train_l2.py
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("gen_ts_training")

DATA_ROOT = Path("D:/WhaleWatch_Data")
TS_POSTS_PATH = DATA_ROOT / "truth_social" / "posts.parquet"
EQUITY_DIR = DATA_ROOT / "equity"
OUT_PATH = DATA_ROOT / "ts_training_data.parquet"
CHECKPOINT_PATH = DATA_ROOT / "ts_training_checkpoint.json"
CLEAN_DATA_PATH = DATA_ROOT / "clean_training_data.parquet"

CHECKPOINT_EVERY = 50       # save checkpoint every N posts
DEFAULT_MODEL = "claude-haiku-4-5-20251001"
STOP_LOSS_PCT = 0.02
TAKE_PROFIT_PCT = 0.04
HOLD_DAYS = 1               # 1 trading day hold = ~1440 minutes
API_CALL_DELAY = 0.3        # seconds between API calls (avoid rate limit)

TICKERS = ["SPY", "QQQ", "VIX"]
FEATURE_NAMES = [
    "poly_price_delta", "poly_price_delta_abs", "poly_volume_spike_pct",
    "poly_yes_direction", "has_poly_signal",
    "ts_keyword_count", "ts_engagement", "has_ts_signal",
    "dual_signal",
    "hour_of_day", "day_of_week", "is_us_market_hours", "is_premarket",
    "vix_level", "vix_percentile",
    "direction_buy", "direction_short",
    "ticker_spy", "ticker_qqq", "ticker_vix",
]


# ---------------------------------------------------------------------------
# Price data loading
# ---------------------------------------------------------------------------

def _load_price_data() -> dict[str, pd.Series]:
    """Load daily close prices for SPY, QQQ, VIX. Returns {ticker: pd.Series}."""
    prices = {}
    for ticker in TICKERS:
        path = EQUITY_DIR / f"{ticker}_1d.parquet"
        if not path.exists():
            logger.warning("Missing daily price file: %s", path)
            continue
        df = pd.read_parquet(path)
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, utc=True)
        elif df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        close_col = "close" if "close" in df.columns else "Close"
        prices[ticker] = df[close_col].sort_index().dropna()
        logger.info("Loaded %s daily prices: %d bars (%s → %s)",
                    ticker, len(prices[ticker]),
                    prices[ticker].index[0].date(), prices[ticker].index[-1].date())
    return prices


def _load_vix_regime() -> tuple[pd.Series | None, pd.Series | None]:
    """Load VIX daily close and rolling percentile for regime features."""
    path = EQUITY_DIR / "VIX_1d.parquet"
    if not path.exists():
        return None, None
    try:
        df = pd.read_parquet(path)
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, utc=True)
        elif df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        close_col = "close" if "close" in df.columns else "Close"
        vix_series = df[close_col].sort_index().dropna()
        vix_pct = vix_series.rank(pct=True).rolling(252, min_periods=10).mean()
        return vix_series, vix_pct
    except Exception as exc:
        logger.warning("Failed to load VIX regime data: %s", exc)
        return None, None


def _vix_at(ts: pd.Timestamp, vix_series: pd.Series, vix_pct: pd.Series) -> tuple[float, float]:
    mask = vix_series.index <= ts
    if not mask.any():
        return 0.0, 0.0
    level = float(vix_series[mask].iloc[-1])
    pct = float(vix_pct[mask].iloc[-1]) if vix_pct is not None else 0.0
    return level, (pct if not np.isnan(pct) else 0.0)


# ---------------------------------------------------------------------------
# Price outcome simulation
# ---------------------------------------------------------------------------

def _simulate_outcome(
    ticker: str,
    direction: str,
    posted_at: pd.Timestamp,
    prices: dict[str, pd.Series],
    tp_pct: float = TAKE_PROFIT_PCT,
    sl_pct: float = STOP_LOSS_PCT,
    hold_days: int = HOLD_DAYS,
) -> tuple[float | None, float | None, str | None, float | None]:
    """Simulate a 1-day trade and return (entry_price, exit_price, outcome, realized_pnl).

    Entry: the close price on the first trading day on or after posted_at.
    Exit:  the close price hold_days trading days after entry.

    Uses daily bars so intraday stop/TP checks are approximated by comparing the
    daily return against the thresholds.
    """
    if ticker not in prices:
        return None, None, None, None

    series = prices[ticker]

    # Find first trading day on or after the post date (normalize to midnight UTC)
    post_date = posted_at.normalize()
    mask_entry = series.index >= post_date
    if not mask_entry.any():
        return None, None, None, None

    entry_iloc = int(np.argmax(mask_entry))
    if entry_iloc + hold_days >= len(series):
        return None, None, None, None  # not enough future data

    entry_price = float(series.iloc[entry_iloc])
    exit_price = float(series.iloc[entry_iloc + hold_days])

    if entry_price <= 0:
        return None, None, None, None

    # Raw return in the signal direction
    if direction == "BUY":
        raw_ret = (exit_price - entry_price) / entry_price
    else:  # SHORT
        raw_ret = (entry_price - exit_price) / entry_price

    # Apply TP/SL clip (approximation: if raw return exceeds TP, cap at TP; if below -SL, stop out)
    if raw_ret >= tp_pct:
        outcome = "WIN"
        realized_pnl = tp_pct
    elif raw_ret <= -sl_pct:
        outcome = "STOP_OUT"
        realized_pnl = -sl_pct
    elif raw_ret > 0:
        outcome = "WIN"
        realized_pnl = raw_ret
    else:
        outcome = "LOSS"
        realized_pnl = raw_ret

    return entry_price, exit_price, outcome, round(realized_pnl, 6)


# ---------------------------------------------------------------------------
# Feature vector builder
# ---------------------------------------------------------------------------

def _build_row(
    post: pd.Series,
    direction: str,
    ticker: str,
    llm_model: str,
    entry_price: float,
    exit_price: float,
    outcome: str,
    realized_pnl: float,
    vix_series: pd.Series | None,
    vix_pct: pd.Series | None,
) -> dict:
    """Build a training row dict compatible with clean_training_data.parquet schema."""
    ts = post["posted_at"]
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    ts = pd.Timestamp(ts)

    # VIX regime at signal time
    vix_level, vix_percentile = (0.0, 0.0)
    if vix_series is not None:
        vix_level, vix_percentile = _vix_at(ts, vix_series, vix_pct)

    # Temporal
    hour = ts.hour
    dow = ts.weekday()
    minute_of_day = ts.hour * 60 + ts.minute
    is_market = int(810 <= minute_of_day < 1200)
    is_pre = int(480 <= minute_of_day < 810)

    return {
        # Metadata (not in FEATURE_NAMES but needed for training pipeline)
        "order_id": str(post["post_id"]),
        "event_id": str(post["post_id"]),
        "created_at": ts,
        "signal_direction": direction,
        "signal_ticker": ticker,
        "confidence": 0.5,             # placeholder — will be replaced by trained model
        "holding_period_min": hold_days_to_minutes(HOLD_DAYS),
        "stop_loss_pct": STOP_LOSS_PCT,
        "take_profit_pct": TAKE_PROFIT_PCT,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "realized_pnl": realized_pnl,
        "outcome": outcome,
        "close_reason": "TS_SIGNAL_LABELED",
        "llm_model": llm_model,

        # Feature columns (FEATURE_NAMES)
        "poly_price_delta": 0.0,
        "poly_price_delta_abs": 0.0,
        "poly_volume_spike_pct": 0.0,
        "poly_yes_direction": 0,
        "has_poly_signal": 0,
        "ts_keyword_count": int(post.get("keyword_count", 0)),
        "ts_engagement": float(post.get("engagement", 0.0)),
        "has_ts_signal": 1,
        "dual_signal": 0,
        "hour_of_day": hour,
        "day_of_week": dow,
        "is_us_market_hours": is_market,
        "is_premarket": is_pre,
        "vix_level": vix_level,
        "vix_percentile": vix_percentile,
        "direction_buy": int(direction == "BUY"),
        "direction_short": int(direction == "SHORT"),
        "ticker_spy": int(ticker == "SPY"),
        "ticker_qqq": int(ticker == "QQQ"),
        "ticker_vix": int(ticker == "VIX"),

        # Transaction cost columns (will be filled by CostModel later)
        "trade_cost": 0.0,
        "net_pnl": realized_pnl,
        "net_outcome": outcome,
    }


def hold_days_to_minutes(days: int) -> int:
    return days * 1440


# ---------------------------------------------------------------------------
# LLM caller
# ---------------------------------------------------------------------------

def _build_llm(model: str):
    from reasoner.layer1_llm.claude_llm import ClaudeLLM
    return ClaudeLLM(model=model, max_retries=3)


def _post_to_event(post: pd.Series):
    from models.raw_events import TruthSocialRawEvent
    ts = post["posted_at"]
    if hasattr(ts, "to_pydatetime"):
        ts = ts.to_pydatetime()
    if ts.tzinfo is None:
        from datetime import timezone as _tz
        ts = ts.replace(tzinfo=_tz.utc)

    keywords = post.get("keywords", [])
    if not isinstance(keywords, list):
        try:
            import ast
            keywords = ast.literal_eval(str(keywords))
        except Exception:
            keywords = []

    return TruthSocialRawEvent(
        post_id=str(post["post_id"]),
        content=str(post.get("content", "")),
        posted_at=ts,
        pulled_at=datetime.now(tz=timezone.utc),
        replies_count=int(post.get("replies_count", 0)),
        reblogs_count=int(post.get("reblogs_count", 0)),
        favourites_count=int(post.get("favourites_count", 0)),
        keywords=keywords,
        is_repost=bool(post.get("is_repost", False)),
        language=post.get("language"),
    )


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _load_checkpoint() -> dict:
    if CHECKPOINT_PATH.exists():
        try:
            return json.loads(CHECKPOINT_PATH.read_text())
        except Exception:
            pass
    return {"processed_ids": [], "rows": []}


def _save_checkpoint(ckpt: dict) -> None:
    CHECKPOINT_PATH.write_text(json.dumps(ckpt, default=str, indent=2))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(
    min_keywords: int = 1,
    model: str = DEFAULT_MODEL,
    merge: bool = False,
    dry_run_limit: int = 0,
) -> pd.DataFrame | None:

    if not TS_POSTS_PATH.exists():
        logger.error("posts.parquet not found at %s", TS_POSTS_PATH)
        return None

    # Load data
    posts = pd.read_parquet(TS_POSTS_PATH)
    if posts["posted_at"].dt.tz is None:
        posts["posted_at"] = posts["posted_at"].dt.tz_localize("UTC")
    posts = posts.sort_values("posted_at").reset_index(drop=True)

    # Filter to keyword-bearing posts only
    qualifying = posts[posts["keyword_count"] >= min_keywords].copy()
    logger.info(
        "Qualifying posts (keyword_count >= %d): %d / %d",
        min_keywords, len(qualifying), len(posts),
    )

    if dry_run_limit > 0:
        qualifying = qualifying.head(dry_run_limit)
        logger.info("--dry-run: limiting to first %d posts", dry_run_limit)

    # Load price data and VIX regime
    prices = _load_price_data()
    vix_series, vix_pct = _load_vix_regime()

    if not prices:
        logger.error("No equity price data found — cannot simulate outcomes")
        return None

    # Load LLM
    llm = _build_llm(model)
    logger.info("LLM: %s", model)

    # Resume from checkpoint
    ckpt = _load_checkpoint()
    processed_ids = set(ckpt.get("processed_ids", []))
    rows = ckpt.get("rows", [])
    logger.info(
        "Checkpoint: %d already processed, %d rows saved",
        len(processed_ids), len(rows),
    )

    # Process posts
    n_hold = 0
    n_no_price = 0
    n_processed = 0
    n_total = len(qualifying)

    for i, (_, post) in enumerate(qualifying.iterrows()):
        pid = str(post["post_id"])

        if pid in processed_ids:
            continue

        # Rate limit
        if n_processed > 0:
            time.sleep(API_CALL_DELAY)

        # Call Layer 1 LLM
        try:
            event = _post_to_event(post)
            l1 = llm.get_signal(event)
        except Exception as exc:
            logger.warning("LLM error for post %s: %s — skipping", pid, exc)
            processed_ids.add(pid)
            n_processed += 1
            continue

        # Skip HOLD
        if l1.direction == "HOLD":
            n_hold += 1
            processed_ids.add(pid)
            n_processed += 1
        else:
            # Simulate trade outcome
            entry, exit_, outcome, pnl = _simulate_outcome(
                ticker=l1.ticker,
                direction=l1.direction,
                posted_at=pd.Timestamp(post["posted_at"]),
                prices=prices,
            )

            if entry is None:
                n_no_price += 1
                processed_ids.add(pid)
                n_processed += 1
                logger.debug("No price data for post %s at %s", pid, post["posted_at"])
            else:
                row = _build_row(
                    post=post,
                    direction=l1.direction,
                    ticker=l1.ticker,
                    llm_model=model,
                    entry_price=entry,
                    exit_price=exit_,
                    outcome=outcome,
                    realized_pnl=pnl,
                    vix_series=vix_series,
                    vix_pct=vix_pct,
                )
                rows.append(row)
                processed_ids.add(pid)
                n_processed += 1

        # Progress log
        total_done = len(processed_ids)
        if total_done % 10 == 0 or total_done == n_total:
            win_rate = sum(1 for r in rows if r.get("outcome") == "WIN") / max(len(rows), 1)
            logger.info(
                "Progress: %d/%d posts | rows=%d | hold=%d | no_price=%d | win_rate=%.1f%%",
                total_done, n_total, len(rows), n_hold, n_no_price, win_rate * 100,
            )

        # Checkpoint
        if n_processed % CHECKPOINT_EVERY == 0 and n_processed > 0:
            ckpt_save = {"processed_ids": list(processed_ids), "rows": rows}
            _save_checkpoint(ckpt_save)

    # Final checkpoint save
    ckpt_save = {"processed_ids": list(processed_ids), "rows": rows}
    _save_checkpoint(ckpt_save)

    if not rows:
        logger.warning("No training rows generated — check API key and price data")
        return None

    df = pd.DataFrame(rows)
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True)

    # Summary report
    outcome_counts = df["outcome"].value_counts()
    win_rate = (df["outcome"] == "WIN").mean()

    print("\n" + "=" * 60)
    print("GENERATION REPORT")
    print("=" * 60)
    print(f"  Posts processed        : {len(processed_ids)}")
    print(f"  HOLD (skipped)         : {n_hold}")
    print(f"  No price data (skipped): {n_no_price}")
    print(f"  Training rows generated: {len(df)}")
    print()
    print("  Outcome distribution:")
    for outcome, cnt in outcome_counts.items():
        print(f"    {outcome:10s}: {cnt:4d}  ({cnt/len(df)*100:.1f}%)")
    print(f"  Overall win rate       : {win_rate:.1%}")
    print()
    print("  Win rate by direction + ticker:")
    grp = df.groupby(["signal_direction", "signal_ticker"])["outcome"].apply(
        lambda x: (x == "WIN").mean()
    ).round(3)
    print(grp.to_string())
    print()
    print("  Date range:")
    print(f"    {df['created_at'].min().date()} → {df['created_at'].max().date()}")
    print("=" * 60 + "\n")

    # Save
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)
    logger.info("TS training data saved → %s  (%d rows)", OUT_PATH, len(df))

    # Optionally merge with existing clean_training_data.parquet
    if merge:
        _merge_into_clean(df)

    return df


def _merge_into_clean(ts_df: pd.DataFrame) -> None:
    """Merge new TS training rows into clean_training_data.parquet.

    Deduplicates on (created_at, signal_ticker, signal_direction) to avoid
    double-counting posts that had multiple keyword groups.
    """
    if not CLEAN_DATA_PATH.exists():
        logger.warning("clean_training_data.parquet not found — saving TS data as-is")
        ts_df.to_parquet(CLEAN_DATA_PATH, index=False)
        return

    existing = pd.read_parquet(CLEAN_DATA_PATH)
    logger.info("Existing clean data: %d rows", len(existing))

    # Align columns — fill missing with 0/None
    for col in ts_df.columns:
        if col not in existing.columns:
            existing[col] = np.nan
    for col in existing.columns:
        if col not in ts_df.columns:
            ts_df[col] = np.nan

    combined = pd.concat([existing, ts_df], ignore_index=True)
    combined["created_at"] = pd.to_datetime(combined["created_at"], utc=True)
    combined = combined.drop_duplicates(
        subset=["created_at", "signal_ticker", "signal_direction"]
    ).sort_values("created_at").reset_index(drop=True)

    combined.to_parquet(CLEAN_DATA_PATH, index=False)
    logger.info(
        "Merged into clean_training_data.parquet: %d → %d rows",
        len(existing), len(combined),
    )

    # New win rate
    win_rate = (combined["outcome"] == "WIN").mean()
    ts_coverage = (combined.get("has_ts_signal", pd.Series(0)) > 0).mean()
    logger.info(
        "Merged dataset: win_rate=%.1f%%  ts_signal_coverage=%.1f%%",
        win_rate * 100, ts_coverage * 100,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate real L2 training data from Trump posts via LLM"
    )
    parser.add_argument(
        "--min-keywords", type=int, default=1,
        help="Minimum keyword count to process a post (default: 1)",
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help=f"Claude model for Layer 1 classification (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--merge", action="store_true",
        help="Merge generated data into clean_training_data.parquet after generation",
    )
    parser.add_argument(
        "--dry-run", type=int, default=0, metavar="N",
        help="Process only the first N posts (no API cost limit check needed)",
    )
    parser.add_argument(
        "--clear-checkpoint", action="store_true",
        help="Delete any existing checkpoint and start fresh",
    )
    args = parser.parse_args()

    if args.clear_checkpoint and CHECKPOINT_PATH.exists():
        CHECKPOINT_PATH.unlink()
        logger.info("Checkpoint cleared — starting fresh")

    logger.info("=" * 60)
    logger.info("WhaleWatch_Alpha — TS Training Data Generator")
    logger.info("Model    : %s", args.model)
    logger.info("Min kw   : %d", args.min_keywords)
    logger.info("Merge    : %s", args.merge)
    logger.info("Dry-run  : %s", f"first {args.dry_run} posts" if args.dry_run else "no")
    logger.info("=" * 60)

    run(
        min_keywords=args.min_keywords,
        model=args.model,
        merge=args.merge,
        dry_run_limit=args.dry_run,
    )
