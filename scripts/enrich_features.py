"""enrich_features.py — Retroactively enrich training data with real signal features.

Problem: clean_training_data.parquet was generated from label_events.py (historical
price labeling). It contains zero signal features because no real scanner events were
matched to the labeled trades.

Fix: For each labeled trade, look back a configurable window in the real Truth Social
posts data and Polymarket price data to populate:

  Truth Social:
    has_ts_signal       — 1 if a qualifying Trump post exists in the lookback window
    ts_keyword_count    — keyword count from the most keyword-rich post in window
    ts_engagement       — engagement score from that post

  Polymarket (optional, --poly flag):
    has_poly_signal     — 1 if any tracked market moved > threshold in lookback window
    poly_price_delta    — largest absolute price delta across tracked markets
    poly_price_delta_abs
    poly_volume_spike_pct — approx volume spike (based on 7-day rolling avg volume)

The enrichment is retrospective — it asks: "On days when Trump posted about markets,
did trading outcomes differ from quiet days?" This is exactly what the L2 model needs
to learn.

Output: D:/WhaleWatch_Data/clean_training_data.parquet (enriched in-place)
        D:/WhaleWatch_Data/enrichment_report.txt

Usage:
    python scripts/enrich_features.py
    python scripts/enrich_features.py --lookback-hours 6
    python scripts/enrich_features.py --poly             # also enrich Polymarket features
    python scripts/enrich_features.py --dry-run          # report only, no write
"""

import argparse
import logging
import os
import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("enrich_features")

DATA_ROOT = Path("D:/WhaleWatch_Data")
CLEAN_DATA_PATH = DATA_ROOT / "clean_training_data.parquet"
TS_POSTS_PATH = DATA_ROOT / "truth_social" / "posts.parquet"
POLY_CATALOG_PATH = DATA_ROOT / "polymarket" / "markets_catalog.parquet"
POLY_PRICES_DIR = DATA_ROOT / "polymarket" / "prices"
REPORT_PATH = DATA_ROOT / "enrichment_report.txt"

# Polymarket: only markets with meaningful volume (filter noise)
POLY_MIN_VOLUME_24H = 5_000.0        # USD — ignore very small markets
POLY_DELTA_THRESHOLD = 0.02          # 2% price move = significant
POLY_SPIKE_LOOKBACK_DAYS = 7         # rolling window for volume spike baseline


# ---------------------------------------------------------------------------
# Truth Social enrichment
# ---------------------------------------------------------------------------

def _enrich_ts(
    df: pd.DataFrame,
    posts: pd.DataFrame,
    lookback_hours: float = 8.0,
    min_keywords: int = 1,
) -> pd.DataFrame:
    """Populate has_ts_signal, ts_keyword_count, ts_engagement for each row.

    For each trade's created_at, looks back `lookback_hours` for any Trump post
    with keyword_count >= min_keywords. Picks the post with the highest keyword_count
    (most signal-rich) within the window.

    Args:
        df:              Training data with 'created_at' column (UTC-aware)
        posts:           posts.parquet DataFrame with 'posted_at', 'keyword_count', 'engagement'
        lookback_hours:  How far back to look from created_at (default 8h)
        min_keywords:    Minimum keyword count to count as a signal (default 1)
    """
    logger.info(
        "Enriching TS features: lookback=%.0fh  min_keywords=%d  posts=%d  trades=%d",
        lookback_hours, min_keywords, len(posts), len(df),
    )

    # Ensure datetime types are timezone-aware
    if posts["posted_at"].dt.tz is None:
        posts = posts.copy()
        posts["posted_at"] = posts["posted_at"].dt.tz_localize("UTC")

    if df["created_at"].dt.tz is None:
        df = df.copy()
        df["created_at"] = df["created_at"].dt.tz_localize("UTC")

    # Filter to qualifying posts only (has market keywords)
    signal_posts = posts[posts["keyword_count"] >= min_keywords].copy()
    signal_posts = signal_posts.sort_values("posted_at").reset_index(drop=True)
    logger.info("  Qualifying posts (keyword_count >= %d): %d / %d",
                min_keywords, len(signal_posts), len(posts))

    # For each trade, find the best (highest keyword_count) post in the lookback window.
    # Use a merge_asof approach: sort trades by created_at, then for each trade find
    # posts within [created_at - lookback_hours, created_at].
    delta = pd.Timedelta(hours=lookback_hours)

    trade_times = df["created_at"].values
    post_times = signal_posts["posted_at"].values
    post_kw = signal_posts["keyword_count"].values
    post_eng = signal_posts["engagement"].values

    has_signal = np.zeros(len(df), dtype=int)
    kw_count = np.zeros(len(df), dtype=float)
    engagement = np.zeros(len(df), dtype=float)

    # Binary search bounds for efficiency
    from bisect import bisect_left, bisect_right

    # Convert to numpy datetime64 for fast comparison
    post_times_np = np.array(post_times, dtype="datetime64[ns]")
    trade_times_np = np.array([np.datetime64(t) for t in trade_times])

    matched = 0
    for i, t in enumerate(trade_times_np):
        lo = t - np.timedelta64(int(lookback_hours * 3600e9), "ns")
        hi = t

        lo_idx = bisect_left(post_times_np, lo)
        hi_idx = bisect_right(post_times_np, hi)

        if hi_idx > lo_idx:
            # Pick the post with the most keywords in the window
            window_kw = post_kw[lo_idx:hi_idx]
            window_eng = post_eng[lo_idx:hi_idx]
            best_idx = int(np.argmax(window_kw))
            has_signal[i] = 1
            kw_count[i] = float(window_kw[best_idx])
            engagement[i] = float(window_eng[best_idx])
            matched += 1

    df = df.copy()
    df["has_ts_signal"] = has_signal
    df["ts_keyword_count"] = kw_count
    df["ts_engagement"] = engagement

    logger.info(
        "  TS enrichment: %d / %d trades matched (%.1f%%)",
        matched, len(df), matched / len(df) * 100,
    )
    return df


# ---------------------------------------------------------------------------
# Polymarket enrichment
# ---------------------------------------------------------------------------

def _load_poly_catalog(min_volume: float = POLY_MIN_VOLUME_24H) -> pd.DataFrame:
    """Load Polymarket markets catalog, filtered to active/liquid markets."""
    if not POLY_CATALOG_PATH.exists():
        logger.warning("Polymarket catalog not found — skipping poly enrichment")
        return pd.DataFrame()

    cat = pd.read_parquet(POLY_CATALOG_PATH)
    # Filter to markets with meaningful liquidity
    if "volume_24h" in cat.columns:
        cat = cat[cat["volume_24h"] >= min_volume]
    if "closed" in cat.columns:
        cat = cat[~cat["closed"]]
    logger.info("Polymarket catalog: %d liquid/active markets", len(cat))
    return cat


def _load_poly_prices(condition_id: str, yes_token_id: str) -> pd.Series | None:
    """Load the YES price time series for a given market."""
    # Try condition_id-based file first, then yes_token_id
    for stem in [condition_id, yes_token_id]:
        path = POLY_PRICES_DIR / f"{stem}_YES.parquet"
        if path.exists():
            try:
                df = pd.read_parquet(path)
                if "yes_price" not in df.columns:
                    return None
                s = df["yes_price"].dropna()
                if not isinstance(s.index, pd.DatetimeIndex):
                    s.index = pd.to_datetime(s.index, utc=True)
                elif s.index.tz is None:
                    s.index = s.index.tz_localize("UTC")
                return s.sort_index()
            except Exception:
                continue
    return None


def _enrich_poly(
    df: pd.DataFrame,
    lookback_days: float = 1.0,
    delta_threshold: float = POLY_DELTA_THRESHOLD,
    max_markets: int = 500,
) -> pd.DataFrame:
    """Populate has_poly_signal, poly_price_delta, poly_price_delta_abs, poly_volume_spike_pct.

    For each trade's created_at, scans the top liquid Polymarket markets for any that moved
    more than delta_threshold in the lookback window. Records the largest move seen.

    This is slow (many files); use max_markets to limit to the most liquid ones.
    """
    catalog = _load_poly_catalog()
    if catalog.empty:
        return df

    # Sort by 24h volume, take top N most liquid
    if "volume_24h" in catalog.columns:
        catalog = catalog.nlargest(max_markets, "volume_24h")

    logger.info("Loading Polymarket prices for %d markets ...", len(catalog))

    # Pre-load all price series into memory (most are small)
    price_map: dict[str, pd.Series] = {}
    loaded = 0
    for _, row in catalog.iterrows():
        cid = row["condition_id"]
        ytid = row.get("yes_token_id", "")
        s = _load_poly_prices(cid, ytid)
        if s is not None and len(s) >= 2:
            price_map[cid] = s
            loaded += 1

    logger.info("  Loaded %d / %d price series", loaded, len(catalog))

    if not price_map:
        logger.warning("No Polymarket price files found — poly features stay zero")
        return df

    # Build a combined daily-price matrix for fast lookups
    # Each series is daily; align to a common date index
    combined = pd.DataFrame(price_map)
    if not isinstance(combined.index, pd.DatetimeIndex):
        combined.index = pd.to_datetime(combined.index, utc=True)
    combined = combined.sort_index()

    # For each trade, compute max abs delta across all markets in lookback window
    delta_lookback = pd.Timedelta(days=lookback_days)

    has_poly = np.zeros(len(df), dtype=int)
    poly_delta = np.zeros(len(df), dtype=float)
    poly_delta_abs = np.zeros(len(df), dtype=float)
    poly_spike_pct = np.zeros(len(df), dtype=float)

    matched = 0
    for i, ts in enumerate(df["created_at"]):
        lo = ts - delta_lookback

        window = combined[(combined.index >= lo) & (combined.index <= ts)]
        if len(window) < 2:
            continue

        # Compute per-column (market) price delta over the window
        first = window.iloc[0]
        last = window.iloc[-1]
        deltas = (last - first).dropna()

        if deltas.empty:
            continue

        # Find largest absolute move
        abs_deltas = deltas.abs()
        max_abs = abs_deltas.max()

        if max_abs >= delta_threshold:
            best_col = abs_deltas.idxmax()
            has_poly[i] = 1
            poly_delta[i] = float(deltas[best_col])
            poly_delta_abs[i] = float(max_abs)

            # Volume spike proxy: compare latest 24h volume vs 7-day rolling average
            # We use price volatility (std of window) as a proxy since we don't have
            # per-market volume time series at this resolution
            window_std = float(window[best_col].std())
            poly_spike_pct[i] = round(window_std * 100, 2)  # % std as spike proxy

            matched += 1

    df = df.copy()
    df["has_poly_signal"] = has_poly
    df["poly_price_delta"] = np.clip(poly_delta, -1.0, 1.0)
    df["poly_price_delta_abs"] = np.clip(poly_delta_abs, 0.0, 1.0)
    df["poly_volume_spike_pct"] = np.clip(poly_spike_pct, 0.0, None)
    # poly_yes_direction: +1 if delta > 0, -1 if delta < 0, 0 otherwise
    df["poly_yes_direction"] = np.sign(poly_delta).astype(int)
    # dual_signal: 1 if both TS and Poly fired
    df["dual_signal"] = ((df["has_ts_signal"] > 0) & (df["has_poly_signal"] > 0)).astype(int)

    logger.info(
        "  Poly enrichment: %d / %d trades matched (%.1f%%)",
        matched, len(df), matched / len(df) * 100,
    )
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(
    lookback_hours: float = 8.0,
    min_keywords: int = 1,
    enrich_poly: bool = False,
    poly_lookback_days: float = 1.0,
    poly_max_markets: int = 300,
    dry_run: bool = False,
) -> pd.DataFrame | None:
    # Load training data
    if not CLEAN_DATA_PATH.exists():
        logger.error("clean_training_data.parquet not found — run clean_data.py first")
        return None
    df = pd.read_parquet(CLEAN_DATA_PATH)
    logger.info("Loaded training data: %d rows", len(df))

    # Snapshot of signal coverage before enrichment
    ts_before = int((df.get("has_ts_signal", pd.Series(0)) > 0).sum())
    poly_before = int((df.get("has_poly_signal", pd.Series(0)) > 0).sum())

    # ---- Truth Social enrichment ----
    if not TS_POSTS_PATH.exists():
        logger.warning("posts.parquet not found at %s — skipping TS enrichment", TS_POSTS_PATH)
    else:
        posts = pd.read_parquet(TS_POSTS_PATH)
        logger.info("Loaded %d Truth Social posts", len(posts))
        df = _enrich_ts(df, posts, lookback_hours=lookback_hours, min_keywords=min_keywords)

    # ---- Polymarket enrichment (optional) ----
    if enrich_poly:
        df = _enrich_poly(
            df,
            lookback_days=poly_lookback_days,
            max_markets=poly_max_markets,
        )
        # Recompute dual_signal
        df["dual_signal"] = ((df["has_ts_signal"] > 0) & (df["has_poly_signal"] > 0)).astype(int)
    else:
        # Without poly enrichment, dual_signal = has_ts_signal (single signal source)
        # Keep existing dual_signal (zeros) — only set True when both actually fired
        pass

    # ---- Report ----
    ts_after = int((df["has_ts_signal"] > 0).sum())
    poly_after = int((df["has_poly_signal"] > 0).sum())
    dual_after = int((df["dual_signal"] > 0).sum())

    print("\n" + "=" * 60)
    print("ENRICHMENT REPORT")
    print("=" * 60)
    print(f"  Total rows             : {len(df)}")
    print(f"  TS signal rows before  : {ts_before}")
    print(f"  TS signal rows after   : {ts_after}  ({ts_after/len(df)*100:.1f}%)")
    print(f"  Poly signal rows before: {poly_before}")
    print(f"  Poly signal rows after : {poly_after}  ({poly_after/len(df)*100:.1f}%)")
    print(f"  Dual signal rows       : {dual_after}  ({dual_after/len(df)*100:.1f}%)")

    if ts_after > 0:
        print()
        print("  TS signal — win rate comparison:")
        ts_win = df[df["has_ts_signal"] > 0]["outcome"].eq("WIN").mean()
        no_ts_win = df[df["has_ts_signal"] == 0]["outcome"].eq("WIN").mean()
        print(f"    With TS signal    : {ts_win:.1%}")
        print(f"    Without TS signal : {no_ts_win:.1%}")
        print(f"    TS signal lift    : {(ts_win - no_ts_win):+.1%}")

        print()
        print("  TS signal — win rate by direction + ticker:")
        grp = (
            df.groupby(["has_ts_signal", "signal_direction", "signal_ticker"])["outcome"]
            .apply(lambda x: (x == "WIN").mean())
            .round(3)
            .rename("win_rate")
        )
        print(grp.to_string())

    print("=" * 60 + "\n")

    report_text = (
        f"Enrichment run\n"
        f"  lookback_hours={lookback_hours}  min_keywords={min_keywords}  "
        f"enrich_poly={enrich_poly}\n"
        f"  rows={len(df)}  ts_after={ts_after}  poly_after={poly_after}  "
        f"dual={dual_after}\n"
        f"  ts_win_rate_with_signal={ts_win if ts_after>0 else 'N/A':.3f}\n"
        f"  ts_win_rate_without={no_ts_win if ts_after>0 else 'N/A':.3f}\n"
    )

    if dry_run:
        logger.info("--dry-run: not writing output file")
        return df

    # Write back
    df.to_parquet(CLEAN_DATA_PATH, index=False)
    logger.info("Enriched data saved → %s", CLEAN_DATA_PATH)

    REPORT_PATH.write_text(report_text)
    logger.info("Report saved → %s", REPORT_PATH)

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enrich training data with real signal features")
    parser.add_argument(
        "--lookback-hours", type=float, default=8.0,
        help="Hours to look back from trade created_at for TS posts (default: 8)",
    )
    parser.add_argument(
        "--min-keywords", type=int, default=1,
        help="Minimum keyword count to count a TS post as a signal (default: 1)",
    )
    parser.add_argument(
        "--poly", action="store_true",
        help="Also enrich Polymarket features (slower — reads ~300 price files)",
    )
    parser.add_argument(
        "--poly-lookback-days", type=float, default=1.0,
        help="Days to look back for Polymarket price delta (default: 1)",
    )
    parser.add_argument(
        "--poly-max-markets", type=int, default=300,
        help="Max number of Polymarket markets to scan (top by volume, default: 300)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Run enrichment and print report but do not write output",
    )
    args = parser.parse_args()

    run(
        lookback_hours=args.lookback_hours,
        min_keywords=args.min_keywords,
        enrich_poly=args.poly,
        poly_lookback_days=args.poly_lookback_days,
        poly_max_markets=args.poly_max_markets,
        dry_run=args.dry_run,
    )
