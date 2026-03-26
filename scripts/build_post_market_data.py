"""build_post_market_data.py — Build Trump post → market reaction dataset.

For each Trump Truth Social post, computes price returns across multiple
holding periods using real market data. No LLM involved — labels are purely
objective market outcomes.

Holding periods:
  Intraday (requires 5m data):  5m, 30m, 1h, 2h, 4h
  Daily    (requires 1d data):  1d

Out-of-hours handling:
  Trump posts ~62% of the time outside US market hours (09:30–16:00 ET).
  For intraday periods the "entry" is the first 5m bar at or after market
  open on the next trading day, not the post timestamp itself.
  For 1d the entry is always the closing price on the post date (or next
  trading day if posted on a weekend/holiday).

Features added beyond posts.parquet:
  - favourites_count, reblogs_count   (raw engagement from TS)
  - engagement                        (log1p of total, already in posts)
  - caps_ratio                        (fraction uppercase — proxy for urgency)
  - content_length
  - Keyword category flags: has_tariff, has_deal, has_china, has_fed,
                            has_energy, has_geopolitical, has_market
  - Temporal: hour_of_day, day_of_week, is_market_hours, is_premarket
  - VIX regime at post time: vix_level, vix_percentile

Outputs:
  D:/WhaleWatch_Data/post_market_data.parquet

Usage:
    python scripts/build_post_market_data.py                     # 5m window (Dec 29+)
    python scripts/build_post_market_data.py --full              # 1d only, all posts
    python scripts/build_post_market_data.py --start 2026-02-01  # custom start
    python scripts/build_post_market_data.py --min-keywords 0    # include all posts
"""

import argparse
import logging
import sys
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
logger = logging.getLogger("build_post_market")

DATA_ROOT   = Path("D:/WhaleWatch_Data")
EQUITY_DIR  = DATA_ROOT / "equity"
POSTS_PATH  = DATA_ROOT / "truth_social" / "posts.parquet"
OUT_PATH    = DATA_ROOT / "post_market_data.parquet"

TICKERS = ["SPY", "QQQ", "VIX", "VIXY"]

# US market hours in UTC: 13:30–20:00
MARKET_OPEN_UTC  = 13 * 60 + 30   # 810 minutes
MARKET_CLOSE_UTC = 20 * 60        # 1200 minutes

# Intraday holding periods in minutes
INTRADAY_PERIODS = [5, 30, 60, 120, 240]   # 5m 30m 1h 2h 4h

# Fade model constants
FADE_ENTRY_LAG = "30m"                    # initial move window before fade entry
FADE_PERIODS   = ["2h", "4h", "1d"]       # holding periods for fade model

# Keyword category flags
KEYWORD_CATEGORIES = {
    "has_tariff":       ["tariff", "tariffs", "trade war", "trade deal", "import", "export", "trade deficit"],
    "has_deal":         ["deal", "agreement", "treaty", "ceasefire"],
    "has_china":        ["china"],
    "has_fed":          ["fed", "federal reserve", "interest rate", "inflation", "rate cut"],
    "has_energy":       ["oil", "gas", "energy", "lng", "opec", "gasoline", "pipeline", "drill"],
    "has_geopolitical": ["ukraine", "russia", "iran", "nato", "war", "military", "sanctions", "north korea", "israel"],
    "has_market":       ["stock", "stocks", "market", "nasdaq", "s&p", "dow", "wall street", "bitcoin", "crypto"],
}


# ---------------------------------------------------------------------------
# Price data loading
# ---------------------------------------------------------------------------

def _load_prices() -> dict:
    """Load all available price series. Returns nested dict:
    {ticker: {resolution: pd.Series(close, index=DatetimeIndex UTC)}}

    Priority for each resolution (first found wins):
      5m  : {TICKER}_5m_poly.parquet  >  {TICKER}_5m_recent.parquet
      1h  : {TICKER}_1h_poly.parquet  >  {TICKER}_1h.parquet
      1d  : {TICKER}_1d.parquet  (no Polygon daily needed — yfinance has full history)
    """
    prices = {}
    for ticker in TICKERS:
        prices[ticker] = {}

        # Daily — yfinance 1d has full history, no need for Polygon
        p = EQUITY_DIR / f"{ticker}_1d.parquet"
        if p.exists():
            df = pd.read_parquet(p)
            df = _ensure_utc_index(df)
            col = "close" if "close" in df.columns else "Close"
            prices[ticker]["1d"] = df[col].sort_index().dropna()

        # 5-minute — prefer Polygon (longer history), fallback to yfinance recent
        for candidate in [f"{ticker}_5m_poly.parquet", f"{ticker}_5m_recent.parquet"]:
            p = EQUITY_DIR / candidate
            if p.exists():
                df = pd.read_parquet(p)
                df = _ensure_utc_index(df)
                col = "close" if "close" in df.columns else "Close"
                prices[ticker]["5m"] = df[col].sort_index().dropna()
                break   # use first found

        # 1-hour — prefer Polygon (Jan 2025+), fallback to yfinance chunked (Apr 2024+)
        for candidate in [f"{ticker}_1h_poly.parquet", f"{ticker}_1h.parquet"]:
            p = EQUITY_DIR / candidate
            if p.exists():
                df = pd.read_parquet(p)
                df = _ensure_utc_index(df)
                col = "close" if "close" in df.columns else "Close"
                prices[ticker]["1h"] = df[col].sort_index().dropna()
                break

        # 2-minute (short-range precision, kept for completeness)
        p = EQUITY_DIR / f"{ticker}_2m.parquet"
        if p.exists():
            df = pd.read_parquet(p)
            df = _ensure_utc_index(df)
            col = "close" if "close" in df.columns else "Close"
            prices[ticker]["2m"] = df[col].sort_index().dropna()

    for ticker, resolutions in prices.items():
        for res, s in resolutions.items():
            logger.info("  Loaded %s %s: %d bars  %s → %s",
                        ticker, res, len(s), s.index[0].date(), s.index[-1].date())

    return prices


def _ensure_utc_index(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
    elif df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    return df


def _load_vix_regime(prices: dict) -> tuple:
    """Return (vix_daily_close, vix_percentile, vixy_daily_close)."""
    s = prices.get("VIX", {}).get("1d")
    if s is None:
        return None, None, None
    pct = s.rank(pct=True).rolling(252, min_periods=10).mean()
    # VIXY: use 1d if available, else 1h as fallback
    vixy = prices.get("VIXY", {}).get("1d") or prices.get("VIXY", {}).get("1h")
    return s, pct, vixy


# ---------------------------------------------------------------------------
# Price lookup helpers
# ---------------------------------------------------------------------------

def _price_at_or_after(series: pd.Series, ts: pd.Timestamp) -> float | None:
    """Return first price at or after ts. Returns None if no data after ts."""
    mask = series.index >= ts
    if not mask.any():
        return None
    return float(series[mask].iloc[0])


def _price_N_minutes_after(series: pd.Series, entry_ts: pd.Timestamp, minutes: int) -> float | None:
    """Return price N minutes after entry_ts using intraday bars.

    Finds the first bar at or after (entry_ts + minutes). Returns None if
    no data exists that far out (e.g. end of available history).
    """
    target = entry_ts + pd.Timedelta(minutes=minutes)
    return _price_at_or_after(series, target)


def _next_market_open(ts: pd.Timestamp, series_1d: pd.Series) -> pd.Timestamp | None:
    """Return the timestamp of market open (13:30 UTC) on the next trading day
    on or after ts. Uses the 1d price series to find valid trading days.
    """
    ts_date = ts.normalize()
    # Check if ts itself is on a trading day AND market hasn't opened yet
    if ts_date in series_1d.index:
        min_of_day = ts.hour * 60 + ts.minute
        if min_of_day < MARKET_OPEN_UTC:
            # Same trading day, market hasn't opened yet
            return ts_date.replace(hour=13, minute=30, second=0, microsecond=0)

    # Otherwise find next trading day
    future_days = series_1d.index[series_1d.index > ts_date]
    if len(future_days) == 0:
        return None
    next_day = future_days[0]
    return next_day.replace(hour=13, minute=30, second=0, microsecond=0)


def _is_market_hours(ts: pd.Timestamp) -> bool:
    m = ts.hour * 60 + ts.minute
    return MARKET_OPEN_UTC <= m < MARKET_CLOSE_UTC


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def _extract_features(post: pd.Series, vix_series, vix_pct, vixy_series) -> dict:
    ts = post["posted_at"]
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")

    content  = str(post.get("content", "")).lower()
    raw_kws  = post.get("keywords", [])
    if isinstance(raw_kws, np.ndarray):
        raw_kws = raw_kws.tolist()
    elif not isinstance(raw_kws, list):
        raw_kws = []

    # Engagement
    favs    = int(post.get("favourites_count", 0))
    reblogs = int(post.get("reblogs_count", 0))
    replies = int(post.get("replies_count", 0))
    engagement = float(np.log1p(favs + reblogs + replies))

    # Content features
    raw_content = str(post.get("content", ""))
    caps_ratio = sum(1 for c in raw_content if c.isupper()) / max(len(raw_content), 1)
    content_length = len(raw_content)

    # Temporal
    hour     = ts.hour
    dow      = ts.weekday()
    min_day  = ts.hour * 60 + ts.minute
    is_mkt   = int(MARKET_OPEN_UTC <= min_day < MARKET_CLOSE_UTC)
    is_pre   = int(480 <= min_day < MARKET_OPEN_UTC)

    # VIX regime (from daily close)
    vix_level, vix_percentile = 0.0, 0.0
    if vix_series is not None:
        mask = vix_series.index <= ts
        if mask.any():
            vix_level = float(vix_series[mask].iloc[-1])
            if vix_pct is not None:
                v = float(vix_pct[mask].iloc[-1])
                vix_percentile = v if not np.isnan(v) else 0.0

    # VIXY price level at post time (most recent bar at or before ts)
    vixy_level = 0.0
    if vixy_series is not None:
        mask_v = vixy_series.index <= ts
        if mask_v.any():
            vixy_level = round(float(vixy_series[mask_v].iloc[-1]), 4)

    # Keyword category flags
    kw_flags = {}
    for flag, keywords in KEYWORD_CATEGORIES.items():
        kw_flags[flag] = int(any(k in content for k in keywords))

    return {
        "post_id":        str(post["post_id"]),
        "posted_at":      ts,
        # Raw engagement
        "favourites_count": favs,
        "reblogs_count":    reblogs,
        "replies_count":    replies,
        "engagement":       round(engagement, 4),
        # Content
        "keyword_count":    int(post.get("keyword_count", 0)),
        "caps_ratio":       round(caps_ratio, 4),
        "content_length":   content_length,
        # Keyword categories
        **kw_flags,
        # Temporal
        "hour_of_day":        hour,
        "day_of_week":        dow,
        "is_market_hours":    is_mkt,
        "is_premarket":       is_pre,
        # VIX regime
        "vix_level":          round(vix_level, 4),
        "vix_percentile":     round(vix_percentile, 4),
        # VIXY level (tradeable VIX proxy — available after Polygon pull)
        "vixy_level":         vixy_level,
    }


# ---------------------------------------------------------------------------
# Return computation
# ---------------------------------------------------------------------------

def _compute_returns(
    ts: pd.Timestamp,
    prices: dict,
    spy_1d: pd.Series,
) -> dict:
    """Compute price returns for all tickers and all holding periods.

    For intraday periods:
      - Entry = first 5m bar at or after market open on the post day
        (handles out-of-hours posts)
    For 1d:
      - Entry = daily close on post date or next trading day
    """
    returns = {}
    entry_meta = {}

    # ---- Intraday entry: next market open on or after post ----
    intra_entry_ts = _next_market_open(ts, spy_1d)

    for ticker in TICKERS:
        t_prices = prices[ticker]

        # ---- 1d return ----
        s_1d = t_prices.get("1d")
        if s_1d is not None:
            # Entry: close on post date (or next trading day)
            post_date = ts.normalize()
            mask_entry = s_1d.index >= post_date
            if mask_entry.any():
                entry_iloc = int(np.argmax(mask_entry))
                if entry_iloc + 1 < len(s_1d):
                    entry_1d = float(s_1d.iloc[entry_iloc])
                    exit_1d  = float(s_1d.iloc[entry_iloc + 1])
                    returns[f"{ticker.lower()}_ret_1d"] = round(
                        (exit_1d - entry_1d) / entry_1d, 6
                    )
                    if ticker == "SPY":
                        entry_meta["entry_price_1d"] = entry_1d
                        entry_meta["entry_date_1d"]  = s_1d.index[entry_iloc].date().isoformat()

        # ---- Intraday returns ----
        # Entry = first actual bar in the 5m series at or after next market open.
        # All T+N offsets are measured from the ENTRY BAR'S real timestamp, not
        # from the hardcoded 13:30 UTC. This handles EST (open=14:30 UTC) vs EDT
        # (open=13:30 UTC) automatically — the bar just has whichever timestamp the
        # exchange actually opened at.
        s_5m = t_prices.get("5m")
        if s_5m is not None and intra_entry_ts is not None and intra_entry_ts >= s_5m.index[0]:
            # Find the actual first bar at or after the estimated open.
            # Guard above ensures the entry date is within the 5m data coverage window —
            # without it, a post from e.g. March 2025 would match ALL Dec 2025+ bars and
            # incorrectly use the first available 5m bar as its intraday entry.
            mask_entry = s_5m.index >= intra_entry_ts
            if mask_entry.any():
                actual_entry_ts    = s_5m.index[mask_entry][0]
                entry_price_5m     = float(s_5m.iloc[int(np.argmax(mask_entry))])

                if ticker == "SPY":
                    entry_meta["entry_price_intra"] = entry_price_5m
                    entry_meta["entry_time_intra"]  = actual_entry_ts.isoformat()

                for minutes in INTRADAY_PERIODS:
                    # Measure from the real entry bar timestamp
                    exit_price = _price_N_minutes_after(s_5m, actual_entry_ts, minutes)
                    if exit_price is not None:
                        label = f"{ticker.lower()}_ret_{minutes}m"
                        if minutes >= 60:
                            h = minutes // 60
                            label = f"{ticker.lower()}_ret_{h}h"
                        returns[label] = round(
                            (exit_price - entry_price_5m) / entry_price_5m, 6
                        )

        # ---- 1h fallback: use when 5m data doesn't cover this date ----
        # Computes 1h, 2h, 4h returns only (can't do 5m/30m from 1h bars).
        # Only fills labels not already set by the 5m block above.
        s_1h = t_prices.get("1h")
        if s_1h is not None and intra_entry_ts is not None and intra_entry_ts >= s_1h.index[0]:
            mask_entry_1h = s_1h.index >= intra_entry_ts
            if mask_entry_1h.any():
                actual_entry_1h   = s_1h.index[mask_entry_1h][0]
                entry_price_1h    = float(s_1h.iloc[int(np.argmax(mask_entry_1h))])

                if ticker == "SPY" and "entry_price_intra" not in entry_meta:
                    entry_meta["entry_price_intra"] = entry_price_1h
                    entry_meta["entry_time_intra"]  = actual_entry_1h.isoformat()

                for hours, col_label in [(1, "1h"), (2, "2h"), (4, "4h")]:
                    dest = f"{ticker.lower()}_ret_{col_label}"
                    if dest in returns:
                        continue   # already computed from 5m — don't overwrite
                    exit_price = _price_N_minutes_after(s_1h, actual_entry_1h, hours * 60)
                    if exit_price is not None:
                        returns[dest] = round(
                            (exit_price - entry_price_1h) / entry_price_1h, 6
                        )

    return returns, entry_meta


# ---------------------------------------------------------------------------
# Fade label computation
# ---------------------------------------------------------------------------

def add_fade_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Add fade (mean-reversion) labels and continuation returns to the dataset.

    Fade thesis: after a Trump post the market makes an initial 30m move. When
    that move is large enough (filtered at training time), fading the direction
    may be profitable as the overshoot reverts.

    Columns added:
      spy_initial_ret        — same as spy_ret_30m; signed initial move
      spy_initial_direction  — +1 or -1
      spy_abs_initial_ret    — |spy_ret_30m|
      spy_ret_cont_{p}       — continuation return (spy_ret_{p} - spy_ret_30m)
                               for p in ["2h", "4h", "1d"]
      spy_fade_label_{p}     — 1 if continuation moved OPPOSITE to initial move
                               (fade worked), 0 if continuation continued in
                               the same direction; NaN if either return missing.
    """
    if "spy_ret_30m" not in df.columns:
        logger.warning("spy_ret_30m not found — skipping fade label computation")
        return df

    df = df.copy()

    df["spy_initial_ret"]       = df["spy_ret_30m"]
    df["spy_initial_direction"] = np.sign(df["spy_ret_30m"])
    df["spy_abs_initial_ret"]   = df["spy_ret_30m"].abs()

    period_col_map = {"2h": "spy_ret_2h", "4h": "spy_ret_4h", "1d": "spy_ret_1d"}

    for period in FADE_PERIODS:
        total_col = period_col_map.get(period)
        cont_col  = f"spy_ret_cont_{period}"
        lbl_col   = f"spy_fade_label_{period}"

        if total_col not in df.columns:
            logger.warning("  %s not found — skipping fade labels for %s", total_col, period)
            continue

        # Continuation return = full-period return minus initial 30m move
        df[cont_col] = df[total_col] - df["spy_ret_30m"]

        # Fade worked = continuation moved opposite to initial direction
        both_valid  = df["spy_ret_30m"].notna() & df[total_col].notna()
        fade_worked = (df[cont_col] * df["spy_ret_30m"]) < 0
        df[lbl_col] = np.where(both_valid, fade_worked.astype(int), np.nan)

    # Summary
    for period in FADE_PERIODS:
        lbl_col = f"spy_fade_label_{period}"
        if lbl_col in df.columns:
            s = df[lbl_col].dropna()
            if len(s) > 0:
                logger.info(
                    "  fade_label_%s: n=%d  fade_worked=%.1f%%  "
                    "initial_ret mean=|%.4f|",
                    period, len(s), float(s.mean()) * 100,
                    float(df["spy_abs_initial_ret"].dropna().mean()),
                )

    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(
    start_date: str | None = None,
    min_keywords: int = 1,
    full_history: bool = False,
) -> pd.DataFrame | None:

    if not POSTS_PATH.exists():
        logger.error("posts.parquet not found: %s", POSTS_PATH)
        return None

    # Load posts
    posts = pd.read_parquet(POSTS_PATH)
    if posts["posted_at"].dt.tz is None:
        posts["posted_at"] = posts["posted_at"].dt.tz_localize("UTC")
    posts = posts.sort_values("posted_at").reset_index(drop=True)

    # Default start: 5m data window unless --full requested
    if full_history:
        default_start = "2025-01-20"
    else:
        default_start = "2025-12-29"   # start of 5m data

    start = pd.Timestamp(start_date or default_start, tz="UTC")
    posts = posts[posts["posted_at"] >= start]

    if min_keywords > 0:
        posts = posts[posts["keyword_count"] >= min_keywords]

    logger.info("Posts to process: %d  (start=%s  min_keywords=%d)",
                len(posts), start.date(), min_keywords)

    # Load prices
    logger.info("Loading price data ...")
    prices = _load_prices()
    vix_series, vix_pct, vixy_series = _load_vix_regime(prices)

    spy_1d = prices["SPY"]["1d"]

    rows = []
    skipped = 0

    for _, post in posts.iterrows():
        ts = post["posted_at"]

        features = _extract_features(post, vix_series, vix_pct, vixy_series)
        returns, entry_meta = _compute_returns(ts, prices, spy_1d)

        if not returns:
            skipped += 1
            continue

        row = {**features, **returns, **entry_meta}
        rows.append(row)

    logger.info("Built %d rows  (%d skipped — no price data)", len(rows), skipped)

    if not rows:
        logger.error("No rows built — check price data coverage")
        return None

    df = pd.DataFrame(rows)
    df["posted_at"] = pd.to_datetime(df["posted_at"], utc=True)

    # ---- Fade labels ----
    logger.info("Computing fade labels ...")
    df = add_fade_labels(df)

    # ---- Summary report ----
    ret_cols_1d    = [c for c in df.columns if c.endswith("_ret_1d")]
    ret_cols_intra = [c for c in df.columns if "_ret_" in c and not c.endswith("_ret_1d")]

    print("\n" + "=" * 65)
    print("POST MARKET DATA SUMMARY")
    print("=" * 65)
    print(f"  Posts processed   : {len(df)}")
    print(f"  Date range        : {df['posted_at'].min().date()} → {df['posted_at'].max().date()}")
    print(f"  In market hours   : {df['is_market_hours'].sum()} ({df['is_market_hours'].mean():.0%})")
    print()

    print("  1-day return stats (entry close → next close):")
    for col in sorted(ret_cols_1d):
        s = df[col].dropna()
        print(f"    {col:20s}  n={len(s):4d}  mean={s.mean():+.4f}  "
              f"std={s.std():.4f}  pos={( s>0).mean():.0%}")

    if ret_cols_intra:
        print()
        print("  Intraday return stats (entry = next market open):")
        for col in sorted(ret_cols_intra):
            s = df[col].dropna()
            print(f"    {col:20s}  n={len(s):4d}  mean={s.mean():+.4f}  "
                  f"std={s.std():.4f}  pos={(s>0).mean():.0%}")

    print()
    print("  Engagement distribution:")
    print(f"    favourites  mean={df['favourites_count'].mean():.0f}  "
          f"median={df['favourites_count'].median():.0f}  "
          f"max={df['favourites_count'].max():.0f}")
    print(f"    reblogs     mean={df['reblogs_count'].mean():.0f}  "
          f"median={df['reblogs_count'].median():.0f}  "
          f"max={df['reblogs_count'].max():.0f}")

    print()
    print("  Keyword category coverage:")
    for flag in KEYWORD_CATEGORIES:
        if flag in df.columns:
            n = df[flag].sum()
            print(f"    {flag:22s}: {n:4d} posts ({n/len(df):.0%})")

    fade_label_cols = [c for c in df.columns if c.startswith("spy_fade_label_")]
    if fade_label_cols:
        print()
        print("  Fade label coverage (fade_worked rate):")
        for col in sorted(fade_label_cols):
            s = df[col].dropna()
            print(f"    {col:26s}: n={len(s):4d}  fade_worked={s.mean():.0%}")

    print("=" * 65 + "\n")

    # Save
    df.to_parquet(OUT_PATH, index=False)
    logger.info("Saved → %s  (%d rows, %d columns)", OUT_PATH, len(df), len(df.columns))
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build Trump post → market reaction dataset"
    )
    parser.add_argument(
        "--start", default=None,
        help="Start date YYYY-MM-DD (default: 2025-12-29 for 5m test, or 2025-01-20 with --full)",
    )
    parser.add_argument(
        "--full", action="store_true",
        help="Use full history (1d only, Jan 2025+). Without this flag: 5m window (Dec 29+).",
    )
    parser.add_argument(
        "--min-keywords", type=int, default=1,
        help="Minimum keyword count to include post (default: 1, use 0 for all posts)",
    )
    args = parser.parse_args()

    run(
        start_date=args.start,
        min_keywords=args.min_keywords,
        full_history=args.full,
    )
