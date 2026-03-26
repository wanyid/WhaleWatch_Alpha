"""build_poly_market_data.py — Build Polymarket anomaly-session → SPY reaction dataset.

Pipeline:
  1. Load hourly YES-price histories for all tracked markets
  2. Detect price-spike anomaly events per market (rolling window delta >= threshold)
  3. Group nearby anomaly events into sessions (60-min window)
     - Same-direction events reinforce; opposite-direction events partially cancel
  4. Compute session feature vector (strength, diversity, market regime)
  5. Join each session to subsequent SPY excess returns → binary BUY/SHORT label

Output: D:/WhaleWatch_Data/poly_market_data.parquet

Usage:
    python scripts/build_poly_market_data.py
    python scripts/build_poly_market_data.py --session-window 60
    python scripts/build_poly_market_data.py --price-delta 0.05
"""

import argparse
import logging
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("build_poly_data")

POLY_DIR   = Path("D:/WhaleWatch_Data/polymarket")
PRICES_DIR = POLY_DIR / "prices"
EQUITY_DIR = Path("D:/WhaleWatch_Data/equity")
OUT_PATH   = Path("D:/WhaleWatch_Data/poly_market_data.parquet")

# Anomaly detection
ROLLING_WINDOW        = 10     # bars in rolling window for delta calc
PRICE_DELTA_THRESHOLD = 0.05   # 5pp YES-price spike = anomaly

# Session aggregation
DEFAULT_SESSION_WINDOW = 60    # minutes: events within this window → one session

# Holding periods to label
HOLDING_PERIODS: dict[str, int] = {
    "5m":  5,
    "30m": 30,
    "1h":  60,
    "2h":  120,
    "4h":  240,
    "1d":  1440,
}

# Dead-zone thresholds (same as Truth Social model)
DEAD_ZONE_PCT: dict[str, float] = {
    "5m":  0.0010,
    "30m": 0.0020,
    "1h":  0.0030,
    "2h":  0.0040,
    "4h":  0.0060,
    "1d":  0.0080,
}

ROLLING_BASELINE_WINDOW = 20   # sessions for excess-return baseline

# Volume spike detection
VOLUME_ROLLING_WINDOW   = 20   # hours for rolling volume baseline
VOLUME_SPIKE_THRESHOLD  = 50.0 # % above rolling avg = volume spike


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def load_market_meta() -> pd.DataFrame:
    path = POLY_DIR / "market_meta.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Market metadata not found: {path}\n"
            "Run pull_polymarket_history.py first."
        )
    return pd.read_parquet(path)


def load_price_file(condition_id: str) -> pd.DataFrame:
    safe_id  = condition_id.replace("0x", "")[:24]
    out_path = PRICES_DIR / f"{safe_id}.parquet"
    if not out_path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_parquet(out_path)
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        return df.sort_index()
    except Exception:
        return pd.DataFrame()


def load_volume_file(condition_id: str) -> pd.Series:
    """Load hourly USDC volume for a market. Returns empty Series if unavailable."""
    safe_id  = condition_id.replace("0x", "")[:24]
    out_path = Path("D:/WhaleWatch_Data/polymarket/volume") / f"{safe_id}.parquet"
    if not out_path.exists():
        return pd.Series(dtype=float)
    try:
        df = pd.read_parquet(out_path)
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        return df["volume_usd"].sort_index()
    except Exception:
        return pd.Series(dtype=float)


def load_spy_prices() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load SPY 5m and 1h from Polygon (preferred) or yfinance fallback."""
    def _load(paths: list[Path]) -> pd.DataFrame:
        for p in paths:
            if p.exists():
                df = pd.read_parquet(p)
                if df.index.tz is None:
                    df.index = df.index.tz_localize("UTC")
                return df.sort_index()
        return pd.DataFrame()

    spy_5m = _load([EQUITY_DIR / "SPY_5m_poly.parquet", EQUITY_DIR / "SPY_5m_recent.parquet"])
    spy_1h = _load([EQUITY_DIR / "SPY_1h_poly.parquet", EQUITY_DIR / "SPY_1h.parquet"])
    return spy_5m, spy_1h


def load_vixy_prices() -> pd.DataFrame:
    """Load VIXY 5m from Polygon (intraday VIX proxy)."""
    path = EQUITY_DIR / "VIXY_5m_poly.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    return df.sort_index()


def load_vix_data() -> tuple[pd.Series, pd.Series]:
    """Load VIX daily close and rolling percentile via yfinance."""
    import yfinance as yf
    raw = yf.download("^VIX", start="2023-01-01", progress=False, auto_adjust=True)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.droplevel(1)
    vix = raw["Close"].squeeze()
    if vix.index.tz is None:
        vix.index = vix.index.tz_localize("UTC")
    vix_pct = vix.rolling(252, min_periods=60).rank(pct=True)
    return vix, vix_pct


# ---------------------------------------------------------------------------
# Step 1 — Anomaly detection
# ---------------------------------------------------------------------------

def detect_anomalies(
    price_series: pd.Series,
    condition_id: str,
    meta_row: pd.Series,
    volume_series: pd.Series | None = None,
) -> pd.DataFrame:
    """Return one row per anomaly event in a market's price history.

    An anomaly is a |YES-price delta| >= PRICE_DELTA_THRESHOLD over the
    last ROLLING_WINDOW bars (i.e., a sustained directional move).

    If volume_series is provided, also computes volume_spike_pct at each
    anomaly timestamp (% above rolling average). This identifies whale-size
    trades: price move + volume spike = high-confidence informed bet.
    """
    if len(price_series) < ROLLING_WINDOW + 1:
        return pd.DataFrame()

    # Price at bar (t - ROLLING_WINDOW)
    lagged = price_series.shift(ROLLING_WINDOW)
    delta  = price_series - lagged
    events = delta[delta.abs() >= PRICE_DELTA_THRESHOLD].dropna()

    if events.empty:
        return pd.DataFrame()

    # Pre-compute volume spike % at each hour (NaN if no volume data)
    vol_spike_map: dict = {}
    if volume_series is not None and len(volume_series) >= 3:
        # Rolling baseline excludes current bar (shift(1) then rolling mean)
        vol_baseline = volume_series.shift(1).rolling(VOLUME_ROLLING_WINDOW, min_periods=3).mean()
        for ts in events.index:
            # Find the hourly volume bar matching this timestamp (floor to hour)
            bar = ts.floor("1h")
            if bar in volume_series.index and bar in vol_baseline.index:
                baseline = vol_baseline.loc[bar]
                vol      = volume_series.loc[bar]
                if pd.notna(baseline) and baseline > 0:
                    vol_spike_map[ts] = ((vol - baseline) / baseline) * 100.0
                else:
                    vol_spike_map[ts] = 0.0
            else:
                vol_spike_map[ts] = None   # volume data absent for this bar

    rows = []
    for ts, d in events.items():
        rows.append({
            "condition_id":    condition_id,
            "question":        meta_row["question"],
            "topic_bucket":    meta_row["topic_bucket"],
            "event_time":      ts,
            "price_before":    float(lagged.loc[ts]),
            "price_after":     float(price_series.loc[ts]),
            "price_delta":     float(d),
            "volume_spike_pct": vol_spike_map.get(ts),   # None if no volume data
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Step 2 — Session aggregation
# ---------------------------------------------------------------------------

def build_sessions(
    anomalies: pd.DataFrame,
    session_window_min: int,
) -> pd.DataFrame:
    """Group anomaly events within session_window_min into signal sessions.

    Reinforcement logic:
    - Dominant direction = sign of cumulative delta across all events
    - Same-direction events (corroborating) strengthen the session
    - Opposite-direction events (conflicting) weaken it
    - n_markets: diversity signal — multiple markets moving is stronger

    Returns one row per session.
    """
    if anomalies.empty:
        return pd.DataFrame()

    anomalies = anomalies.sort_values("event_time").reset_index(drop=True)
    sessions  = []
    i = 0

    while i < len(anomalies):
        anchor_time = anomalies.loc[i, "event_time"]
        window_end  = anchor_time + pd.Timedelta(minutes=session_window_min)

        mask   = (anomalies["event_time"] >= anchor_time) & (anomalies["event_time"] <= window_end)
        events = anomalies[mask].copy()

        # Net direction: positive = YES markets rising (often risk-on or specific event likely)
        cum_delta    = float(events["price_delta"].sum())
        dom_sign     = 1 if cum_delta >= 0 else -1
        same_dir     = events[events["price_delta"] * dom_sign > 0]
        opp_dir      = events[events["price_delta"] * dom_sign < 0]

        # Topic composition
        topic_counts   = events["topic_bucket"].value_counts()
        dominant_topic = topic_counts.index[0] if not topic_counts.empty else "other"

        duration_min = (
            (events["event_time"].max() - anchor_time).total_seconds() / 60
            if len(events) > 1
            else 0.0
        )

        # Volume spike aggregation (NaN where volume data is absent)
        vol_col = events["volume_spike_pct"].dropna() if "volume_spike_pct" in events.columns else pd.Series(dtype=float)
        max_vol_spike  = float(vol_col.max())     if not vol_col.empty else None
        avg_vol_spike  = float(vol_col.mean())    if not vol_col.empty else None
        # n_volume_spikes: events where both price AND volume anomaly fired
        n_vol_spikes   = int((vol_col >= VOLUME_SPIKE_THRESHOLD).sum()) if not vol_col.empty else 0

        sessions.append({
            "session_time":           anchor_time,
            "dominant_topic":         dominant_topic,
            # Session strength features
            "max_price_delta":        float(events["price_delta"].abs().max()),
            "cumulative_delta":       cum_delta,
            "net_delta_abs":          abs(cum_delta),
            "dominant_direction":     dom_sign,          # +1 YES rising / -1 YES falling
            "n_events":               len(events),
            "n_markets":              int(events["condition_id"].nunique()),
            "n_corroborating":        len(same_dir),
            "n_opposing":             len(opp_dir),
            "corroboration_ratio":    len(same_dir) / max(len(events), 1),
            "session_duration_min":   duration_min,
            # Volume spike features — whale bet size signal
            "max_volume_spike_pct":   max_vol_spike,   # None if volume data unavailable
            "avg_volume_spike_pct":   avg_vol_spike,
            "n_volume_spikes":        n_vol_spikes,    # events with price + volume anomaly
            # Topic flags (one-hot, non-exclusive)
            "has_tariff":             int((events["topic_bucket"] == "tariff").any()),
            "has_geopolitical":       int((events["topic_bucket"] == "geopolitical").any()),
            "has_fed":                int((events["topic_bucket"] == "fed").any()),
            "has_energy":             int((events["topic_bucket"] == "energy").any()),
            "has_executive":          int((events["topic_bucket"] == "executive").any()),
        })

        # Advance past the current window
        after = anomalies[anomalies["event_time"] > window_end]
        i     = after.index[0] if not after.empty else len(anomalies)

    return pd.DataFrame(sessions)


# ---------------------------------------------------------------------------
# Step 3 — Temporal and VIX features
# ---------------------------------------------------------------------------

def add_temporal_features(sessions: pd.DataFrame) -> pd.DataFrame:
    local = sessions["session_time"].dt.tz_convert("America/New_York")
    sessions["hour_of_day"]     = local.dt.hour
    sessions["day_of_week"]     = local.dt.dayofweek
    sessions["is_market_hours"] = (
        (local.dt.hour >= 9) &
        (local.dt.hour < 16) &
        (local.dt.dayofweek < 5)
    ).astype(int)
    return sessions


def add_vix_features(
    sessions: pd.DataFrame,
    vix_close: pd.Series,
    vix_pct: pd.Series,
    vixy_5m: pd.DataFrame,
) -> pd.DataFrame:
    date_strs = sessions["session_time"].dt.date.astype(str)
    vix_map   = {str(d): v for d, v in zip(vix_close.index.date, vix_close.values)}
    pct_map   = {str(d): v for d, v in zip(vix_pct.index.date, vix_pct.values)}
    sessions["vix_level"]      = date_strs.map(vix_map)
    sessions["vix_percentile"] = date_strs.map(pct_map)

    # VIXY intraday price: nearest 5m bar at or before session start
    if not vixy_5m.empty:
        vixy_levels = []
        for ts in sessions["session_time"]:
            past = vixy_5m.index[vixy_5m.index <= ts]
            vixy_levels.append(float(vixy_5m.loc[past[-1], "close"]) if len(past) > 0 else None)
        sessions["vixy_level"] = vixy_levels
    else:
        sessions["vixy_level"] = None

    return sessions


# ---------------------------------------------------------------------------
# Step 4 — SPY forward returns
# ---------------------------------------------------------------------------

def _next_bar(ts: pd.Timestamp, spy: pd.DataFrame) -> pd.Timestamp | None:
    future = spy.index[spy.index >= ts]
    return future[0] if len(future) > 0 else None


def compute_spy_returns(
    sessions: pd.DataFrame,
    spy_5m: pd.DataFrame,
    spy_1h: pd.DataFrame,
    session_window_min: int = DEFAULT_SESSION_WINDOW,
) -> pd.DataFrame:
    """Add raw SPY forward return columns for each holding period.

    Entry is anchored at T + session_window_min (i.e., after the session
    window has closed), not at T+0 (first anomaly).

    Rationale: session features (n_events, corroboration_ratio, etc.) are
    computed across the full session window. At T+0 only the first anomaly is
    known; the complete feature vector isn't available until T+60min. Using
    T+60min as the entry ensures the model is predicting using only information
    that would be available at inference time.
    """
    for period_name, minutes in HOLDING_PERIODS.items():
        spy     = spy_5m if minutes <= 60 else spy_1h
        ret_col = f"ret_{period_name}"
        returns = []

        for ts in sessions["session_time"]:
            # Wait for the session window to close before entering
            entry_ts  = ts + pd.Timedelta(minutes=session_window_min)
            entry_bar = _next_bar(entry_ts, spy)
            if entry_bar is None or entry_bar < spy.index[0]:
                returns.append(None)
                continue

            entry_price = float(spy.loc[entry_bar, "close"])
            exit_ts     = entry_bar + pd.Timedelta(minutes=minutes)
            exit_bars   = spy.index[spy.index >= exit_ts]
            if len(exit_bars) == 0:
                returns.append(None)
                continue

            exit_price = float(spy.loc[exit_bars[0], "close"])
            returns.append((exit_price - entry_price) / entry_price)

        sessions[ret_col] = returns

    return sessions


# ---------------------------------------------------------------------------
# Step 5 — Excess return labels
# ---------------------------------------------------------------------------

def add_excess_labels(sessions: pd.DataFrame) -> pd.DataFrame:
    """Compute rolling excess returns and raw-return direction labels.

    Two things happen here:
      1. excess_ret_* columns are computed and stored for exploration and the
         fade model (not used to define training labels).
      2. label_* = sign of the RAW return (1 = SPY up, 0 = SPY down).

    Using raw returns for labels avoids the rolling-baseline absorption
    problem: if signals cluster in time the rolling mean partially cancels the
    very signal it should be measuring.  The dead zone (filter on |raw_ret|)
    is applied at training time so thresholds can be changed without a rebuild.

    Rolling baseline notes:
      - shift(1) ensures no lookahead: row i's baseline uses sessions 0..i-1
      - excess_ret_* are informational; they are NOT in the model feature set
    """
    for period_name in HOLDING_PERIODS:
        ret_col   = f"ret_{period_name}"
        ex_col    = f"excess_ret_{period_name}"
        lbl_col   = f"label_{period_name}"

        if ret_col not in sessions.columns:
            continue

        # Excess return stored for reference / fade model (shift(1) = no lookahead)
        baseline         = sessions[ret_col].shift(1).rolling(
            ROLLING_BASELINE_WINDOW, min_periods=5
        ).mean()
        sessions[ex_col] = sessions[ret_col] - baseline

        # Label = sign of raw return; NaN only where ret is NaN
        sessions[lbl_col] = np.where(
            sessions[ret_col].notna(),
            (sessions[ret_col] > 0).astype(int),
            np.nan,
        )

    return sessions


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(session_window_min: int, price_delta: float) -> None:
    global PRICE_DELTA_THRESHOLD
    PRICE_DELTA_THRESHOLD = price_delta

    logger.info("Loading market metadata...")
    meta = load_market_meta()
    logger.info("  %d markets", len(meta))

    logger.info("Loading SPY prices...")
    spy_5m, spy_1h = load_spy_prices()
    if spy_5m.empty and spy_1h.empty:
        logger.error("No SPY data found. Run pull_polygon_history.py first.")
        return
    logger.info("  SPY 5m: %d bars  1h: %d bars", len(spy_5m), len(spy_1h))

    logger.info("Loading VIX/VIXY data...")
    vix_close, vix_pct = load_vix_data()
    vixy_5m = load_vixy_prices()
    logger.info("  VIXY 5m: %d bars%s", len(vixy_5m), "" if not vixy_5m.empty else " (not found — vixy_level will be null)")

    # Detect anomalies for every market
    all_anomalies: list[pd.DataFrame] = []
    n_loaded = n_with_volume = 0
    for _, row in meta.iterrows():
        prices = load_price_file(row["condition_id"])
        if prices.empty:
            continue
        volume = load_volume_file(row["condition_id"])
        if not volume.empty:
            n_with_volume += 1
        anomalies = detect_anomalies(
            prices["price"], row["condition_id"], row,
            volume_series=volume if not volume.empty else None,
        )
        if not anomalies.empty:
            all_anomalies.append(anomalies)
            n_loaded += 1

    logger.info("  %d/%d markets have volume data", n_with_volume, n_loaded)

    if not all_anomalies:
        logger.error("No anomaly events detected. Run pull_polymarket_history.py first.")
        return

    anomalies_df = pd.concat(all_anomalies, ignore_index=True).sort_values("event_time")
    logger.info(
        "Detected %d anomaly events across %d markets (delta threshold=%.2f)",
        len(anomalies_df), n_loaded, PRICE_DELTA_THRESHOLD,
    )

    # Topic breakdown
    for bucket, cnt in anomalies_df["topic_bucket"].value_counts().items():
        logger.info("  %-14s  %d events", bucket, cnt)

    # Build sessions
    logger.info("Building sessions (window=%d min)...", session_window_min)
    sessions = build_sessions(anomalies_df, session_window_min)
    logger.info("  %d sessions", len(sessions))

    # Feature engineering
    sessions = add_temporal_features(sessions)
    sessions = add_vix_features(sessions, vix_close, vix_pct, vixy_5m)

    # SPY returns (entry anchored at T + session_window_min)
    logger.info("Computing SPY forward returns (entry at T+%dmin)...", session_window_min)
    sessions = compute_spy_returns(sessions, spy_5m, spy_1h, session_window_min)

    # Excess return labels
    sessions = add_excess_labels(sessions)

    # Drop rows with no SPY price data at all
    ret_cols = [f"ret_{p}" for p in HOLDING_PERIODS]
    sessions = sessions.dropna(subset=ret_cols, how="all")

    sessions.to_parquet(OUT_PATH)
    logger.info("Saved %d sessions → %s", len(sessions), OUT_PATH)

    # Summary
    logger.info("\n--- Label summary ---")
    for period_name in HOLDING_PERIODS:
        lbl_col = f"label_{period_name}"
        if lbl_col not in sessions.columns:
            continue
        valid = sessions[lbl_col].dropna()
        if len(valid) > 0:
            logger.info(
                "  %-4s  n=%4d  pos_rate=%.1f%%",
                period_name, len(valid), valid.mean() * 100,
            )
        else:
            logger.info("  %-4s  n=0  (no valid labels)", period_name)

    logger.info("\nNext: python scripts/train_poly_model.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build Polymarket anomaly-session → SPY dataset")
    parser.add_argument(
        "--session-window", type=int, default=DEFAULT_SESSION_WINDOW,
        help="Minutes to group co-occurring anomalies into one session (default: 60)",
    )
    parser.add_argument(
        "--price-delta", type=float, default=PRICE_DELTA_THRESHOLD,
        help="Minimum YES-price move to flag as anomaly (default: 0.05 = 5pp)",
    )
    args = parser.parse_args()
    run(session_window_min=args.session_window, price_delta=args.price_delta)
