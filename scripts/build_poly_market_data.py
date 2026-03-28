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
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load data_start_date from settings
_settings_path = PROJECT_ROOT / "config" / "settings.yaml"
with open(_settings_path) as _f:
    _cfg = yaml.safe_load(_f)
DATA_START_DATE = pd.Timestamp(_cfg["data"]["data_start_date"], tz="UTC")

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

# Fade model: initial-move window used as entry lag
FADE_ENTRY_LAG  = "30m"                      # observe first 30m, then enter opposite
FADE_PERIODS    = ["2h", "4h", "1d"]         # continuation periods to label

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

    # Ensure true 1h bars — some sources deliver mixed 30m/1h intervals.
    # Resample from 5m if available (more precise); otherwise OHLCV-aggregate the 1h file.
    if not spy_5m.empty:
        spy_1h = spy_5m.resample("1h").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
        ).dropna(subset=["close"])
    elif not spy_1h.empty:
        spy_1h = spy_1h.resample("1h").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
        ).dropna(subset=["close"])

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
    """Load VIX daily close and rolling percentile via yfinance.

    Percentile uses shift(1) before ranking so the current day's VIX
    is not included in its own percentile calculation (no lookahead).
    """
    import yfinance as yf
    raw = yf.download("^VIX", start="2023-01-01", progress=False, auto_adjust=True)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.droplevel(1)
    vix = raw["Close"].squeeze()
    if vix.index.tz is None:
        vix.index = vix.index.tz_localize("UTC")
    # shift(1) prevents lookahead: today's rank uses only prior closes
    vix_pct = vix.shift(1).rolling(252, min_periods=60).rank(pct=True)
    return vix, vix_pct


# ---------------------------------------------------------------------------
# Step 0.5 — Price-derived activity proxy
# ---------------------------------------------------------------------------

def compute_price_activity(price_series: pd.Series) -> pd.Series:
    """Derive an hourly activity series from price data.

    Computes |price_change| per bar, which is a proxy for trading activity:
    hours with large price moves indicate more trading.  This replaces the
    CLOB trades-based USDC volume (no longer publicly available) while
    preserving the same spike-detection logic downstream.

    Returns a Series with the same index as the input, values = abs(Δprice).
    """
    activity = price_series.diff().abs()
    activity.iloc[0] = 0.0  # first bar has no diff
    return activity


# ---------------------------------------------------------------------------
# Step 1 — Anomaly detection
# ---------------------------------------------------------------------------

def detect_anomalies(
    price_series: pd.Series,
    condition_id: str,
    meta_row: pd.Series,
    volume_series: pd.Series | None = None,
    activity_series: pd.Series | None = None,
) -> pd.DataFrame:
    """Return one row per anomaly event in a market's price history.

    An anomaly is a |YES-price delta| >= PRICE_DELTA_THRESHOLD over the
    last ROLLING_WINDOW bars (i.e., a sustained directional move).

    Activity/volume spike:
      1. If volume_series is provided (real USDC volume), use it.
      2. Otherwise fall back to activity_series (price-derived proxy).
      3. If neither is available, volume_spike_pct = 0.

    The spike metric is always: % above rolling baseline — whether the
    underlying data is real volume or price activity.
    """
    if len(price_series) < ROLLING_WINDOW + 1:
        return pd.DataFrame()

    # Price at bar (t - ROLLING_WINDOW)
    lagged = price_series.shift(ROLLING_WINDOW)
    delta  = price_series - lagged
    events = delta[delta.abs() >= PRICE_DELTA_THRESHOLD].dropna()

    if events.empty:
        return pd.DataFrame()

    # Select the best available activity signal
    # Prefer real volume; fall back to price-derived activity proxy
    act_series = None
    if volume_series is not None and len(volume_series) >= 3:
        act_series = volume_series
    elif activity_series is not None and len(activity_series) >= 3:
        act_series = activity_series

    # Pre-compute spike % at each anomaly timestamp
    spike_map: dict = {}
    if act_series is not None:
        # Rolling baseline excludes current bar (shift(1) then rolling mean)
        baseline = act_series.shift(1).rolling(VOLUME_ROLLING_WINDOW, min_periods=3).mean()
        for ts in events.index:
            bar = ts.floor("1h")
            if bar in act_series.index and bar in baseline.index:
                bl = baseline.loc[bar]
                val = act_series.loc[bar]
                if pd.notna(bl) and bl > 0:
                    spike_map[ts] = ((val - bl) / bl) * 100.0
                else:
                    spike_map[ts] = 0.0

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
            "volume_spike_pct": spike_map.get(ts, 0.0),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Step 2 — Session aggregation
# ---------------------------------------------------------------------------

def build_sessions(
    anomalies: pd.DataFrame,
    session_window_min: int,
    meta: pd.DataFrame | None = None,
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

    # Pre-build market quality lookup from metadata
    vol_lookup: dict[str, float] = {}
    start_lookup: dict[str, pd.Timestamp] = {}   # store start_date, compute age per-session
    if meta is not None:
        for _, row in meta.iterrows():
            cid = row["condition_id"]
            vol_lookup[cid] = float(row.get("volume", 0) or 0)
            start_str = str(row.get("start_date", ""))[:10]
            if start_str and start_str != "nan":
                try:
                    start_lookup[cid] = pd.Timestamp(start_str, tz="UTC")
                except Exception:
                    pass

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

        # Volume spike aggregation — NaN → 0 for explicit imputation
        vol_col = events["volume_spike_pct"].fillna(0.0) if "volume_spike_pct" in events.columns else pd.Series([0.0])
        max_vol_spike  = float(vol_col.max())
        avg_vol_spike  = float(vol_col.mean())
        n_vol_spikes   = int((vol_col >= VOLUME_SPIKE_THRESHOLD).sum())

        # Market quality features — volume and age of markets in this session
        # Age is computed relative to session time (not build time) to avoid lookahead
        session_cids = events["condition_id"].unique()
        mkt_vols = [vol_lookup.get(c, 0) for c in session_cids]
        mkt_ages = [
            max((anchor_time - start_lookup[c]).days, 0)
            for c in session_cids if c in start_lookup
        ]

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
            # Volume spike features — imputed to 0 when volume data absent
            "max_volume_spike_pct":   max_vol_spike,
            "avg_volume_spike_pct":   avg_vol_spike,
            "n_volume_spikes":        n_vol_spikes,
            # Topic flags (one-hot, non-exclusive)
            "has_tariff":             int((events["topic_bucket"] == "tariff").any()),
            "has_geopolitical":       int((events["topic_bucket"] == "geopolitical").any()),
            "has_fed":                int((events["topic_bucket"] == "fed").any()),
            "has_energy":             int((events["topic_bucket"] == "energy").any()),
            "has_executive":          int((events["topic_bucket"] == "executive").any()),
            # Market quality features
            "avg_market_volume":      float(np.mean(mkt_vols)) if mkt_vols else 0.0,
            "max_market_volume":      float(np.max(mkt_vols)) if mkt_vols else 0.0,
            "avg_market_age_days":    float(np.mean(mkt_ages)) if mkt_ages else 0.0,
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


def add_spy_context_features(
    sessions: pd.DataFrame,
    spy_5m: pd.DataFrame,
) -> pd.DataFrame:
    """Add SPY price context at session time — what the equity market is doing.

    Features:
      spy_ret_1h:             SPY return over prior 1h (short-term momentum)
      spy_ret_4h:             SPY return over prior 4h (trend)
      spy_range_pct:          intraday (high - low) / open (extension)
      spy_dist_from_open_pct: (last close - day open) / day open (intraday bias)

    All use strictly past data (no lookahead): the bar AT session_time is
    excluded (< ts, not <= ts) to avoid including concurrent market activity.

    Uses merge_asof for O(n log m) instead of per-row O(n × m).
    """
    ctx_cols = ["spy_ret_1h", "spy_ret_4h", "spy_range_pct", "spy_dist_from_open_pct"]
    if spy_5m.empty:
        for col in ctx_cols:
            sessions[col] = np.nan
        return sessions

    spy = spy_5m.copy()
    spy.index.name = "bar_time"

    # Pre-compute 1h and 4h lagged closes using shift (strictly past)
    # shift(12) = 12 bars × 5min = 1h ago; shift(48) = 4h ago
    spy["close_1h_ago"] = spy["close"].shift(12)
    spy["close_4h_ago"] = spy["close"].shift(48)

    # NY calendar date for intraday features
    spy_ny = spy.index.tz_convert("America/New_York")
    spy["ny_date"] = spy_ny.date

    # Cumulative intraday high/low/open per NY day (strictly expanding)
    spy["day_open"]     = spy.groupby("ny_date")["open"].transform("first")
    spy["day_high_cum"] = spy.groupby("ny_date")["high"].cummax()
    spy["day_low_cum"]  = spy.groupby("ny_date")["low"].cummin()

    # Build a lookup frame with one row per 5m bar
    spy_ctx = spy[["close", "close_1h_ago", "close_4h_ago",
                   "day_open", "day_high_cum", "day_low_cum"]].copy()
    spy_ctx = spy_ctx.reset_index()

    # Prepare session keys for merge_asof (strict past: use the bar BEFORE session_time)
    sess_keys = sessions[["session_time"]].copy()
    sess_keys["_idx"] = sess_keys.index
    sess_keys = sess_keys.sort_values("session_time")

    # merge_asof: find the most recent bar strictly before session_time
    merged = pd.merge_asof(
        sess_keys,
        spy_ctx,
        left_on="session_time",
        right_on="bar_time",
        direction="backward",
        tolerance=pd.Timedelta(hours=18),  # reject if no bar within 18h
    )

    # Compute features from the merged bar
    c      = merged["close"]
    c_1h   = merged["close_1h_ago"]
    c_4h   = merged["close_4h_ago"]
    d_open = merged["day_open"]
    d_high = merged["day_high_cum"]
    d_low  = merged["day_low_cum"]

    merged["spy_ret_1h"]             = (c - c_1h) / c_1h
    merged["spy_ret_4h"]             = (c - c_4h) / c_4h
    merged["spy_range_pct"]          = np.where(d_open > 0, (d_high - d_low) / d_open, np.nan)
    merged["spy_dist_from_open_pct"] = np.where(d_open > 0, (c - d_open) / d_open, np.nan)

    # Restore original order and assign
    merged = merged.sort_values("_idx")
    for col in ctx_cols:
        sessions[col] = merged[col].values

    return sessions


def add_momentum_features(sessions: pd.DataFrame) -> pd.DataFrame:
    """Add cross-session momentum and clustering features.

    Uses only strictly past sessions (shift-safe via chronological sort).

    Features:
      sessions_last_24h:        count of sessions in the 24h before this session
      cumulative_delta_24h:     net cumulative_delta across those prior sessions
      hours_since_last_session: hours since the immediately preceding session
    """
    if sessions.empty:
        return sessions

    sessions = sessions.sort_values("session_time").reset_index(drop=True)
    times   = sessions["session_time"].values    # numpy datetime64
    deltas  = sessions["cumulative_delta"].values

    s24h  = []
    cd24h = []
    hours_since = []

    for i in range(len(sessions)):
        ts_i = times[i]
        ts_24h_ago = ts_i - np.timedelta64(24, "h")

        # Look only at strictly prior sessions (j < i)
        prior_mask = (times[:i] > ts_24h_ago) & (times[:i] < ts_i)
        s24h.append(int(prior_mask.sum()))
        cd24h.append(float(deltas[:i][prior_mask].sum()) if prior_mask.any() else 0.0)

        if i > 0:
            hrs = (ts_i - times[i - 1]) / np.timedelta64(1, "h")
            hours_since.append(float(hrs))
        else:
            hours_since.append(np.nan)

    sessions["sessions_last_24h"]        = s24h
    sessions["cumulative_delta_24h"]     = cd24h
    sessions["hours_since_last_session"] = hours_since
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

def _next_bar(ts: pd.Timestamp, spy: pd.DataFrame,
              max_gap_hours: float = 18) -> pd.Timestamp | None:
    """Find the first SPY bar >= ts, rejecting if the gap exceeds max_gap_hours.

    A large gap means the session predates the available SPY data or falls on
    a long holiday — either way the matched bar does not represent a real fill.
    """
    future = spy.index[spy.index >= ts]
    if len(future) == 0:
        return None
    bar = future[0]
    if (bar - ts).total_seconds() / 3600 > max_gap_hours:
        return None
    return bar


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
            if entry_bar is None:
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
    """Compute rolling excess returns and signal-relative direction labels.

    Two things happen here:
      1. excess_ret_* columns are computed and stored for exploration and the
         fade model (not used to define training labels).
      2. label_* = signal-relative label: 1 if SPY moved in the direction the
         whale signal predicted, 0 otherwise.

         Specifically: label = int(dominant_direction * raw_ret > 0).
         - dominant_direction = +1 when Polymarket YES prices are rising
           (bullish signal → expect SPY up)
         - dominant_direction = -1 when YES prices are falling
           (bearish signal → expect SPY down)

         This directly answers "should I follow this signal?" rather than
         the weaker question "will SPY go up?"

    Rolling baseline notes:
      - shift(1) ensures no lookahead: row i's baseline uses sessions 0..i-1
      - excess_ret_* are informational; they are NOT in the model feature set
    """
    dom_dir = sessions["dominant_direction"]

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

        # Signal-relative label: did SPY move in the whale-signal direction?
        # dominant_direction * ret > 0 means the whale signal was correct.
        sessions[lbl_col] = np.where(
            sessions[ret_col].notna(),
            (dom_dir * sessions[ret_col] > 0).astype(int),
            np.nan,
        )

    return sessions


# ---------------------------------------------------------------------------
# Step 6 — Fade labels (continuation return after initial move)
# ---------------------------------------------------------------------------

def add_fade_labels(sessions: pd.DataFrame) -> pd.DataFrame:
    """Compute continuation returns and fade direction labels.

    After the standard session entry (T+60min), the model waits FADE_ENTRY_LAG
    (30m) to observe the initial market reaction, then enters in the OPPOSITE
    direction.

    Columns added:
      initial_ret          : SPY return during the entry-lag window (= ret_30m)
      initial_direction    : +1 if initial move up, -1 if down
      abs_initial_ret      : absolute magnitude of the initial move

      ret_cont_{period}    : SPY return from T+90min onward
                             = ret_{period} - ret_30m
      fade_label_{period}  : 1 if continuation reverses the initial direction
                             (the fade worked), 0 if it continued, NaN if data
                             unavailable
    """
    initial_col = f"ret_{FADE_ENTRY_LAG}"
    if initial_col not in sessions.columns:
        logger.warning("add_fade_labels: %s column missing — skipping", initial_col)
        return sessions

    sessions["initial_ret"]       = sessions[initial_col]
    sessions["initial_direction"] = np.sign(sessions[initial_col])
    sessions["abs_initial_ret"]   = sessions[initial_col].abs()

    for period_name in FADE_PERIODS:
        total_col = f"ret_{period_name}"
        cont_col  = f"ret_cont_{period_name}"
        lbl_col   = f"fade_label_{period_name}"

        if total_col not in sessions.columns:
            continue

        # Continuation = total return (from T+60min) minus the initial move
        sessions[cont_col] = sessions[total_col] - sessions[initial_col]

        # Fade worked if continuation is opposite sign to initial move
        both_valid = sessions[cont_col].notna() & sessions[initial_col].notna()
        fade_worked = (sessions[cont_col] * sessions[initial_col]) < 0
        sessions[lbl_col] = np.where(both_valid, fade_worked.astype(int), np.nan)

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
    n_loaded = n_with_volume = n_with_activity = 0
    for _, row in meta.iterrows():
        prices = load_price_file(row["condition_id"])
        if prices.empty:
            continue

        # Prefer real USDC volume; fall back to price-derived activity proxy
        volume = load_volume_file(row["condition_id"])
        activity = None
        if not volume.empty:
            n_with_volume += 1
        else:
            # Compute price-activity proxy from the hourly price data
            activity = compute_price_activity(prices["price"])
            if len(activity) >= 3:
                n_with_activity += 1

        anomalies = detect_anomalies(
            prices["price"], row["condition_id"], row,
            volume_series=volume if not volume.empty else None,
            activity_series=activity,
        )
        if not anomalies.empty:
            all_anomalies.append(anomalies)
            n_loaded += 1

    logger.info(
        "  %d/%d markets: %d with real volume, %d with price-activity proxy",
        n_loaded, len(meta), n_with_volume, n_with_activity,
    )

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

    # Build sessions (pass meta for market quality features)
    logger.info("Building sessions (window=%d min)...", session_window_min)
    sessions = build_sessions(anomalies_df, session_window_min, meta=meta)
    logger.info("  %d sessions (all dates)", len(sessions))

    # Filter to data_start_date — pre-regime sessions lack SPY coverage and
    # mix in market dynamics from a different presidential term
    n_before = len(sessions)
    sessions = sessions[sessions["session_time"] >= DATA_START_DATE].reset_index(drop=True)
    logger.info("  %d sessions after data_start_date filter (%s) — dropped %d",
                len(sessions), DATA_START_DATE.date(), n_before - len(sessions))

    # Feature engineering
    sessions = add_temporal_features(sessions)
    sessions = add_vix_features(sessions, vix_close, vix_pct, vixy_5m)

    logger.info("Computing SPY context features at session time...")
    sessions = add_spy_context_features(sessions, spy_5m)

    logger.info("Computing cross-session momentum features...")
    sessions = add_momentum_features(sessions)

    # SPY returns (entry anchored at T + session_window_min)
    logger.info("Computing SPY forward returns (entry at T+%dmin)...", session_window_min)
    sessions = compute_spy_returns(sessions, spy_5m, spy_1h, session_window_min)

    # Signal-relative labels: P(whale signal direction was correct)
    sessions = add_excess_labels(sessions)

    # Fade labels (continuation return after initial move)
    sessions = add_fade_labels(sessions)

    # Drop rows with no SPY price data at all
    ret_cols = [f"ret_{p}" for p in HOLDING_PERIODS]
    sessions = sessions.dropna(subset=ret_cols, how="all")

    sessions.to_parquet(OUT_PATH)
    logger.info("Saved %d sessions → %s", len(sessions), OUT_PATH)

    # Summary — directional labels (signal-relative: P(whale signal correct))
    logger.info("\n--- Directional label summary (signal-relative) ---")
    for period_name in HOLDING_PERIODS:
        lbl_col = f"label_{period_name}"
        if lbl_col not in sessions.columns:
            continue
        valid = sessions[lbl_col].dropna()
        if len(valid) > 0:
            logger.info(
                "  %-4s  n=%4d  signal_correct=%.1f%%",
                period_name, len(valid), valid.mean() * 100,
            )
        else:
            logger.info("  %-4s  n=0  (no valid labels)", period_name)

    # Summary — fade labels
    logger.info("\n--- Fade label summary (entry lag: %s) ---", FADE_ENTRY_LAG)
    for period_name in FADE_PERIODS:
        lbl_col = f"fade_label_{period_name}"
        if lbl_col not in sessions.columns:
            continue
        valid = sessions[lbl_col].dropna()
        if len(valid) > 0:
            logger.info(
                "  %-4s  n=%4d  fade_rate=%.1f%%  (initial_ret mean=%.3f%%)",
                period_name, len(valid), valid.mean() * 100,
                sessions["initial_ret"].dropna().mean() * 100,
            )
        else:
            logger.info("  %-4s  n=0  (no valid fade labels)", period_name)

    logger.info("\nNext: python scripts/train_poly_model.py")
    logger.info("      python scripts/train_poly_fade_model.py")


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
