"""Feature engineering for Layer 2 predictor.

DEPRECATED — used only by the legacy StatPredictor (logistic regression).
New models use the feature sets defined in:
  - reasoner/layer2_predictor/post_features.py  (Truth Social)
  - reasoner/layer2_predictor/poly_features.py  (Polymarket)
Do not add features here.


Converts a SignalEvent into a numeric feature vector for model input.
All features are derived from data available at signal time — no lookahead.

Feature groups:
  - Polymarket signal strength (price delta, volume spike, outcome direction)
  - Truth Social signal (keyword count, engagement, repost flag)
  - Dual signal flag
  - Temporal context (hour-of-day, day-of-week, market session)
  - L1 output encoding (direction × ticker)
"""

from pathlib import Path

import numpy as np
import pandas as pd

from models.signal_event import SignalEvent

# ---------------------------------------------------------------------------
# VIX regime context — loaded once at import time from stored daily parquet
# ---------------------------------------------------------------------------
_VIX_DAILY: pd.Series | None = None
_VIX_ROLLING_252: pd.Series | None = None

def _load_vix() -> None:
    global _VIX_DAILY, _VIX_ROLLING_252
    vix_path = Path("D:/WhaleWatch_Data/equity/VIX_1d.parquet")
    if not vix_path.exists():
        return
    try:
        df = pd.read_parquet(vix_path)
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, utc=True)
        elif df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        close_col = "Close" if "Close" in df.columns else df.columns[3]
        _VIX_DAILY = df[close_col].sort_index()
        _VIX_ROLLING_252 = _VIX_DAILY.rank(pct=True).rolling(252, min_periods=10).mean()
    except Exception:
        pass

_load_vix()


def _vix_at(ts: pd.Timestamp) -> tuple[float, float]:
    """Return (vix_level, vix_percentile) at the given timestamp.

    vix_level: raw VIX close value (e.g. 18.5)
    vix_percentile: rolling 252-day rank (0=historically low, 1=historically high)
    """
    if _VIX_DAILY is None:
        return 0.0, 0.0
    mask = _VIX_DAILY.index <= ts
    if not mask.any():
        return 0.0, 0.0
    level = float(_VIX_DAILY[mask].iloc[-1])
    pct = float(_VIX_ROLLING_252[mask].iloc[-1]) if _VIX_ROLLING_252 is not None else 0.0
    return level, pct if not np.isnan(pct) else 0.0


# ---------------------------------------------------------------------------
# Feature names — must stay in sync with build_feature_vector()
# ---------------------------------------------------------------------------
FEATURE_NAMES = [
    # Polymarket
    "poly_price_delta",         # signed, 0 if no poly signal
    "poly_price_delta_abs",     # absolute magnitude
    "poly_volume_spike_pct",    # % above rolling avg, 0 if no poly signal
    "poly_yes_direction",       # +1 if YES moved up, -1 if down, 0 if no signal
    "has_poly_signal",          # 1 if polymarket signal present

    # Truth Social
    "ts_keyword_count",         # number of market keywords detected
    "ts_engagement",            # log1p(favourites + reblogs)
    "has_ts_signal",            # 1 if truth social signal present

    # Combined
    "dual_signal",              # 1 if both signals fired

    # Temporal (at signal creation time)
    "hour_of_day",              # 0–23 UTC
    "day_of_week",              # 0=Mon … 6=Sun
    "is_us_market_hours",       # 1 if 13:30–20:00 UTC (9:30–16:00 ET)
    "is_premarket",             # 1 if 08:00–13:30 UTC

    # Market regime at signal time (from CLAUDE.md feature spec)
    "vix_level",                # VIX close price at signal time (0 if unavailable)
    "vix_percentile",           # VIX rank in trailing 252-day window (0–1)

    # L1 signal encoding (one-hot style but kept as individual features)
    "direction_buy",            # 1 if BUY
    "direction_short",          # 1 if SHORT
    "ticker_spy",               # 1 if SPY
    "ticker_qqq",               # 1 if QQQ
    "ticker_vix",               # 1 if VIX
]

N_FEATURES = len(FEATURE_NAMES)


def build_feature_vector(event: SignalEvent) -> np.ndarray:
    """Convert a SignalEvent to a 1-D float32 feature array.

    All missing / None fields are safely replaced with 0.
    """
    dt = event.created_at

    # ------ Polymarket ------
    has_poly = int(event.poly_market_id is not None)
    poly_delta = float(event.poly_price_delta or 0.0)
    poly_spike = float(event.poly_volume_spike_pct or 0.0)
    poly_yes_dir = 0
    if has_poly:
        poly_yes_dir = 1 if (event.poly_outcome_token or "").upper() == "YES" and poly_delta > 0 else \
                       -1 if poly_delta < 0 else 0

    # ------ Truth Social ------
    has_ts = int(event.ts_post_id is not None)
    ts_kw_count = len(event.ts_post_keywords or [])
    ts_engagement = float(np.log1p(0))     # default
    # Note: raw engagement (likes/reblogs) not stored on SignalEvent;
    # keyword count is the proxy signal strength for now.

    # ------ Temporal ------
    hour = dt.hour
    dow = dt.weekday()
    # US market hours in UTC: 13:30–20:00
    is_market = int(dt.hour * 60 + dt.minute >= 810 and dt.hour * 60 + dt.minute < 1200)
    is_pre = int(dt.hour * 60 + dt.minute >= 480 and dt.hour * 60 + dt.minute < 810)

    # ------ VIX regime ------
    ts = pd.Timestamp(dt).tz_localize("UTC") if dt.tzinfo is None else pd.Timestamp(dt)
    vix_level, vix_pct = _vix_at(ts)

    # ------ L1 encoding ------
    direction = (event.signal_direction or "HOLD").upper()
    ticker = (event.signal_ticker or "SPY").upper()

    vec = np.array([
        poly_delta,
        abs(poly_delta),
        poly_spike,
        poly_yes_dir,
        has_poly,
        ts_kw_count,
        ts_engagement,
        has_ts,
        int(event.dual_signal),
        hour,
        dow,
        is_market,
        is_pre,
        vix_level,
        vix_pct,
        int(direction == "BUY"),
        int(direction == "SHORT"),
        int(ticker == "SPY"),
        int(ticker == "QQQ"),
        int(ticker == "VIX"),
    ], dtype=np.float32)

    assert len(vec) == N_FEATURES, f"Feature count mismatch: {len(vec)} vs {N_FEATURES}"
    return vec


def events_to_dataframe(events: list[SignalEvent]) -> pd.DataFrame:
    """Convert a list of resolved SignalEvents to a feature DataFrame.

    Only includes events with a known outcome (WIN / LOSS / STOP_OUT).
    The target column 'won' is 1 for WIN, 0 for LOSS/STOP_OUT.
    The target column 'holding_minutes' is the actual holding duration.
    """
    rows = []
    for ev in events:
        if ev.outcome not in ("WIN", "LOSS", "STOP_OUT"):
            continue
        vec = build_feature_vector(ev)
        row = dict(zip(FEATURE_NAMES, vec))
        row["won"] = int(ev.outcome == "WIN")
        # Holding duration: derive from market prices if available, else use predicted
        if ev.market_price_at_signal and ev.market_price_exit and ev.holding_period_minutes:
            row["holding_minutes"] = float(ev.holding_period_minutes)
        else:
            row["holding_minutes"] = float(ev.holding_period_minutes or 60)
        rows.append(row)

    return pd.DataFrame(rows)
