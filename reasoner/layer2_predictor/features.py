"""Feature engineering for Layer 2 predictor.

Converts a SignalEvent into a numeric feature vector for model input.
All features are derived from data available at signal time — no lookahead.

Feature groups:
  - Polymarket signal strength (price delta, volume spike, outcome direction)
  - Truth Social signal (keyword count, engagement, repost flag)
  - Dual signal flag
  - Temporal context (hour-of-day, day-of-week, market session)
  - L1 output encoding (direction × ticker)
"""

import numpy as np
import pandas as pd

from models.signal_event import SignalEvent


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
