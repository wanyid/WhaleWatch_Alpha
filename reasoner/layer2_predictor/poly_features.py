"""Shared feature definitions for Polymarket L2 models.

Imported by:
  - scripts/train_poly_model.py       (directional model)
  - scripts/train_poly_fade_model.py  (fade / mean-reversion model)
  - scanners/polymarket_scanner.py    (live feature construction)

Keeping definitions here ensures training and inference always use an
identical feature list — the session manager builds the same keys that
the trained model was fitted on.
"""

# ---------------------------------------------------------------------------
# Feature groups — Polymarket session directional model
# ---------------------------------------------------------------------------

# Session strength — size and breadth of the whale move
STRENGTH_FEATURES = [
    "max_price_delta",       # single largest price move — whale size proxy
    "cumulative_delta",      # net directional sum (signed)
    "net_delta_abs",         # absolute cumulative magnitude
    "dominant_direction",    # +1 YES rising / -1 YES falling
    "n_events",              # total anomaly events in session
    "n_markets",             # distinct markets involved
    "n_corroborating",       # events in dominant direction
    "n_opposing",            # counter-events
    "corroboration_ratio",   # n_corroborating / n_events
    "session_duration_min",  # short+large = single entry; long = accumulation
    "max_volume_spike_pct",  # largest volume anomaly in session
    "avg_volume_spike_pct",  # average volume spike across session events
    "n_volume_spikes",       # events with BOTH price AND volume anomaly
]

# Topic composition — which buckets fired
TOPIC_FEATURES = [
    "has_tariff",
    "has_geopolitical",
    "has_fed",
    "has_energy",
    "has_executive",
]

# Market regime + temporal at session time
REGIME_FEATURES = [
    "vix_level",
    "vix_percentile",
    "vixy_level",
    "is_market_hours",
    "hour_of_day",
    "day_of_week",
]

# SPY context at session time — what equity is doing when signal fires
SPY_CONTEXT_FEATURES = [
    "spy_ret_1h",              # SPY return over prior 1h (momentum)
    "spy_ret_4h",              # SPY return over prior 4h (trend)
    "spy_range_pct",           # intraday high-low / open (extension)
    "spy_dist_from_open_pct",  # (close - open) / open (intraday bias)
]

# Market quality — not all Polymarket markets are equal
MARKET_QUALITY_FEATURES = [
    "avg_market_volume",       # mean lifetime USD volume of markets in session
    "max_market_volume",       # largest market in session by volume
    "avg_market_age_days",     # mean age of markets in session
]

# Cross-market momentum + session clustering
MOMENTUM_FEATURES = [
    "sessions_last_24h",       # number of sessions in prior 24h (activity burst)
    "cumulative_delta_24h",    # net delta across all sessions in prior 24h
    "hours_since_last_session", # recency — lower = cluster, higher = isolated signal
]

# Canonical order for directional model (34 features total)
ALL_DIRECTIONAL_FEATURES = (
    STRENGTH_FEATURES
    + TOPIC_FEATURES
    + REGIME_FEATURES
    + SPY_CONTEXT_FEATURES
    + MARKET_QUALITY_FEATURES
    + MOMENTUM_FEATURES
)

# ---------------------------------------------------------------------------
# Fade-specific features (appended to directional set)
# ---------------------------------------------------------------------------

# Initial 30m SPY move at T+90min; only available at fade entry time.
FADE_FEATURES = [
    "initial_ret",        # signed 30m return — direction + magnitude
    "initial_direction",  # +1 or -1; explicit directional indicator
    "abs_initial_ret",    # magnitude alone
]

# Full feature set for fade model (37 features total)
ALL_FADE_FEATURES = ALL_DIRECTIONAL_FEATURES + FADE_FEATURES
