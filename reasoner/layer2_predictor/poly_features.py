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

# Canonical order for directional model (24 features total)
ALL_DIRECTIONAL_FEATURES = (
    STRENGTH_FEATURES
    + TOPIC_FEATURES
    + REGIME_FEATURES
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

# Full feature set for fade model (27 features total)
ALL_FADE_FEATURES = ALL_DIRECTIONAL_FEATURES + FADE_FEATURES
