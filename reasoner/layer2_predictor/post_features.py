"""Shared feature definitions for Truth Social (post) L2 models.

Imported by:
  - scripts/train_post_model.py       (directional model)
  - scripts/train_post_fade_model.py  (fade / mean-reversion model)

Keeping definitions here ensures training and any future inference
wrapper always use an identical feature list.
"""

# ---------------------------------------------------------------------------
# Feature groups — Truth Social directional model
# ---------------------------------------------------------------------------

KEYWORD_FEATURES = [
    "has_tariff",
    "has_deal",
    "has_china",
    "has_fed",
    "has_energy",
    "has_geopolitical",
    "has_market",
]

POST_FEATURES = [
    # engagement counts (favourites_count, reblogs_count, engagement) are excluded:
    # historical parquet records accumulated counts at scrape time, not at T=0,
    # making them lookahead features in training.  caps_ratio, content_length,
    # and keyword_count are derivable from post content alone.
    "caps_ratio",
    "content_length",
    "keyword_count",
]

TEMPORAL_FEATURES = [
    "hour_of_day",
    "day_of_week",
    "is_market_hours",
    "is_premarket",
]

MARKET_FEATURES = [
    "vix_level",
    "vix_percentile",
    "vixy_level",
]

# Canonical order for directional model (17 features total)
ALL_DIRECTIONAL_FEATURES = (
    KEYWORD_FEATURES
    + POST_FEATURES
    + TEMPORAL_FEATURES
    + MARKET_FEATURES
)

# ---------------------------------------------------------------------------
# Fade-specific features (appended to directional set)
# ---------------------------------------------------------------------------

# Initial 30m SPY move observed at T_open+30m; these are only available at
# fade entry time, not at the directional model's entry time (T_open).
FADE_FEATURES = [
    "spy_initial_ret",        # signed 30m return — direction + magnitude
    "spy_initial_direction",  # +1 or -1; explicit directional indicator
    "spy_abs_initial_ret",    # magnitude alone; larger overshoot = stronger candidate
]

# Full feature set for fade model (20 features total)
ALL_FADE_FEATURES = ALL_DIRECTIONAL_FEATURES + FADE_FEATURES
