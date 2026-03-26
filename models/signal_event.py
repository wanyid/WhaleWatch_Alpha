from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional


@dataclass
class L1Signal:
    """Output of Reasoner Layer 1 (LLM).

    Contains only the tradeable direction and instrument — no free-text
    reasoning is stored. The llm_model field records which model produced
    the signal so results can be compared across model versions.
    """
    direction: str          # "BUY" | "SHORT" | "HOLD"
    ticker: str             # "SPY" | "QQQ" | "VIX"
    llm_model: str          # e.g. "claude-opus-4-6"
    source_event_type: str  # "polymarket" | "truth_social"


@dataclass
class SignalEvent:
    """Fully-resolved trading signal, from raw scanner event to execution outcome.

    Populated incrementally as the event moves through the pipeline:
      1. Scanner fields filled on creation
      2. L1 fields filled by Reasoner Layer 1
      3. L2 fields filled by Reasoner Layer 2
      4. Risk fields set by RiskManager
      5. Market impact fields filled post-execution (for backtesting / L2 training)
    """
    # --- Identity ---
    event_id: str           # UUID4
    created_at: datetime

    # --- Signal A: Polymarket (None if signal came from Truth Social only) ---
    poly_market_id: Optional[str] = None
    poly_market_slug: Optional[str] = None
    poly_market_question: Optional[str] = None
    poly_outcome_token: Optional[str] = None        # "YES" | "NO"
    poly_price_before: Optional[float] = None       # implied probability 0.0–1.0
    poly_price_after: Optional[float] = None
    poly_price_delta: Optional[float] = None        # signed change
    poly_volume_24h: Optional[float] = None         # USD
    poly_volume_spike_pct: Optional[float] = None   # % above 7-day rolling avg

    # --- Signal B: Truth Social (None if signal came from Polymarket only) ---
    ts_post_id: Optional[str] = None
    ts_post_content: Optional[str] = None
    ts_post_timestamp: Optional[datetime] = None
    ts_post_keywords: Optional[List[str]] = None    # keyword tags extracted pre-LLM

    # --- Correlation ---
    dual_signal: bool = False   # True if both A and B contributed

    # --- Reasoner Layer 1: LLM ---
    signal_direction: Optional[str] = None         # "BUY" | "SHORT" | "HOLD"
    signal_ticker: Optional[str] = None            # "SPY" | "QQQ" | "VIX"
    llm_model: Optional[str] = None

    # --- Reasoner Layer 2: Predictor ---
    confidence: Optional[float] = None             # estimated win rate 0.0–1.0
    holding_period_minutes: Optional[int] = None   # predicted hold (1–4320)

    # --- Risk (set by RiskManager before execution) ---
    stop_loss_pct: Optional[float] = None           # e.g. 0.02 = 2%
    take_profit_pct: Optional[float] = None

    # --- Market impact (filled post-facto for backtesting / L2 training) ---
    market_price_at_signal: Optional[float] = None
    market_price_exit: Optional[float] = None
    realized_pnl: Optional[float] = None
    outcome: Optional[str] = None   # "WIN" | "LOSS" | "STOP_OUT" | "OPEN"
