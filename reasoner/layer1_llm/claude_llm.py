"""Reasoner Layer 1 — Anthropic Claude implementation.

Takes a raw scanner event (Polymarket move or Truth Social post) and asks
Claude to classify it as BUY / SHORT / HOLD on SPY, QQQ, or VIX.
Returns a compact L1Signal — no reasoning text is stored.
"""

import json
import logging
import os
from typing import Union

from models.raw_events import PolymarketRawEvent, TruthSocialRawEvent
from models.signal_event import L1Signal
from reasoner.layer1_llm.base_llm import BaseLLM

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT = """\
You are a trading signal classifier for a US equity volatility strategy.

STRATEGY CONTEXT:
- We trade SPY (S&P 500 ETF), QQQ (Nasdaq-100 ETF), or VIX (volatility index).
- Holding period: minutes to 3 days.
- Core thesis: political/macro events cause short-term overreactions that partially mean-revert.
  However, if news is confirmed and markets continue moving, we exit immediately ("true news stop").

INSTRUMENTS:
- BUY VIX   → expect a spike in market fear / volatility (conflict, sudden shock, crisis)
- SHORT SPY  → expect broad US equity weakness (tariffs, trade war, sanctions, bad macro data)
- SHORT QQQ  → expect tech-specific weakness (antitrust, China tech, supply chain disruption)
- BUY SPY    → expect broad US equity strength (deal announced, de-escalation, positive policy)
- BUY QQQ    → expect tech-led rally (deregulation, AI policy, rate cut signal)
- HOLD       → event is too ambiguous, already priced in, or unlikely to move markets materially

RULES:
1. Default to HOLD when uncertain.
2. Choose the SINGLE most relevant ticker — do not hedge across instruments.
3. Return ONLY valid JSON, no other text, no markdown code fences.

OUTPUT FORMAT (strict):
{"direction": "BUY" | "SHORT" | "HOLD", "ticker": "SPY" | "QQQ" | "VIX"}
"""

_POLYMARKET_TEMPLATE = """\
EVENT TYPE: Polymarket prediction market — anomalous price/volume move

MARKET QUESTION: {question}

PRICE MOVE:
  Before: {price_before:.2f} (implied probability)
  After:  {price_after:.2f}
  Delta:  {price_delta:+.3f} ({direction_hint} move on {outcome_token} outcome)

VOLUME:
  24h volume: ${volume_24h:,.0f} USD
  Volume spike: {volume_spike_pct:+.1f}% vs 7-day average

KEYWORDS implied by market topic: {keywords}

Classify the likely short-term directional impact on US equity markets.
"""

_TRUTHSOCIAL_TEMPLATE = """\
EVENT TYPE: Truth Social post from @realDonaldTrump

POSTED AT: {posted_at} UTC
CONTENT: {content}
DETECTED KEYWORDS: {keywords}
ENGAGEMENT: {favourites} likes · {reblogs} reposts

Classify the likely short-term directional impact on US equity markets.
"""


def _polymarket_keywords(question: str) -> list[str]:
    """Extract implied topic keywords from the market question."""
    q = question.lower()
    found = []
    for kw in ["tariff", "china", "trade", "ukraine", "russia", "iran", "nato",
               "fed", "rate", "inflation", "cabinet", "sanctions", "oil", "crypto"]:
        if kw in q:
            found.append(kw)
    return found


class ClaudeLLM(BaseLLM):
    """Layer 1 signal classifier using Anthropic Claude.

    Args:
        model: Claude model ID (default: claude-opus-4-6).
        max_retries: Number of API retry attempts on transient errors.
    """

    _VALID_DIRECTIONS = {"BUY", "SHORT", "HOLD"}
    _VALID_TICKERS = {"SPY", "QQQ", "VIX"}

    def __init__(
        self,
        model: str = "claude-opus-4-6",
        max_retries: int = 3,
    ) -> None:
        self._model = model
        self._max_retries = max_retries
        self._client = self._build_client()

    def model_id(self) -> str:
        return self._model

    def get_signal(
        self,
        event: Union[TruthSocialRawEvent, PolymarketRawEvent],
    ) -> L1Signal:
        """Call Claude and return a direction + ticker for the given event."""
        source_type, user_prompt = self._build_prompt(event)

        last_exc: Exception = RuntimeError("No attempts made")
        for attempt in range(1, self._max_retries + 1):
            try:
                response = self._client.messages.create(
                    model=self._model,
                    max_tokens=32,
                    system=_SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": user_prompt}],
                )
                raw_text = response.content[0].text.strip()
                direction, ticker = self._parse_response(raw_text)
                logger.info(
                    "L1 signal: %s %s (source=%s, model=%s)",
                    direction, ticker, source_type, self._model,
                )
                return L1Signal(
                    direction=direction,
                    ticker=ticker,
                    llm_model=self._model,
                    source_event_type=source_type,
                )
            except (ValueError, KeyError) as exc:
                logger.warning("L1 parse error (attempt %d/%d): %s", attempt, self._max_retries, exc)
                last_exc = exc
            except Exception as exc:
                logger.warning("L1 API error (attempt %d/%d): %s", attempt, self._max_retries, exc)
                last_exc = exc

        # All retries exhausted — default to HOLD to avoid trading on bad data
        logger.error("L1 all retries failed (%s), defaulting to HOLD SPY", last_exc)
        return L1Signal(
            direction="HOLD",
            ticker="SPY",
            llm_model=self._model,
            source_event_type=source_type,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_client(self):
        try:
            import anthropic
        except ImportError as exc:
            raise ImportError("anthropic package not installed. Run: pip install anthropic") from exc

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError("ANTHROPIC_API_KEY must be set in environment.")
        return anthropic.Anthropic(api_key=api_key)

    def _build_prompt(
        self,
        event: Union[TruthSocialRawEvent, PolymarketRawEvent],
    ) -> tuple[str, str]:
        """Return (source_event_type, user_prompt_text)."""
        if isinstance(event, PolymarketRawEvent):
            direction_hint = "bullish" if event.price_delta > 0 else "bearish"
            keywords = _polymarket_keywords(event.market_question)
            prompt = _POLYMARKET_TEMPLATE.format(
                question=event.market_question,
                price_before=event.price_before,
                price_after=event.price_after,
                price_delta=event.price_delta,
                direction_hint=direction_hint,
                outcome_token=event.outcome_token,
                volume_24h=event.volume_24h,
                volume_spike_pct=event.volume_spike_pct,
                keywords=", ".join(keywords) if keywords else "none detected",
            )
            return "polymarket", prompt

        elif isinstance(event, TruthSocialRawEvent):
            prompt = _TRUTHSOCIAL_TEMPLATE.format(
                posted_at=event.posted_at.strftime("%Y-%m-%d %H:%M"),
                content=event.content[:800],    # cap at 800 chars to control token usage
                keywords=", ".join(event.keywords) if event.keywords else "none detected",
                favourites=event.favourites_count,
                reblogs=event.reblogs_count,
            )
            return "truth_social", prompt

        else:
            raise TypeError(f"Unsupported event type: {type(event)}")

    def _parse_response(self, raw_text: str) -> tuple[str, str]:
        """Parse the JSON response and validate fields."""
        data = json.loads(raw_text)
        direction = str(data.get("direction", "")).upper()
        ticker = str(data.get("ticker", "")).upper()

        if direction not in self._VALID_DIRECTIONS:
            raise ValueError(f"Invalid direction: {direction!r}")
        if ticker not in self._VALID_TICKERS:
            raise ValueError(f"Invalid ticker: {ticker!r}")

        return direction, ticker
