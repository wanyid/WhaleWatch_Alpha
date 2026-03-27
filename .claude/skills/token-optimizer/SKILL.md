---
name: token-optimizer
description: |
  Anthropic API token cost optimization for WhaleWatch LLM calls.
  TRIGGER when: optimizing Claude API costs, implementing prompt caching, reducing token usage,
  analyzing token spend, reviewing claude_llm.py, changing LLM model selection,
  adding new prompt templates, implementing batch processing for historical signals,
  setting max_tokens, reviewing API call frequency, token budget monitoring.
  DO NOT TRIGGER when: working on non-LLM code, scanner logic, executor, or predictor.
license: MIT
metadata:
  category: llm-optimization
  version: "1.0.0"
---

# Token Optimization — WhaleWatch LLM Layer

## Core Principle

L1 generates **one Claude call per signal event**. In production this fires whenever
the Polymarket scanner or Truth Social scanner triggers. Token cost compounds fast —
optimize at the call site in `reasoner/layer1_llm/claude_llm.py`.

---

## 1. Prompt Caching (highest-impact, implement first)

`_SYSTEM_PROMPT` in `claude_llm.py` is **static across every call** — it never changes.
This is the ideal target for Anthropic prompt caching (`cache_control: ephemeral`).

**How to apply:**

```python
response = self._client.messages.create(
    model=self._model,
    max_tokens=32,
    system=[
        {
            "type": "text",
            "text": _SYSTEM_PROMPT,
            "cache_control": {"type": "ephemeral"},   # ← add this
        }
    ],
    messages=[{"role": "user", "content": user_prompt}],
)
```

**Impact:** System prompt is ~300 tokens. At Sonnet pricing, caching hits save ~90%
of input token cost on that segment. With 50+ signals/day, this adds up quickly.

**Cache duration:** 5 minutes per cache entry. Re-used as long as calls are frequent
(which they will be in live trading). No action needed on cache expiry — it refreshes
automatically on the next call.

**Verify cache hits** by checking `response.usage.cache_read_input_tokens > 0`
after the first call succeeds.

---

## 2. Model Selection by Signal Type

Not all signals need Opus. Use the cheapest model that reliably produces valid JSON.

| Signal source | Recommended model | Rationale |
|---------------|-------------------|-----------|
| Polymarket (structured numbers) | `claude-haiku-4-5-20251001` | Structured input → low ambiguity |
| Truth Social (free text) | `claude-sonnet-4-6` | Nuanced language, sarcasm, policy context |
| Dual signal (both sources) | `claude-sonnet-4-6` | Higher-stakes call; use better model |

**Implementation:** Read model from `settings.yaml → reasoner.layer1.model` per source type,
or pass `model` as a param to `ClaudeLLM.__init__`. Keep Opus only for manual review tasks.

---

## 3. `max_tokens` is Already Correct — Do Not Increase

Output is `{"direction": "BUY", "ticker": "SPY"}` — max ~35 characters.
`max_tokens=32` is correct. Never raise this for the L1 call.

If adding a reasoning field to the output schema, budget it explicitly and document the
new token ceiling. Do not use open-ended `max_tokens`.

---

## 4. User Prompt Compression

Current templates are lean. Rules to stay that way:

- **Truth Social content cap:** `event.content[:800]` is already applied. Keep it.
  If the post is longer, the signal-relevant part is almost always in the first 500 chars.
  Consider dropping to `[:500]` if cost is a concern — test parse accuracy first.
- **Polymarket template:** drop `volume_24h` from the prompt if `volume_spike_pct` is
  present — they're correlated and the spike % is more decision-relevant.
- **Keywords field:** `_polymarket_keywords()` filters to ~14 topic keywords. This is good.
  Do not expand the keyword list or pass the raw question twice.

---

## 5. Token Counting Before Shipping a New Prompt

Use `client.messages.count_tokens()` to verify token cost before deploying any prompt change:

```python
count = self._client.messages.count_tokens(
    model=self._model,
    system=_SYSTEM_PROMPT,
    messages=[{"role": "user", "content": user_prompt}],
)
logger.debug("Prompt tokens: %d", count.input_tokens)
```

Run this in a test script against a sample Polymarket event and a sample Truth Social post.
Record baseline token counts. Any PR that touches prompts must include before/after counts.

---

## 6. Batch API for Historical / Backtest Use

Do **not** use `messages.create()` in a loop when processing historical events for
labeling or backtest analysis. Use the Batch API instead:

```python
import anthropic

requests = [
    {"custom_id": event.event_id, "params": {"model": ..., "messages": [...]}}
    for event in historical_events
]
batch = client.messages.batches.create(requests=requests)
# Poll batch.id until processing_status == "ended"
```

Batch API: 50% cheaper than real-time, up to 100k requests per batch.
Use for: `label_events.py`, any bulk re-classification, hyperparameter search over prompt variants.

---

## 7. Retry Logic — Avoid Double-Spend on Parse Errors

Current retry in `ClaudeLLM.get_signal()` retries on both API errors **and** parse errors.
Parse errors (bad JSON) are rare but shouldn't cost full token price on retry.

**Better pattern:**
```python
# On parse error: retry with a brief correction message appended
# instead of re-sending the full prompt
messages = [
    {"role": "user",    "content": user_prompt},
    {"role": "assistant", "content": raw_text},       # the bad response
    {"role": "user",    "content": 'Invalid JSON. Return only: {"direction": "...", "ticker": "..."}'},
]
```
This continuation costs ~15 tokens instead of re-sending the full system + user prompt.

---

## 8. Monitoring Token Spend

Log `response.usage` after every successful call:

```python
logger.debug(
    "L1 tokens: in=%d (cached=%d) out=%d",
    response.usage.input_tokens,
    getattr(response.usage, "cache_read_input_tokens", 0),
    response.usage.output_tokens,
)
```

Track cumulative daily spend in `daily_pnl` table or a separate `api_cost` table.
Alert if daily token spend exceeds a threshold (e.g., $5/day = likely runaway loop).
