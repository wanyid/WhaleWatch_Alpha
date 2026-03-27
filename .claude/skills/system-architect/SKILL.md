---
name: system-architect
description: |
  WhaleWatch system design review, code quality, and API connection patterns.
  TRIGGER when: adding a new module or component, reviewing architecture decisions,
  implementing a new scanner or executor, designing a new API connection,
  reviewing code for quality issues, adding retry/backoff/timeout logic,
  implementing auth token refresh (Schwab, Polymarket), reviewing interface compliance,
  refactoring pipeline components, designing data flow between layers,
  debugging API hangs or connection failures, reviewing new scripts.
  DO NOT TRIGGER when: writing model training loops, prompt engineering, UI work.
license: MIT
metadata:
  category: architecture
  version: "1.0.0"
---

# System Architecture & Code Quality — WhaleWatch

## Pipeline Mental Model

```
[PolymarketScanner]  ──┐                     Raw events → SessionManager (L2 fast path)
                        ├──► [ClaudeLLM L1] ──► [StatPredictor L2] ──► [RiskManager] ──► [Executor]
[TruthSocialScanner] ──┘
```

**Key invariants** — verify these hold whenever modifying the pipeline:
1. Scanners are independent: either alone can trigger the pipeline
2. L1 output is always `BUY | SHORT | HOLD` × `SPY | QQQ | VIX` — never raw text
3. L2 output is always `confidence ∈ [0,1]` + `holding_period_minutes ∈ [1, 4320]`
4. RiskManager gates execution — minimum confidence 0.60
5. Executor is the only component that touches money/positions
6. All resolved SignalEvents are persisted for L2 retraining

---

## 1. Interface Compliance Checklist

Before merging any new component, verify it implements the correct base class:

| Component | Base class | Required methods |
|-----------|-----------|-----------------|
| Scanner | `base_scanner.py` | `scan()` → raw event |
| Market data | `base_provider.py` | `get_ohlcv()`, `get_latest_price()` |
| LLM | `base_llm.py` | `get_signal()`, `model_id()` |
| Predictor | `base_predictor.py` | `predict()`, `is_trained()` |
| Executor | `base_executor.py` | `submit_signal()`, `close_position()`, `close_expired_positions()`, `open_positions()`, `check_true_news_stop()`, `session_summary()` |

**Common mistake:** Implementing a method with the right name but wrong return type.
Always check: does the new class return the exact type the caller expects?
`close_expired_positions()` must return `list[float]` (P&L values), not `int`.

---

## 2. API Connection Patterns

### 2a. Timeout — Always Use Tuple Form

```python
# WRONG — flat timeout fires on connect but not on slow body transfer
resp = requests.get(url, timeout=15)

# CORRECT — (connect_timeout, read_timeout)
resp = requests.get(url, timeout=(5, 12))
```

This was the root cause of the CLOB hang fixed in commit e7bc647.
Apply this pattern to **every** `requests.get/post` call in the codebase.

### 2b. Retry with Exponential Backoff

For any external API call that can transiently fail:

```python
import time

def _api_call_with_retry(fn, max_retries=3, base_delay=1.0):
    for attempt in range(max_retries):
        try:
            return fn()
        except requests.HTTPError as exc:
            if exc.response.status_code in (429, 500, 502, 503, 504):
                time.sleep(base_delay * (2 ** attempt))
                continue
            raise   # 4xx client errors — don't retry
        except requests.ConnectionError:
            time.sleep(base_delay * (2 ** attempt))
    raise RuntimeError(f"API call failed after {max_retries} retries")
```

Do **not** retry 400 Bad Request — these are logic errors (wrong params), not transient.

### 2c. Schwab Token Refresh (7-day expiry)

Schwab refresh tokens expire every 7 days. The pipeline will fail silently on Monday
morning if the token wasn't refreshed over the weekend.

**Required:** Add a startup check in `SchwabExecutor.__init__`:
```python
# Warn if token file is older than 6 days
token_age_days = (time.time() - Path(token_path).stat().st_mtime) / 86400
if token_age_days > 6:
    logger.warning(
        "Schwab token is %.1f days old — expires in <1 day. "
        "Run: python scripts/setup_schwab_auth.py", token_age_days
    )
```

Set up a weekly reminder (cron or calendar) to run `setup_schwab_auth.py` before
market open on Monday.

### 2d. Polymarket CLOB — Known Failure Modes

| Symptom | Root cause | Fix |
|---------|------------|-----|
| Request hangs indefinitely | Flat timeout doesn't fire on body stall | Use `timeout=(5, 12)` |
| HTTP 400 on new markets | `startTs` predates market creation | Clamp `fetch_start` to `market_start_date` |
| HTTP 400 on `clobTokenIds` | Field is a JSON string, not a list | `json.loads(raw)` before indexing |
| Silent empty response | CLOB caps responses to ~28 days | Use 25-day chunks |

All four are fixed. If a new 400 appears, check the market's `startDate` / `created_at`
field name — Gamma API field names are inconsistent across market types.

---

## 3. Code Quality Rules for Trading Systems

### 3a. No Side Effects in Signal Builders

`_build_signal_event()` and `_build_prompt()` must be pure functions:
- No database writes
- No API calls
- No logging of mutable state
- Deterministic given the same inputs

Side effects belong only in: `submit_signal()`, `close_position()`, and the scanner poll loops.

### 3b. Determinism in L1

`temperature=0.0` must remain set in `claude_llm.py`. The L1 call must be deterministic
so that replaying a SignalEvent with the same inputs produces the same direction.
Never raise temperature for production inference — only in experimental branches.

### 3c. Idempotent DB Writes

All SQLite inserts must use `INSERT OR IGNORE` or `ON CONFLICT DO UPDATE` patterns
(as in `daily_pnl` upsert). Never assume a row doesn't exist. The pipeline can restart
mid-session and replay events.

### 3d. HOLD Is the Safe Default

Any error path in the pipeline that cannot produce a valid signal **must** emit HOLD,
never raise an exception that halts the loop. The loop must survive:
- L1 API timeout → HOLD
- L2 model not trained → HOLD (skip execution)
- RiskManager circuit breaker → HOLD
- Executor connection failure → log + continue (do not crash the scanner threads)

### 3e. Feature Leakage Guard

Before adding any new feature to L2 predictor input, ask:
> "Is this value available at the exact moment the signal fires, or does it accumulate afterwards?"

**Banned features** (lookahead bias — values change after signal time):
- `favourites_count`, `reblogs_count`, `engagement` (accumulate post-scrape)
- Any price that is fetched after `signal_time + ε`
- Outcome labels from the same event batch

---

## 4. Adding a New Component — Checklist

When adding a new scanner, executor, predictor, or data provider:

- [ ] Inherits from the correct abstract base class
- [ ] All abstract methods implemented with correct return types
- [ ] `timeout=(connect, read)` on every external HTTP call
- [ ] Retry logic for transient errors (429, 5xx); no retry for 4xx
- [ ] Auth credentials loaded from `.env`, never hardcoded
- [ ] Settings loaded from `config/settings.yaml`, not hardcoded constants
- [ ] Logs at INFO for normal events, DEBUG for polling noise, ERROR for failures
- [ ] New settings documented with a comment in `settings.yaml`
- [ ] `requirements.txt` updated if new package added
- [ ] Syntax-checked before commit: `python -c "import ast; ast.parse(open('file.py').read())"`

---

## 5. Data Flow Integrity

The `SignalEvent` dataclass is the single contract between all pipeline layers.
When modifying it:

- Add new fields as `Optional[...]` with `default=None` — never break existing callers
- Update `_DDL` in both `paper_executor.py` and `schwab_executor.py`
- Update `_migrate_db()` in `paper_executor.py` to add new columns to existing DBs
- Update `build_poly_market_data.py` and `build_post_market_data.py` if new fields
  become training features

---

## 6. Swap Checklist (from CLAUDE.md)

| Component | Switch path |
|-----------|-------------|
| Market data: yfinance → Polygon | Add `polygon_provider.py`, update `settings.yaml → data.market_data_provider` |
| LLM: Claude → OpenAI | Add `openai_llm.py`, update `settings.yaml → reasoner.layer1.provider` |
| Predictor: logistic → XGBoost → NN | Update `settings.yaml → reasoner.layer2.predictor` |
| Executor: paper → Schwab | `settings.yaml → executor.provider: "schwab"` + run `setup_schwab_auth.py` |

Never swap components by modifying the concrete class — always route through the factory
or settings-based instantiation so the swap is a one-line config change.
