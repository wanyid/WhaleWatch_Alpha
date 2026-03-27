# WhaleWatch_Alpha — Project Architecture Reference

A trading bot that uses Polymarket "whale" activity and Trump Truth Social posts as independent
leading indicators for SPY / QQQ / VIX volatility trades.

---

## Pipeline Overview

```
[Scanner A: Polymarket]  ──┐
                            ├──► [Reasoner L1: LLM] ──► [Reasoner L2: Predictor] ──► [Executor]
[Scanner B: Truth Social] ──┘
```

Signals are **independent**. Either scanner alone is sufficient to trigger the pipeline.
When both fire on the same theme (dual_signal), confidence is naturally higher in Layer 2.

---

## Project Structure

```
WhaleWatch_Alpha/
├── CLAUDE.md
├── requirements.txt
├── .env.example
├── config/
│   └── settings.yaml                  # thresholds, tickers, LLM config, data_start_date
├── scanners/
│   ├── base_scanner.py                # Abstract base class for all scanners
│   ├── polymarket_scanner.py          # Signal A: Polymarket CLOB API anomaly detector
│   ├── truthsocial_scanner.py         # Signal B: truthbrush / scraper for @realDonaldTrump
│   └── market_data/
│       ├── base_provider.py           # Abstract interface — swap yfinance → Polygon later
│       └── yfinance_provider.py       # MVP free implementation
├── reasoner/
│   ├── layer1_llm/
│   │   ├── base_llm.py                # Abstract LLM interface (provider-agnostic)
│   │   └── claude_llm.py             # Default: Anthropic Claude
│   └── layer2_predictor/
│       ├── base_predictor.py          # Abstract predictor interface
│       ├── stat_predictor.py          # Logistic regression baseline
│       └── nn_predictor.py            # Neural network (future upgrade)
├── executor/
│   ├── base_executor.py               # Abstract executor interface
│   ├── paper_executor.py              # Paper trading + P&L log (current)
│   └── alpaca_executor.py             # Live broker stub (future)
├── models/
│   └── signal_event.py                # SignalEvent dataclass + enums
├── risk/
│   └── risk_manager.py                # Stop-loss, drawdown guardrails
├── backtest/
│   └── backtester.py                  # Replay historical SignalEvents
└── main.py                            # Orchestration entry point
```

---

## Module Responsibilities

### Scanners
- `polymarket_scanner.py` — polls Polymarket CLOB API, detects price delta and volume spikes
  above a configurable rolling-average threshold; emits raw `PolymarketEvent`
- `truthsocial_scanner.py` — monitors @realDonaldTrump via `truthbrush`; extracts keywords
  and emits raw `TruthSocialEvent`
- `market_data/base_provider.py` — defines `get_ohlcv(ticker, start, end, interval)` interface
  so the data source can be swapped without touching Reasoner or Backtest code

### Reasoner — Layer 1 (LLM)
Input: raw scanner event(s)
Output: `signal_direction` (`BUY` | `SHORT` | `HOLD`) + `signal_ticker` (`SPY` | `QQQ` | `VIX`)

The LLM receives a structured prompt containing the scanner payload and returns a JSON with
direction and ticker only — no free-text reasoning stored.

`base_llm.py` defines the interface; swap implementations via `settings.yaml → llm.provider`.

### Reasoner — Layer 2 (Predictor)
Input: Layer 1 output + feature vector (poly delta, volume spike, dual_signal flag, time-of-day,
VIX level at signal time, day-of-week)
Output: `confidence` (estimated win rate, 0.0–1.0) + `holding_period_minutes` (1–4320)

Trained on historical `SignalEvent` records with known `outcome`. Start with logistic regression
(`stat_predictor.py`); graduate to a feedforward NN (`nn_predictor.py`) once enough labeled data
exists (target: ≥ 200 resolved events).

Training data window: **January 20, 2025 onward** (configurable via `data_start_date` in
`settings.yaml`). This anchors the regime to Trump's second presidency and avoids mixing in
pre-inauguration market dynamics.

### Executor
`base_executor.py` defines `submit_signal(event: SignalEvent) → str` (order ID) and
`close_position(order_id: str, reason: str)`.

`paper_executor.py` — logs entries/exits to a local CSV/SQLite, tracks simulated P&L.
`alpaca_executor.py` — stub; implement when switching to live trading.

### Risk Manager
Applied between Layer 2 output and Executor. Sets `stop_loss_pct` and `take_profit_pct` on the
`SignalEvent`, and enforces circuit breakers before forwarding to the executor.

---

## SignalEvent Schema

Defined in `models/signal_event.py`. All resolved events are persisted and used as the training
corpus for Layer 2.

```python
@dataclass
class SignalEvent:
    # Identity
    event_id: str                           # UUID4
    created_at: datetime

    # Signal A — Polymarket (None if signal came from Truth Social only)
    poly_market_id: Optional[str]
    poly_market_slug: Optional[str]
    poly_market_question: Optional[str]
    poly_outcome_token: Optional[str]       # "YES" | "NO"
    poly_price_before: Optional[float]      # implied probability 0.0–1.0
    poly_price_after: Optional[float]
    poly_price_delta: Optional[float]       # signed change
    poly_volume_24h: Optional[float]        # USD
    poly_volume_spike_pct: Optional[float]  # % above 7-day rolling average

    # Signal B — Truth Social (None if signal came from Polymarket only)
    ts_post_id: Optional[str]
    ts_post_content: Optional[str]
    ts_post_timestamp: Optional[datetime]
    ts_post_keywords: Optional[List[str]]   # keyword tags extracted pre-LLM

    # Correlation
    dual_signal: bool                       # True if both A and B contributed

    # Reasoner Layer 1 — LLM
    signal_direction: str                   # "BUY" | "SHORT" | "HOLD"
    signal_ticker: str                      # "SPY" | "QQQ" | "VIX"
    llm_model: str                          # model ID that produced this signal

    # Reasoner Layer 2 — Predictor
    confidence: float                       # estimated win rate 0.0–1.0
    holding_period_minutes: int             # predicted hold time (1–4320)

    # Risk (set by RiskManager before execution)
    stop_loss_pct: float                    # e.g. 0.02 = 2%
    take_profit_pct: float

    # Market Impact — filled post-facto for backtesting and Layer 2 training
    market_price_at_signal: Optional[float]
    market_price_exit: Optional[float]
    realized_pnl: Optional[float]
    outcome: Optional[str]                  # "WIN" | "LOSS" | "STOP_OUT" | "OPEN"
```

---

## Safety Guardrails

| Rule | Value | Rationale |
|------|-------|-----------|
| Min confidence to execute | ≥ 0.60 | Filter low-conviction Layer 2 outputs |
| Hard stop-loss | 2% per position | Caps single-trade downside |
| Daily drawdown halt | −3% portfolio | Circuit breaker; no new signals until next session |
| Max holding cap | 4320 min (3 days) | Hard exit regardless of predictor; enforced by executor |
| Min holding floor | 1 min | Prevents accidental market-order churn |
| "True News" stop | Polymarket probability continues moving **with** the signal direction post-entry → exit immediately | The mean-reversion / fade thesis is invalidated when the market confirms the news |
| Training data floor | Jan 20, 2025 | Configurable via `data_start_date` in `settings.yaml` |

---

## Abstraction Points (Swap Checklist)

| Component | Interface | Current Implementation | Swap Path |
|-----------|-----------|----------------------|-----------|
| Market data | `base_provider.py` | `yfinance_provider.py` (5m chunked) | Add `polygon_provider.py`, update `settings.yaml → data.provider` |
| LLM | `base_llm.py` | `claude_llm.py` | Add `openai_llm.py`, update `settings.yaml → llm.provider` |
| Predictor | `base_predictor.py` | `stat_predictor.py` | Switch to `nn_predictor.py` after ≥ 200 labeled events |
| Executor | `base_executor.py` | `paper_executor.py` | Switch to `alpaca_executor.py` for live trading |

---

## Development Phases

1. **Data collection** — wire up scanners, persist raw events to SQLite
2. **Layer 1** — implement LLM signal direction + ticker labeling
3. **Labeling** — backfill market prices for historical events; compute `outcome`
4. **Layer 2** — train logistic regression on labeled events; validate win rate
5. **Paper trading** — run full pipeline end-to-end with paper executor
6. **Backtesting** — replay historical `SignalEvent` corpus through risk + executor
7. **Live trading** — swap in Alpaca executor; monitor vs. paper baseline

---

## GitHub Workflow

**Repository:** `wanyid/WhaleWatch_Alpha` on GitHub

### Commit requirements
- Every meaningful unit of work (new module, feature, bug fix, config change) gets its own commit
- Commit messages must be clear, specific, and follow this format:

```
<type>(<scope>): <short summary>

<optional body — what changed and why>
```

Types: `feat`, `fix`, `refactor`, `docs`, `config`, `test`
Scopes: `scanner`, `reasoner`, `executor`, `risk`, `backtest`, `models`, `config`, `ci`

Examples:
```
feat(scanner): add Polymarket anomaly detection with rolling volume baseline
fix(scanner): handle Gamma API pagination edge case on last page
docs(claude): add GitHub workflow requirements
config: add data_start_date and LLM provider settings to settings.yaml
```

### Push cadence
- Push to `main` after every logical feature or module is complete and working
- Never leave uncommitted changes when switching to a new module
- Use feature branches (`feat/reasoner-layer1`) for larger multi-session work if needed

### What NOT to commit
- **`.env` and `.env.example` — NEVER commit either file. Both are gitignored.**
  `.env.example` may contain real credentials and must never reach GitHub.
- `data/`, `*.db`, `*.sqlite` — local artifacts
- `__pycache__/`, `.claude/` — see `.gitignore`

---

## Model Design & Testing Skills

Three skills are configured for all Layer 2 predictor work. Invoke them before writing
or reviewing any model training, evaluation, or diagnostic code.

| Task | Skill to invoke |
|------|----------------|
| Training pipelines, CV splits, calibration, `TimeSeriesSplit`, `CalibratedClassifierCV` | `scientific-skills:scikit-learn` |
| Walk-forward diagnostics, Brier decomposition, statistical tests on OOS results | `scientific-skills:statsmodels` |
| Feature importance, SHAP values for XGBoost confidence model explainability | `scientific-skills:shap` |

### When to invoke

- **Before writing any new training script** or modifying `train_*.py` — invoke `scikit-learn`
- **Before evaluating OOS results or diagnosing model degradation** — invoke `statsmodels`
- **Before running feature selection or debugging a low-Brier model** — invoke `shap`
- **When `check_model_staleness.py` flags STALE** — invoke all three in sequence:
  1. `shap` to identify which features degraded
  2. `statsmodels` to test whether regime shift is statistically significant
  3. `scikit-learn` to retrain with updated hyperparameters

### Layer 2 model files
- `reasoner/layer2_predictor/stat_predictor.py` — logistic regression baseline
- `reasoner/layer2_predictor/nn_predictor.py` — neural network (future)
- `scripts/train_post_model.py` / `train_poly_model.py` — directional models
- `scripts/train_post_fade_model.py` / `train_poly_fade_model.py` — fade models
- `scripts/check_model_staleness.py` — staleness checker (rolling Brier vs training Brier)
