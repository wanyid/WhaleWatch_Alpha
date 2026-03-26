# WhaleWatch_Alpha — Session Log

---

## Session 1 — 2026-03-25

### Participants
- User (Quantitative Developer)
- Claude Sonnet 4.6 (AI pair programmer)

---

### Project Brief

Build **WhaleWatch_Alpha** — a trading bot that uses Polymarket "whale" activity and Trump Truth Social posts as leading indicators for SPY / QQQ / VIX volatility trades.

---

### Key Design Decisions Made

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Signal architecture | Independent dual-signal (A + B) | Either signal alone triggers the pipeline; both together boosts Layer 2 confidence |
| Holding window | 1 min – 3 days (4320 min) | No HFT, no long-term holds |
| Reasoner structure | Two-layer (LLM → direction/ticker; ML → win rate + hold period) | Separates interpretation from prediction |
| Layer 2 baseline | Logistic regression | Upgrade to NN after ≥200 labeled events |
| LLM provider | Configurable (Claude default) | Swappable via `base_llm.py` interface |
| Market data | yfinance 5m chunked | Free, no API key; 50-day windows stitched from 2024-01-01 |
| Execution | Paper trading first | Abstract `base_executor.py` for future Alpaca live swap |
| Training data floor | Jan 20 2025 (Trump inauguration) | Keeps political/market regime consistent |
| Position sizing | Deferred | Focus on win rate first; bet sizing to be optimized later |
| VIX spike reducer | Removed | VIX spikes are the signal target, not a risk filter |

---

### Architecture Defined

```
[Scanner A: Polymarket CLOB]  ──┐
                                 ├──► [Reasoner L1: LLM] ──► [Reasoner L2: Predictor] ──► [Executor]
[Scanner B: Truth Social]     ──┘
```

**Reasoner Layer 1 (LLM):**
Input: raw scanner event → Output: `BUY/SHORT/HOLD` + `SPY/QQQ/VIX`

**Reasoner Layer 2 (Predictor):**
Input: L1 output + feature vector → Output: `confidence` (win rate 0–1) + `holding_period_minutes` (1–4320)

---

### Safety Guardrails Defined

| Rule | Value |
|------|-------|
| Min confidence to trade | ≥ 0.60 |
| Hard stop-loss | 2% per position |
| Daily drawdown halt | −3% portfolio |
| Max holding cap | 4320 min (3 days) |
| Min holding floor | 1 min |
| "True News" stop | Exit if Polymarket probability continues moving with signal direction post-entry |
| Training data floor | Jan 20, 2025 |

---

### Files Built This Session

```
WhaleWatch_Alpha/
├── CLAUDE.md                                   ← architecture reference + GitHub workflow rules
├── SESSIONS.md                                 ← this file
├── requirements.txt                            ← all dependencies
├── .env.example                                ← credential template
├── .gitignore
├── config/
│   └── settings.yaml                           ← thresholds, providers, tickers
├── models/
│   ├── raw_events.py                           ← TruthSocialRawEvent, PolymarketRawEvent
│   └── signal_event.py                         ← L1Signal, SignalEvent dataclasses
├── scanners/
│   ├── base_scanner.py                         ← abstract BaseScanner
│   ├── truthsocial_scanner.py                  ← Signal B: polls @realDonaldTrump via truthbrush
│   ├── polymarket_scanner.py                   ← Signal A: Gamma discovery + CLOB anomaly detection
│   └── market_data/
│       ├── base_provider.py                    ← abstract get_ohlcv() / get_latest_price()
│       └── yfinance_provider.py                ← 5m chunked + 1h single-request implementations
├── reasoner/
│   ├── layer1_llm/
│   │   ├── base_llm.py                         ← abstract BaseLLM interface
│   │   └── claude_llm.py                       ← Claude implementation, structured JSON prompt
│   └── layer2_predictor/
│       ├── base_predictor.py                   ← abstract BasePredictor interface
│       ├── features.py                         ← 18-feature vector engineering
│       └── stat_predictor.py                   ← LogisticRegression (win rate) + Ridge (hold period)
└── scripts/
    └── pull_historical_data.py                 ← one-time data pull to D:/WhaleWatch_Data/
```

---

### Data Pull Script

**Output:** `D:/WhaleWatch_Data/`

| File | Source | Resolution | Notes |
|------|--------|------------|-------|
| `equity/SPY_5m.parquet` | yfinance | 5-minute | ~1.5M rows |
| `equity/QQQ_5m.parquet` | yfinance | 5-minute | ~1.5M rows |
| `equity/VIX_5m.parquet` | yfinance | 5-minute | index, no free 1m source |
| `polymarket/markets_catalog.parquet` | Gamma API | — | all relevant market metadata |
| `polymarket/prices/*.parquet` | CLOB API | 1-hour | one file per market |
| `truth_social/trump_posts.jsonl` | truthbrush | — | all posts since 2024-01-01 |

**Run:**
```bash
conda activate whalewatch
cd C:\ClaudeCode\WhaleWatch_Alpha
pip install -r requirements.txt
python scripts/pull_historical_data.py
```

---

### Data Source Decisions

| Data | Source | Reason |
|------|--------|--------|
| SPY / QQQ / VIX bars | yfinance (5m chunked) | Free, no signup; 50-day request windows stitched from 2024 |
| Polymarket prices | CLOB API (public) | No auth needed for read-only data |
| Polymarket discovery | Gamma API (public) | Tag/keyword filtering for political markets |
| Truth Social | truthbrush (Stanford IO) | Python package for @realDonaldTrump posts |
| Live execution (future) | Alpaca | Abstract executor stub ready; activate when going live |
| Upgrade path (future) | Polygon.io | Swap via `base_provider.py` when 1m real-time data needed |

---

### GitHub

**Repo:** https://github.com/wanyid/WhaleWatch_Alpha

**Commits this session:**

| Hash | Message |
|------|---------|
| `f61c911` | `feat(init)`: initial project scaffold — architecture, scanners, models |
| `d403e83` | `feat(reasoner)`: Reasoner Layer 1 + market data provider + historical data pull script |
| `90400cc` | `feat(data)`: switch equity pull to 1-minute bars via Alpaca + add AlpacaProvider |
| `57b1c58` | `feat(reasoner)`: Reasoner Layer 2 — logistic regression win-rate + holding-period predictor |
| `d842635` | `refactor(data)`: replace Alpaca with yfinance 5m chunked — no API key needed |

---

## Session 2 — 2026-03-26

### Participants
- User (Quantitative Developer)
- Claude Sonnet 4.6 (AI pair programmer)

---

### Key Decisions / Changes

| Area | Decision | Rationale |
|------|----------|-----------|
| Conda env | SSL fixed by copying `libssl-1_1-x64.dll` + `libcrypto-1_1-x64.dll` from `anaconda3/Library/bin/` to `anaconda3/DLLs/`; fresh `whalewatch` env created with Python 3.11 | Base Anaconda env had broken OpenSSL DLL |
| Equity data | Switched to `1d` (unlimited) + `5m period="60d"` (recent only) | yfinance hard-limits: 5m=60 days, 1h=730 days; 2024-01-01 → today is ~815 days — 1h fails |
| Polymarket CLOB API | Parameter names changed: `token_id` → `market`, `start_time`/`end_time` → `start_ts`/`end_ts` | Breaking API change on Polymarket's side |
| Polymarket fidelity | Changed `fidelity=60` → `fidelity=1440` (daily) | Hourly (60) only returns last ~28 days; daily gives full market lifetime history |
| Polymarket discovery | Added `end_date_min=2024-01-01` + `volume_num_min=1000` for closed markets | Without filters: 150k+ closed markets since 2024 → hours to page through; volume filter cuts to manageable set of whale-relevant markets |

---

### Data Pull Status

| File | Status | Details |
|------|--------|---------|
| `equity/SPY_1d.parquet` | ✅ Done | 559 rows, 2024-01-01 → today |
| `equity/QQQ_1d.parquet` | ✅ Done | 559 rows |
| `equity/VIX_1d.parquet` | ✅ Done | 559 rows |
| `equity/SPY_5m_recent.parquet` | ✅ Done | 4,558 rows, last 60 days |
| `equity/QQQ_5m_recent.parquet` | ✅ Done | 4,556 rows |
| `equity/VIX_5m_recent.parquet` | ✅ Done | 8,887 rows |
| `polymarket/markets_catalog.parquet` | ⏳ Re-run needed | Catalog from previous broken run; needs re-pull with fixed params |
| `polymarket/prices/*.parquet` | ⏳ Pending | Price pull not yet completed; script fixed and ready |
| `truth_social/trump_posts.jsonl` | ⏳ Pending | Needs TRUTHSOCIAL_USERNAME + TRUTHSOCIAL_PASSWORD in .env |

---

### Commits This Session

| Hash | Message |
|------|---------|
| `f61c911` | `feat(init)`: initial project scaffold — architecture, scanners, models |
| `d403e83` | `feat(reasoner)`: Reasoner Layer 1 + market data provider + historical data pull script |
| `90400cc` | `feat(data)`: switch equity pull to 1-minute bars via Alpaca + add AlpacaProvider |
| `57b1c58` | `feat(reasoner)`: Reasoner Layer 2 — logistic regression win-rate + holding-period predictor |
| `d842635` | `refactor(data)`: replace Alpaca with yfinance 5m chunked — no API key needed |
| `a07c4cc` | `refactor(data)`: rewrite pull script — dual resolution equity + correct Gamma camelCase fields |
| `7577ff9` | `fix(data)`: update CLOB API params — token_id→market, fidelity 60→1440 |
| *(pending)* | `fix(data)`: add Gamma discovery filters (end_date_min + volume_num_min) |

---

### Next Steps

- [ ] **Re-run data pull** — `python scripts/pull_historical_data.py` in `whalewatch` conda env
  - Active markets discovery: ~90s (367 pages)
  - Closed markets discovery: ~minutes with new filters
  - Price fetch: ~10-20 min depending on market count
  - Truth Social: needs credentials in `.env`
- [ ] Wire `main.py` orchestration loop (scanners → reasoner → executor)
- [ ] Build `executor/paper_executor.py` — paper trading + P&L log
- [ ] Build `risk/risk_manager.py` — enforces guardrails before execution
- [ ] Build `backtest/backtester.py` — replay historical SignalEvents
- [ ] Label historical events (backfill market prices → compute outcome)
- [ ] Train Layer 2 on labeled data once ≥10 resolved events exist
- [ ] Validate win rate; then optimize bet sizing

---

### Known Issues / Open Items

- **Conda SSL fix**: Copy `libssl-1_1-x64.dll` + `libcrypto-1_1-x64.dll` from `C:\Users\wangy\anaconda3\Library\bin\` to `C:\Users\wangy\anaconda3\DLLs\` if SSL errors recur in the `whalewatch` env.
- **Polymarket data depth**: CLOB `/prices-history` with `fidelity=1440` returns data only from market creation date (not from 2024-01-01 for newer markets). This is expected — prices don't exist before a market opens.
- **VIX 5m data**: yfinance returns ~2× more rows than SPY/QQQ for the same period (8,887 vs ~4,557) — likely because VIX trades extended hours. Filter to market hours if needed for signal alignment.
- **Truth Social rate limiting**: Cloudflare blocks after ~40 rapid requests. Script pauses 25s every 40 posts automatically. Needs credentials in `.env`.
- **SECRETS — NEVER PUSH TO GITHUB**: `.env` AND `.env.example` are both gitignored and must never be committed. `.env.example` may contain real credentials. Both files are blocked in `.gitignore`.

---

## Session 3 — 2026-03-26 (overnight build)

### Key Decisions / Changes

| Area | Decision | Rationale |
|------|----------|-----------|
| Truth Social | Confirmed `truthbrush` works with `LoeWongtoe` account; `created_after` requires `datetime` object not string | Tested live — auth tokens issued, posts fetched |
| Secrets | `.env.example` gitignored permanently; credentials scrubbed from file | Real creds were accidentally stored in example file |
| Gamma closed filter | Tightened to `end_date_min=2025-01-20` + `volume_num_min=50000` | `volume_num_min=1000` still left 100k+ pages; 50k limit cuts to whale-relevant markets while keeping the inauguration-aligned training window |
| `main.py` architecture | Scanner threads + Queue + pipeline function + periodic sweep | Decoupled from scanner implementation; graceful SIGINT/SIGTERM shutdown |
| Paper executor storage | SQLite (not CSV) with `positions` + `daily_pnl` tables | Supports concurrent writes, easy ad-hoc queries, natural upsert for daily aggregation |
| Backtester price lookup | Uses stored `{TICKER}_1d.parquet` OHLCV; falls back to entry price if missing | Keeps backtester self-contained; no live API calls needed for replay |

### Files Built This Session

| File | Status | Description |
|------|--------|-------------|
| `executor/base_executor.py` | ✅ Done | Abstract interface |
| `executor/paper_executor.py` | ✅ Done | SQLite paper trading + P&L |
| `executor/alpaca_executor.py` | ✅ Stub | Future live broker |
| `risk/risk_manager.py` | ✅ Done | Confidence gate + circuit breaker |
| `backtest/backtester.py` | ✅ Done | Historical replay from SQLite |
| `main.py` | ✅ Done | Full orchestration loop |
| `tests/test_risk_executor.py` | ✅ Done | 12 passing smoke tests |

### Commits This Session

| Hash | Message |
|------|---------|
| `6b4f888` | `config`: block .env.example from git — never commit secrets |
| `56cad2b` | `feat(executor,risk,backtest)`: paper executor + risk manager + backtester + main loop |
| `80ca6d2` | `test(executor,risk)`: 12 smoke tests for RiskManager + PaperExecutor |

### Data Pull Status (Session 3)

| Source | Status | Notes |
|--------|--------|-------|
| Equity (all 6 files) | ✅ Done | Unchanged |
| Polymarket catalog | ⏳ Running | Closed market discovery in progress with tightened filters |
| Polymarket prices | ⏳ Pending | Will start after discovery completes |
| Truth Social | ⏳ Pending | Will run after Polymarket; credentials confirmed working |

### Next Steps (for next session)

- [ ] Verify data pull completed successfully — check `D:/WhaleWatch_Data/polymarket/prices/` file count
- [ ] Label historical Polymarket events (map closed markets → SPY/QQQ/VIX moves at signal time → `outcome`)
- [ ] Build `scripts/label_events.py` — backfill `market_price_at_signal`, `market_price_exit`, `realized_pnl`, `outcome`
- [ ] Train Layer 2 once ≥10 labeled events exist (`stat_predictor.py` `fit()` method)
- [ ] Add ANTHROPIC_API_KEY to `.env` and do an end-to-end `python main.py --once` smoke test
- [ ] Monitor paper trading session when market opens

---

## Session 4 — 2026-03-26

### Participants
- User (Quantitative Developer)
- Claude Sonnet 4.6 (AI pair programmer)

---

### Key Work Completed

#### Morning: Truth Social L2 Pipeline — Fixed + Trained

| Area | Decision / Fix | Rationale |
|------|---------------|-----------|
| Label method | Excess return vs 20-post rolling baseline (dead-zone filtered) | Isolates post-driven alpha from SPY drift; removes ambiguous near-zero moves |
| Training labels | Measured SPY price reactions (not LLM output) | Objective ground truth; avoids baking LLM subjectivity into training |
| Dead zone | Period-specific thresholds (5m: ±0.10% → 1d: ±0.80%) | Drops 70–86% of rows where post-attribution is unclear |
| Model architecture | XGBoost + isotonic calibration (replaces LogisticRegression) | XGBoost handles non-linear VIX × day-of-week interactions; isotonic calibration gives well-calibrated probabilities |
| CV method | TimeSeriesSplit 5-fold (no lookahead) | Prevents future data leaking into training folds |
| Timezone bug (fixed) | Intraday offsets anchored from actual entry bar timestamp, not hardcoded 13:30 UTC | EST/EDT shift caused 90%+ of T+5m returns to = 0 |
| Post filter | Keyword-bearing posts only (`--min-keywords 1`) | AUC 0.60 → 0.64 for 1d by removing golf/endorsement posts |
| Regime split | Separate high-VIX (≥20) and low-VIX (<20) models | Intraday signal quality collapses in crisis periods |
| Decay weighting | 180-day half-life option | Modest benefit for 4h; neutral elsewhere |
| Polygon.io | Used for SPY/QQQ/VIXY 5m + 1h (Jan 2025 → present) | yfinance 5m limited to 60 days — insufficient for full training window |

**Truth Social L2 Model Results (trained on 2,207 posts, Jan 21 2025 → Mar 25 2026):**

| Period | Baseline AUC | Best model | Best AUC |
|--------|-------------|------------|----------|
| 5m  | 0.437 | low_vix | 0.565 |
| 30m | 0.460 | low_vix | 0.615 |
| 1h  | 0.483 | low_vix | 0.550 |
| 2h  | 0.499 | high_vix | 0.553 |
| 4h  | 0.538 | baseline | 0.538 |
| **1d**  | **0.638** | **baseline** | **0.638** |

**Key insight:** Low-VIX regime gives strong intraday signal (30m AUC 0.615). High-VIX
crisis periods suppress intraday signal — only 1d and 2h remain tradeable.

---

#### Afternoon: Polymarket L2 Pipeline Built

New scripts and data pipeline for the Polymarket signal arm:

| Script | Purpose |
|--------|---------|
| `scripts/pull_polymarket_history.py` | Gamma API discovery + CLOB daily price history per market |
| `scripts/pull_polymarket_trades.py` | CLOB trade history → hourly USDC volume buckets |
| `scripts/build_poly_market_data.py` | Session aggregation: group anomaly events → features + SPY labels |
| `scripts/train_poly_model.py` | XGBoost + isotonic calibration, same interface as Truth Social models |

**Session aggregation design:**
- Whale anomaly events within a configurable window are grouped into "sessions"
- Session-level features: `max_price_delta`, `cumulative_delta`, `dominant_direction`, `n_events`, `n_markets`, `corroboration_ratio`, `session_duration_min`
- Volume features: `max_volume_spike_pct`, `avg_volume_spike_pct`, `n_volume_spikes`
- Topic composition: `has_tariff`, `has_geopolitical`, `has_fed`, `has_energy`, `has_executive`
- Regime: `vix_level`, `vix_percentile`, `vixy_level`, `is_market_hours`, `hour_of_day`, `day_of_week`

**Model output:** `P(SPY excess return > 0 after session)` — same interface as Truth Social models.
Saved as `poly_direction_{period}.pkl` with high/low VIX and decay-weighted variants.

---

### Docs Created / Updated

| File | Content |
|------|---------|
| `docs/training_decisions.md` | 12-section design rationale: label method, dead zones, model architecture, regime split, timezone bug, etc. |
| `docs/model_performance.md` | Full Truth Social L2 results with model routing table for inference time |

---

### Files Built / Updated This Session

```
scripts/
├── pull_polymarket_history.py   ← Gamma discovery + CLOB daily prices
├── pull_polymarket_trades.py    ← CLOB trade history → hourly USDC volume
├── build_poly_market_data.py    ← session aggregation + SPY label engineering
└── train_poly_model.py          ← XGBoost L2 trainer (same interface as train_post_model.py)
docs/
├── training_decisions.md        ← 12 design decisions with rationale
└── model_performance.md         ← Truth Social L2 model results + routing table
models/saved/
├── spy_direction_{period}.pkl          ← baseline (all 6 periods)
├── spy_direction_{period}_high_vix.pkl ← VIX ≥ 20 split
├── spy_direction_{period}_low_vix.pkl  ← VIX < 20 split
└── spy_direction_{period}_weighted.pkl ← 180-day decay-weighted
```

---

### Commits This Session

| Hash | Message |
|------|---------|
| `082a3b8` | `fix(data)`: label_events + clean_data + train_l2 pipeline fixes |
| `9aa5a0c` | `feat(data)`: generate real L2 training data from Trump posts via LLM |
| `9df81ab` | `feat(reasoner)`: L2 model pipeline — Polygon data, alpha labeling, regime split |
| `d2b4144` | `feat(scanner)`: Polymarket L2 pipeline — session aggregation + SPY model |
| `9ff71ad` | `feat(scanner)`: add dominant_direction + vixy_level features to Polymarket model |
| `321d79f` | `feat(scanner)`: add volume/trade-size features to Polymarket session model |

---

### Next Steps (for next session)

- [ ] **Run Polymarket data pipeline** in `whalewatch` conda env:
  ```
  python scripts/pull_polymarket_history.py   # discovery + daily prices
  python scripts/pull_polymarket_trades.py    # hourly USDC volume
  python scripts/build_poly_market_data.py   # session aggregation
  python scripts/train_poly_model.py --regime  # train poly_direction_*.pkl
  ```
- [ ] Document Polymarket L2 model results in `docs/model_performance.md`
- [ ] Compare Truth Social vs Polymarket signal quality across holding periods
- [ ] Wire Polymarket model into `stat_predictor.py` inference path (use `poly_direction_1d.pkl` as secondary scorer)
- [ ] Run end-to-end `python main.py --once` smoke test with ANTHROPIC_API_KEY in `.env`
- [ ] Begin paper trading; monitor first live session

---

### Model Routing at Inference Time (Current Best)

```
VIX >= 20 (elevated/crisis):
  1d  → spy_direction_1d.pkl              (AUC 0.638)
  2h  → spy_direction_2h_high_vix.pkl     (AUC 0.553)
  Others → SKIP

VIX < 20 (calm):
  30m → spy_direction_30m_low_vix.pkl     (AUC 0.615)  ← best intraday
  1h  → spy_direction_1h_low_vix.pkl      (AUC 0.550)
  5m  → spy_direction_5m_low_vix.pkl      (AUC 0.565)
  1d  → spy_direction_1d.pkl             (AUC 0.638)

Min confidence threshold to act: 0.60
```
