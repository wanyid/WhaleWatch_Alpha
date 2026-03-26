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

### Next Steps

- [ ] Fix conda SSL issue to run `pip install -r requirements.txt` and execute data pull
- [ ] Wire `main.py` orchestration loop (scanners → reasoner → executor)
- [ ] Build `executor/paper_executor.py` — paper trading + P&L log
- [ ] Build `risk/risk_manager.py` — enforces guardrails before execution
- [ ] Build `backtest/backtester.py` — replay historical SignalEvents
- [ ] Label historical events (backfill market prices → compute outcome)
- [ ] Train Layer 2 on labeled data once ≥10 resolved events exist
- [ ] Validate win rate; then optimize bet sizing

---

### Known Issues / Open Items

- Conda Python environments on user's machine have a broken OpenSSL DLL — affects all pip installs.
  **Fix:** Run `conda install -c anaconda openssl && conda update --all` in Anaconda Prompt (as Admin), then create fresh `whalewatch` env.
- VIX 1-minute data: no free public source exists for an index. Using 5m via yfinance. For live strategy, consider polling CBOE directly or using a VIX ETF proxy (UVXY/VIXY) as a substitute.
- Truth Social rate limiting: Cloudflare blocks after ~40 rapid requests. Script pauses 25s every 40 posts automatically.
