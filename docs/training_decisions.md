# L2 Model Training — Design Decisions

This document records the key decisions made during the design and training of the
Layer 2 (L2) SPY direction predictor, including rationale and alternatives considered.

---

## 1. Training Signal: Market Reaction, Not LLM Labels

**Decision:** Labels are derived from measured SPY price returns after each Trump post,
not from LLM-assigned direction labels.

**Rationale:** Using LLM output as training labels would bake LLM subjectivity into the
ground truth. The model would be learning to imitate the LLM, not to predict market outcomes.
By using actual market reactions, labels are objective and independent of any model.

**Alternative considered:** Call Layer 1 (LLM) on each post to get BUY/SHORT, then measure
whether the market moved in that direction. Rejected because it links training quality to LLM
quality and costs API tokens for every historical post.

---

## 2. Alpha Isolation: Excess Return Labels

**Decision:** Labels use excess returns (actual return minus 20-post rolling baseline) rather
than raw returns.

**Rationale:** SPY has a positive drift (beta) that varies by regime. In a bull market, a model
trained on raw returns would learn to predict "BUY" most of the time regardless of post content.
The excess return isolates the post-driven alpha component.

**Dead-zone thresholds** (moves too small to be attributable to a post):

| Period | Threshold |
|--------|-----------|
| 5m  | ±0.10% |
| 30m | ±0.20% |
| 1h  | ±0.30% |
| 2h  | ±0.40% |
| 4h  | ±0.60% |
| 1d  | ±0.80% |

Rows within the dead zone are dropped from training. Typically 70–86% of rows are dropped,
leaving only posts with clear market-relevant excess moves.

**Effect:** Pos-rate (fraction of BUY labels) moved from 55–63% (raw returns, beta-biased)
to 46–52% (excess returns, near 50/50) across all periods.

---

## 3. Output: Win Probability, Not Direction + Confidence

**Decision:** The model outputs P(SPY excess return > 0 after post), i.e. the probability that
BUY is the correct direction. HOLD is not a model output.

**Rationale:** Avoids the need to define a HOLD class (ambiguous threshold) and gives a
calibrated probability that the executor can threshold however it likes. The executor applies
a minimum confidence threshold (default: 60%) to decide whether to act at all.

---

## 4. Ticker: SPY Only

**Decision:** Train on SPY direction only. VIX is used as an input feature, not an output.
QQQ returns are in the dataset but not used as training labels.

**Rationale:** Adding multi-ticker output (SPY vs QQQ vs VIX) significantly increases model
complexity without clear benefit at this stage. SPY is the most liquid and directly tradeable.
VIX is a sentiment/regime indicator that improves prediction but is not the trading target.

---

## 5. Post Filter: Keywords Only

**Decision:** Train only on posts containing at least one market-relevant keyword
(`--min-keywords 1`). Non-keyword posts are excluded from training.

**Keyword categories:** has_tariff, has_deal, has_china, has_fed, has_energy,
has_geopolitical, has_market.

**Effect:** AUC improved from 0.602 to 0.645 for the 1d model by removing ~60% of posts
(golf, endorsements, personal) that have no predictive relationship to market moves.

**At inference time:** Non-keyword posts should route to HOLD without consulting Layer 2.

---

## 6. Anti-Lookahead: TimeSeriesSplit

**Decision:** All cross-validation uses `TimeSeriesSplit` (validation folds always come
*after* training folds in time). `StratifiedKFold(shuffle=True)` was explicitly rejected.

**Rationale:** Financial ML with shuffled CV allows future data to inform training, producing
optimistic AUC estimates that collapse in live trading. TimeSeriesSplit enforces a strict
walk-forward evaluation.

---

## 7. Model Architecture: XGBoost with Isotonic Calibration

**Decision:** XGBoost binary classifier wrapped in `CalibratedClassifierCV(method='isotonic')`.

**Rationale:** XGBoost handles non-linear interactions between VIX level, day-of-week, and
keyword flags. Isotonic calibration ensures that predicted probabilities are well-calibrated
(pred=0.60 means ~60% actual win rate). Logistic regression was tested as a baseline but
produced lower AUC across all periods.

---

## 8. Regime-Split Models (Option B)

**Decision:** Train separate high-VIX (≥20) and low-VIX (<20) models in addition to the
full-data baseline.

**Finding:** The relationship between Trump posts and SPY returns is **regime-dependent**:

| Period | Baseline | High VIX (≥20) | Low VIX (<20) |
|--------|----------|----------------|---------------|
| 5m  | 0.44 | 0.45 | **0.56** |
| 30m | 0.46 | 0.51 | **0.62** |
| 1h  | 0.48 | 0.41 | 0.55 |
| 2h  | 0.50 | 0.55 | 0.52 |
| 4h  | 0.54 | 0.48 | **0.55** |
| 1d  | **0.64** | **0.64** | 0.61 |

Key insight: Intraday models perform better in **calm (low-VIX) markets**. In high-VIX
crisis periods, SPY reacts to everything (macro, news, flows) and Trump posts lose their
relative signal strength. The 1d model is robust across both regimes.

**Deployment rule:**
- VIX < 20: use `spy_direction_{period}_low_vix.pkl`
- VIX ≥ 20: use `spy_direction_{period}_high_vix.pkl` for 2h only; skip other intraday signals
- 1d: always use `spy_direction_1d.pkl` (full dataset, AUC=0.64)

---

## 9. Decay-Weighted Models (Option C)

**Decision:** Train decay-weighted models with 180-day half-life alongside the baseline.

**Rationale:** Market regimes shift over time. Exponential decay gives recent posts more
influence without discarding old data entirely.

**Finding:** Modest improvement for 4h (0.54→0.55), neutral or slightly worse elsewhere.
Provides a practical alternative when regime routing is undesirable.

Saved as `spy_direction_{period}_weighted.pkl`.

---

## 10. Data Sources

| Data | Source | Coverage |
|------|--------|----------|
| Trump posts | Truth Social (scraper) | Jan 20, 2025 → present |
| SPY/QQQ 5m | Polygon.io free tier | Jan 21, 2025 → present |
| SPY/QQQ 1h | Polygon.io free tier | Jan 21, 2025 → present |
| VIXY 5m/1h | Polygon.io free tier | Jan 21, 2025 → present |
| VIX daily | yfinance | Jan 2024 → present |
| SPY/QQQ daily | yfinance | Jan 2024 → present |

**Note:** Polygon free tier does not provide VIX index (I:VIX) data. VIX level and
percentile features use yfinance daily closes. VIXY ETF (Polygon) is used as the
intraday tradeable VIX proxy.

**yfinance intraday limit:** 5m data max 60 days; 1h data max 730 days. For full
Jan 2025+ intraday history, Polygon.io is required.

---

## 11. Timezone Bug (Fixed)

**Bug:** `_next_market_open()` returned 13:30 UTC for all posts. For Dec–Mar posts
(EST), actual market open is 14:30 UTC. T+5m targets pointed to the same bar as
entry, producing return = 0 for ~90% of intraday rows.

**Fix:** Anchor all intraday offsets from the **actual entry bar's real timestamp**
(first bar at or after the estimated open), not from the hardcoded 13:30 UTC.

**Coverage bug (fixed):** For the full Jan 2025 dataset, posts before Dec 2025
had `intra_entry_ts` in early 2025 which matched *all* Dec 2025+ 5m bars
(since all are "after" early 2025). Fixed by guarding: only compute intraday
returns if `intra_entry_ts >= s_5m.index[0]`.

---

## 12. Model File Naming Convention

```
models/saved/spy_direction_{period}.pkl          # baseline (full data)
models/saved/spy_direction_{period}_high_vix.pkl # VIX >= 20
models/saved/spy_direction_{period}_low_vix.pkl  # VIX < 20
models/saved/spy_direction_{period}_weighted.pkl # decay-weighted (half-life 180d)
```

Each pickle contains: `model`, `features`, `period`, `ret_col`, `label_method`,
`rolling_window`, `dead_zone_pct`, `n_train`, `pos_rate`, `cv_metrics`, `model_type`,
plus regime-specific metadata where applicable.
