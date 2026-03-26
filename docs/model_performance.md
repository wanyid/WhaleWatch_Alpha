# L2 Model Performance Summary

Last updated: 2026-03-26

## Training data

- **Posts:** 2,207 keyword-bearing Trump Truth Social posts (Jan 21, 2025 → Mar 25, 2026)
- **Price data:** Polygon.io 5m/1h for SPY, QQQ, VIXY; yfinance 1d for VIX
- **Label method:** Excess return vs 20-post rolling baseline, dead-zone filtered
- **CV method:** TimeSeriesSplit (5 folds, no lookahead)

---

## Baseline models (full dataset, all regimes)

| Period | AUC | Accuracy | n_train | pos_rate | File |
|--------|-----|----------|---------|----------|------|
| 5m  | 0.437 | 0.470 | 360 | 50.0% | spy_direction_5m.pkl |
| 30m | 0.460 | 0.482 | 665 | 46.6% | spy_direction_30m.pkl |
| 1h  | 0.483 | 0.498 | 569 | 46.2% | spy_direction_1h.pkl |
| 2h  | 0.499 | 0.490 | 744 | 51.3% | spy_direction_2h.pkl |
| 4h  | 0.538 | 0.552 | 557 | 49.4% | spy_direction_4h.pkl |
| **1d** | **0.638** | **0.606** | **648** | **46.8%** | **spy_direction_1d.pkl** ✓ |

---

## Regime-split models (VIX threshold = 20)

### High VIX (≥ 20) — current market regime as of 2026-03-26

| Period | AUC | n_train | Vs baseline | File |
|--------|-----|---------|-------------|------|
| 5m  | 0.454 | 224 | +2pts | spy_direction_5m_high_vix.pkl |
| 30m | 0.507 | 351 | +5pts | spy_direction_30m_high_vix.pkl |
| 1h  | 0.406 | 277 | -8pts | spy_direction_1h_high_vix.pkl |
| **2h** | **0.553** | **338** | **+5pts** | **spy_direction_2h_high_vix.pkl** ✓ |
| 4h  | 0.478 | 305 | -6pts | spy_direction_4h_high_vix.pkl |
| **1d** | **0.644** | **379** | **+1pt** | **spy_direction_1d_high_vix.pkl** ✓ |

### Low VIX (< 20) — calm market regime

| Period | AUC | n_train | Vs baseline | File |
|--------|-----|---------|-------------|------|
| **5m**  | **0.565** | **136** | **+13pts** | **spy_direction_5m_low_vix.pkl** ✓ |
| **30m** | **0.615** | **314** | **+16pts** | **spy_direction_30m_low_vix.pkl** ✓ |
| **1h**  | **0.550** | **292** | **+7pts**  | **spy_direction_1h_low_vix.pkl** ✓ |
| 2h  | 0.525 | 406 | +3pts | spy_direction_2h_low_vix.pkl |
| **4h**  | **0.549** | **252** | **+1pt**   | **spy_direction_4h_low_vix.pkl** |
| 1d  | 0.606 | 269 | -3pts | spy_direction_1d_low_vix.pkl |

---

## Decay-weighted models (180-day half-life)

| Period | AUC | Vs baseline |
|--------|-----|-------------|
| 5m  | 0.402 | -4pts |
| 30m | 0.466 | +1pt |
| 1h  | 0.497 | +1pt |
| 2h  | 0.505 | +1pt |
| **4h** | **0.553** | **+2pts** |
| 1d  | 0.597 | -4pts |

---

## Recommended model routing at inference time

```
Current VIX → model selection:

VIX >= 20 (elevated/crisis):
  1d signal  → spy_direction_1d.pkl         (AUC 0.638, primary)
  2h signal  → spy_direction_2h_high_vix.pkl (AUC 0.553, secondary)
  Others     → SKIP (AUC < 0.51, unreliable)

VIX < 20 (calm):
  5m  signal → spy_direction_5m_low_vix.pkl  (AUC 0.565)
  30m signal → spy_direction_30m_low_vix.pkl (AUC 0.615, best intraday)
  1h  signal → spy_direction_1h_low_vix.pkl  (AUC 0.550)
  2h  signal → spy_direction_2h_low_vix.pkl  (AUC 0.525)
  4h  signal → spy_direction_4h_low_vix.pkl  (AUC 0.549)
  1d  signal → spy_direction_1d.pkl          (AUC 0.638, full dataset preferred)
```

Minimum confidence threshold to act: **0.60** (set in config/settings.yaml)

---

## Key features (top by importance, 1d model)

1. `vix_level` — current VIX absolute level
2. `day_of_week` — Mon–Fri market dynamics differ significantly
3. `vix_percentile` — VIX relative to 252-day history
4. `vixy_level` — VIXY ETF price (intraday VIX proxy)
5. `is_market_hours` — whether post was during regular trading hours
6. `hour_of_day` — time of day of post
7. `has_energy` — energy/oil/gas keyword present
8. `has_deal` — deal/agreement/treaty keyword present

---

## Notes

- AUC > 0.60 is considered meaningful signal for financial ML
- AUC < 0.51 is below random and should not be traded
- All intraday models become unreliable in high-VIX crisis periods
- Models will improve as more Polygon data accumulates (currently 14 months)
- Retrain recommended after every 3 months or major regime shift
