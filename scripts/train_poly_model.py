"""train_poly_model.py — Train Polymarket session → signal correctness predictor.

Input:  D:/WhaleWatch_Data/poly_market_data.parquet
Output: models/saved/poly_direction_{period}.pkl          (baseline)
        models/saved/poly_direction_{period}_high_vix.pkl (VIX >= threshold)
        models/saved/poly_direction_{period}_low_vix.pkl  (VIX < threshold)
        models/saved/poly_direction_{period}_weighted.pkl (decay-weighted)

Uses XGBoost + isotonic calibration, TimeSeriesSplit CV.
Output: P(whale signal direction is correct) — label=1 means SPY moved in
        the direction predicted by the dominant Polymarket whale activity.

Usage:
    python scripts/train_poly_model.py
    python scripts/train_poly_model.py --min-events 1 --regime --vix-threshold 20
    python scripts/train_poly_model.py --decay-half 180
"""

import argparse
import logging
import pickle
from datetime import datetime, timezone
from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Train/OOS split cutoff
# ---------------------------------------------------------------------------
TRAIN_CUTOFF = pd.Timestamp("2026-02-28", tz="UTC")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("train_poly_model")
warnings.filterwarnings("ignore", category=UserWarning)

DATA_PATH   = Path("D:/WhaleWatch_Data/poly_market_data.parquet")
MODELS_DIR  = PROJECT_ROOT / "models" / "saved"

HOLDING_PERIODS = ["5m", "30m", "1h", "2h", "4h", "1d"]

# Dead-zone thresholds (moved here from builder so they can be varied without
# rebuilding the dataset — same values as before).
DEAD_ZONE: dict[str, float] = {
    "5m":  0.0010,   # 0.10%
    "30m": 0.0020,   # 0.20%
    "1h":  0.0030,   # 0.30%
    "2h":  0.0040,   # 0.40%
    "4h":  0.0060,   # 0.60%
    "1d":  0.0080,   # 0.80%
}

# Hyperparameter grid for --tune mode (27 candidates).
TUNE_GRID = [
    {"max_depth": d, "n_estimators": n, "learning_rate": lr}
    for d  in [2, 3, 4]
    for n  in [100, 200, 300]
    for lr in [0.03, 0.05, 0.10]
]

# ---------------------------------------------------------------------------
# Feature definitions — imported from shared module
# ---------------------------------------------------------------------------
from reasoner.layer2_predictor.poly_features import (
    STRENGTH_FEATURES,
    TOPIC_FEATURES,
    REGIME_FEATURES,
    SPY_CONTEXT_FEATURES,
    MARKET_QUALITY_FEATURES,
    MOMENTUM_FEATURES,
    ALL_DIRECTIONAL_FEATURES,
)

ALL_FEATURES = ALL_DIRECTIONAL_FEATURES


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _decay_weights(df: pd.DataFrame, half_life_days: int) -> np.ndarray:
    """Exponential decay weights: recent sessions get weight 1.0, older decay."""
    now       = df["session_time"].max()
    days_ago  = (now - df["session_time"]).dt.total_seconds() / 86400
    return np.exp(-np.log(2) * days_ago / half_life_days).values


def _cross_validate(
    X: pd.DataFrame,
    y: pd.Series,
    model_factory,
    sample_weight: np.ndarray | None = None,
) -> dict:
    """TimeSeriesSplit 5-fold CV. Returns mean AUC and accuracy."""
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import roc_auc_score, accuracy_score

    tscv    = TimeSeriesSplit(n_splits=5)
    aucs    = []
    accs    = []

    for train_idx, val_idx in tscv.split(X):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        w_tr        = sample_weight[train_idx] if sample_weight is not None else None

        if y_tr.nunique() < 2 or y_val.nunique() < 2:
            continue

        clf = model_factory()
        clf.fit(X_tr, y_tr, **({} if w_tr is None else {"sample_weight": w_tr}))

        probs = clf.predict_proba(X_val)[:, 1]
        preds = (probs >= 0.5).astype(int)

        try:
            aucs.append(roc_auc_score(y_val, probs))
            accs.append(accuracy_score(y_val, preds))
        except ValueError:
            pass

    if not aucs:
        return {"auc": 0.5, "accuracy": 0.5, "n_folds": 0}
    return {
        "auc":      round(float(np.mean(aucs)), 3),
        "accuracy": round(float(np.mean(accs)), 3),
        "n_folds":  len(aucs),
    }


def _tune_xgb_params(X: pd.DataFrame, y: pd.Series, n_folds: int = 5) -> dict:
    """Grid search over TUNE_GRID using TimeSeriesSplit, minimising mean Brier score."""
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import brier_score_loss
    from xgboost import XGBClassifier

    tscv        = TimeSeriesSplit(n_splits=n_folds)
    best_brier  = float("inf")
    best_params = TUNE_GRID[0]

    logger.info("  Tuning XGBoost: %d candidates × %d folds ...", len(TUNE_GRID), n_folds)

    for params in TUNE_GRID:
        fold_briers = []
        for train_idx, val_idx in tscv.split(X):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
            if y_tr.nunique() < 2:
                continue
            clf = XGBClassifier(
                **params, subsample=0.8, colsample_bytree=0.8,
                eval_metric="logloss", random_state=42, verbosity=0,
            )
            clf.fit(X_tr, y_tr)
            probs = clf.predict_proba(X_val)[:, 1]
            fold_briers.append(float(brier_score_loss(y_val, probs)))

        if not fold_briers:
            continue
        mean_b = float(np.mean(fold_briers))
        if mean_b < best_brier:
            best_brier  = mean_b
            best_params = params

    logger.info("  Best params: %s  (mean Brier=%.4f)", best_params, best_brier)
    return best_params


def _walk_forward(X: pd.DataFrame, y: pd.Series, n_folds: int = 5) -> pd.DataFrame:
    """Expanding-window walk-forward: train on 0..k, test on k+1.

    Diagnostic only — shows whether Brier / AUC is stable over time.
    """
    from sklearn.metrics import roc_auc_score, brier_score_loss
    from xgboost import XGBClassifier

    n         = len(y)
    block     = n // (n_folds + 1)
    rows      = []

    for k in range(1, n_folds + 1):
        tr_end  = k * block
        val_end = min(tr_end + block, n)

        X_tr, y_tr   = X.iloc[:tr_end], y.iloc[:tr_end]
        X_val, y_val = X.iloc[tr_end:val_end], y.iloc[tr_end:val_end]

        if y_tr.nunique() < 2 or y_val.nunique() < 2:
            continue

        clf = XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="logloss", random_state=42, verbosity=0,
        )
        clf.fit(X_tr, y_tr)
        probs = clf.predict_proba(X_val)[:, 1]

        rows.append({
            "fold":     k,
            "train_n":  tr_end,
            "val_n":    val_end - tr_end,
            "pos_rate": round(float(y_val.mean()), 4),
            "brier":    round(float(brier_score_loss(y_val, probs)), 4),
            "auc":      round(float(roc_auc_score(y_val, probs)), 4)
                        if y_val.nunique() > 1 else float("nan"),
        })

    return pd.DataFrame(rows) if rows else pd.DataFrame()


def _build_calibrated_model(
    X: pd.DataFrame,
    y: pd.Series,
    sample_weight: np.ndarray | None = None,
    xgb_params: dict | None = None,
):
    """Fit XGBoost with early stopping + probability calibration.

    Two-step process:
      1. Fit XGBoost with early stopping on a time-based train/val split
         (last 20% as validation). This prevents overfitting by selecting
         the optimal number of boosting rounds automatically.
      2. Calibrate the fitted model's probabilities using sigmoid/isotonic
         calibration on the full dataset (FrozenEstimator + 5-fold CV).

    scale_pos_weight accounts for class imbalance (especially relevant for
    regime sub-models which train on a fraction of the full dataset).
    """
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.frozen import FrozenEstimator
    from xgboost import XGBClassifier

    n_pos    = int(y.sum())
    n_neg    = len(y) - n_pos
    pos_rate = n_pos / max(len(y), 1)
    spw = min(n_neg / max(n_pos, 1), 5.0) if (pos_rate < 0.35 or pos_rate > 0.65) else 1.0

    defaults = dict(
        n_estimators=500, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, scale_pos_weight=spw,
        eval_metric="logloss", random_state=42, verbosity=0,
    )
    if xgb_params:
        defaults.update(xgb_params)
        defaults["n_estimators"] = max(defaults["n_estimators"], 500)

    base = XGBClassifier(**defaults)

    # Step 1: Fit with early stopping on time-based val split (last 20%)
    n = len(X)
    val_size = max(int(n * 0.2), 20)
    if n > val_size + 30:
        X_tr, X_val = X.iloc[:-val_size], X.iloc[-val_size:]
        y_tr, y_val = y.iloc[:-val_size], y.iloc[-val_size:]
        w_tr = sample_weight[:-val_size] if sample_weight is not None else None
        fit_kw = {} if w_tr is None else {"sample_weight": w_tr}
        base.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
                 verbose=False, **fit_kw)
        # best_iteration is set by early stopping; refit on full data with that count
        best_n = getattr(base, "best_iteration", defaults["n_estimators"]) + 1
        base = XGBClassifier(**{**defaults, "n_estimators": best_n})

    fit_kw_full = {} if sample_weight is None else {"sample_weight": sample_weight}
    base.fit(X, y, **fit_kw_full)

    # Step 2: Calibrate probabilities on full data (model already fitted)
    method = "isotonic" if len(y) >= 500 else "sigmoid"
    clf = CalibratedClassifierCV(FrozenEstimator(base), method=method, cv=5)
    clf.fit(X, y)
    return clf


def _save_model(
    model,
    features: list[str],
    period: str,
    cv_metrics: dict,
    n_train: int,
    pos_rate: float,
    ret_col: str,
    label_method: str,
    suffix: str = "",
    extra_meta: dict | None = None,
) -> Path:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"poly_direction_{period}{suffix}.pkl"
    out_path = MODELS_DIR / filename

    payload = {
        "model":        model,
        "features":     features,
        "period":       period,
        "ret_col":      ret_col,
        "label_method": label_method,
        "n_train":      n_train,
        "pos_rate":     round(pos_rate, 3),
        "cv_metrics":   cv_metrics,
        "model_type":   "xgboost_isotonic",
        "trained_at":   datetime.now(tz=timezone.utc).isoformat(),
        "source":       "polymarket_sessions",
    }
    if extra_meta:
        payload.update(extra_meta)

    with open(out_path, "wb") as f:
        pickle.dump(payload, f)

    logger.info(
        "  Saved %-40s  AUC=%.3f  acc=%.3f  n=%d  pos=%.1f%%",
        filename, cv_metrics["auc"], cv_metrics["accuracy"], n_train, pos_rate * 100,
    )
    return out_path


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def _oos_metrics_poly(model, X_oos: pd.DataFrame, y_oos: pd.Series) -> dict:
    """Evaluate a fitted model on the held-out OOS set."""
    from sklearn.metrics import roc_auc_score, accuracy_score, brier_score_loss
    if len(y_oos) == 0 or y_oos.nunique() < 2:
        return {"oos_n": int(len(y_oos)), "oos_note": "insufficient OOS data"}
    probs = model.predict_proba(X_oos)[:, 1]
    preds = (probs >= 0.5).astype(int)
    return {
        "oos_n":        int(len(y_oos)),
        "oos_pos_rate": round(float(y_oos.mean()), 3),
        "oos_auc":      round(float(roc_auc_score(y_oos, probs)), 3),
        "oos_accuracy": round(float(accuracy_score(y_oos, preds)), 3),
        "oos_brier":    round(float(brier_score_loss(y_oos, probs)), 3),
    }


def train_period(
    df_train: pd.DataFrame,
    df_oos: pd.DataFrame,
    period: str,
    train_regime: bool,
    vix_threshold: float,
    decay_half: int,
    tune: bool = False,
    walk_forward: bool = False,
) -> None:
    lbl_col = f"label_{period}"
    exc_col = f"excess_ret_{period}"
    ret_col = f"ret_{period}"

    if lbl_col not in df_train.columns:
        logger.warning("  %s: label column missing — skipping", period)
        return

    # Soft dead zone: instead of dropping rows with small moves, keep ALL
    # samples but down-weight those where |ret| < threshold. This preserves
    # 5-10x more training data while still emphasising large, unambiguous moves.
    #
    # Weight = min(|ret| / threshold, 1.0)  — linear ramp from 0 to 1
    # Samples at or above threshold get weight 1.0; smaller moves get
    # proportionally lower weight. No samples are discarded.
    dead_zone = DEAD_ZONE.get(period, 0.0)
    mask_valid = df_train[lbl_col].notna()
    if ret_col in df_train.columns:
        mask_valid = mask_valid & df_train[ret_col].notna()

    df_valid = df_train[mask_valid].copy()
    df_valid[lbl_col] = df_valid[lbl_col].astype(int)

    # Compute soft dead-zone weights
    if ret_col in df_valid.columns and dead_zone > 0:
        dz_weights = np.clip(df_valid[ret_col].abs().values / dead_zone, 0.0, 1.0)
        n_full_weight = int((dz_weights >= 1.0).sum())
        n_down_weighted = len(dz_weights) - n_full_weight
        logger.info(
            "  %s: n=%d  full_weight=%d  down_weighted=%d (dz=%.4f)",
            period, len(df_valid), n_full_weight, n_down_weighted, dead_zone,
        )
    else:
        dz_weights = np.ones(len(df_valid))
        logger.info("  %s: n=%d  (no dead-zone weighting)", period, len(df_valid))

    features = [f for f in ALL_FEATURES if f in df_valid.columns]
    X = df_valid[features]
    y = df_valid[lbl_col]

    if len(y) < 30:
        logger.warning("  %s: only %d valid rows — skipping (need >= 30)", period, len(y))
        return

    pos_rate = float(y.mean())
    logger.info("  %s: n_train=%d  signal_correct=%.1f%%", period, len(y), pos_rate * 100)

    from xgboost import XGBClassifier
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.model_selection import TimeSeriesSplit

    # model_factory is used inside _cross_validate where each fold trains on a
    # subset. Use sigmoid calibration (stable at small N) since fold sizes are
    # always smaller than the full training set. scale_pos_weight is set from
    # the full-fold class ratio as a reasonable approximation.
    n_pos_full  = int(y.sum())
    n_neg_full  = len(y) - n_pos_full
    pos_rate_full = n_pos_full / max(len(y), 1)
    spw_approx  = (
        min(n_neg_full / max(n_pos_full, 1), 5.0)
        if (pos_rate_full < 0.35 or pos_rate_full > 0.65) else 1.0
    )

    def model_factory():
        base = XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=spw_approx,
            eval_metric="logloss", random_state=42, verbosity=0,
        )
        # Use sigmoid in CV context: fold sizes are ~1/5 of training set,
        # making isotonic calibration unreliable within each fold.
        return CalibratedClassifierCV(base, method="sigmoid", cv=TimeSeriesSplit(n_splits=3))

    # Build OOS arrays (shared across all variants for this period)
    X_oos_p: pd.DataFrame = pd.DataFrame()
    y_oos_p: pd.Series    = pd.Series(dtype=int)
    if len(df_oos) > 0 and lbl_col in df_oos.columns:
        df_oos_v = df_oos[df_oos[lbl_col].notna()].copy()
        df_oos_v[lbl_col] = df_oos_v[lbl_col].astype(int)
        oos_feats = [f for f in features if f in df_oos_v.columns]
        if oos_feats and len(df_oos_v) > 0:
            X_oos_p = df_oos_v[oos_feats]
            y_oos_p = df_oos_v[lbl_col]
            logger.info("  %s: n_oos=%d  oos_pos_rate=%.1f%%",
                        period, len(y_oos_p), float(y_oos_p.mean()) * 100)

    # Walk-forward stability check (diagnostic)
    if walk_forward:
        wf_df = _walk_forward(X, y)
        if not wf_df.empty:
            logger.info("  Walk-forward stability (%s):\n%s", period, wf_df.to_string(index=False))

    # Optional hyperparameter tuning
    tuned_params = _tune_xgb_params(X, y) if tune else None

    # --- Baseline (with soft dead-zone weights) ---
    cv = _cross_validate(X, y, model_factory, sample_weight=dz_weights)
    model = _build_calibrated_model(X, y, sample_weight=dz_weights, xgb_params=tuned_params)
    oos_m = _oos_metrics_poly(model, X_oos_p, y_oos_p) if len(X_oos_p) > 0 else {}
    _save_model(
        model, features, period, cv, len(y), pos_rate, ret_col,
        label_method="signal_relative", suffix="",
        extra_meta={"train_cutoff": str(TRAIN_CUTOFF.date()), "oos_metrics": oos_m},
    )
    if oos_m:
        logger.info("  %s OOS → AUC=%.3f  acc=%.3f  n=%d",
                    period, oos_m.get("oos_auc", 0), oos_m.get("oos_accuracy", 0), oos_m.get("oos_n", 0))

    # --- Regime split ---
    if train_regime and "vix_level" in df_valid.columns:
        for regime_name, mask_fn in [
            ("high_vix", lambda d: d["vix_level"] >= vix_threshold),
            ("low_vix",  lambda d: d["vix_level"] <  vix_threshold),
        ]:
            mask   = mask_fn(df_valid)
            X_r    = X[mask]
            y_r    = y[mask]
            w_r    = dz_weights[mask.values]
            if len(y_r) < 30 or y_r.nunique() < 2:
                logger.info("  %s %s: %d rows — skip", period, regime_name, len(y_r))
                continue
            cv_r    = _cross_validate(X_r, y_r, model_factory, sample_weight=w_r)
            model_r = _build_calibrated_model(X_r, y_r, sample_weight=w_r)
            # OOS for regime sub-model
            oos_r: dict = {}
            if len(X_oos_p) > 0 and "vix_level" in df_oos.columns:
                df_oos_r = df_oos[mask_fn(df_oos)] if len(df_oos) > 0 else pd.DataFrame()
                if len(df_oos_r) > 0 and lbl_col in df_oos_r.columns:
                    df_oos_rv = df_oos_r[df_oos_r[lbl_col].notna()].copy()
                    df_oos_rv[lbl_col] = df_oos_rv[lbl_col].astype(int)
                    X_oos_r = df_oos_rv[[f for f in features if f in df_oos_rv.columns]]
                    y_oos_r = df_oos_rv[lbl_col]
                    oos_r   = _oos_metrics_poly(model_r, X_oos_r, y_oos_r)
            _save_model(
                model_r, features, period, cv_r, len(y_r), float(y_r.mean()),
                ret_col, label_method="signal_relative",
                suffix=f"_{regime_name}",
                extra_meta={"vix_threshold": vix_threshold, "regime": regime_name,
                            "train_cutoff": str(TRAIN_CUTOFF.date()), "oos_metrics": oos_r},
            )

    # --- Decay-weighted (combined with dead-zone weights) ---
    if decay_half > 0:
        decay_w  = _decay_weights(df_valid, decay_half)
        combined = dz_weights * decay_w
        cv_w    = _cross_validate(X, y, model_factory, sample_weight=combined)
        model_w = _build_calibrated_model(X, y, sample_weight=combined)
        oos_w   = _oos_metrics_poly(model_w, X_oos_p, y_oos_p) if len(X_oos_p) > 0 else {}
        _save_model(
            model_w, features, period, cv_w, len(y), pos_rate,
            ret_col, label_method="signal_relative",
            suffix="_weighted",
            extra_meta={"decay_half_life_days": decay_half,
                        "train_cutoff": str(TRAIN_CUTOFF.date()), "oos_metrics": oos_w},
        )


def run(
    min_events: int,
    train_regime: bool,
    vix_threshold: float,
    decay_half: int,
    tune: bool = False,
    walk_forward: bool = False,
) -> None:
    if not DATA_PATH.exists():
        logger.error("Dataset not found: %s\nRun build_poly_market_data.py first.", DATA_PATH)
        return

    df = pd.read_parquet(DATA_PATH)
    logger.info("Loaded %d sessions from %s", len(df), DATA_PATH)

    # Filter by minimum events per session
    if "n_events" in df.columns and min_events > 0:
        before = len(df)
        df     = df[df["n_events"] >= min_events]
        logger.info("Filtered to n_events >= %d: %d → %d sessions", min_events, before, len(df))

    # Ensure chronological order (required by TimeSeriesSplit)
    if "session_time" in df.columns:
        df = df.sort_values("session_time").reset_index(drop=True)
        logger.info(
            "Date range: %s → %s",
            df["session_time"].min().date(),
            df["session_time"].max().date(),
        )

    # Train / OOS split
    if "session_time" in df.columns:
        df_train = df[df["session_time"] <= TRAIN_CUTOFF].copy()
        df_oos   = df[df["session_time"] >  TRAIN_CUTOFF].copy()
        logger.info(
            "Train/OOS split at %s  →  train=%d sessions  OOS=%d sessions",
            TRAIN_CUTOFF.date(), len(df_train), len(df_oos),
        )
    else:
        logger.warning("No 'session_time' column — using all data for training, no OOS split")
        df_train = df.copy()
        df_oos   = df.iloc[0:0].copy()

    logger.info("\nFeature set (%d features):", len(ALL_FEATURES))
    logger.info("  Strength:  %s", STRENGTH_FEATURES)
    logger.info("  Topics:    %s", TOPIC_FEATURES)
    logger.info("  Regime:    %s", REGIME_FEATURES)
    logger.info("  SPY ctx:   %s", SPY_CONTEXT_FEATURES)
    logger.info("  Mkt qual:  %s", MARKET_QUALITY_FEATURES)
    logger.info("  Momentum:  %s", MOMENTUM_FEATURES)

    logger.info("\n--- Training ---")
    for period in HOLDING_PERIODS:
        logger.info("\nPeriod: %s", period)
        train_period(df_train, df_oos, period, train_regime, vix_threshold, decay_half, tune, walk_forward)

    # -----------------------------------------------------------------------
    # Quality-weighted ensemble: only include models with OOS AUC > 0.52
    # (above chance). Weight each model's vote by its OOS AUC so stronger
    # models have more influence on the direction decision.
    # -----------------------------------------------------------------------
    MIN_ENSEMBLE_AUC = 0.52
    logger.info("\n--- Building quality-weighted ensemble (min AUC=%.2f) ---", MIN_ENSEMBLE_AUC)
    ensemble_models = {}
    quality_weights = {}
    for period in HOLDING_PERIODS:
        path = MODELS_DIR / f"poly_direction_{period}.pkl"
        if path.exists():
            try:
                with open(path, "rb") as fh:
                    payload = pickle.load(fh)
                oos = payload.get("oos_metrics") or {}
                oos_auc = oos.get("oos_auc", 0.5)
                if oos_auc >= MIN_ENSEMBLE_AUC:
                    ensemble_models[period] = payload
                    quality_weights[period] = oos_auc
                    logger.info("  %-4s  OOS AUC=%.3f  → included (weight=%.3f)",
                                period, oos_auc, oos_auc)
                else:
                    logger.info("  %-4s  OOS AUC=%.3f  → excluded (below %.2f)",
                                period, oos_auc, MIN_ENSEMBLE_AUC)
            except Exception:
                pass

    if len(ensemble_models) >= 1:
        period_map = {"5m": 5, "30m": 30, "1h": 60, "2h": 120, "4h": 240, "1d": 1440}
        ensemble_payload = {
            "model_type":      "ensemble_quality_weighted",
            "periods":         list(ensemble_models.keys()),
            "models":          ensemble_models,
            "quality_weights": quality_weights,
            "period_map":      period_map,
            "features":        ALL_FEATURES,
            "trained_at":      datetime.now(tz=timezone.utc).isoformat(),
            "train_cutoff":    str(TRAIN_CUTOFF.date()),
        }
        ens_path = MODELS_DIR / "poly_direction_ensemble.pkl"
        with open(ens_path, "wb") as f:
            pickle.dump(ensemble_payload, f)
        logger.info("  Saved ensemble (%d/%d periods) → %s",
                    len(ensemble_models), len(HOLDING_PERIODS), ens_path.name)
    else:
        logger.warning("  Ensemble skipped: no models met AUC threshold %.2f", MIN_ENSEMBLE_AUC)

    # Summary table
    logger.info("\n--- Model summary (train cutoff: %s) ---", TRAIN_CUTOFF.date())
    model_files = sorted(MODELS_DIR.glob("poly_direction_*.pkl"))
    results = []
    for f in model_files:
        try:
            with open(f, "rb") as fh:
                m = pickle.load(fh)
            if m.get("model_type") == "ensemble_best_confidence":
                continue  # skip ensemble in per-period table
            oos = m.get("oos_metrics") or {}
            results.append({
                "file":     f.name,
                "period":   m.get("period", ""),
                "cv_auc":   m["cv_metrics"]["auc"],
                "oos_auc":  oos.get("oos_auc", "—"),
                "n_train":  m.get("n_train", 0),
                "n_oos":    oos.get("oos_n", "—"),
                "pos_pct":  round(m.get("pos_rate", 0.5) * 100, 1),
            })
        except Exception:
            pass

    if results:
        summary = pd.DataFrame(results).sort_values(["period", "file"])
        logger.info("\n%s", summary.to_string(index=False))

    logger.info("\nModels saved to %s", MODELS_DIR)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Polymarket L2 SPY direction predictor")
    parser.add_argument("--min-events", type=int, default=1,
                        help="Min anomaly events per session to include in training (default: 1)")
    parser.add_argument("--regime", action="store_true",
                        help="Train separate high/low VIX regime models")
    parser.add_argument("--vix-threshold", type=float, default=20.0,
                        help="VIX threshold for high/low regime split (default: 20)")
    parser.add_argument("--decay-half", type=int, default=0,
                        help="Exponential decay half-life in days (0 = off; try 180)")
    parser.add_argument("--tune", action="store_true",
                        help="Grid-search XGBoost hyperparams before final fit (slower)")
    parser.add_argument("--walk-forward", action="store_true",
                        help="Print expanding-window fold stability table (diagnostic only)")
    args = parser.parse_args()

    run(
        min_events=args.min_events,
        train_regime=args.regime,
        vix_threshold=args.vix_threshold,
        decay_half=args.decay_half,
        tune=args.tune,
        walk_forward=args.walk_forward,
    )
