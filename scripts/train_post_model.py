"""train_post_model.py — Train SPY direction predictor on Trump post → market reaction data.

Labels are based on **excess returns** (actual return minus rolling baseline) to isolate
alpha from beta market drift.  A dead-zone threshold removes rows where the move is too
small to plausibly be post-driven.

For each holding period (5m, 30m, 1h, 2h, 4h, 1d):
  - rolling_mean = 20-post trailing average of spy_ret_X (shift-1, no lookahead)
  - excess_return = spy_ret_X - rolling_mean
  - Label: 1 (BUY) if excess_return > +threshold, 0 (SHORT) if < -threshold
  - Rows in the dead zone (|excess_return| <= threshold) are dropped
  - Train XGBoost binary classifier with isotonic calibration
  - TimeSeriesSplit CV to avoid lookahead bias
  - Report AUC, Brier score, calibration table, feature importance
  - Save best model (by AUC) to models/saved/spy_direction_{period}.pkl

Output: confidence = P(SPY excess return is positive after post)
HOLD decision is made externally by the executor using a confidence threshold.

Usage:
    python scripts/train_post_model.py
    python scripts/train_post_model.py --period 2h          # single period
    python scripts/train_post_model.py --cv-folds 10
    python scripts/train_post_model.py --min-keywords 1     # only posts with keywords
    python scripts/train_post_model.py --report-only        # CV only, don't save
    python scripts/train_post_model.py --rolling-window 30  # rolling baseline window
"""

import argparse
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Train/OOS split cutoff
# ---------------------------------------------------------------------------
# Data up to and including this date is used for training.
# Data after this date is held out as the out-of-sample (OOS) test set.
TRAIN_CUTOFF = pd.Timestamp("2026-02-28")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("train_post_model")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_PATH  = Path("D:/WhaleWatch_Data/post_market_data.parquet")
MODELS_DIR = Path("models/saved")
FI_DIR     = Path("D:/WhaleWatch_Data")

# ---------------------------------------------------------------------------
# Feature set
# ---------------------------------------------------------------------------
KEYWORD_FEATURES = [
    "has_tariff", "has_deal", "has_china", "has_fed",
    "has_energy", "has_geopolitical", "has_market",
]
POST_FEATURES = [
    "favourites_count", "reblogs_count", "engagement",
    "caps_ratio", "content_length", "keyword_count",
]
TEMPORAL_FEATURES = [
    "hour_of_day", "day_of_week",
    "is_market_hours",   # 1 = post during regular trading hours, 0 = pre/after/overnight
    "is_premarket",
]
MARKET_FEATURES = [
    "vix_level", "vix_percentile",
    "vixy_level",   # tradeable VIX proxy (available after Polygon pull; 0 if not yet pulled)
]

ALL_FEATURES = KEYWORD_FEATURES + POST_FEATURES + TEMPORAL_FEATURES + MARKET_FEATURES

# Holding periods → raw SPY return column
HOLDING_PERIODS = {
    "5m":  "spy_ret_5m",
    "30m": "spy_ret_30m",
    "1h":  "spy_ret_1h",
    "2h":  "spy_ret_2h",
    "4h":  "spy_ret_4h",
    "1d":  "spy_ret_1d",
}

# Dead-zone thresholds (fraction, not percent): moves smaller than this are dropped.
# Scaled roughly by sqrt(time) to match expected noise magnitude.
DEAD_ZONE = {
    "5m":  0.0010,   # 0.10%
    "30m": 0.0020,   # 0.20%
    "1h":  0.0030,   # 0.30%
    "2h":  0.0040,   # 0.40%
    "4h":  0.0060,   # 0.60%
    "1d":  0.0080,   # 0.80%
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_data(min_keywords: int = 0) -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"post_market_data.parquet not found at {DATA_PATH}. "
            "Run scripts/build_post_market_data.py first."
        )
    df = pd.read_parquet(DATA_PATH)
    logger.info("Loaded %d rows from %s", len(df), DATA_PATH)

    if min_keywords > 0:
        before = len(df)
        df = df[df["keyword_count"] >= min_keywords].copy()
        logger.info(
            "Keyword filter (>= %d): %d → %d rows (%.0f%% retained)",
            min_keywords, before, len(df), len(df) / before * 100,
        )

    # Must be chronological for TimeSeriesSplit and rolling baselines
    if "posted_at" in df.columns:
        df = df.sort_values("posted_at").reset_index(drop=True)

    return df


# ---------------------------------------------------------------------------
# Alpha isolation: rolling-mean adjustment + dead-zone filter
# ---------------------------------------------------------------------------

def _compute_excess_returns(df: pd.DataFrame, rolling_window: int = 20) -> pd.DataFrame:
    """Add excess_ret_X columns for each holding period.

    excess_ret = actual_ret - rolling_mean(actual_ret, window, shift-1)

    shift(1) ensures we only use past observations — no lookahead.
    min_periods=10 prevents NaN-dropping too many early rows.
    """
    df = df.copy()
    for period, ret_col in HOLDING_PERIODS.items():
        if ret_col not in df.columns:
            continue
        rolling_baseline = (
            df[ret_col]
            .shift(1)
            .rolling(window=rolling_window, min_periods=max(5, rolling_window // 4))
            .mean()
        )
        df[f"excess_{ret_col}"] = df[ret_col] - rolling_baseline
        df[f"baseline_{ret_col}"] = rolling_baseline
    return df


def _build_label(
    df: pd.DataFrame,
    period: str,
    ret_col: str,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Return (X, y, df_valid) after applying excess-return labeling + dead-zone filter.

    y = 1 (BUY)   if excess_ret > +threshold
    y = 0 (SHORT) if excess_ret < -threshold
    rows with |excess_ret| <= threshold are dropped (dead zone)
    """
    excess_col = f"excess_{ret_col}"
    threshold  = DEAD_ZONE[period]

    if excess_col not in df.columns:
        raise ValueError(f"excess column {excess_col} not found — call _compute_excess_returns first")

    # Drop rows where excess return is NaN (rolling window not yet full) or in dead zone
    mask_valid   = df[excess_col].notna()
    mask_buy     = df[excess_col] > +threshold
    mask_short   = df[excess_col] < -threshold
    mask_signal  = mask_valid & (mask_buy | mask_short)

    df_valid = df[mask_signal].copy()

    missing = [f for f in ALL_FEATURES if f not in df_valid.columns]
    if missing:
        logger.warning("Missing features (zero-filled): %s", missing)
        for f in missing:
            df_valid[f] = 0.0

    X = df_valid[ALL_FEATURES].fillna(0).values.astype(np.float32)
    y = mask_buy[mask_signal].astype(int).values

    return X, y, df_valid


# ---------------------------------------------------------------------------
# Model building
# ---------------------------------------------------------------------------

def _build_pipeline(model_type: str = "xgboost"):
    from sklearn.pipeline import Pipeline

    if model_type == "xgboost":
        try:
            from xgboost import XGBClassifier
            clf = XGBClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric="logloss",
                random_state=42,
                verbosity=0,
                use_label_encoder=False,
            )
        except ImportError:
            logger.warning("xgboost not installed — falling back to logistic regression")
            model_type = "logistic"

    if model_type == "logistic":
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(
            C=1.0, class_weight="balanced", max_iter=1000, random_state=42
        )

    return Pipeline([("scaler", StandardScaler()), ("clf", clf)])


def _build_calibrated_model(X, y, model_type: str = "xgboost", sample_weight=None):
    pipe   = _build_pipeline(model_type)
    method = "isotonic" if len(y) >= 50 else "sigmoid"
    # Use TimeSeriesSplit so calibration folds respect time order — avoids
    # future data leaking into earlier calibration folds.
    cal    = CalibratedClassifierCV(pipe, method=method, cv=TimeSeriesSplit(n_splits=3))
    fit_kw = {"clf__sample_weight": sample_weight} if sample_weight is not None else {}
    cal.fit(X, y, **fit_kw)
    return cal


def _oos_metrics(model, X_oos: np.ndarray, y_oos: np.ndarray) -> dict:
    """Evaluate a fitted model on the held-out OOS set."""
    if len(y_oos) == 0 or len(np.unique(y_oos)) < 2:
        return {"oos_n": len(y_oos), "oos_note": "insufficient OOS data"}
    y_prob = model.predict_proba(X_oos)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        "oos_n":         int(len(y_oos)),
        "oos_pos_rate":  round(float(y_oos.mean()), 4),
        "oos_auc_roc":   round(float(roc_auc_score(y_oos, y_prob)), 4),
        "oos_accuracy":  round(float(accuracy_score(y_oos, y_pred)), 4),
        "oos_brier":     round(float(brier_score_loss(y_oos, y_prob)), 4),
        "oos_log_loss":  round(float(log_loss(y_oos, y_prob)), 4),
    }


def _decay_weights(df_valid: pd.DataFrame, decay_half_days: int) -> np.ndarray:
    """Exponential decay weights — most recent post has weight=1, halves every decay_half_days."""
    ref = df_valid["posted_at"].max()
    days_ago = (ref - df_valid["posted_at"]).dt.days.values
    return np.exp(-np.log(2) * days_ago / decay_half_days).astype(np.float32)


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------

def _cross_validate(
    X, y, n_folds: int, model_type: str = "xgboost", sample_weight=None
) -> dict:
    tscv   = TimeSeriesSplit(n_splits=n_folds)
    y_prob = np.full(len(y), np.nan, dtype=float)

    for train_idx, val_idx in tscv.split(X):
        if len(np.unique(y[train_idx])) < 2:
            continue
        pipe = _build_pipeline(model_type)
        fit_kw = {}
        if sample_weight is not None:
            fit_kw = {"clf__sample_weight": sample_weight[train_idx]}
        pipe.fit(X[train_idx], y[train_idx], **fit_kw)
        y_prob[val_idx] = pipe.predict_proba(X[val_idx])[:, 1]

    val_mask = ~np.isnan(y_prob)
    if val_mask.sum() == 0:
        return {"error": "no validation predictions"}

    y_pred = (y_prob[val_mask] >= 0.5).astype(int)
    return {
        "n_samples":   int(val_mask.sum()),
        "pos_rate":    round(float(y[val_mask].mean()), 4),
        "cv_folds":    n_folds,
        "cv_type":     "TimeSeriesSplit",
        "model":       model_type,
        "accuracy":    round(float(accuracy_score(y[val_mask], y_pred)), 4),
        "auc_roc":     round(float(roc_auc_score(y[val_mask], y_prob[val_mask])), 4)
                       if len(np.unique(y[val_mask])) > 1 else 0.0,
        "brier_score": round(float(brier_score_loss(y[val_mask], y_prob[val_mask])), 4),
        "log_loss":    round(float(log_loss(y[val_mask], y_prob[val_mask])), 4),
    }


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

def _calibration_report(model, X, y, n_bins: int = 5) -> pd.DataFrame:
    y_prob = model.predict_proba(X)[:, 1]
    try:
        frac_pos, mean_pred = calibration_curve(y, y_prob, n_bins=n_bins, strategy="quantile")
    except ValueError:
        return pd.DataFrame()
    return pd.DataFrame({
        "predicted_prob":  np.round(mean_pred, 4).tolist(),
        "actual_win_rate": np.round(frac_pos, 4).tolist(),
        "calibration_err": np.round(frac_pos - mean_pred, 4).tolist(),
    })


def _feature_importance(model) -> pd.DataFrame | None:
    try:
        importances = []
        for est in model.calibrated_classifiers_:
            inner = est.estimator.named_steps["clf"]
            if hasattr(inner, "feature_importances_"):
                importances.append(inner.feature_importances_)
            elif hasattr(inner, "coef_"):
                importances.append(np.abs(inner.coef_[0]))
        if importances:
            avg = np.mean(importances, axis=0)
            return (
                pd.DataFrame({"feature": ALL_FEATURES, "importance": avg})
                .sort_values("importance", ascending=False)
                .reset_index(drop=True)
            )
    except Exception as exc:
        logger.debug("Could not extract feature importances: %s", exc)
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _save_model(
    cal_model, period: str, ret_col: str, n: int, n_pos: int,
    metrics: dict, rolling_window: int, model_type: str,
    suffix: str = "", extra_meta: dict | None = None,
    oos_metrics: dict | None = None,
) -> None:
    model_path = MODELS_DIR / f"spy_direction_{period}{suffix}.pkl"
    payload = {
        "model":          cal_model,
        "features":       ALL_FEATURES,
        "period":         period,
        "ret_col":        ret_col,
        "label_method":   "excess_return",
        "rolling_window": rolling_window,
        "dead_zone_pct":  DEAD_ZONE[period] * 100,
        "n_train":        n,
        "train_cutoff":   str(TRAIN_CUTOFF.date()),
        "pos_rate":       round(n_pos / n, 4) if n > 0 else 0.0,
        "cv_metrics":     metrics,
        "oos_metrics":    oos_metrics or {},
        "model_type":     model_type,
        **(extra_meta or {}),
    }
    with open(model_path, "wb") as f:
        pickle.dump(payload, f)
    logger.info("Model saved → %s", model_path)


def run(
    periods: list[str] | None = None,
    n_folds: int = 5,
    min_keywords: int = 0,
    model_type: str = "xgboost",
    rolling_window: int = 20,
    report_only: bool = False,
    regime: bool = False,
    vix_threshold: float = 20.0,
    decay_half_days: int = 0,
) -> None:
    """
    regime=True       → also train high-VIX / low-VIX split models
    decay_half_days>0 → also train exponential-decay-weighted model
    Both can be combined with the baseline run.
    """
    df_raw = _load_data(min_keywords=min_keywords)
    # Compute excess returns on the full dataset so rolling baselines for OOS
    # rows are anchored on real prior observations (no leakage — shift-1 used).
    df     = _compute_excess_returns(df_raw, rolling_window=rolling_window)

    # Train / OOS split
    ts_col = "posted_at" if "posted_at" in df.columns else None
    if ts_col:
        df_train = df[df[ts_col] <= TRAIN_CUTOFF].copy()
        df_oos   = df[df[ts_col] >  TRAIN_CUTOFF].copy()
        logger.info(
            "Train/OOS split at %s  →  train=%d rows  OOS=%d rows",
            TRAIN_CUTOFF.date(), len(df_train), len(df_oos),
        )
    else:
        logger.warning("No 'posted_at' column — using all data for training, no OOS split")
        df_train = df.copy()
        df_oos   = df.iloc[0:0].copy()   # empty

    # Log baseline stats for transparency
    logger.info(
        "Rolling baseline window: %d posts  dead-zone thresholds: %s",
        rolling_window,
        {k: f"{v*100:.2f}%" for k, v in DEAD_ZONE.items()},
    )

    target_periods = periods if periods else list(HOLDING_PERIODS.keys())
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    summary_rows = []

    for period in target_periods:
        ret_col = HOLDING_PERIODS.get(period)
        if ret_col is None or ret_col not in df_train.columns:
            logger.warning("Unknown/missing period '%s' — skipping", period)
            continue

        try:
            X, y, df_valid = _build_label(df_train, period, ret_col)
        except Exception as exc:
            logger.error("Label build failed for %s: %s", period, exc)
            continue

        n       = len(y)
        n_pos   = int(y.sum())
        n_total = int(df_train[ret_col].notna().sum())
        dropped = n_total - n

        logger.info(
            "Period %-4s  train: raw=%-4d  after_dead_zone=%-4d (dropped %d=%.0f%%)  "
            "pos_rate=%.1f%%  (BUY=%-3d  SHORT=%-3d)",
            period, n_total, n, dropped, dropped / max(n_total, 1) * 100,
            n_pos / n * 100 if n > 0 else 0, n_pos, n - n_pos,
        )

        if n < 2 * n_folds:
            logger.warning("Too few samples (%d) for %d-fold CV — skipping %s", n, n_folds, period)
            continue

        metrics = _cross_validate(X, y, n_folds, model_type)

        print(f"\n{'=' * 65}")
        print(f"  Period: {period}  |  dead_zone=±{DEAD_ZONE[period]*100:.2f}%  |  train n={n}")
        print(f"{'=' * 65}")
        for k, v in metrics.items():
            print(f"  {k:25s}: {v}")

        # Build OOS labels using the same dead-zone rules
        oos_m: dict = {}
        if len(df_oos) > 0 and ret_col in df_oos.columns:
            try:
                X_oos, y_oos, _ = _build_label(df_oos, period, ret_col)
                logger.info("  OOS: %d rows after dead-zone filter", len(y_oos))
            except Exception:
                X_oos, y_oos = np.empty((0, len(ALL_FEATURES))), np.array([])

        summary_rows.append({
            "period":         period,
            "n_train":        n,
            "pct_kept":       round(n / max(n_total, 1) * 100, 1),
            **{k: v for k, v in metrics.items()
               if k not in ("cv_folds", "cv_type", "model", "error", "n_samples")},
        })

        if report_only:
            continue

        # Fit final calibrated model on training data only
        logger.info("Fitting final calibrated model for %s ...", period)
        try:
            cal_model = _build_calibrated_model(X, y, model_type)
        except Exception as exc:
            logger.error("Failed to fit calibrated model for %s: %s", period, exc)
            continue

        # OOS evaluation
        if len(df_oos) > 0:
            oos_m = _oos_metrics(cal_model, X_oos, y_oos)
            print(f"\n  OOS evaluation (after {TRAIN_CUTOFF.date()}):")
            for k, v in oos_m.items():
                print(f"    {k:25s}: {v}")

        # In-sample calibration sanity check
        if n >= 20:
            cal_df = _calibration_report(cal_model, X, y)
            if not cal_df.empty:
                print(f"\n  In-sample calibration (sanity check):")
                for _, row in cal_df.iterrows():
                    print(
                        f"    pred={row['predicted_prob']:.2f}  "
                        f"actual={row['actual_win_rate']:.2f}  "
                        f"err={row['calibration_err']:+.2f}"
                    )

        # Feature importance
        fi = _feature_importance(cal_model)
        if fi is not None:
            fi_path = FI_DIR / f"feature_importance_{period}.csv"
            fi.to_csv(fi_path, index=False)
            print(f"\n  Top features ({period}):")
            for _, row in fi.head(8).iterrows():
                print(f"    {row['feature']:25s}  {row['importance']:.4f}")
            logger.info("Feature importance saved → %s", fi_path)

        # Save baseline model
        _save_model(cal_model, period, ret_col, n, n_pos,
                    metrics, rolling_window, model_type, suffix="",
                    oos_metrics=oos_m)

        # ---- Option B: regime-split models ----
        if regime:
            for regime_name, vix_mask_fn in [
                ("high_vix", lambda d: d["vix_level"] >= vix_threshold),
                ("low_vix",  lambda d: d["vix_level"] <  vix_threshold),
            ]:
                r_mask    = vix_mask_fn(df_valid)
                X_r, y_r  = X[r_mask], y[r_mask]
                n_r       = len(y_r)
                if n_r < 2 * n_folds or len(np.unique(y_r)) < 2:
                    logger.warning(
                        "  Regime %s/%s: only %d samples — skipping", period, regime_name, n_r
                    )
                    continue
                m_r = _cross_validate(X_r, y_r, n_folds, model_type)
                print(f"\n  [Regime: {regime_name} VIX{'>='+str(vix_threshold) if 'high' in regime_name else '<'+str(vix_threshold)}]"
                      f"  n={n_r}  AUC={m_r.get('auc_roc','?')}")
                cal_r = _build_calibrated_model(X_r, y_r, model_type)
                # OOS for regime sub-model
                oos_r: dict = {}
                if len(df_oos) > 0 and len(X_oos) > 0:
                    oos_mask = vix_mask_fn(df_oos) if "vix_level" in df_oos.columns else pd.Series(True, index=df_oos.index)
                    # Re-derive OOS labels for this regime slice from df_oos filtered
                    df_oos_r = df_oos[oos_mask]
                    if len(df_oos_r) > 0:
                        try:
                            X_oos_r, y_oos_r, _ = _build_label(df_oos_r, period, ret_col)
                            oos_r = _oos_metrics(cal_r, X_oos_r, y_oos_r)
                        except Exception:
                            pass
                _save_model(cal_r, period, ret_col, n_r, int(y_r.sum()),
                            m_r, rolling_window, model_type,
                            suffix=f"_{regime_name}",
                            extra_meta={"vix_threshold": vix_threshold,
                                        "regime": regime_name},
                            oos_metrics=oos_r)

        # ---- Option C: decay-weighted model ----
        if decay_half_days > 0:
            w = _decay_weights(df_valid, decay_half_days)
            m_w = _cross_validate(X, y, n_folds, model_type, sample_weight=w)
            print(f"\n  [Decay-weighted half_life={decay_half_days}d]"
                  f"  AUC={m_w.get('auc_roc','?')}")
            cal_w = _build_calibrated_model(X, y, model_type, sample_weight=w)
            _save_model(cal_w, period, ret_col, n, n_pos,
                        m_w, rolling_window, model_type,
                        suffix="_weighted",
                        extra_meta={"decay_half_days": decay_half_days},
                        oos_metrics=_oos_metrics(cal_w, X_oos, y_oos) if len(df_oos) > 0 and len(X_oos) > 0 else {})

    # Summary table
    if summary_rows:
        print(f"\n{'=' * 65}")
        print(f"SUMMARY — train cutoff: {TRAIN_CUTOFF.date()}  |  OOS: after that date")
        print(f"{'=' * 65}")
        summary_df = pd.DataFrame(summary_rows)
        print(summary_df.to_string(index=False))
        print()

        if "auc_roc" in summary_df.columns and not summary_df["auc_roc"].isna().all():
            best = summary_df.loc[summary_df["auc_roc"].idxmax()]
            logger.info(
                "Best period: %s  AUC=%.4f  accuracy=%.4f  n_train=%d (%.0f%% of data kept)",
                best["period"], best["auc_roc"], best.get("accuracy", float("nan")),
                best["n_train"], best["pct_kept"],
            )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train alpha-adjusted SPY direction predictor from Trump posts"
    )
    parser.add_argument(
        "--period",
        choices=list(HOLDING_PERIODS.keys()),
        default=None,
        help="Single holding period to train (default: all)",
    )
    parser.add_argument(
        "--cv-folds", type=int, default=5,
        help="Number of TimeSeriesSplit folds (default 5)",
    )
    parser.add_argument(
        "--min-keywords", type=int, default=0,
        help="Only train on posts with >= N keyword matches (default 0 = all posts)",
    )
    parser.add_argument(
        "--model", choices=["xgboost", "logistic"], default="xgboost",
        help="Model architecture (default: xgboost)",
    )
    parser.add_argument(
        "--rolling-window", type=int, default=20,
        help="Number of prior posts for rolling baseline (default 20)",
    )
    parser.add_argument(
        "--report-only", action="store_true",
        help="Print CV report without fitting or saving models",
    )
    parser.add_argument(
        "--regime", action="store_true",
        help="Also train high-VIX / low-VIX regime-split models",
    )
    parser.add_argument(
        "--vix-threshold", type=float, default=20.0,
        help="VIX split threshold for --regime mode (default 20)",
    )
    parser.add_argument(
        "--decay-half", type=int, default=0,
        help="Exponential decay half-life in days for weighted model (0=off)",
    )
    args = parser.parse_args()

    periods = [args.period] if args.period else None

    run(
        periods=periods,
        n_folds=args.cv_folds,
        min_keywords=args.min_keywords,
        model_type=args.model,
        rolling_window=args.rolling_window,
        report_only=args.report_only,
        regime=args.regime,
        vix_threshold=args.vix_threshold,
        decay_half_days=args.decay_half,
    )
