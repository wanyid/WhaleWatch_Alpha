"""train_l2.py — Fit the Layer 2 predictor on clean labeled data.

Pipeline:
  1. Load D:/WhaleWatch_Data/clean_training_data.parquet
     (if not found, runs clean_data.py automatically)
  2. Cross-validate the logistic regression + Ridge models
  3. Fit final models on full dataset
  4. Save to models/saved/

Outputs a training report including:
  - CV win-rate accuracy (classifier)
  - CV MAE for holding-period regressor
  - Feature importances (log-odds coefficients)
  - Calibration check (predicted probabilities vs actual win rate)

Usage:
    python scripts/train_l2.py
    python scripts/train_l2.py --cv-folds 10    # more cross-validation folds
    python scripts/train_l2.py --report-only    # show CV report, don't save models
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reasoner.layer2_predictor.features import FEATURE_NAMES
from reasoner.layer2_predictor.stat_predictor import StatPredictor
from scripts.clean_data import run as clean_run

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("train_l2")

CLEAN_DATA_PATH = Path("D:/WhaleWatch_Data/clean_training_data.parquet")
SETTINGS_PATH = Path("config/settings.yaml")


def _load_data() -> pd.DataFrame:
    if not CLEAN_DATA_PATH.exists():
        logger.info("Clean data not found — running clean_data.py first ...")
        df = clean_run()
        if df is None or df.empty:
            raise RuntimeError("clean_data.py produced no data — check label_events.py ran first")
        return df
    df = pd.read_parquet(CLEAN_DATA_PATH)
    logger.info("Loaded clean data: %d rows from %s", len(df), CLEAN_DATA_PATH)
    return df


def _cross_validate(
    X: np.ndarray,
    y_win: np.ndarray,
    y_hold: np.ndarray,
    n_folds: int = 5,
    model: str = "logistic",
) -> dict:
    """Run time-series k-fold CV and return a metrics dict.

    Uses TimeSeriesSplit instead of StratifiedKFold to avoid lookahead bias:
    validation folds always come AFTER their training folds in time.
    """
    tscv = TimeSeriesSplit(n_splits=n_folds)

    if model == "xgboost":
        try:
            from xgboost import XGBClassifier, XGBRegressor
            # Scale pos_weight to handle class imbalance (equiv to class_weight='balanced')
            n_neg = int((y_win == 0).sum())
            n_pos = int((y_win == 1).sum())
            scale_pw = n_neg / max(n_pos, 1)
            clf_pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", XGBClassifier(
                    n_estimators=200, max_depth=4, learning_rate=0.05,
                    scale_pos_weight=scale_pw, subsample=0.8, colsample_bytree=0.8,
                    eval_metric="logloss", random_state=42, verbosity=0,
                    use_label_encoder=False,
                )),
            ])
            reg_pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("reg", XGBRegressor(
                    n_estimators=200, max_depth=4, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8,
                    random_state=42, verbosity=0,
                )),
            ])
        except ImportError:
            logger.warning("xgboost not installed — falling back to logistic regression")
            model = "logistic"

    if model == "logistic":
        clf_pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=1.0, class_weight="balanced",
                                       max_iter=1000, random_state=42)),
        ])
        reg_pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("reg", Ridge(alpha=1.0)),
        ])

    # Classifier CV — manual loop since cross_val_predict doesn't support TimeSeriesSplit
    # well with predict_proba and varying fold sizes
    y_prob = np.zeros(len(y_win), dtype=float)
    hold_preds = np.zeros_like(y_hold, dtype=float)

    for train_idx, val_idx in tscv.split(X):
        clf_pipe.fit(X[train_idx], y_win[train_idx])
        y_prob[val_idx] = clf_pipe.predict_proba(X[val_idx])[:, 1]

        reg_pipe.fit(X[train_idx], y_hold[train_idx])
        hold_preds[val_idx] = reg_pipe.predict(X[val_idx])

    # Only evaluate on rows that were in a validation fold
    # (TimeSeriesSplit skips the first fold's train portion as val)
    val_mask = y_prob > 0  # rows that got a prediction

    y_pred = (y_prob >= 0.5).astype(int)
    cv_acc = float(accuracy_score(y_win[val_mask], y_pred[val_mask]))
    cv_auc = (
        float(roc_auc_score(y_win[val_mask], y_prob[val_mask]))
        if len(np.unique(y_win[val_mask])) > 1 else 0.0
    )
    cv_brier = float(brier_score_loss(y_win[val_mask], y_prob[val_mask]))
    cv_logloss = float(log_loss(y_win[val_mask], y_prob[val_mask]))
    cv_hold_mae = float(mean_absolute_error(y_hold[val_mask], hold_preds[val_mask]))

    return {
        "cv_folds": n_folds,
        "cv_type": "TimeSeriesSplit",
        "model": model,
        "win_accuracy": round(cv_acc, 4),
        "win_auc_roc": round(cv_auc, 4),
        "brier_score": round(cv_brier, 4),      # lower is better (0=perfect, 0.25=no skill)
        "log_loss": round(cv_logloss, 4),
        "hold_mae_minutes": round(cv_hold_mae, 1),
    }


def _feature_importance_report(
    clf_pipe: Pipeline,
    feature_names: list[str],
) -> pd.DataFrame:
    """Return log-odds coefficients for the fitted logistic regression."""
    clf = clf_pipe.named_steps["clf"]
    scaler = clf_pipe.named_steps["scaler"]
    # Scale coefficients back to original feature scale for interpretability
    coefs = clf.coef_[0] / scaler.scale_
    return pd.DataFrame({
        "feature": feature_names,
        "log_odds": coefs.tolist(),
        "abs_importance": np.abs(coefs).tolist(),
    }).sort_values("abs_importance", ascending=False).reset_index(drop=True)


def _calibration_report(
    clf_pipe: Pipeline,
    X: np.ndarray,
    y_win: np.ndarray,
    n_bins: int = 5,
) -> pd.DataFrame:
    """Check how well predicted probabilities match actual win rates."""
    y_prob = clf_pipe.predict_proba(X)[:, 1]
    fraction_positive, mean_predicted = calibration_curve(y_win, y_prob, n_bins=n_bins)
    return pd.DataFrame({
        "predicted_prob": mean_predicted.tolist(),
        "actual_win_rate": fraction_positive.tolist(),
        "calibration_error": (fraction_positive - mean_predicted).tolist(),
    })


def run(n_folds: int = 5, report_only: bool = False, model: str = "logistic") -> None:
    df = _load_data()

    # Available features (may be a subset if some were absent at labeling time)
    missing = [f for f in FEATURE_NAMES if f not in df.columns]
    if missing:
        logger.warning("Missing features (will be zero-filled): %s", missing)
        for f in missing:
            df[f] = 0.0

    X = df[FEATURE_NAMES].values.astype(np.float32)
    y_win = df["outcome"].map({"WIN": 1, "LOSS": 0, "STOP_OUT": 0}).fillna(0).values.astype(int)
    y_hold = df["holding_period_min"].clip(1, 4320).values.astype(np.float32)

    n_samples = len(X)
    n_wins = int(y_win.sum())
    logger.info(
        "Training data: %d samples  win_rate=%.1f%%  features=%d  model=%s",
        n_samples, n_wins / n_samples * 100, len(FEATURE_NAMES), model,
    )

    # Signal coverage summary
    ts_rows = int((df.get("has_ts_signal", pd.Series(0)) > 0).sum())
    poly_rows = int((df.get("has_poly_signal", pd.Series(0)) > 0).sum())
    dual_rows = int((df.get("dual_signal", pd.Series(0)) > 0).sum())
    logger.info(
        "Signal coverage: ts=%d (%.0f%%)  poly=%d (%.0f%%)  dual=%d (%.0f%%)",
        ts_rows, ts_rows/n_samples*100,
        poly_rows, poly_rows/n_samples*100,
        dual_rows, dual_rows/n_samples*100,
    )

    # Cross-validation report
    if n_samples >= 2 * n_folds:
        logger.info("Running %d-fold TimeSeriesSplit cross-validation (model=%s) ...", n_folds, model)
        cv_metrics = _cross_validate(X, y_win, y_hold, n_folds, model=model)

        print("\n" + "=" * 60)
        print("CROSS-VALIDATION REPORT  (TimeSeriesSplit — no lookahead)")
        print("=" * 60)
        for k, v in cv_metrics.items():
            print(f"  {k:30s}: {v}")
        print("=" * 60 + "\n")
    else:
        logger.warning(
            "Only %d samples — need ≥ %d for %d-fold CV. Skipping CV.",
            n_samples, 2 * n_folds, n_folds,
        )
        cv_metrics = {}

    if report_only:
        logger.info("--report-only: not fitting or saving models")
        return

    # Fit final models on all data
    df_train = df.copy()
    if "won" not in df_train.columns:
        df_train["won"] = df_train["outcome"].map({"WIN": 1, "LOSS": 0, "STOP_OUT": 0}).fillna(0).astype(int)
    if "holding_minutes" not in df_train.columns:
        df_train["holding_minutes"] = df_train.get("holding_period_min", pd.Series(60)).clip(1, 4320)

    predictor = StatPredictor()
    predictor.train(df_train)

    if not predictor.is_trained():
        logger.error("Training failed — check data quality")
        sys.exit(1)

    # Feature importance (logistic regression has interpretable coefficients)
    if hasattr(predictor._clf.named_steps.get("clf", None), "coef_"):
        fi = _feature_importance_report(predictor._clf, FEATURE_NAMES)
        print("\nFEATURE IMPORTANCE (log-odds, sorted by absolute value):")
        print(fi.to_string(index=False))
        print()
        fi_path = Path("D:/WhaleWatch_Data/feature_importance.csv")
        fi.to_csv(fi_path, index=False)
        logger.info("Feature importance saved → %s", fi_path)
    elif hasattr(predictor._clf.named_steps.get("clf", None), "feature_importances_"):
        # XGBoost feature importance
        clf = predictor._clf.named_steps["clf"]
        fi = pd.DataFrame({
            "feature": FEATURE_NAMES,
            "importance": clf.feature_importances_,
        }).sort_values("importance", ascending=False).reset_index(drop=True)
        print("\nFEATURE IMPORTANCE (XGBoost gain, sorted descending):")
        print(fi.to_string(index=False))
        print()
        fi_path = Path("D:/WhaleWatch_Data/feature_importance.csv")
        fi.to_csv(fi_path, index=False)
        logger.info("Feature importance saved → %s", fi_path)

    # Calibration check
    if n_samples >= 20:
        cal = _calibration_report(predictor._clf, X, y_win)
        print("CALIBRATION CHECK (predicted vs actual win rate):")
        print(cal.to_string(index=False))
        print()

    logger.info("Layer 2 training complete. Models saved to models/saved/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Layer 2 predictor")
    parser.add_argument("--cv-folds", type=int, default=5,
                        help="Number of CV folds (default 5)")
    parser.add_argument("--report-only", action="store_true",
                        help="Print CV report without fitting final models")
    parser.add_argument("--model", choices=["logistic", "xgboost"], default="logistic",
                        help="Model architecture (default: logistic)")
    args = parser.parse_args()
    run(n_folds=args.cv_folds, report_only=args.report_only, model=args.model)
