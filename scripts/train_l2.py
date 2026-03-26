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
from sklearn.model_selection import StratifiedKFold, cross_val_predict
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
) -> dict:
    """Run stratified k-fold CV and return a metrics dict."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    clf_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=1.0, class_weight="balanced",
                                   max_iter=1000, random_state=42)),
    ])
    reg_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("reg", Ridge(alpha=1.0)),
    ])

    # Classifier CV
    y_prob = cross_val_predict(clf_pipe, X, y_win, cv=skf, method="predict_proba")[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    cv_acc = float(accuracy_score(y_win, y_pred))
    cv_auc = float(roc_auc_score(y_win, y_prob)) if len(np.unique(y_win)) > 1 else 0.0
    cv_brier = float(brier_score_loss(y_win, y_prob))
    cv_logloss = float(log_loss(y_win, y_prob))

    # Regressor CV (use KFold indices derived from StratifiedKFold on y_win)
    hold_preds = np.zeros_like(y_hold, dtype=float)
    for train_idx, val_idx in skf.split(X, y_win):
        reg_pipe.fit(X[train_idx], y_hold[train_idx])
        hold_preds[val_idx] = reg_pipe.predict(X[val_idx])
    cv_hold_mae = float(mean_absolute_error(y_hold, hold_preds))

    return {
        "cv_folds": n_folds,
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


def run(n_folds: int = 5, report_only: bool = False) -> None:
    df = _load_data()

    # Available features (may be a subset if some were absent at labeling time)
    available_features = [f for f in FEATURE_NAMES if f in df.columns]
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
        "Training data: %d samples  win_rate=%.1f%%  features=%d",
        n_samples, n_wins / n_samples * 100, len(FEATURE_NAMES),
    )

    # Cross-validation report
    if n_samples >= 2 * n_folds:
        logger.info("Running %d-fold cross-validation ...", n_folds)
        cv_metrics = _cross_validate(X, y_win, y_hold, n_folds)

        print("\n" + "=" * 60)
        print("CROSS-VALIDATION REPORT")
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
    predictor = StatPredictor()
    predictor.train(df)

    if not predictor.is_trained():
        logger.error("Training failed — check data quality")
        sys.exit(1)

    # Feature importance
    fi = _feature_importance_report(predictor._clf, FEATURE_NAMES)
    print("\nFEATURE IMPORTANCE (log-odds, sorted by absolute value):")
    print(fi.to_string(index=False))
    print()

    # Calibration check
    if n_samples >= 20:
        cal = _calibration_report(predictor._clf, X, y_win)
        print("CALIBRATION CHECK (predicted vs actual win rate):")
        print(cal.to_string(index=False))
        print()

    logger.info("Layer 2 training complete. Models saved to models/saved/")

    # Save feature importance report
    fi_path = Path("D:/WhaleWatch_Data/feature_importance.csv")
    fi.to_csv(fi_path, index=False)
    logger.info("Feature importance saved → %s", fi_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Layer 2 predictor")
    parser.add_argument("--cv-folds", type=int, default=5,
                        help="Number of cross-validation folds (default 5)")
    parser.add_argument("--report-only", action="store_true",
                        help="Print CV report without fitting final models")
    args = parser.parse_args()
    run(n_folds=args.cv_folds, report_only=args.report_only)
