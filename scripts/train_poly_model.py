"""train_poly_model.py — Train Polymarket session → SPY direction predictor.

Input:  D:/WhaleWatch_Data/poly_market_data.parquet
Output: models/saved/poly_direction_{period}.pkl          (baseline)
        models/saved/poly_direction_{period}_high_vix.pkl (VIX >= threshold)
        models/saved/poly_direction_{period}_low_vix.pkl  (VIX < threshold)
        models/saved/poly_direction_{period}_weighted.pkl (decay-weighted)

Uses XGBoost + isotonic calibration, TimeSeriesSplit CV.
Output: P(SPY excess return > 0 after session) — same interface as
        the Truth Social L2 model (spy_direction_*.pkl).

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

# ---------------------------------------------------------------------------
# Feature definitions
# ---------------------------------------------------------------------------

# Session strength — how strong and broad was the Polymarket move
STRENGTH_FEATURES = [
    "max_price_delta",       # single largest price move — whale size proxy
    "cumulative_delta",      # net directional sum (signed)
    "net_delta_abs",         # absolute cumulative magnitude
    "dominant_direction",    # +1 YES rising / -1 YES falling (separates direction from magnitude)
    "n_events",              # total anomaly events in session
    "n_markets",             # distinct markets involved (cross-market coordination = whale thesis)
    "n_corroborating",       # events in dominant direction (reinforcement count)
    "n_opposing",            # counter-events (uncertainty / split-whale signal)
    "corroboration_ratio",   # n_corroborating / n_events — purity of directional bet
    "session_duration_min",  # short+large = single whale entry; long = accumulation
]

# Topic composition — which buckets fired
TOPIC_FEATURES = [
    "has_tariff",
    "has_geopolitical",
    "has_fed",
    "has_energy",
    "has_executive",
]

# Market regime at session time
REGIME_FEATURES = [
    "vix_level",        # VIX absolute level — low VIX = cleaner signal environment
    "vix_percentile",   # VIX relative to history
    "vixy_level",       # VIXY intraday price — real-time VIX proxy (fear barometer at session time)
    "is_market_hours",
    "hour_of_day",
    "day_of_week",
]

ALL_FEATURES = STRENGTH_FEATURES + TOPIC_FEATURES + REGIME_FEATURES


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


def _build_calibrated_model(
    X: pd.DataFrame,
    y: pd.Series,
    sample_weight: np.ndarray | None = None,
):
    """Fit XGBoost + isotonic calibration on the full dataset."""
    from sklearn.calibration import CalibratedClassifierCV
    from xgboost import XGBClassifier

    base = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
        verbosity=0,
    )
    clf = CalibratedClassifierCV(base, method="isotonic", cv=3)
    fit_kwargs = {} if sample_weight is None else {"sample_weight": sample_weight}
    clf.fit(X, y, **fit_kwargs)
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

def train_period(
    df: pd.DataFrame,
    period: str,
    train_regime: bool,
    vix_threshold: float,
    decay_half: int,
) -> None:
    lbl_col = f"label_{period}"
    ret_col = f"ret_{period}"

    if lbl_col not in df.columns:
        logger.warning("  %s: label column missing — skipping", period)
        return

    df_valid = df[df[lbl_col].notna()].copy()
    df_valid[lbl_col] = df_valid[lbl_col].astype(int)

    features = [f for f in ALL_FEATURES if f in df_valid.columns]
    X = df_valid[features]
    y = df_valid[lbl_col]

    if len(y) < 30:
        logger.warning("  %s: only %d valid rows — skipping (need >= 30)", period, len(y))
        return

    pos_rate = float(y.mean())

    from xgboost import XGBClassifier
    from sklearn.calibration import CalibratedClassifierCV

    def model_factory():
        base = XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="logloss", random_state=42, verbosity=0,
        )
        return CalibratedClassifierCV(base, method="isotonic", cv=3)

    # --- Baseline (full data) ---
    cv = _cross_validate(X, y, model_factory)
    model = _build_calibrated_model(X, y)
    _save_model(
        model, features, period, cv, len(y), pos_rate, ret_col,
        label_method="excess_return_rolling20", suffix="",
    )

    # --- Regime split ---
    if train_regime and "vix_level" in df_valid.columns:
        for regime_name, mask_fn in [
            ("high_vix", lambda d: d["vix_level"] >= vix_threshold),
            ("low_vix",  lambda d: d["vix_level"] <  vix_threshold),
        ]:
            mask   = mask_fn(df_valid)
            X_r    = X[mask]
            y_r    = y[mask]
            if len(y_r) < 30 or y_r.nunique() < 2:
                logger.info("  %s %s: %d rows — skip", period, regime_name, len(y_r))
                continue
            cv_r   = _cross_validate(X_r, y_r, model_factory)
            model_r = _build_calibrated_model(X_r, y_r)
            _save_model(
                model_r, features, period, cv_r, len(y_r), float(y_r.mean()),
                ret_col, label_method="excess_return_rolling20",
                suffix=f"_{regime_name}",
                extra_meta={"vix_threshold": vix_threshold, "regime": regime_name},
            )

    # --- Decay-weighted ---
    if decay_half > 0:
        weights = _decay_weights(df_valid, decay_half)
        cv_w    = _cross_validate(X, y, model_factory, sample_weight=weights)
        model_w = _build_calibrated_model(X, y, sample_weight=weights)
        _save_model(
            model_w, features, period, cv_w, len(y), pos_rate,
            ret_col, label_method="excess_return_rolling20",
            suffix="_weighted",
            extra_meta={"decay_half_life_days": decay_half},
        )


def run(
    min_events: int,
    train_regime: bool,
    vix_threshold: float,
    decay_half: int,
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

    # Date range
    if "session_time" in df.columns:
        logger.info(
            "Date range: %s → %s",
            df["session_time"].min().date(),
            df["session_time"].max().date(),
        )

    logger.info("\nFeature set (%d features):", len(ALL_FEATURES))
    logger.info("  Strength:  %s", STRENGTH_FEATURES)
    logger.info("  Topics:    %s", TOPIC_FEATURES)
    logger.info("  Regime:    %s", REGIME_FEATURES)

    logger.info("\n--- Training ---")
    for period in HOLDING_PERIODS:
        logger.info("\nPeriod: %s", period)
        train_period(df, period, train_regime, vix_threshold, decay_half)

    # Summary table
    logger.info("\n--- Model summary ---")
    model_files = sorted(MODELS_DIR.glob("poly_direction_*.pkl"))
    results = []
    for f in model_files:
        try:
            with open(f, "rb") as fh:
                m = pickle.load(fh)
            results.append({
                "file":    f.name,
                "period":  m.get("period", ""),
                "auc":     m["cv_metrics"]["auc"],
                "n_train": m.get("n_train", 0),
                "pos_pct": round(m.get("pos_rate", 0.5) * 100, 1),
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
    args = parser.parse_args()

    run(
        min_events=args.min_events,
        train_regime=args.regime,
        vix_threshold=args.vix_threshold,
        decay_half=args.decay_half,
    )
