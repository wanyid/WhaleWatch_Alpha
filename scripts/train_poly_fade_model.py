"""train_poly_fade_model.py — Train Polymarket fade (mean-reversion) predictor.

Thesis: after a Polymarket whale session closes, the market makes an initial
move (first 30 minutes). That move sometimes overshoots and partially reverts.
This model predicts whether fading (betting opposite to) the initial move will
be profitable over 2h, 4h, or 1d holding periods.

Timeline (all times relative to first anomaly T):
  T+0      → first Polymarket anomaly detected
  T+60min  → session window closes; full session features available; SPY entry
  T+90min  → initial 30m move observed; fade entry (opposite direction)
  T+60+2h  → 2h hold exits
  T+60+4h  → 4h hold exits
  T+60+1d  → 1d hold exits

Input:  D:/WhaleWatch_Data/poly_market_data.parquet  (built by build_poly_market_data.py)
Output: models/saved/poly_fade_{period}.pkl
        models/saved/poly_fade_{period}_high_vix.pkl
        models/saved/poly_fade_{period}_low_vix.pkl

Two dead-zone filters applied at training time:
  1. |initial_ret| >= OVERSHOOT_MIN  — only fade moves large enough to overshoot
  2. |ret_cont|   >= CONT_DEAD_ZONE  — only train on meaningful continuations

Usage:
    python scripts/train_poly_fade_model.py
    python scripts/train_poly_fade_model.py --regime
    python scripts/train_poly_fade_model.py --overshoot-min 0.003
    python scripts/train_poly_fade_model.py --report-only
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
logger = logging.getLogger("train_poly_fade")
warnings.filterwarnings("ignore", category=UserWarning)

DATA_PATH  = Path("D:/WhaleWatch_Data/poly_market_data.parquet")
MODELS_DIR = PROJECT_ROOT / "models" / "saved"

# Train / OOS split — same cutoff as directional model
TRAIN_CUTOFF = pd.Timestamp("2026-02-28")

# Fade-specific holding periods (must be longer than FADE_ENTRY_LAG=30m)
FADE_PERIODS = ["2h", "4h", "1d"]

# Dead-zone thresholds on the continuation return
CONT_DEAD_ZONE: dict[str, float] = {
    "2h": 0.0040,   # 0.40%
    "4h": 0.0060,   # 0.60%
    "1d": 0.0080,   # 0.80%
}

# Default minimum initial move to be worth fading (0.30% for 30m)
DEFAULT_OVERSHOOT_MIN = 0.0030


# ---------------------------------------------------------------------------
# Feature definitions — imported from shared module
# ---------------------------------------------------------------------------
from reasoner.layer2_predictor.poly_features import (
    STRENGTH_FEATURES,
    TOPIC_FEATURES,
    REGIME_FEATURES,
    FADE_FEATURES,
    ALL_FADE_FEATURES,
)

# Local alias — STRENGTH_FEATURES was called SESSION_FEATURES in the old inline version
SESSION_FEATURES = STRENGTH_FEATURES

ALL_FEATURES = ALL_FADE_FEATURES


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _decay_weights(df: pd.DataFrame, half_life_days: int) -> np.ndarray:
    now      = df["session_time"].max()
    days_ago = (now - df["session_time"]).dt.total_seconds() / 86400
    return np.exp(-np.log(2) * days_ago / half_life_days).values


def _cross_validate(
    X: pd.DataFrame,
    y: pd.Series,
    model_factory,
    sample_weight: np.ndarray | None = None,
) -> dict:
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import roc_auc_score, accuracy_score

    tscv = TimeSeriesSplit(n_splits=5)
    aucs, accs = [], []

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
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.model_selection import TimeSeriesSplit
    from xgboost import XGBClassifier

    n_pos    = int(y.sum())
    n_neg    = len(y) - n_pos
    pos_rate = n_pos / max(len(y), 1)
    spw      = min(n_neg / max(n_pos, 1), 5.0) if (pos_rate < 0.35 or pos_rate > 0.65) else 1.0

    base = XGBClassifier(
        n_estimators=200,
        max_depth=3,          # shallower than directional model — less data for fade
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=spw,
        eval_metric="logloss",
        random_state=42,
        verbosity=0,
    )
    method = "isotonic" if len(y) >= 500 else "sigmoid"
    clf    = CalibratedClassifierCV(base, method=method, cv=TimeSeriesSplit(n_splits=3))
    fit_kw = {} if sample_weight is None else {"sample_weight": sample_weight}
    clf.fit(X, y, **fit_kw)
    return clf


def _oos_metrics(model, X_oos: pd.DataFrame, y_oos: pd.Series) -> dict:
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


def _save_model(
    model,
    features: list[str],
    period: str,
    cv_metrics: dict,
    n_train: int,
    pos_rate: float,
    overshoot_min: float,
    suffix: str = "",
    extra_meta: dict | None = None,
    oos_metrics: dict | None = None,
) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"poly_fade_{period}{suffix}.pkl"
    out_path = MODELS_DIR / filename

    payload = {
        "model":          model,
        "features":       features,
        "period":         period,
        "fade_entry_lag": "30m",
        "label_method":   "fade_reversion",
        "overshoot_min":  overshoot_min,
        "cont_dead_zone": CONT_DEAD_ZONE.get(period, 0.0),
        "n_train":        n_train,
        "train_cutoff":   str(TRAIN_CUTOFF.date()),
        "pos_rate":       round(pos_rate, 3),
        "cv_metrics":     cv_metrics,
        "oos_metrics":    oos_metrics or {},
        "model_type":     "xgboost_fade",
        "trained_at":     datetime.now(tz=timezone.utc).isoformat(),
    }
    if extra_meta:
        payload.update(extra_meta)

    with open(out_path, "wb") as f:
        pickle.dump(payload, f)
    logger.info(
        "  Saved %-38s  cv_auc=%.3f  n=%d  pos=%.1f%%",
        filename, cv_metrics["auc"], n_train, pos_rate * 100,
    )


# ---------------------------------------------------------------------------
# Per-period training
# ---------------------------------------------------------------------------

def _filter_fade(
    df: pd.DataFrame,
    period: str,
    overshoot_min: float,
) -> pd.DataFrame:
    """Apply both dead-zone filters and return the valid training subset."""
    lbl_col  = f"fade_label_{period}"
    cont_col = f"ret_cont_{period}"

    if lbl_col not in df.columns or cont_col not in df.columns:
        return pd.DataFrame()

    cont_dead = CONT_DEAD_ZONE.get(period, 0.0)

    mask = (
        df[lbl_col].notna() &
        df["initial_ret"].notna() &
        (df["initial_ret"].abs() >= overshoot_min) &   # meaningful overshoot
        (df[cont_col].abs() >= cont_dead)               # meaningful continuation
    )
    df_valid = df[mask].copy()
    df_valid[lbl_col] = df_valid[lbl_col].astype(int)

    n_total   = int(df[lbl_col].notna().sum())
    n_dropped = n_total - len(df_valid)
    logger.info(
        "  %s: raw=%d  after_filters=%d (dropped %d=%.0f%%)  "
        "overshoot>=%.2f%%  cont_dead>=%.2f%%",
        period, n_total, len(df_valid),
        n_dropped, n_dropped / max(n_total, 1) * 100,
        overshoot_min * 100, cont_dead * 100,
    )
    return df_valid


def train_period(
    df_train: pd.DataFrame,
    df_oos: pd.DataFrame,
    period: str,
    overshoot_min: float,
    train_regime: bool,
    vix_threshold: float,
    report_only: bool,
) -> None:
    from xgboost import XGBClassifier
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.model_selection import TimeSeriesSplit

    lbl_col = f"fade_label_{period}"

    df_valid = _filter_fade(df_train, period, overshoot_min)
    if len(df_valid) < 30:
        logger.warning("  %s: only %d rows after filters — skipping", period, len(df_valid))
        return

    features = [f for f in ALL_FEATURES if f in df_valid.columns]
    X = df_valid[features]
    y = df_valid[lbl_col]

    pos_rate   = float(y.mean())
    n_pos      = int(y.sum())
    n_neg      = len(y) - n_pos
    spw_approx = (
        min(n_neg / max(n_pos, 1), 5.0)
        if (pos_rate < 0.35 or pos_rate > 0.65) else 1.0
    )

    logger.info(
        "  %s: n_train=%d  pos_rate=%.1f%%  (fade_worked=%d  not_faded=%d)",
        period, len(y), pos_rate * 100, n_pos, n_neg,
    )

    def model_factory():
        base = XGBClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=spw_approx,
            eval_metric="logloss", random_state=42, verbosity=0,
        )
        return CalibratedClassifierCV(base, method="sigmoid", cv=TimeSeriesSplit(n_splits=3))

    cv = _cross_validate(X, y, model_factory)
    logger.info("  %s: CV  auc=%.3f  acc=%.3f  folds=%d",
                period, cv["auc"], cv["accuracy"], cv["n_folds"])

    if report_only:
        return

    # OOS set
    df_oos_v = _filter_fade(df_oos, period, overshoot_min) if len(df_oos) > 0 else pd.DataFrame()
    X_oos = df_oos_v[[f for f in features if f in df_oos_v.columns]] if len(df_oos_v) > 0 else pd.DataFrame()
    y_oos = df_oos_v[lbl_col] if len(df_oos_v) > 0 else pd.Series(dtype=int)

    # --- Baseline model ---
    model   = _build_calibrated_model(X, y)
    oos_m   = _oos_metrics(model, X_oos, y_oos) if len(X_oos) > 0 else {}
    if oos_m:
        logger.info("  %s: OOS auc=%.3f  n=%d", period, oos_m.get("oos_auc", 0), oos_m.get("oos_n", 0))
    _save_model(model, features, period, cv, len(y), pos_rate,
                overshoot_min, suffix="", oos_metrics=oos_m)

    # --- Regime split ---
    if train_regime and "vix_level" in df_valid.columns:
        for regime_name, mask_fn in [
            ("high_vix", lambda d: d["vix_level"] >= vix_threshold),
            ("low_vix",  lambda d: d["vix_level"] <  vix_threshold),
        ]:
            mask   = mask_fn(df_valid)
            X_r, y_r = X[mask], y[mask]
            if len(y_r) < 30 or y_r.nunique() < 2:
                logger.info("  %s %s: %d rows — skip", period, regime_name, len(y_r))
                continue
            cv_r    = _cross_validate(X_r, y_r, model_factory)
            model_r = _build_calibrated_model(X_r, y_r)

            oos_r: dict = {}
            if len(df_oos_v) > 0 and "vix_level" in df_oos_v.columns:
                df_oos_r = df_oos_v[mask_fn(df_oos_v)]
                if len(df_oos_r) > 0:
                    X_oos_r = df_oos_r[[f for f in features if f in df_oos_r.columns]]
                    oos_r   = _oos_metrics(model_r, X_oos_r, df_oos_r[lbl_col])

            logger.info("  %s %s: cv_auc=%.3f  n=%d",
                        period, regime_name, cv_r["auc"], len(y_r))
            _save_model(model_r, features, period, cv_r, len(y_r), float(y_r.mean()),
                        overshoot_min, suffix=f"_{regime_name}",
                        extra_meta={"vix_threshold": vix_threshold, "regime": regime_name},
                        oos_metrics=oos_r)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run(
    overshoot_min: float,
    train_regime: bool,
    vix_threshold: float,
    min_events: int,
    report_only: bool,
) -> None:
    if not DATA_PATH.exists():
        logger.error("Dataset not found: %s\nRun build_poly_market_data.py first.", DATA_PATH)
        return

    df = pd.read_parquet(DATA_PATH)
    logger.info("Loaded %d sessions from %s", len(df), DATA_PATH)

    # Verify fade columns exist
    missing_fade = [f"fade_label_{p}" for p in FADE_PERIODS if f"fade_label_{p}" not in df.columns]
    if missing_fade:
        logger.error(
            "Fade label columns missing: %s\n"
            "Re-run build_poly_market_data.py to generate them.", missing_fade,
        )
        return

    if "n_events" in df.columns and min_events > 0:
        before = len(df)
        df     = df[df["n_events"] >= min_events]
        logger.info("Filtered to n_events >= %d: %d → %d", min_events, before, len(df))

    if "session_time" in df.columns:
        df = df.sort_values("session_time").reset_index(drop=True)
        logger.info("Date range: %s → %s",
                    df["session_time"].min().date(), df["session_time"].max().date())

    # Train / OOS split
    if "session_time" in df.columns:
        df_train = df[df["session_time"] <= TRAIN_CUTOFF].copy()
        df_oos   = df[df["session_time"] >  TRAIN_CUTOFF].copy()
        logger.info("Train/OOS split at %s  →  train=%d  OOS=%d",
                    TRAIN_CUTOFF.date(), len(df_train), len(df_oos))
    else:
        df_train, df_oos = df.copy(), df.iloc[0:0].copy()

    logger.info("Overshoot minimum: %.2f%%  (only fade initial moves above this)",
                overshoot_min * 100)
    logger.info("Feature set: %d features  (%d fade-specific)",
                len(ALL_FEATURES), len(FADE_FEATURES))

    for period in FADE_PERIODS:
        logger.info("\n--- Period: %s ---", period)
        train_period(df_train, df_oos, period, overshoot_min,
                     train_regime, vix_threshold, report_only)

    # Summary table
    logger.info("\n--- Fade model summary ---")
    model_files = sorted(MODELS_DIR.glob("poly_fade_*.pkl"))
    results = []
    for f in model_files:
        try:
            with open(f, "rb") as fh:
                m = pickle.load(fh)
            oos = m.get("oos_metrics") or {}
            results.append({
                "file":    f.name,
                "period":  m.get("period", ""),
                "cv_auc":  m["cv_metrics"]["auc"],
                "oos_auc": oos.get("oos_auc", "—"),
                "n_train": m.get("n_train", 0),
                "n_oos":   oos.get("oos_n", "—"),
            })
        except Exception:
            pass

    if results:
        summary = pd.DataFrame(results).sort_values(["period", "file"])
        logger.info("\n%s", summary.to_string(index=False))

    logger.info("\nFade models saved to %s", MODELS_DIR)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Polymarket fade (mean-reversion) model"
    )
    parser.add_argument("--overshoot-min", type=float, default=DEFAULT_OVERSHOOT_MIN,
                        help="Minimum |initial_ret| to qualify as a fade candidate "
                             "(default: 0.003 = 0.30%%)")
    parser.add_argument("--regime", action="store_true",
                        help="Train separate high/low VIX regime models")
    parser.add_argument("--vix-threshold", type=float, default=20.0,
                        help="VIX split threshold for regime models (default: 20)")
    parser.add_argument("--min-events", type=int, default=1,
                        help="Min anomaly events per session (default: 1)")
    parser.add_argument("--report-only", action="store_true",
                        help="Print CV results without saving models")
    args = parser.parse_args()

    run(
        overshoot_min=args.overshoot_min,
        train_regime=args.regime,
        vix_threshold=args.vix_threshold,
        min_events=args.min_events,
        report_only=args.report_only,
    )
