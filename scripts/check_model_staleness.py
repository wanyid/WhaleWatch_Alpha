"""check_model_staleness.py — Detect when saved L2 models have degraded on recent data.

Loads each saved directional model, evaluates it on the most recent N days of
the training dataset (post_market_data.parquet or poly_market_data.parquet),
and compares the rolling OOS Brier score against the training Brier stored in
the model pickle.  Prints a warning when degradation exceeds the threshold.

Usage:
    python scripts/check_model_staleness.py
    python scripts/check_model_staleness.py --window-days 30
    python scripts/check_model_staleness.py --threshold 0.05
    python scripts/check_model_staleness.py --model-glob "spy_direction_*.pkl"
"""

import argparse
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("staleness_check")

MODELS_DIR       = PROJECT_ROOT / "models" / "saved"
POST_DATA_PATH   = Path("D:/WhaleWatch_Data/post_market_data.parquet")
POLY_DATA_PATH   = Path("D:/WhaleWatch_Data/poly_market_data.parquet")

# Default: evaluate on the most recent 30 calendar days of data
DEFAULT_WINDOW_DAYS = 30
# Warn if rolling Brier exceeds training Brier by more than this amount
DEFAULT_THRESHOLD   = 0.05


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_model(pkl_path: Path) -> dict | None:
    try:
        with open(pkl_path, "rb") as f:
            return pickle.load(f)
    except Exception as exc:
        logger.warning("Could not load %s: %s", pkl_path.name, exc)
        return None


def _training_brier(payload: dict) -> float | None:
    """Return stored CV Brier score from model payload."""
    cv = payload.get("cv_metrics") or {}
    # post model uses "brier_score"; poly model uses "brier"
    return cv.get("brier_score") or cv.get("brier")


def _evaluate_on_recent(
    payload: dict,
    df: pd.DataFrame,
    window_days: int,
    ts_col: str,
    lbl_col: str,
) -> dict | None:
    """Score the model on data from the last window_days in df.

    Returns dict with brier, auc, n, or None if insufficient data.
    """
    from sklearn.metrics import brier_score_loss, roc_auc_score

    features = payload.get("features")
    model    = payload.get("model")
    if features is None or model is None:
        return None

    if ts_col not in df.columns or lbl_col not in df.columns:
        return None

    cutoff = df[ts_col].max() - pd.Timedelta(days=window_days)
    recent = df[df[ts_col] > cutoff].copy()
    recent = recent[recent[lbl_col].notna()].copy()
    recent[lbl_col] = recent[lbl_col].astype(int)

    if len(recent) < 10:
        return {"n": len(recent), "note": "too few recent samples"}

    missing = [f for f in features if f not in recent.columns]
    for f in missing:
        recent[f] = 0.0

    X = recent[features].fillna(0).values.astype(np.float32)
    y = recent[lbl_col].values

    if len(np.unique(y)) < 2:
        return {"n": len(y), "note": "single class in recent window"}

    try:
        probs = model.predict_proba(X)[:, 1]
        return {
            "n":     int(len(y)),
            "brier": round(float(brier_score_loss(y, probs)), 4),
            "auc":   round(float(roc_auc_score(y, probs)), 4),
        }
    except Exception as exc:
        return {"n": len(y), "note": str(exc)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(
    model_glob: str = "spy_direction_*.pkl",
    window_days: int = DEFAULT_WINDOW_DAYS,
    threshold: float = DEFAULT_THRESHOLD,
) -> None:
    # Load datasets
    post_df = pd.DataFrame()
    poly_df = pd.DataFrame()

    if POST_DATA_PATH.exists():
        post_df = pd.read_parquet(POST_DATA_PATH)
        if "posted_at" in post_df.columns and post_df["posted_at"].dt.tz is None:
            post_df["posted_at"] = post_df["posted_at"].dt.tz_localize("UTC")
        logger.info("Loaded post data: %d rows", len(post_df))
    else:
        logger.warning("post_market_data.parquet not found — skipping Truth Social models")

    if POLY_DATA_PATH.exists():
        poly_df = pd.read_parquet(POLY_DATA_PATH)
        if "session_time" in poly_df.columns and poly_df["session_time"].dt.tz is None:
            poly_df["session_time"] = poly_df["session_time"].dt.tz_localize("UTC")
        logger.info("Loaded poly data: %d rows", len(poly_df))
    else:
        logger.warning("poly_market_data.parquet not found — skipping Polymarket models")

    pkl_files = sorted(MODELS_DIR.glob(model_glob))
    if not pkl_files:
        logger.error("No models found matching '%s' in %s", model_glob, MODELS_DIR)
        return

    results = []

    for pkl_path in pkl_files:
        payload = _load_model(pkl_path)
        if payload is None:
            continue

        train_brier = _training_brier(payload)
        period      = payload.get("period", "?")
        source      = payload.get("source", "")  # "polymarket_sessions" or absent (post)
        is_poly     = "poly" in pkl_path.name or source == "polymarket_sessions"

        if is_poly and len(poly_df) > 0:
            lbl_col = f"label_{period}"
            recent  = _evaluate_on_recent(payload, poly_df, window_days, "session_time", lbl_col)
        elif not is_poly and len(post_df) > 0:
            lbl_col = f"spy_ret_{period}"
            # Post model uses raw return column; convert to binary label for scoring
            df_copy = post_df.copy()
            dead = {"5m":0.001,"30m":0.002,"1h":0.003,"2h":0.004,"4h":0.006,"1d":0.008}
            dz   = dead.get(period, 0.003)
            if lbl_col in df_copy.columns:
                df_copy["_lbl"] = np.where(
                    df_copy[lbl_col].abs() < dz, np.nan,
                    (df_copy[lbl_col] > 0).astype(float),
                )
                recent = _evaluate_on_recent(payload, df_copy, window_days, "posted_at", "_lbl")
            else:
                recent = None
        else:
            recent = None

        degraded = False
        if recent and "brier" in recent and train_brier is not None:
            delta    = recent["brier"] - train_brier
            degraded = delta > threshold

        results.append({
            "model":        pkl_path.name,
            "period":       period,
            "train_brier":  round(train_brier, 4) if train_brier else "—",
            "recent_brier": recent.get("brier", "—") if recent else "—",
            "recent_n":     recent.get("n", "—") if recent else "—",
            "note":         recent.get("note", "") if recent else "no data",
            "status":       "STALE" if degraded else "ok",
        })

    if not results:
        logger.info("No results to report.")
        return

    df_out = pd.DataFrame(results)
    print("\n" + "=" * 80)
    print(f"MODEL STALENESS REPORT  |  window={window_days}d  |  threshold=Δ{threshold:.3f}")
    print("=" * 80)
    print(df_out.to_string(index=False))
    print()

    stale = df_out[df_out["status"] == "STALE"]
    if len(stale) > 0:
        logger.warning(
            "%d model(s) flagged as STALE — consider retraining:\n%s",
            len(stale),
            stale[["model", "train_brier", "recent_brier"]].to_string(index=False),
        )
    else:
        logger.info("All models within threshold — no retraining needed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Check saved L2 models for calibration degradation on recent data"
    )
    parser.add_argument(
        "--model-glob", default="spy_direction_*.pkl",
        help="Glob pattern for model files to check (default: spy_direction_*.pkl)",
    )
    parser.add_argument(
        "--window-days", type=int, default=DEFAULT_WINDOW_DAYS,
        help=f"Evaluate on last N calendar days of data (default: {DEFAULT_WINDOW_DAYS})",
    )
    parser.add_argument(
        "--threshold", type=float, default=DEFAULT_THRESHOLD,
        help=f"Brier degradation that triggers STALE warning (default: {DEFAULT_THRESHOLD})",
    )
    args = parser.parse_args()

    run(
        model_glob=args.model_glob,
        window_days=args.window_days,
        threshold=args.threshold,
    )
