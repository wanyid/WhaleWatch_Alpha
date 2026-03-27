"""run_pipeline.py — Full Polymarket data + training pipeline orchestrator.

Runs these steps in sequence, auto-restarting on crash:
  1. pull_polymarket_history.py
  2. pull_polymarket_trades.py
  3. build_poly_market_data.py
  4. build_post_market_data.py
  5. train_poly_model.py      --tune --walk-forward
  6. train_poly_fade_model.py --tune --walk-forward
  7. train_post_model.py      --tune --walk-forward
  8. train_post_fade_model.py --tune --walk-forward

Output strategy (Windows-compatible — no pipes):
  - Each step's stdout+stderr goes directly to pipeline.log via file handle
  - pull_polymarket_history.py also writes to pull_history.log for easy monitoring
  - On crash (non-zero exit): retried up to MAX_RESTARTS times

Usage:
    python scripts/run_pipeline.py
"""

import subprocess
import sys
import os
import threading
import time
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PYTHON       = sys.executable
SCRIPTS_DIR  = Path(__file__).resolve().parent
PIPELINE_LOG = Path("D:/WhaleWatch_Data/pipeline.log")
HISTORY_LOG  = Path("D:/WhaleWatch_Data/pull_history.log")

MAX_RESTARTS = 15

STEPS = [
    ("pull_polymarket_history.py",  []),
    ("pull_polymarket_trades.py",   []),
    ("build_poly_market_data.py",   []),
    ("build_post_market_data.py",   []),
    ("train_poly_model.py",         ["--tune", "--walk-forward"]),
    ("train_poly_fade_model.py",    ["--tune", "--walk-forward"]),
    ("train_post_model.py",         ["--tune", "--walk-forward"]),
    ("train_post_fade_model.py",    ["--tune", "--walk-forward"]),
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ts():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg):
    line = f"[{ts()}]  {msg}\n"
    with open(PIPELINE_LOG, "a", encoding="utf-8") as f:
        f.write(line)
    print(line, end="", flush=True)


def run_step(script_name: str, extra_args: list) -> bool:
    """Run a script, writing output directly to log file (no pipes).
    Returns True on success, False if max restarts exceeded."""

    cmd = [PYTHON, str(SCRIPTS_DIR / script_name)] + extra_args
    is_history = (script_name == "pull_polymarket_history.py")

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    # History pull gets a watchdog: if pull_history.log is silent for STALL_SECS,
    # kill the process and restart (incremental — already-done markets are skipped).
    STALL_SECS = 900  # 15 minutes — covers ~12-min Gamma discovery silence + buffer

    for attempt in range(1, MAX_RESTARTS + 2):
        if attempt > MAX_RESTARTS:
            log(f"FATAL: {script_name} failed {MAX_RESTARTS} times. Pipeline stopped.")
            return False

        label = f"(attempt {attempt})" if attempt > 1 else ""
        log(f"===== START: {script_name} {label} =====")

        # On restarts of history pull, skip the 12-min Gamma discovery
        if is_history and attempt > 1:
            cmd = [PYTHON, str(SCRIPTS_DIR / script_name)] + extra_args + ["--use-cached-catalog"]
        else:
            cmd = [PYTHON, str(SCRIPTS_DIR / script_name)] + extra_args

        # Overwrite history log on first attempt, append on restarts
        hlog_mode = "w" if (is_history and attempt == 1) else "a"
        hlog_path = HISTORY_LOG if is_history else PIPELINE_LOG

        with open(hlog_path, hlog_mode, encoding="utf-8") as out_fh:
            proc = subprocess.Popen(cmd, stdout=out_fh, stderr=out_fh, env=env)

            if is_history:
                # Watchdog thread: monitors log size; kills proc if silent too long
                killed_by_watchdog = threading.Event()

                def _watchdog():
                    last_size = 0
                    last_active = time.time()
                    while proc.poll() is None:
                        time.sleep(30)
                        try:
                            cur_size = HISTORY_LOG.stat().st_size
                        except OSError:
                            cur_size = 0
                        if cur_size > last_size:
                            last_size = cur_size
                            last_active = time.time()
                        elif time.time() - last_active > STALL_SECS:
                            log(f"WATCHDOG: no log activity for {STALL_SECS}s — killing stalled process")
                            try:
                                proc.kill()
                            except Exception:
                                pass
                            killed_by_watchdog.set()
                            return

                wd = threading.Thread(target=_watchdog, daemon=True)
                wd.start()
                proc.wait()
                wd.join(timeout=5)

                # Append history log to pipeline log
                try:
                    with open(PIPELINE_LOG, "a", encoding="utf-8") as plog:
                        plog.write(HISTORY_LOG.read_text(encoding="utf-8"))
                except Exception:
                    pass

                if killed_by_watchdog.is_set():
                    log("Restarting after watchdog kill (incremental — skips done markets)...")
                    time.sleep(3)
                    continue  # retry loop
            else:
                proc.wait()

        if proc.returncode == 0:
            log(f"===== DONE: {script_name} (exit 0) =====")
            return True

        log(f"===== FAILED: {script_name} (exit {proc.returncode}) — retrying in 5s =====")
        time.sleep(5)

    return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    PIPELINE_LOG.parent.mkdir(parents=True, exist_ok=True)

    log("=" * 60)
    log("  WhaleWatch pipeline started")
    log(f"  Python: {PYTHON}")
    log(f"  Steps:  {len(STEPS)}")
    log("=" * 60)

    for script, args in STEPS:
        if not run_step(script, args):
            log(f"Pipeline aborted at: {script}")
            sys.exit(1)

    log("")
    log("=" * 60)
    log("  ALL PIPELINE STEPS COMPLETE")
    log("  Models saved to: models/saved/")
    log("=" * 60)
