#!/usr/bin/env bash
# run_pipeline.sh — Full Polymarket data + training pipeline.
#
# Steps:
#   0. pull_polymarket_history.py  (optional — skip with --skip-pull)
#   1. build_poly_market_data.py
#   2. build_post_market_data.py
#   3. train_poly_model.py      --tune --walk-forward
#   4. train_poly_fade_model.py --tune --walk-forward
#   5. train_post_model.py      --tune --walk-forward
#   6. train_post_fade_model.py --tune --walk-forward
#
# Note: pull_polymarket_trades.py is disabled — CLOB trades API
# now requires auth (401). Volume is replaced by price-activity proxy.
#
# All output is appended to PIPELINE_LOG.
# On any step failure the script stops and logs the error.

set -euo pipefail

SKIP_PULL=false
for arg in "$@"; do
    case "$arg" in
        --skip-pull) SKIP_PULL=true ;;
    esac
done

PYTHON=/c/Users/wangy/anaconda3/envs/whalewatch/python.exe
SCRIPTS=/c/ClaudeCode/WhaleWatch_Alpha/scripts
HISTORY_LOG=/d/WhaleWatch_Data/pull_history.log
PIPELINE_LOG=/d/WhaleWatch_Data/pipeline.log

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')]  $*" | tee -a "$PIPELINE_LOG"
}

run_step() {
    local label="$1"; shift
    log "===== START: $label ====="
    if PYTHONUNBUFFERED=1 "$PYTHON" "$@" >> "$PIPELINE_LOG" 2>&1; then
        log "===== DONE:  $label ====="
    else
        log "===== FAILED: $label — pipeline stopped ====="
        exit 1
    fi
}

# ---------------------------------------------------------------------------
# Step 0 — run (or wait for) pull_polymarket_history.py to completion
# ---------------------------------------------------------------------------
log "Pipeline started."

if [ "$SKIP_PULL" = true ]; then
    log "Skipping history pull (--skip-pull)."
else
    # Start the history pull.
    # Note: on Windows + Git Bash, kill -0 doesn't track native Python processes.
    # We detect crashes by watching log activity: if the log hasn't grown in
    # STALL_SECS seconds and "Done." isn't present, we assume a crash and restart.
    STALL_SECS=900     # 15 minutes of silence = assume crashed (discovery phase takes ~12 min)
    MAX_RESTARTS=5

    _start_history_pull() {
        > "$HISTORY_LOG"
        PYTHONUNBUFFERED=1 "$PYTHON" "$SCRIPTS/pull_polymarket_history.py" >> "$HISTORY_LOG" 2>&1 &
        log "History pull started (PID hint: $!)"
    }

    _log_size() {
        wc -c < "$HISTORY_LOG" 2>/dev/null || echo 0
    }

    log "Starting pull_polymarket_history.py..."
    _start_history_pull
    restarts=0
    last_size=$(_log_size)
    last_activity=$(date +%s)

    while true; do
        sleep 30

        if grep -q "Done\." "$HISTORY_LOG" 2>/dev/null; then
            log "History pull complete."
            break
        fi

        current_size=$(_log_size)
        now=$(date +%s)

        if [ "$current_size" -gt "$last_size" ]; then
            # Log grew — process is alive
            last_size=$current_size
            last_activity=$now
        else
            stall=$(( now - last_activity ))
            if [ "$stall" -ge "$STALL_SECS" ]; then
                restarts=$(( restarts + 1 ))
                if [ "$restarts" -gt "$MAX_RESTARTS" ]; then
                    log "FATAL: history pull stalled/crashed $MAX_RESTARTS times. Stopping."
                    exit 1
                fi
                log "WARNING: no log activity for ${stall}s (restart $restarts/$MAX_RESTARTS). Restarting..."
                tail -3 "$HISTORY_LOG" >> "$PIPELINE_LOG"
                _start_history_pull
                last_size=$(_log_size)
                last_activity=$(date +%s)
            fi
        fi
    done
fi

# NOTE: pull_polymarket_trades.py is disabled — CLOB trades API requires
# auth (401 since ~March 2026). Volume replaced by price-activity proxy
# computed in build_poly_market_data.py.

# ---------------------------------------------------------------------------
# Step 1 — build datasets
# ---------------------------------------------------------------------------
run_step "build_poly_market_data.py" "$SCRIPTS/build_poly_market_data.py"
run_step "build_post_market_data.py" "$SCRIPTS/build_post_market_data.py"

# ---------------------------------------------------------------------------
# Step 2 — train models
# ---------------------------------------------------------------------------
run_step "train_poly_model.py"      "$SCRIPTS/train_poly_model.py"      --tune --walk-forward
run_step "train_poly_fade_model.py" "$SCRIPTS/train_poly_fade_model.py" --tune --walk-forward
run_step "train_post_model.py"      "$SCRIPTS/train_post_model.py"      --tune --walk-forward
run_step "train_post_fade_model.py" "$SCRIPTS/train_post_fade_model.py" --tune --walk-forward

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
log ""
log "======================================================="
log "  ALL PIPELINE STEPS COMPLETE"
log "  Models saved to: models/saved/"
log "  Check pipeline.log for full output."
log "======================================================="
