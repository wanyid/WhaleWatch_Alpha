#!/usr/bin/env bash
# monitor_pull.sh — Watch pull_polymarket_history.py for stalls and report progress.
#
# Usage: bash scripts/monitor_pull.sh
#
# Prints a progress snapshot every CHECK_INTERVAL seconds.
# Warns if no new price files have been written for STALL_THRESHOLD seconds.
# Exits when the pull log shows "Done." or the Python process disappears.

LOG=/d/WhaleWatch_Data/pull_history.log
PRICES_DIR=/d/WhaleWatch_Data/polymarket/prices
CHECK_INTERVAL=120    # check every 2 minutes
STALL_THRESHOLD=600   # warn if no new file in 10 minutes

last_count=0
last_new_file_time=$(date +%s)

echo "=== Polymarket pull monitor started at $(date) ==="
echo "    Log:      $LOG"
echo "    Checking every ${CHECK_INTERVAL}s, stall warning after ${STALL_THRESHOLD}s"
echo ""

while true; do
    # Count current price files
    current_count=$(ls "$PRICES_DIR" 2>/dev/null | grep -v "^0x" | wc -l)
    now=$(date +%s)

    # Detect new files written since last check
    if [ "$current_count" -gt "$last_count" ]; then
        new_files=$(( current_count - last_count ))
        last_new_file_time=$now
        stall_flag=""
    else
        new_files=0
        stall_secs=$(( now - last_new_file_time ))
        if [ "$stall_secs" -ge "$STALL_THRESHOLD" ]; then
            stall_flag="  *** STALL WARNING: no new files in ${stall_secs}s ***"
        else
            stall_flag="  (no new files in ${stall_secs}s)"
        fi
    fi

    # Grab last meaningful log line
    last_log=$(grep -E "\[|Done\.|ERROR|WARNING" "$LOG" 2>/dev/null | tail -1)

    echo "[$(date '+%H:%M:%S')]  new-format files: $current_count  (+${new_files} since last check)${stall_flag}"
    echo "           Last log: ${last_log}"
    echo ""

    # Exit conditions
    if grep -q "^.*Done\." "$LOG" 2>/dev/null; then
        echo "=== Pull completed! ==="
        echo "Next step: python scripts/pull_polymarket_trades.py"
        break
    fi

    # Check if python process is still alive
    if ! pgrep -f "pull_polymarket_history" > /dev/null 2>&1; then
        echo "=== Python process not found — pull may have finished or crashed ==="
        echo "Last 10 log lines:"
        tail -10 "$LOG"
        break
    fi

    last_count=$current_count
    sleep "$CHECK_INTERVAL"
done
