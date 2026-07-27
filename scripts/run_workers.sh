#!/bin/bash
# Launch N crash-restarting workers for run_sim.py against a
# pilot_slope_power-style scenario CSV (ROC-per-slope-steepness pilot).
#
# Each worker supervises its own run_sim.py, restarting on crash (e.g. Julia
# GC segfaults) and stopping once run_sim.py prints "DONE". Resumable:
# cached light curves (.npz) and per-fit result JSONs are skipped on
# restart, so relaunching after any crash or manual stop is always safe.
#
# Light curves and fit results are ID-keyed, so multiple CSVs (e.g. the
# original 20-rep pilot and a later extra-reps extension) can safely share
# the same --lc-dir/--out-dir as long as their ID ranges don't overlap.
#
# Usage (from anywhere; launches all workers detached, then returns):
#   ./run_workers.sh <data-dir> <csv-name> <n-workers>
#
# Internal (used by the script to re-invoke itself per worker; do not call
# directly):
#   ./run_workers.sh <data-dir> <csv-name> <n-workers> <worker-id>
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA="$1"
CSV="$2"
NWORKERS="$3"
# tag log files by csv stem so runs against different CSVs don't clobber
# each other's logs when sharing the same data dir
TAG="${CSV%.csv}"

if [ $# -ge 4 ]; then
    WORKER="$4"
    LOG="$DATA/logs/sim_${TAG}_w${WORKER}.log"
    n=0
    while true; do
        python "$SCRIPT_DIR/run_sim.py" \
            --config-csv "$DATA/$CSV" \
            --lc-dir "$DATA/lightcurves" \
            --out-dir "$DATA/results" \
            --n-sims 100000 --stride "$NWORKERS" --worker "$WORKER" \
            --filter-value -1.0 \
            --enforce-leakage-margin false \
            --models drw \
            >> "$LOG" 2>&1
        if tail -50 "$LOG" | grep -q "^DONE"; then
            echo "$(date): finished cleanly" >> "$LOG"
            break
        fi
        n=$((n + 1))
        if [ "$n" -ge 100 ]; then
            echo "$(date): giving up after 100 restarts" >> "$LOG"
            break
        fi
        echo "$(date): process died, restart $n/100" >> "$LOG"
        sleep 10
    done
    exit 0
fi

mkdir -p "$DATA/lightcurves" "$DATA/results" "$DATA/logs"
for ((I = 0; I < NWORKERS; I++)); do
    nohup "$0" "$DATA" "$CSV" "$NWORKERS" "$I" </dev/null >/dev/null 2>&1 &
    disown
done
echo "Launched $NWORKERS workers against $DATA/$CSV (logs: $DATA/logs/sim_${TAG}_w*.log)"
