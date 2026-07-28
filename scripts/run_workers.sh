#!/bin/bash
# Launch N crash-restarting workers for run_sim.py against a scenario CSV
# (e.g. one built by make_slope_robustness_csv.py).
#
# Each worker supervises its own run_sim.py, restarting on crash (e.g. Julia
# GC segfaults) and stopping once run_sim.py prints "DONE". Resumable:
# cached light curves (.npz) and per-fit result JSONs are skipped on
# restart, so relaunching after any crash or manual stop is always safe.
#
# Light curves and fit results are ID-keyed, so multiple CSVs (e.g. a base
# scenario and a later extra-reps extension) can safely share the same
# --lc-dir/--out-dir as long as their ID ranges don't overlap.
#
# Overridable via environment variables (defaults match the DRW-robustness
# sensitivity study's scenario CSVs, all of which have lowalpha=-1.0):
#   MODELS         comma list for run_sim.py --models (default: drw)
#   FILTER_COL     run_sim.py --filter-col   (default: lowalpha)
#   FILTER_VALUE   run_sim.py --filter-value (default: -1.0)
#   ENFORCE_LEAKAGE  true/false, run_sim.py --enforce-leakage-margin
#                    (default: false -- see changes_and_decisions.md re: S1
#                    at NumofWINDOW=20)
#   CONDA_ENV      conda environment to run in (default: pioran-periodicity,
#                  matching docs/installation.md)
#
# Usage (from anywhere; launches all workers detached, then returns):
#   ./run_workers.sh <data-dir> <csv-path> <n-workers>
# <csv-path> may be a bare filename (resolved relative to <data-dir>) or a
# full/relative path to anywhere.
#
# Internal (used by the script to re-invoke itself per worker; do not call
# directly):
#   ./run_workers.sh <data-dir> <csv-path> <n-workers> <worker-id>
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA="$1"
CSV="$2"
NWORKERS="$3"
MODELS="${MODELS:-drw}"
FILTER_COL="${FILTER_COL:-lowalpha}"
FILTER_VALUE="${FILTER_VALUE:--1.0}"
ENFORCE_LEAKAGE="${ENFORCE_LEAKAGE:-false}"
CONDA_ENV="${CONDA_ENV:-pioran-periodicity}"

case "$CSV" in
    /*) CSV_PATH="$CSV" ;;
    *) CSV_PATH="$DATA/$CSV" ;;
esac
# tag log files by csv stem so runs against different CSVs don't clobber
# each other's logs when sharing the same data dir
TAG="$(basename "${CSV%.csv}")"

if [ $# -ge 4 ]; then
    WORKER="$4"
    LOG="$DATA/logs/sim_${TAG}_w${WORKER}.log"
    n=0
    while true; do
        conda run -n "$CONDA_ENV" python "$SCRIPT_DIR/run_sim.py" \
            --config-csv "$CSV_PATH" \
            --lc-dir "$DATA/lightcurves" \
            --out-dir "$DATA/results" \
            --n-sims 100000 --stride "$NWORKERS" --worker "$WORKER" \
            --filter-col "$FILTER_COL" --filter-value "$FILTER_VALUE" \
            --enforce-leakage-margin "$ENFORCE_LEAKAGE" \
            --models "$MODELS" \
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
echo "Launched $NWORKERS workers (models=$MODELS, env=$CONDA_ENV) against $CSV_PATH (logs: $DATA/logs/sim_${TAG}_w*.log)"
