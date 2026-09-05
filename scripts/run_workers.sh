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
#   CADENCE_LIBRARY  path to a CadenceLibrary.to_cache() dir, passed as
#                    run_sim.py --cadence-library (default: unset -- only
#                    needed for a cadence_source CSV, e.g. the real ZTF/LSST
#                    cadence campaigns; see pioran_periodicity.cadence)
#   N_SAMPLES      run_sim.py --n-samples override (default: unset -- only
#                  needed to re-simulate a synthetic-window CSV at a longer
#                  length to fix an S1 violation; cadence_source CSVs pick
#                  their own longer default automatically)
#   CONDA_ENV      conda environment to run in (default: pioran-periodicity,
#                  matching docs/installation.md)
#   MULTIBAND      true/false, passes run_sim.py --multiband (default: false
#                  -- only meaningful against a band_amp_beta CSV; see
#                  scripts/REMOTE_RUN.md before a large multiband campaign)
#   MAX_NCALLS     run_sim.py --max-ncalls (default: unset -> run_sim.py's
#                  1000000). MUST be raised for 6-band LSST multiband runs,
#                  which need 2.7M-4.9M; at the default they are silently
#                  truncated and their logz is not an evidence estimate.
#                  Use 8000000 for MULTIBAND=true on LSST cadences.
#   CHECKPOINT_DIR run_sim.py --checkpoint-dir (default: unset -> no
#                  checkpointing). Makes a truncated fit resumable rather
#                  than throwaway; needs h5py. Recommended for any run where
#                  a single fit takes more than a few minutes.
#
#   WATCHDOG       true/false, start scripts/watchdog.py alongside the
#                  workers (default: true). The supervisor loop below only
#                  restarts a worker that *exits*; it cannot see one that
#                  hangs without exiting, which is what Julia GC stalls do
#                  (8 of 12 workers lost that way on 2026-08-31, each
#                  burning ~90% CPU for 30+ h while writing nothing).
#                  Needs a python3 on PATH (stdlib only) and is most useful
#                  with CHECKPOINT_DIR set, so a killed fit resumes.
#   WATCHDOG_STALL_MIN  minutes without checkpoint progress before a worker
#                  is SIGKILLed (default: 45). Must exceed the slowest
#                  normal gap between checkpoint writes.
#   WATCHDOG_POLL  watchdog poll interval in seconds (default: 300)
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
CADENCE_LIBRARY="${CADENCE_LIBRARY:-}"
N_SAMPLES="${N_SAMPLES:-}"
CONDA_ENV="${CONDA_ENV:-pioran-periodicity}"
MULTIBAND="${MULTIBAND:-false}"
MAX_NCALLS="${MAX_NCALLS:-}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-}"
WATCHDOG="${WATCHDOG:-true}"
WATCHDOG_STALL_MIN="${WATCHDOG_STALL_MIN:-45}"
WATCHDOG_POLL="${WATCHDOG_POLL:-300}"

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
    # Built as one array (never empty -- always has the base args below) and
    # extended conditionally, rather than splicing in a possibly-empty
    # array: bash 3.2 (macOS's default /bin/bash) mishandles
    # "${EMPTY_ARRAY[@]}" under `set -u`, either erroring as an unbound
    # variable or (with the :- fallback) injecting a spurious empty-string
    # argument that would break run_sim.py's argument parsing.
    ARGS=(
        --config-csv "$CSV_PATH"
        --lc-dir "$DATA/lightcurves"
        --out-dir "$DATA/results"
        --n-sims 100000 --stride "$NWORKERS" --worker "$WORKER"
        --filter-col "$FILTER_COL" --filter-value "$FILTER_VALUE"
        --enforce-leakage-margin "$ENFORCE_LEAKAGE"
        --models "$MODELS"
    )
    [ -n "$CADENCE_LIBRARY" ] && ARGS+=(--cadence-library "$CADENCE_LIBRARY")
    [ -n "$N_SAMPLES" ] && ARGS+=(--n-samples "$N_SAMPLES")
    [ "$MULTIBAND" = "true" ] && ARGS+=(--multiband)
    [ -n "$MAX_NCALLS" ] && ARGS+=(--max-ncalls "$MAX_NCALLS")
    [ -n "$CHECKPOINT_DIR" ] && ARGS+=(--checkpoint-dir "$CHECKPOINT_DIR")
    n=0
    while true; do
        # --no-capture-output: without it `conda run` BUFFERS the child's
        # stdout/stderr and writes the lot only when the process exits. A
        # healthy worker therefore has a ZERO-BYTE log for its entire run --
        # hours -- and content appears only on a crash or on DONE, which
        # makes an empty log indistinguishable from a dead one and makes
        # `tail -f` useless for watching progress.
        conda run --no-capture-output -n "$CONDA_ENV" \
            python -u "$SCRIPT_DIR/run_sim.py" "${ARGS[@]}" \
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
# Stall watchdog. The per-worker supervisor above restarts run_sim.py when it
# *exits* (e.g. a Julia GC segfault); a worker that hangs without exiting is
# invisible to it. watchdog.py kills a worker whose checkpoint has stopped
# advancing so that supervisor can do its job. One watchdog covers a data dir,
# so don't start a second when extending a campaign with another CSV.
if [ "$WATCHDOG" = "true" ]; then
    if pgrep -f "watchdog.py $DATA" >/dev/null 2>&1; then
        echo "Stall watchdog already running for $DATA"
    elif ! command -v python3 >/dev/null 2>&1; then
        echo "WARNING: python3 not found -- stall watchdog NOT started; a hung worker will go unnoticed" >&2
    else
        nohup python3 "$SCRIPT_DIR/watchdog.py" "$DATA" \
            --stall-min "$WATCHDOG_STALL_MIN" --poll "$WATCHDOG_POLL" \
            </dev/null >/dev/null 2>&1 &
        disown
        echo "Started stall watchdog (stall_min=${WATCHDOG_STALL_MIN}min, log: $DATA/logs/watchdog.log)"
    fi
fi
echo "Launched $NWORKERS workers (models=$MODELS, multiband=$MULTIBAND, max_ncalls=${MAX_NCALLS:-default}, env=$CONDA_ENV) against $CSV_PATH (logs: $DATA/logs/sim_${TAG}_w*.log)"
