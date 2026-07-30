#!/bin/bash
# Runs all six real-cadence-campaign CSVs (3 null, 3 signal) SEQUENTIALLY,
# unattended, for as long as it takes -- see scripts/REMOTE_RUN.md for the
# full picture. Meant to run inside tmux or under nohup+disown so it
# survives SSH disconnect (this script does not detach itself).
#
# Each campaign is launched via run_workers.sh (which backgrounds its own
# crash-restarting workers via nohup+disown and returns immediately), then
# this script POLLS that campaign's worker logs until all of them report a
# terminal state (finished cleanly, or gave up after 100 restarts) before
# moving to the next campaign.
#
# Crash-resistant at every level:
#   - a single fit crashing (e.g. a Julia GC segfault) -> that worker's own
#     while-loop in run_workers.sh restarts it (up to 100x), unaffected by
#     this script.
#   - this orchestrator process itself dying (not expected, but not assumed
#     impossible) -> just rerun it from scratch. Every campaign is resumable
#     (run_sim.py skips cached .npz light curves and existing result JSONs),
#     so a campaign that already finished will see nothing left to do and
#     its workers will report "finished cleanly" almost immediately, and the
#     script proceeds to wherever it actually left off. No manual bookkeeping
#     needed to figure out where it was.
#
# Usage:
#   ROOT=~/work/data/quasar_cadences N_WORKERS=10 \
#       bash scripts/orchestrate_campaign.sh
#
# Env vars:
#   ROOT         data root holding cadence_library/, scenario_csvs/,
#                simulations/ (required)
#   N_WORKERS    workers per campaign (required) -- keep this the SAME
#                across every invocation for a given campaign; the
#                completion check looks for exactly this many worker logs
#   CONDA_ENV    default: pioran-periodicity
#   POLL_SECONDS how often to check for campaign completion (default 300)

set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:?set ROOT, e.g. ~/work/data/quasar_cadences}"
N_WORKERS="${N_WORKERS:?set N_WORKERS, e.g. 10}"
CONDA_ENV="${CONDA_ENV:-pioran-periodicity}"
POLL_SECONDS="${POLL_SECONDS:-300}"

SIMS="$ROOT/simulations"
ORCH_LOG="$ROOT/orchestrator.log"
mkdir -p "$SIMS"

log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') $*" | tee -a "$ORCH_LOG"
}

# name|extra-env(space-separated VAR=val, expanded to a single string here)
# Null cases first (cheap, ~1/3 the rows -- fast sanity check before the
# much larger signal runs).
CAMPAIGNS=(
    "ztf_real_cadence_null_case|CADENCE_LIBRARY=$ROOT/cadence_library"
    "lsst_real_cadence_null_case|CADENCE_LIBRARY=$ROOT/cadence_library"
    "original_cadence_longsim_null_case|N_SAMPLES=8388608 ENFORCE_LEAKAGE=true"
    "ztf_real_cadence_signal_case|CADENCE_LIBRARY=$ROOT/cadence_library"
    "lsst_real_cadence_signal_case|CADENCE_LIBRARY=$ROOT/cadence_library"
    "original_cadence_longsim_signal_case|N_SAMPLES=8388608 ENFORCE_LEAKAGE=true"
)

run_campaign() {
    local name="$1"
    local extra_env="$2"
    local data="$SIMS/$name"
    local csv="$ROOT/scenario_csvs/$name.csv"

    log "=== launching $name ($N_WORKERS workers) ==="
    env MODELS=drw,obpl CONDA_ENV="$CONDA_ENV" $extra_env \
        bash "$SCRIPT_DIR/run_workers.sh" "$data" "$csv" "$N_WORKERS"

    while true; do
        local done_count=0
        local w
        for w in $(seq 0 $((N_WORKERS - 1))); do
            local wlog="$data/logs/sim_${name}_w${w}.log"
            if [ -f "$wlog" ] && grep -qE "finished cleanly|giving up after" "$wlog"; then
                done_count=$((done_count + 1))
            fi
        done
        if [ "$done_count" -ge "$N_WORKERS" ]; then
            break
        fi
        sleep "$POLL_SECONDS"
    done
    log "=== finished $name ==="
}

log "orchestrator starting: ${#CAMPAIGNS[@]} campaigns, $N_WORKERS workers each"
for entry in "${CAMPAIGNS[@]}"; do
    name="${entry%%|*}"
    extra_env="${entry#*|}"
    run_campaign "$name" "$extra_env"
done
log "orchestrator: ALL CAMPAIGNS DONE"
