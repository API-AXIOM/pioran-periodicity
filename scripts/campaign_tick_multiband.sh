#!/bin/bash
# Idempotent single-shot "tick" for the multi-band FP (null) campaign --
# same pattern as campaign_tick.sh (advances at most one step per
# invocation, cron-driven, no long-lived process to die mid-campaign), kept
# as a separate script rather than folding into campaign_tick.sh's
# CAMPAIGNS array because these two runs need --multiband threaded through
# (MULTIBAND=true, see run_workers.sh) and have a very different measured
# cost profile -- see workspace/multiband_cost/ in pioran_periodicity_ai
# and [[multiband-campaign-cost-model]]: OBPL multiband on LSST is ~4x
# DRW's cost (obpl+obpl+sine ~142 min/LC, all four models ~176 min/LC for
# a 6-band LSST object with mu_b).
#
# Launches the STAGE-1 CSVs (*_n20.csv, 20 reps/cell) first. Once those
# finish, re-run against the full *_case.csv (100 reps/cell) to add the
# remaining 80 reps/cell -- run_sim.py/run_workers.sh skip already-cached
# .npz/result files, so this is a safe, resumable extension, not a rerun.
#
# Usage (crontab -e):
#   */5 * * * * ROOT=/home/user/work/data/quasar_cadences N_WORKERS=10 \
#       CONDA_ENV=pioran-periodicity \
#       bash /path/to/pioran-periodicity/scripts/campaign_tick_multiband.sh \
#       >> /home/user/work/data/quasar_cadences/tick_multiband_cron.log 2>&1
#
# Env vars: ROOT, N_WORKERS, CONDA_ENV (same meaning as campaign_tick.sh).
#   Pass STAGE=full (default: n20) once you're ready to extend to 100
#   reps/cell.

set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:?set ROOT, e.g. ~/work/data/quasar_cadences}"
N_WORKERS="${N_WORKERS:?set N_WORKERS, e.g. 10}"
CONDA_ENV="${CONDA_ENV:-pioran-periodicity}"
STAGE="${STAGE:-n20}"  # n20 (initial, 60 LCs/survey) or full (100/cell)

SIMS="$ROOT/simulations"
ORCH_LOG="$ROOT/orchestrator_multiband.log"
LOCK_DIR="$ROOT/.campaign_tick_multiband.lock"
STALE_LOCK_SECONDS=600
mkdir -p "$SIMS"

if [ -d "$LOCK_DIR" ]; then
    lock_age=$(( $(date +%s) - $(date -r "$LOCK_DIR" +%s 2>/dev/null || echo 0) ))
    if [ "$lock_age" -gt "$STALE_LOCK_SECONDS" ]; then
        rmdir "$LOCK_DIR" 2>/dev/null
    fi
fi
if ! mkdir "$LOCK_DIR" 2>/dev/null; then
    exit 0
fi
trap 'rmdir "$LOCK_DIR" 2>/dev/null' EXIT

log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') $*" | tee -a "$ORCH_LOG"
}

suffix=""
[ "$STAGE" = "n20" ] && suffix="_n20"
CAMPAIGNS=(
    "ztf_multiband_null_case"
    "lsst_multiband_null_case"
)

campaign_is_finished() {
    local data="$1" name="$2"
    local w done_count=0
    for w in $(seq 0 $((N_WORKERS - 1))); do
        local wlog="$data/logs/sim_${name}${suffix}_w${w}.log"
        if [ -f "$wlog" ] && grep -qE "finished cleanly|giving up after" "$wlog"; then
            done_count=$((done_count + 1))
        fi
    done
    [ "$done_count" -ge "$N_WORKERS" ]
}

for name in "${CAMPAIGNS[@]}"; do
    data="$SIMS/$name"
    csv="$ROOT/scenario_csvs/${name}${suffix}.csv"
    marker="$data/.${STAGE}_launched"

    if [ ! -f "$marker" ]; then
        log "=== launching $name stage=$STAGE ($N_WORKERS workers) ==="
        env MODELS=drw,obpl MULTIBAND=true CADENCE_LIBRARY="$ROOT/cadence_library" \
            CONDA_ENV="$CONDA_ENV" \
            bash "$SCRIPT_DIR/run_workers.sh" "$data" "$csv" "$N_WORKERS"
        mkdir -p "$data"
        touch "$marker"
        exit 0
    fi

    if campaign_is_finished "$data" "$name"; then
        continue
    fi

    exit 0
done

log "orchestrator: multiband null campaign (stage=$STAGE) ALL DONE"
