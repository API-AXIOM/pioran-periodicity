#!/bin/bash
# Idempotent single-shot "tick" for the real-cadence campaign sequence:
# advances by AT MOST one step (launches the next not-yet-started campaign,
# or does nothing if the current one is still in progress), then exits.
# Meant to be invoked periodically by cron rather than run as one
# long-lived process -- see scripts/REMOTE_RUN.md.
#
# Why this replaces orchestrate_campaign.sh: that script ran as ONE
# long-lived process for the whole ~week campaign, sleeping in a poll loop
# between launches. That process itself was a single point of failure --
# observed 2026-08-03: campaigns 1-2 (ztf/lsst null case) finished (their
# own workers, launched via run_workers.sh's nohup+disown, survived and
# completed independently), but nothing ever launched campaign 3, because
# the orchestrator process that was supposed to notice and launch it had
# died silently (most likely cause: systemd-logind / session-cgroup
# cleanup on SSH disconnect, which can kill a user's whole process tree,
# tmux included, unless linger is enabled -- but the exact cause doesn't
# matter, see below).
#
# cron is a system service, not tied to any login session -- a script it
# invokes doesn't have this failure mode: even if one invocation's process
# dies for any reason, the next tick (default every 5 min) runs fresh and
# picks up wherever things actually are on disk. Every check here is
# idempotent (reads state from disk: does this campaign's directory exist,
# do all its worker logs show a terminal state), never in-memory, so
# there's no state that can be lost between ticks.
#
# A single-run-at-a-time lock guards against overlapping ticks (e.g. a slow
# tick still finishing when the next cron fire lands). Implemented with
# `mkdir` rather than `flock`: mkdir is atomic on any POSIX filesystem and
# needs no extra utility (flock is Linux/util-linux-specific and this
# script's target OS isn't guaranteed).
#
# Usage (crontab -e):
#   */5 * * * * ROOT=/home/user/work/data/quasar_cadences N_WORKERS=10 \
#       CONDA_ENV=pioran-periodicity \
#       bash /path/to/pioran-periodicity/scripts/campaign_tick.sh \
#       >> /home/user/work/data/quasar_cadences/tick_cron.log 2>&1
#
# Env vars: same as orchestrate_campaign.sh (ROOT, N_WORKERS, CONDA_ENV).

set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:?set ROOT, e.g. ~/work/data/quasar_cadences}"
N_WORKERS="${N_WORKERS:?set N_WORKERS, e.g. 10}"
CONDA_ENV="${CONDA_ENV:-pioran-periodicity}"

SIMS="$ROOT/simulations"
ORCH_LOG="$ROOT/orchestrator.log"
LOCK_DIR="$ROOT/.campaign_tick.lock"
STALE_LOCK_SECONDS=600  # a tick only ever mkdir+launches or checks logs;
                          # it should never legitimately hold the lock this
                          # long, so treat an older lock as abandoned by a
                          # killed-mid-run tick (e.g. `kill -9`, which skips
                          # the EXIT trap below) rather than block forever.
mkdir -p "$SIMS"

if [ -d "$LOCK_DIR" ]; then
    lock_age=$(( $(date +%s) - $(date -r "$LOCK_DIR" +%s 2>/dev/null || echo 0) ))
    if [ "$lock_age" -gt "$STALE_LOCK_SECONDS" ]; then
        rmdir "$LOCK_DIR" 2>/dev/null
    fi
fi
if ! mkdir "$LOCK_DIR" 2>/dev/null; then
    exit 0  # another tick is already in flight; nothing to do
fi
trap 'rmdir "$LOCK_DIR" 2>/dev/null' EXIT

log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') $*" | tee -a "$ORCH_LOG"
}

# Same six campaigns, same order, as orchestrate_campaign.sh.
CAMPAIGNS=(
    "ztf_real_cadence_null_case|CADENCE_LIBRARY=$ROOT/cadence_library"
    "lsst_real_cadence_null_case|CADENCE_LIBRARY=$ROOT/cadence_library"
    "original_cadence_longsim_null_case|N_SAMPLES=8388608 ENFORCE_LEAKAGE=true"
    "ztf_real_cadence_signal_case|CADENCE_LIBRARY=$ROOT/cadence_library"
    "lsst_real_cadence_signal_case|CADENCE_LIBRARY=$ROOT/cadence_library"
    "original_cadence_longsim_signal_case|N_SAMPLES=8388608 ENFORCE_LEAKAGE=true"
)

campaign_is_finished() {
    local data="$1" name="$2"
    local w done_count=0
    for w in $(seq 0 $((N_WORKERS - 1))); do
        local wlog="$data/logs/sim_${name}_w${w}.log"
        if [ -f "$wlog" ] && grep -qE "finished cleanly|giving up after" "$wlog"; then
            done_count=$((done_count + 1))
        fi
    done
    [ "$done_count" -ge "$N_WORKERS" ]
}

for entry in "${CAMPAIGNS[@]}"; do
    name="${entry%%|*}"
    extra_env="${entry#*|}"
    data="$SIMS/$name"
    csv="$ROOT/scenario_csvs/$name.csv"

    if [ ! -d "$data" ]; then
        log "=== launching $name ($N_WORKERS workers) ==="
        env MODELS=drw,obpl CONDA_ENV="$CONDA_ENV" $extra_env \
            bash "$SCRIPT_DIR/run_workers.sh" "$data" "$csv" "$N_WORKERS"
        exit 0  # one launch per tick; next tick will check on it
    fi

    if campaign_is_finished "$data" "$name"; then
        continue  # already done -- check whether the NEXT one needs starting
    fi

    # $data exists but isn't finished yet: still in progress. Its own
    # workers self-heal via run_workers.sh's restart loop; nothing for
    # this tick to do.
    exit 0
done

log "orchestrator: ALL CAMPAIGNS DONE"
