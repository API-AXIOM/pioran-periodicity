# Real-cadence robustness campaign — remote launch

Three campaigns, each with a signal (detection-power) and null (FPR)
variant, checking whether the periodicity-detection method's calibration —
built entirely on the original synthetic seasonal-window cadence — holds up
under (a) real ZTF cadences, (b) real LSST/Rubin cadences, and (c) the
*original* synthetic cadence re-simulated at a longer TK95 baseline that
actually clears the 10x leakage-margin check (S1), instead of the
`--enforce-leakage-margin false` override the original run used. All six
CSVs share the same highalpha sweep, period/A1 triads, and 100 reps/cell —
directly comparable to each other and, for (c), to the original
`signal_case.csv`/`null_case.csv`. **DRW and OBPL only** (no CARMA, per the
2026-07-30 scoping decision).

This machine (the one preparing this package) does **not** run the
campaign — no remote access from here. This doc is everything needed to
launch it on a machine that does.

## 1. What to rsync over

- This repo (`pioran-periodicity`), specifically needs at least:
  `pioran_periodicity/`, `scripts/run_sim.py`, `scripts/run_workers.sh`,
  `scripts/orchestrate_campaign.sh`, `pyproject.toml`. Simplest: rsync the
  whole repo.
- The prebuilt cadence library cache (~58 MB, gzip CSV — see
  `pioran_periodicity.cadence.CadenceLibrary.to_cache`):
  ```
  ~/work/data/quasar_cadences/cadence_library/
  ```
  This is all the real-cadence campaigns need at run time — the raw
  `ztf_data/`/`lsst_data/` directories (light curve CSVs, hundreds of MB)
  do NOT need to go along; the library was already built from them.
- The scenario CSVs (~few MB total):
  ```
  ~/work/data/quasar_cadences/scenario_csvs/{ztf,lsst,original_cadence_longsim}_real_cadence_*.csv
  ```
  (the `original_cadence_longsim_*` files don't reference real cadences at
  all despite the shared directory — naming follows the campaign, not the
  file's dependency on the library)

Suggested layout on the remote host (matching the existing campaigns'
convention under `~/work/data/quasar_cadences/simulations/`):

```
<remote-data-root>/
  cadence_library/            # rsynced cache
  scenario_csvs/              # rsynced CSVs
  simulations/
    ztf_real_cadence_signal_case/{lightcurves,results,logs}/
    ztf_real_cadence_null_case/{lightcurves,results,logs}/
    lsst_real_cadence_signal_case/{lightcurves,results,logs}/
    lsst_real_cadence_null_case/{lightcurves,results,logs}/
    original_cadence_longsim_signal_case/{lightcurves,results,logs}/
    original_cadence_longsim_null_case/{lightcurves,results,logs}/
```

`run_workers.sh` creates the `lightcurves`/`results`/`logs` subdirectories
itself — no need to pre-create them.

## 2. Environment

Follow `docs/installation.md` (the package's own documented procedure —
don't try to clone the local ad-hoc dev env):

```bash
conda create -n pioran-periodicity python=3.11
conda activate pioran-periodicity
pip install -e ".[simulation,test]"
conda run -n pioran-periodicity pytest tests/
```

`pioranpy` pulls in Julia + `Pioran.jl` precompilation on first import
(needs network access, a few minutes, one-time). If the remote conda env
name differs from `pioran-periodicity`, pass `CONDA_ENV=<name>` in every
`run_workers.sh` invocation below.

## 3. Launch: unattended, sequential, survives SSH disconnect

For "start it and check back after a week," use `scripts/orchestrate_campaign.sh`
— it runs all six CSVs in order (nulls first, cheap and fast; then the three
much larger signal runs), waiting for each campaign to fully finish before
starting the next, and logs progress to `$ROOT/orchestrator.log`. It does
**not** detach itself, so run it inside `tmux` (recommended — lets you
reattach and watch it later) or `nohup`+`disown` if `tmux` isn't available.

```bash
ROOT=<root>   # e.g. ~/work/data/quasar_cadences -- holds cadence_library/,
              # scenario_csvs/, and where simulations/ will be created
```

**Option A — tmux (recommended):**

```bash
tmux new -s cadence_campaign
# inside the tmux session:
cd <path-to-pioran-periodicity>
ROOT=<root> N_WORKERS=<8-10> CONDA_ENV=pioran-periodicity \
    bash scripts/orchestrate_campaign.sh
# detach: Ctrl-b then d -- the session (and everything in it) keeps running.
```

Reattach any time, from any SSH session, with `tmux attach -t cadence_campaign`.
If the session already looks detached-but-alive when you reattach, that's
expected — it just means it's between campaigns or mid-poll.

**Option B — nohup + disown (no tmux):**

```bash
cd <path-to-pioran-periodicity>
ROOT=<root> N_WORKERS=<8-10> CONDA_ENV=pioran-periodicity \
    nohup bash scripts/orchestrate_campaign.sh > "<root>/orchestrator_stdout.log" 2>&1 &
disown
```

Either way, `<N_WORKERS>` should be your full 8-10 CPUs — the orchestrator
runs campaigns one at a time, so there's no benefit to splitting cores
across simultaneous campaigns, and running each to completion first means
results for earlier campaigns land sooner rather than all six trickling in
together.

**If the machine or the orchestrator process itself dies** (not the
individual fit workers, which already self-heal via `run_workers.sh`'s own
100-restart loop) — just rerun the exact same command again. Verified
locally: a full rerun after everything already finished completes in the
same second, because `run_sim.py` skips every already-cached light curve
and already-written result file. There's no "where did it leave off"
bookkeeping to do by hand.

**Under the hood**, each campaign is one `run_workers.sh` call; if you ever
want to (re)launch just one campaign manually instead of the whole
orchestrator (e.g. to reprioritize), here's the equivalent for the ZTF
signal case — see `orchestrate_campaign.sh`'s `CAMPAIGNS` array for the
other five (each campaign gets its OWN `$ROOT/simulations/<name>/`
directory; don't point two campaigns at the same one):

```bash
MODELS=drw,obpl CADENCE_LIBRARY=$ROOT/cadence_library \
bash scripts/run_workers.sh "$ROOT/simulations/ztf_real_cadence_signal_case" \
    "$ROOT/scenario_csvs/ztf_real_cadence_signal_case.csv" <N_WORKERS>
```

`ENFORCE_LEAKAGE` defaults to `false` in `run_workers.sh` (documented there
for the *original* NumofWINDOW=20 campaign, which needed the override at
the old N_SAMPLES) — the orchestrator already passes `true` explicitly for
`original_cadence_longsim`, now that the longer simulation clears the
margin; the `*_real_cadence_*` campaigns don't need it touched at all.

## 4. Runtime estimate (measured, not guessed)

One full light curve (DRW + DRW+sine + OBPL + OBPL+sine, 4 fits) took
**~183s** in local timing (43s DRW family, 140s OBPL family) — but this
varies with epoch count, which spans a wide range in the real-cadence pools
(84-4764 for ZTF/LSST field objects, up to 47,950 for the ~1 LSST Deep
Drilling Field object each real-cadence pool picked up).

- Total light curves: 18,000 (6000 per campaign x 3 campaigns).
- Sequential: 18,000 x 183s ~= 915 hours (~38 days).
- **With 8 workers: ~4.8 days. With 10 workers: ~3.8 days.** Fits inside a
  one-week budget with some margin, but the DDF-object variance above could
  push this up -- if wall time is tracking noticeably worse than this
  estimate after the first day, that's the likely cause, not a bug.

## 5. Monitoring

```bash
# overall progress: which campaign is running/done, since when
tail -20 $ROOT/orchestrator.log

# fit-result files so far for the campaign in progress (each fully-fit row
# -> 4 files: drw, drw_sine, obpl, obpl_sine)
ls $ROOT/simulations/ztf_real_cadence_signal_case/results/ | wc -l

# tail a worker's log
tail -f $ROOT/simulations/ztf_real_cadence_signal_case/logs/sim_ztf_real_cadence_signal_case_w0.log
```

If `orchestrator.log` shows a campaign was "finished" unusually fast, check
that campaign's worker logs for `giving up after 100 restarts` rather than
`finished cleanly` — that means a worker kept crashing on the same row
(genuine bug, not resolved by restarting) and the orchestrator moved on
anyway, since it only checks for a terminal state, not which one:

```bash
grep -L "finished cleanly" $ROOT/simulations/*/logs/*.log
```

## 6. Not part of this campaign

Aggregation (`scripts/aggregate_results.py`) and plotting
(`pioran_periodicity.visualization`) are unchanged and will work against
these results once they land, the same way they work against
`signal_case`/`null_case` -- not run here, since the campaign itself hasn't
run yet.
