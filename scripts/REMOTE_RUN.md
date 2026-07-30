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
  `pyproject.toml`. Simplest: rsync the whole repo.
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

## 3. Launch commands

Run the null (cheap, ~1/3 the rows) variants first — fast sanity check and
FPR results land within hours, before committing to the much larger signal
runs. All commands assume `cd` into the repo. `<root>` = the remote data
root from step 1 (holds `cadence_library/` and `scenario_csvs/`); each
campaign gets its OWN data directory under `<root>/simulations/` (matching
the existing campaigns' convention and `run_workers.sh`'s own `--lc-dir`/
`--out-dir` = `$DATA/lightcurves`/`$DATA/results` layout) — do NOT pass
`<root>` itself as `$DATA`, or all six campaigns' light curves/results will
land in one shared, unsorted directory.

```bash
ROOT=<root>   # e.g. ~/work/data/quasar_cadences
SIMS=$ROOT/simulations

# --- null cases (600 rows each, ~1800 total) ---

MODELS=drw,obpl \
CADENCE_LIBRARY=$ROOT/cadence_library \
bash scripts/run_workers.sh "$SIMS/ztf_real_cadence_null_case" \
    "$ROOT/scenario_csvs/ztf_real_cadence_null_case.csv" <N_WORKERS>

MODELS=drw,obpl \
CADENCE_LIBRARY=$ROOT/cadence_library \
bash scripts/run_workers.sh "$SIMS/lsst_real_cadence_null_case" \
    "$ROOT/scenario_csvs/lsst_real_cadence_null_case.csv" <N_WORKERS>

MODELS=drw,obpl \
N_SAMPLES=8388608 \
ENFORCE_LEAKAGE=true \
bash scripts/run_workers.sh "$SIMS/original_cadence_longsim_null_case" \
    "$ROOT/scenario_csvs/original_cadence_longsim_null_case.csv" <N_WORKERS>

# --- signal cases (5400 rows each, ~16200 total -- the bulk of the runtime) ---

MODELS=drw,obpl \
CADENCE_LIBRARY=$ROOT/cadence_library \
bash scripts/run_workers.sh "$SIMS/ztf_real_cadence_signal_case" \
    "$ROOT/scenario_csvs/ztf_real_cadence_signal_case.csv" <N_WORKERS>

MODELS=drw,obpl \
CADENCE_LIBRARY=$ROOT/cadence_library \
bash scripts/run_workers.sh "$SIMS/lsst_real_cadence_signal_case" \
    "$ROOT/scenario_csvs/lsst_real_cadence_signal_case.csv" <N_WORKERS>

MODELS=drw,obpl \
N_SAMPLES=8388608 \
ENFORCE_LEAKAGE=true \
bash scripts/run_workers.sh "$SIMS/original_cadence_longsim_signal_case" \
    "$ROOT/scenario_csvs/original_cadence_longsim_signal_case.csv" <N_WORKERS>
```

`<N_WORKERS>`: with 8-10 CPUs available, use all of them per invocation
(run each campaign to completion before starting the next) rather than
splitting cores across simultaneous campaigns — same total core-hours
either way, but results land sooner per campaign for early inspection.
`ENFORCE_LEAKAGE` defaults to `false` in `run_workers.sh` (documented there
for the *original* NumofWINDOW=20 campaign, which needed the override at
the old N_SAMPLES); pass `true` explicitly for `original_cadence_longsim`
as shown above, now that the longer simulation actually clears the margin.
The two `*_real_cadence_*` campaigns don't need `ENFORCE_LEAKAGE` or
`N_SAMPLES` set at all — `cadence_source` rows pick the longer simulation
length and the margin check passes automatically (verified locally).

Each `run_workers.sh` call launches `<N_WORKERS>` detached, crash-restarting
workers and returns immediately (`nohup`+`disown`) — safe to run repeatedly
or after any crash; cached `.npz` light curves and per-fit result JSONs are
skipped on restart.

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
# fit-result files so far (each fully-fit row -> 4 files: drw, drw_sine, obpl, obpl_sine)
ls $SIMS/ztf_real_cadence_signal_case/results/ | wc -l

# tail a worker's log
tail -f $SIMS/ztf_real_cadence_signal_case/logs/sim_ztf_real_cadence_signal_case_w0.log
```

`grep -L DONE ... /logs/*.log` after all workers report `finished cleanly`
would indicate an incomplete worker (hit the 100-restart giveup limit) --
investigate that worker's log rather than assuming the campaign is done
just because the launcher returned.

## 6. Not part of this campaign

Aggregation (`scripts/aggregate_results.py`) and plotting
(`pioran_periodicity.visualization`) are unchanged and will work against
these results once they land, the same way they work against
`signal_case`/`null_case` -- not run here, since the campaign itself hasn't
run yet.
