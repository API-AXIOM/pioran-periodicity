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
  `scripts/campaign_tick.sh`, `pyproject.toml`. Simplest: rsync the whole
  repo.
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

Use `scripts/campaign_tick.sh` via **cron**, not a long-lived foreground/
`tmux`/`nohup` process. An earlier version of this doc recommended
`scripts/orchestrate_campaign.sh`, one process that stays alive for the
whole ~week campaign in a sleep-poll loop — that has a real failure mode
(observed 2026-08-03): the process itself can be silently killed (e.g.
`systemd-logind` reaping a user's whole process tree on session end, which
can take `tmux` and `nohup`'d children down too, depending on system
config), and once it's gone, sequencing just stops — even though each
already-launched campaign's own workers survive independently and finish
normally. `campaign_tick.sh` has no such single point of failure: it does
one idempotent check ("what's on disk — launch the next unstarted campaign,
or do nothing if the current one is still running"), then exits. `cron`
itself is a system service, not tied to any login session, so it doesn't
matter if any individual tick's process dies — the next one just runs.
`orchestrate_campaign.sh` is no longer the recommended launch path (kept in
the repo only for the record).

```bash
ROOT=<root>   # e.g. ~/work/data/quasar_cadences -- holds cadence_library/,
              # scenario_csvs/, and where simulations/ will be created
```

**Set up the cron job:**

```bash
crontab -e
```

Add a line (adjust paths, `<N_WORKERS>` = your full 8-10 CPUs, `CONDA_ENV`
if it's not literally `pioran-periodicity`):

```
*/5 * * * * ROOT=<root> N_WORKERS=<8-10> CONDA_ENV=pioran-periodicity bash <path-to-pioran-periodicity>/scripts/campaign_tick.sh >> <root>/tick_cron.log 2>&1
```

Every 5 minutes, cron checks whether the current campaign has finished and,
if so, launches the next one; if a campaign is still running it does
nothing. **Progress and campaign transitions are logged to
`$ROOT/orchestrator.log`** (not `tick_cron.log`, which just captures each
tick invocation's own stdout/stderr — normally empty). Once all six
campaigns finish, `orchestrator.log` gets a line reading `ALL CAMPAIGNS
DONE` — **remove the cron entry at that point** (`crontab -e`, delete the
line), otherwise it keeps ticking (harmlessly, but noisily) forever.

**If you're recovering from a stalled `orchestrate_campaign.sh` run** (as
happened here): no need to clean anything up first. Check
`ps aux | grep orchestrate_campaign` and `tmux ls` — if the old process or
session is still alive, kill/close it (it's not doing anything useful, but
no need to race it); if it's already dead, nothing to do. Then just set up
the cron job above. The first tick will look at `$ROOT/simulations/` on
disk, see that `ztf_real_cadence_null_case` and `lsst_real_cadence_null_case`
already have all `N_WORKERS` logs reporting `finished cleanly`, skip both
without touching them, and launch `original_cadence_longsim_null_case` —
exactly where the original run actually stalled. Verified locally with a
synthetic 3-campaign test replicating this exact "first two already
finished" state before recommending this fix.

**Under the hood**, each tick's campaign launch is one `run_workers.sh`
call — see `campaign_tick.sh`'s `CAMPAIGNS` array for the full list, in
order (nulls first, cheap and fast; then the three larger signal runs; each
gets its own `$ROOT/simulations/<name>/` directory). If you ever want to
(re)launch just one campaign manually instead of waiting for cron (e.g. to
reprioritize), here's the equivalent for the ZTF signal case:

```bash
MODELS=drw,obpl CADENCE_LIBRARY=$ROOT/cadence_library \
bash scripts/run_workers.sh "$ROOT/simulations/ztf_real_cadence_signal_case" \
    "$ROOT/scenario_csvs/ztf_real_cadence_signal_case.csv" <N_WORKERS>
```

`ENFORCE_LEAKAGE` defaults to `false` in `run_workers.sh` (documented there
for the *original* NumofWINDOW=20 campaign, which needed the override at
the old N_SAMPLES) — `campaign_tick.sh` already passes `true` explicitly for
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

## 7. Multi-band FP (null) campaign -- staged, priority

A second, separate campaign validating the multi-band model
(`pioran_periodicity.multiband`, shared-latent-process rescale trick) on
real ZTF/LSST cadences -- FPR calibration only for now (the analogous
detection-power/signal campaign is built, see below, but deliberately not
launched yet). Same real-cadence pools and highalpha convention as
sections 1-6, plus a fixed `band_amp_beta=0.35` column (one realistic
colour contrast; `run_sim.py`'s `--multiband` reads this column
automatically) and `--multiband` passed to every fit.

**Additional rsync**: `scenario_csvs/{ztf,lsst}_multiband_null_case.csv`
(100 reps/cell, 3 highalpha x 100 = 300 rows each) and their `_n20.csv`
stage-1 subsets (20 reps/cell, 60 rows each -- the first 20 rows of every
highalpha block, so extending to the full 100 later just means re-pointing
at the full CSV; already-fit IDs are skipped, not redone). IDs 70000+
(ZTF) / 80000+ (LSST), clear of every existing range (which tops out at
66399).

**Launch** (STAGE=n20 by default -- do NOT set STAGE=full until the n20
stage has actually finished and you've confirmed the real per-LC cost):

```
*/5 * * * * ROOT=<root> N_WORKERS=<see below> CONDA_ENV=pioran-periodicity \
    bash <path-to-pioran-periodicity>/scripts/campaign_tick_multiband.sh \
    >> <root>/tick_multiband_cron.log 2>&1
```

Progress: `$ROOT/orchestrator_multiband.log`. This is a separate cron
line/script from section 3's `campaign_tick.sh` -- the two campaigns are
independent and can run concurrently if the box has the cores, or
sequentially if not (just don't set N_WORKERS so high across both that
they starve each other).

**Cost (measured 2026-08-18, LSST 6-band + mu_b, one object)**: DRW
combined (drw+drw_sine) 34.6 min/LC, OBPL combined (obpl+obpl_sine) 141.8
min/LC -- **all four models together ~176 min/LC**. ZTF is unmeasured but
expected much cheaper (ZTF's DRW multiband was only 1.9x its merged-fit
cost, vs LSST's much steeper per-dimension scaling). LSST dominates
sizing: the n20 stage is 60 LSST LCs, ~176 core-hours --

| N_WORKERS | wall time (LSST n20 stage) |
|---|---|
| 10 | ~17.6 hr |
| 20 | ~8.8 hr |
| 30 | ~5.9 hr |

Extending to the full 100/cell later (300 LSST LCs) is ~880 core-hours
(~22-44 hr at 20-40 workers).

**Signal campaign (built, NOT launched)**: same treatment,
`scenario_csvs/{ztf,lsst}_multiband_signal_case.csv`, IDs 71000+ (ZTF) /
81000+ (LSST), 27 cells (3 highalpha x 3 periods x 3 A1) x 100 reps = 2700
rows each -- 9x the null campaign's cell count. Not added to any cron
script. Launching this is a deliberate follow-on decision once the null
campaign's actual (not probe) cost is in hand, not something to start
alongside it.
