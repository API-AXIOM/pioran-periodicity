# Scripts

CLI drivers built on the `pioran_periodicity` package's public API. Each
script is standalone (`argparse`-driven) and takes explicit `--data-dir`,
`--config-csv`, `--results-dir` / `--out-dir` paths — none of them assume a
particular repository layout, so point them at wherever your inputs/outputs
live (a `PIORAN_DATA_DIR` environment variable convention is recommended if
you script these together, but not required).

Run all scripts via `conda run -n <env> python scripts/<script>.py --help`
for the full argument list; each has a usage example in its module
docstring. Deferred/heavy imports (`ultranest`, `pioranpy`) mean each script
needs an environment with the full runtime dependencies installed
(`pip install -e ".[simulation]"` plus a working `pioranpy`).

## `run_sim.py`

Simulates light curves from a bending-power-law PSD (given a scenario config
CSV) and fits DRW / CARMA / OBPL noise models (each with/without a sine
mean) via nested sampling. Resumable — reuses cached light curves and skips
existing fit-result JSONs. Splittable across workers with `--stride`/`--worker`.

Expected config CSV columns: `ID, bendfreq, lowalpha, highalpha, sharpness,
rms, NumofWINDOW, NightsperWINDOW, OBSperiod, WINDOWwidth, dataLOSSfrac,
noiseSIGMA, simSEED, sampleSEED`, plus optionally `period, A1` — when
present (and `A1 != 0`) a true sine signal (amplitude `A1` directly, zero
phase) is injected into the simulated light curve, making the scenario a
detection-power study rather than a null/false-positive-rate one. The sine
model's period prior is automatically widened to cover the CSV's own max
`period` (see the module docstring for both conventions).

Outputs: `<lc-dir>/<ID>.npz` (cached light curves), `<out-dir>/<ID>_<model>.json`
(FitResult JSONs, `<model>` in `drw`, `drw_sine`, `carma`, `carma_sine`,
`obpl`, `obpl_sine`).

## `make_slope_robustness_csv.py`

Builds a `run_sim.py` config CSV for the DRW-robustness sensitivity study:
crosses a swept PSD high-frequency slope (`highalpha`) with either no true
signal (`--variant null`, a false-positive-rate scenario) or period-specific
injected-sine amplitude triads (`--variant signal`, a detection-power
scenario; default triads were empirically calibrated per period, not
guessed — see the module docstring for the calibration batches behind them).

## `run_workers.sh`

Launches N crash-restarting `run_sim.py` workers against a scenario CSV,
splitting it by `--stride`/`--worker`. Resumable (safe to relaunch after any
crash or manual stop — cached light curves and existing fit JSONs are
skipped). Defaults (`MODELS=drw`, `FILTER_COL=lowalpha`,
`FILTER_VALUE=-1.0`, `ENFORCE_LEAKAGE=false`) match the DRW-robustness
study's scenario CSVs; override via environment variables for other
scenarios (e.g. `MODELS=all FILTER_VALUE=0.0 ./run_workers.sh ...`). See the
script header for the full variable list.

    ./run_workers.sh <data-dir> <csv-path> <n-workers>

## `plot_bf_results.py`

CLI for `pioran_periodicity.visualization` — turns `aggregate_results.py`
summary JSONs into strip / power-curve / FPR-calibration / ROC figures
(`strip`, `power-curves`, `fpr-calibration`, `roc` subcommands).

## `run_realdata.py`

Fits PG 1302-102 and PG 1553+113 real light curves with DRW / OBPL noise
models against sine / linear / sine+linear mean alternatives. Needs a
`--data-dir` containing `graham2015data.csv` and `PG1553_113_logbase.txt`
(small samples of both ship in `tests/data/AGNobsdata/` for testing; use the
full survey data for a real re-analysis).

Outputs: `<out-dir>/run_config.json` (prior configs + approximation
diagnostics) and `<out-dir>/<source>/<model>.json` FitResult files.

## `aggregate_results.py`

Aggregates a scenario's `<ID>_<model>.json` FitResults (as produced by
`run_sim.py`) into per-configuration log10 Bayes factors (red-noise vs
red-noise+sine), classified with Kass-Raftery thresholds.

## `aggregate_bendfreq.py`

Aggregates `log10_fbend` posteriors across a set of `run_sim.py` fits into
an "average histogram" summary (bend-frequency estimation studies).

## `aggregate_realdata.py`

Aggregates `run_realdata.py` FitResults into a summary JSON (posterior
medians, Bayes factors) and an optional markdown report.
