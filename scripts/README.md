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
noiseSIGMA, simSEED, sampleSEED`.

Outputs: `<lc-dir>/<ID>.npz` (cached light curves), `<out-dir>/<ID>_<model>.json`
(FitResult JSONs, `<model>` in `drw`, `drw_sine`, `carma`, `carma_sine`,
`obpl`, `obpl_sine`).

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
