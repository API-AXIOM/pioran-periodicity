# Paper

Figure-generation scripts for publications built on `pioran_periodicity`.
These read the FitResult JSONs produced by `scripts/run_realdata.py` (and,
for `plot_fits.py`, re-evaluate the GP posterior-predictive via pioranpy) —
run the relevant `scripts/` driver first.

## `plot_fits.py`

Posterior-predictive panels (data + median curve + posterior draws) for the
DRW/OBPL fits to PG 1302-102 and PG 1553+113, one PNG per source per noise
family (`<source>_<drw|obpl>_panels.png`). Requires pioranpy (re-evaluates
the GP for each posterior draw).

```
conda run -n <env> python paper/plot_fits.py \
    --data-dir tests/data/AGNobsdata --results-dir <results-dir> --out-dir <out-dir>
```

## `plot_corners.py`

Corner plots of the posterior samples stored in each FitResult JSON — no
pioranpy/Julia call involved, just reads the JSON.

```
conda run -n <env> python paper/plot_corners.py \
    --results-dir <results-dir> --out-dir <out-dir>/corner
```

## `plot_drw_robustness.py`

FPR, detection-power, and ROC figures for the DRW-robustness sensitivity
study (`null_case`/`signal_case` from `scripts/make_slope_robustness_csv.py`
+ `scripts/run_sim.py`). Groups results off each FitResult's own `meta` via
`aggregate_results.build_table` rather than a config CSV, since these
results directories are assembled from several pilot/extension CSVs with
non-contiguous ID ranges. Uses `pioran_periodicity.visualization`'s
`plot_detection_rate`/`plot_power_curves`/`plot_matched_roc`.

```
conda run -n <env> python paper/plot_drw_robustness.py \
    --null-dir <sim-dir>/null_case/results \
    --signal-dir <sim-dir>/signal_case/results \
    --out-dir paper/figures
```
