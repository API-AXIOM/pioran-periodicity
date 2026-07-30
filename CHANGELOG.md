# Changelog

All notable changes to `pioran_periodicity`. Analysis-level decisions (prior
choices, leakage acceptance, per-run scope) are recorded separately in
`comparison_reports/changes_and_decisions.md`.

## [0.1.3] — 2026-07-29
### Added
- `visualization.py`: `plot_detection_rate` (single-curve P(detect)-vs-one
  swept-column with binomial standard-error bars; the no-series sibling of
  `plot_power_curves`), `plot_matched_roc` (ROC pairing a null and an
  alternative table by a *shared* swept column, e.g. both tables swept over
  `highalpha`, rather than one null held fixed across an alt series like
  `plot_roc`), `filter_table` (subset a summary table by a predicate on its
  parsed key). `plot_detection_rate`/`plot_power_curves` gained an
  `invert_xaxis` flag (default off).
- `scripts/aggregate_results.py`: `--config-csv` is now optional. Core logic
  extracted into an importable `build_table(results_dir, group_cols,
  config_csv=None)`; omitting `--config-csv` groups directly off each
  FitResult JSON's own `meta` (`highalpha`, `true_period`, `true_A1`)
  instead of joining a CSV -- needed when a results directory was
  assembled from several pilot/extension config CSVs with non-contiguous
  ID ranges never reconciled into one canonical CSV.
- `paper/plot_drw_robustness.py` (new): FPR/detection-power/ROC figures for
  the DRW-robustness sensitivity study, output to `paper/figures/`.
### Fixed
- `plot_detection_rate`/`plot_power_curves` now support the
  "steepens-to-the-right" x-axis convention used by slope-sweep studies via
  `invert_xaxis=True`, instead of only the library's default ascending
  order.

## [0.1.2] — 2026-07-22
### Changed
- `SamplerSettings.step_sampler_min_ndim` default 6 → **1**: a slice sampler
  is now attached to **every** model, so all fits in a study share one
  sampling method (no region-vs-slice heterogeneity). Motivated by ndim-5
  DRW+sine fits occasionally stalling under region rejection sampling.
### Fixed
- Two unit tests updated for the new default (settings key-set; resampling
  stub disables the step sampler).

## [0.1.1] — 2026-07-17
### Added
- Slice sampler (`ultranest.stepsampler.SliceSampler` with
  `generate_mixture_random_direction`) attached for ndim ≥
  `step_sampler_min_ndim` (default 6 at this version). Fixes region-rejection
  sampling stalls that truncated high-dimensional fits at `max_ncalls`.
- `FitResult.ess` (posterior effective sample size) and `FitResult.converged`;
  `run_nested` warns loudly on truncation or low ESS (logZ/intervals
  unreliable).

## [0.1.0] — 2026-07-17
### Added
- Initial release: cleaned common core for GP periodicity detection with
  Pioran.jl (via pioranpy) and ultranest. Modules `priors`, `means`,
  `kernels`, `models`, `inference`, `simulate`, `data`. Implements the M1–M7 /
  S1–S4 / B1–B5 fixes documented in `comparison_reports/bugfix_report.tex`.
