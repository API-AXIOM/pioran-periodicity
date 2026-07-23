# Changelog

All notable changes to `pioran_periodicity`. Analysis-level decisions (prior
choices, leakage acceptance, per-run scope) are recorded separately in
`comparison_reports/changes_and_decisions.md`.

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
