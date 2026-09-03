# Changelog

All notable changes to `pioran_periodicity`. Analysis-level decisions (prior
choices, leakage acceptance, per-run scope) are recorded separately in
`comparison_reports/changes_and_decisions.md`.

## [0.3.0] — 2026-09-03

Breaking. Adds one sampled parameter to **every** model.

### Changed
- The process mean is now **fitted, not fixed**. Every model previously had
  its constant offset fixed by subtracting `np.median(y)` (or the reference
  band's median) before evaluating a hard zero-mean GP. Fixing it at a
  data-derived point estimate discards that estimate's own uncertainty, which
  is entangled with the same red-noise kernel parameters — `log10_fbend` in
  particular — that the fit is trying to recover, giving overconfident
  timescale posteriors. `mu0 ~ Normal(0, process_mean_scale)` is now present
  in every variant and fit jointly with the kernel:
  single-band it is added to the mean function; multi-band it becomes
  `band_mu[0]`, the reference band's own absolute offset (previously
  hardcoded 0.0). `a_ref = 1` remains pinned — a separate identifiability
  constraint against `log10_variance`.
  Median-centring is deliberately **kept**: it is a change of origin chosen
  for conditioning, not a statistical assumption, and `mu0` is the offset
  relative to it. Removing the centring would require a prior spanning the
  data's absolute level (~19 mag for real photometry).
- `linear` / `sine+linear` variants drop `intercept`, with which `mu0` would
  be exactly degenerate. `PriorConfig.intercept` is removed.
- Costs one dimension on every model (~1.2–1.3× per fit). Null and
  alternative both carry it, so Bayes factors are largely unaffected.

### Fixed
- `scripts/run_realdata.py` passed an `intercept=` kwarg to `PriorConfig`
  that no longer exists — a breakage nothing caught because no test imported
  the module. Added a test that the campaign configs construct.

### Notes
- Tests cover that `mu0` is wired correctly: present in every variant, the
  same shared `Parameter` object across variants (M1/B4), applied on the
  correct side of the `a_b` rescale (it is in observed units, so it belongs
  in `band_mu[0]` and never in `mean_func`, which is scaled by `a_b`), and
  that fits run end to end. They deliberately do **not** attempt to verify
  the statistical claim about `log10_fbend` coverage.
- Supersedes the unmerged `fix/free-mean-parameter` (734b785), which dropped
  the centring and built `mu0` inside the per-variant loop, breaking the
  M1/B4 invariant.

## [0.2.0] — 2026-09-03

Breaking. Renamed sampled parameters, changed prior families, and changed
campaign defaults. **Every simulated light curve and every fit produced
before this release is invalid** — see the invalidation list below. Full
write-up, with the model equations and a parameter glossary, in
`comparison_reports/bugfix_report.tex` (MB-series).

### Fixed
- **MB1** `kernels.gp_log_likelihood_multiband` subtracted the mean function
  *before* dividing by `a_b`, fitting `y_b = mu_b + m(t) + a_b*x(t)` — a
  band-independent sine against band-dependent red noise — while
  `simulate.sample_real_cadence` injects `a_b*(x + m)`. Both docstrings
  claimed the two matched. Verified against a dense multivariate normal.
  This is what made the steep-slope false positives vanish in the multi-band
  LSST null relative to the single-band campaign. The rescale Jacobian and
  the `yerr/a` scaling were already correct.
- **MB3.3** ZTF `magerr` was consumed as a fractional *flux* error; added
  `simulate.MAG_TO_FRACTIONAL_FLUX = 0.4*ln(10)`, so both survey noise
  prescriptions are now expressed in the same units.
- **MB3.4** the sine period bound relied on `max(4.0, nan)` returning 4.0,
  true only by Python's argument order; replaced with `np.isfinite`
  filtering in the new `run_sim.resolve_period_prior`.

### Changed
- **MB2** sine model parameters `A1`/`A2` → **`A_cos`/`A_sin`** (they are
  coefficients, and `A1` collided with the scenario-CSV column of the same
  name, which is an injected *amplitude* and was passed into the `A2` slot).
  New `means.sine_amplitude()` gives the derived amplitude — the only
  quantity comparable to the CSV's `A1`. Result files written earlier keep
  the old sample keys; `paper/plot_fits.py` and `plot_corners.py` read both.
- **MB3.1** the sine period prior is now the fixed `run_sim.PERIOD_PRIOR =
  (0.2, 8.0)` yr for every scenario, overridable only via `--period-max`. It
  used to be derived per CSV, silently giving nulls `(0.2, 4.0)` and signals
  `(0.2, 8.0)`, so false-positive thresholds were calibrated under a
  different model than detection power (~0.7 nat).
- **MB3.5** campaign CSVs set `sharpness = 1.0` (was 10.0). Pioran's
  `SingleBendingPowerLaw` has no sharpness parameter, so simulating at 10
  put a knee in the data the fitted OBPL cannot represent — a factor 1.87
  (0.27 dex) PSD discrepancy at the bend, inside the science band.
- **MB3.2** `HIGHALPHA_DEFAULT` now `-2.0,-2.3,-2.6,-2.9,-3.2,-3.5`. A truth
  of `-4.0` maps to `alpha_high = 4.0`, exactly the prior bound, which is
  itself the SHO/n=20 basis-accuracy limit (3% error at 4.0, 35% at 4.5).
- Sine amplitude prior is now **hierarchical**: `A_cos, A_sin ~ Normal(0,
  f0*sigma)` with `sigma` a sampled parameter (new
  `priors.ProcessRelativeNormal`), so the prior is on the dimensionless
  `f = A/sigma` and is scale-free across objects. `PriorConfig.
  sine_amplitude_fraction = 1.2`; `sine_amplitude_scale` is retained as the
  CARMA fallback (CARMA has no sampled process variance) and the choice is
  recorded in `meta["sine_amplitude_prior"]`.
- Sine period prior is now **log-uniform** (period is a scale parameter).

### Added
- `beta_sine` (`build_family(fit_sine_colour=True)`, `run_sim
  --fit-sine-colour`, off by default): gives the periodic component its own
  per-band amplitude `c_b = (lambda_b/lambda_ref)**-beta_sine`, independent
  of the noise's `a_b`. Without it the sine and the noise share a colour and
  multi-band data cannot distinguish a real signal from red-noise leakage.
  One parameter regardless of filter count; measured cost ndim 11→12,
  runtime ×1.33.
- `tests/test_model_conventions.py` (66 tests) pins the simulator/likelihood
  contract. Each test checks against an independent reference **and**
  asserts the rival convention differs for that input, so it cannot
  degenerate the way the old `mean_func=None` brute-force check did.

### Invalidates
- All simulated light curves (MB3.5 changed the injection PSD — re-simulate,
  not refit), all multi-band fits (MB1; every multi-band CSV uses
  `beta=0.35`), all null fits (MB3.1), all ZTF simulations (MB3.3).

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
