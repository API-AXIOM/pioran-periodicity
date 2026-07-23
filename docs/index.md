# pioran-periodicity

Bayesian periodicity detection in irregularly sampled AGN light curves:
Gaussian-process red-noise models (DRW, CARMA, one-bend power law via
[Pioran.jl](https://github.com/mlefkir/Pioran.jl) / pioranpy) compared
against periodic alternatives with nested sampling (ultranest).

- [Installation](installation.md)
- [API reference](api.md)
- Tutorials: `tutorials/README.md` in the repository
- Scripts: `scripts/README.md` in the repository

## Design rules

1. **Shared parameters, shared priors.** A noise model's parameters are
   defined once and reused verbatim by the non-periodic model and every
   periodic/trend alternative. Bayes factors inside a `ModelFamily` compare
   models that differ *only* by their mean function.
2. **No data peeking.** Process variances are sampled parameters with priors
   set from unit conventions — kernels never normalise themselves against
   the light curve being fitted.
3. **Measurable approximations.** OBPL kernels use an explicit
   `FrequencyBand` (robust to sampling-pattern outliers), a component-density
   check, and `psd_approximation_error()` to quantify basis-expansion accuracy.
4. **Reproducibility.** Base-10 logs everywhere, explicit ultranest settings
   recorded per fit, seeded posterior resampling.
