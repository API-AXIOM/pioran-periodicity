# pioran-periodicity

Bayesian periodicity detection in irregularly sampled AGN light curves:
Gaussian-process red-noise models (DRW, CARMA, one-bend power law via
[Pioran.jl](https://github.com/mlefkir/Pioran.jl) / pioranpy) compared against
periodic alternatives with nested sampling (ultranest).

## Design rules

1. **Shared parameters, shared priors.** A noise model's parameters are
   defined once and reused verbatim by the non-periodic model and every
   periodic/trend alternative. Bayes factors inside a `ModelFamily` compare
   models that differ *only* by their mean function.
2. **No data peeking.** Process variances are sampled parameters with priors
   set from unit conventions — kernels never normalise themselves against the
   light curve being fitted.
3. **Measurable approximations.** OBPL kernels use an explicit
   `FrequencyBand` (robust to sampling-pattern outliers), a component-density
   check, and `psd_approximation_error()` to quantify basis-expansion accuracy.
4. **Reproducibility.** base-10 logs everywhere, explicit ultranest settings
   recorded per fit, seeded posterior resampling.

## Quick start

```python
import numpy as np
from pioran_periodicity import (PriorConfig, FrequencyBand, build_family,
                                fit_family, log10_bayes_factors,
                                SamplerSettings)

t, y, yerr = ...  # time (years), flux, uncertainties

cfg = PriorConfig(log10_variance=(-4, 1), log10_fbend=(-3, 2),
                  sine_amplitude_scale=0.3, period=(0.1, 8.0))
band = FrequencyBand.from_times(t)            # robust f_max (median spacing)
family = build_family("obpl", cfg, variants=("plain", "sine"),
                      band=band, n_components=20)

results = fit_family(family, t, y, yerr, settings=SamplerSettings(seed=1))
print(log10_bayes_factors(results))           # {'obpl/obpl+sine': ...}
```

Simulation utilities (`pioran_periodicity.simulate`) require the optional
`stingray` dependency and are imported separately:

```python
from pioran_periodicity.simulate import simulate_lightcurve, sample_seasonal_pattern
```

## Install

```
pip install -e ".[simulation,test]"
```

Requires a working pioranpy (juliacall + Pioran.jl) installation — see
[docs/installation.md](docs/installation.md).

## Repository layout

- `pioran_periodicity/` — the package (this is what gets installed).
- `tests/` — package unit tests (`pytest`).
- `tutorials/` — introductory examples.
- `scripts/` — CLI drivers for simulation studies and real-data re-analysis
  (see `scripts/README.md`).
- `paper/` — figure-generation scripts for publications built on this
  package (see `paper/README.md`).
- `docs/` — documentation source (mkdocs-material).

## Provenance

Extracted as the clean, general-purpose core of a thesis-replication
project. The original/legacy analysis code that this package superseded is
not included here.
