"""Nested-sampling inference and model comparison.

Fixes baked in:

* S3 -- posterior resampling is seeded and reproducible.
* S4 -- every sampler setting that affects logZ is set explicitly and
  recorded in the result, so runs are comparable and auditable.
* Non-finite likelihoods are mapped to a large negative penalty instead of
  crashing the sampler.
"""

from __future__ import annotations

import json
import time as _time
from dataclasses import asdict, dataclass, field

import numpy as np

from .models import ModelFamily, ModelSpec

__all__ = [
    "SamplerSettings",
    "FitResult",
    "run_nested",
    "fit_family",
    "log10_bayes_factors",
    "save_result",
    "load_result",
]

_LOW = -1e300  # likelihood penalty for non-finite values


@dataclass(frozen=True)
class SamplerSettings:
    """Explicit ultranest settings (fix S4: no silent defaults).

    ``step_sampler_min_ndim``: from this dimensionality on, a slice sampler is
    attached (``nsteps = step_nsteps_per_dim * ndim``). Ultranest's default
    region rejection sampling (MLFriends) collapses in efficiency for
    ndim >~ 8 with tight correlated posteriors and can exhaust ``max_ncalls``
    before the evidence integral converges (observed on the PG 1553+113 sine
    models, 2026-07-17, and on the ndim-5 DRW+sine simulation fits,
    2026-07-22); slice sampling matches the behaviour of the original jaxns
    analysis. The default of 1 slice-samples EVERY model, so all fits in a
    study share one sampling method (no region-vs-slice heterogeneity between
    models of different dimensionality). Set to a large value to disable.
    """

    min_num_live_points: int = 400
    frac_remain: float = 0.01
    max_ncalls: int = 1_000_000
    max_num_improvement_loops: int = 0
    seed: int = 42  # seeds posterior resampling (fix S3)
    step_sampler_min_ndim: int = 1  # slice-sample everything (consistency)
    step_nsteps_per_dim: int = 2

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class FitResult:
    model: str
    logz: float
    logzerr: float
    samples: dict[str, list[float]]
    settings: dict
    meta: dict = field(default_factory=dict)
    ncall: int = 0
    runtime_s: float = 0.0
    # posterior effective sample size 1/sum(w^2); a collapsed (truncated) run
    # shows ess ~ 1 -- do not quote its medians as intervals
    ess: float = float("nan")
    # False when the run was truncated (max_ncalls) before termination
    converged: bool = True

    def as_dict(self) -> dict:
        return {
            "model": self.model,
            "logz": self.logz,
            "logzerr": self.logzerr,
            "samples": self.samples,
            "settings": self.settings,
            "meta": self.meta,
            "ncall": self.ncall,
            "runtime_s": self.runtime_s,
            "ess": self.ess,
            "converged": self.converged,
        }


def run_nested(
    spec: ModelSpec,
    t,
    y,
    yerr,
    settings: SamplerSettings = SamplerSettings(),
    n_posterior_samples: int = 10_000,
    show_status: bool = False,
    log_dir: str | None = None,
) -> FitResult:
    """Run ultranest on one model and return a reproducible FitResult."""
    import ultranest  # deferred: heavy import

    t = np.asarray(t, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    yerr = np.asarray(yerr, dtype=np.float64)

    names = spec.param_names
    prior = spec.prior

    def loglike_vec(params):
        pdict = prior.as_dict(params)
        try:
            val = spec.loglike(pdict, t, y, yerr)
        except (FloatingPointError, ValueError, RuntimeError):
            return _LOW
        if not np.isfinite(val):
            return _LOW
        return float(val)

    sampler = ultranest.ReactiveNestedSampler(
        names,
        loglike_vec,
        lambda cube: prior(cube),
        log_dir=log_dir,
        resume="overwrite",
    )
    # Slice sampling for higher-dimensional problems: region rejection
    # sampling stalls there (see SamplerSettings docstring).
    if len(names) >= settings.step_sampler_min_ndim:
        import ultranest.stepsampler as _ss

        sampler.stepsampler = _ss.SliceSampler(
            nsteps=settings.step_nsteps_per_dim * len(names),
            generate_direction=_ss.generate_mixture_random_direction,
        )
    start = _time.time()
    result = sampler.run(
        min_num_live_points=settings.min_num_live_points,
        frac_remain=settings.frac_remain,
        max_ncalls=settings.max_ncalls,
        max_num_improvement_loops=settings.max_num_improvement_loops,
        show_status=show_status,
        viz_callback="auto" if show_status else False,
    )
    runtime = _time.time() - start

    # Seeded equal-weight resampling (fix S3)
    rng = np.random.default_rng(settings.seed)
    points = result["weighted_samples"]["points"]
    weights = result["weighted_samples"]["weights"]
    w = weights / weights.sum()
    idx = rng.choice(len(w), size=min(n_posterior_samples, len(w)), p=w, replace=True)
    samples = {name: points[idx, i].tolist() for i, name in enumerate(names)}

    # Convergence accounting: a run truncated by max_ncalls has not finished
    # integrating; its weights are typically dominated by a few live points
    # (ess collapses) and neither logz nor posterior intervals are reliable.
    ess = float(1.0 / np.sum(w**2))
    ncall = int(result.get("ncall", 0))
    converged = ncall < settings.max_ncalls
    if not converged or ess < 10 * len(names):
        import warnings

        warnings.warn(
            f"fit '{spec.name}': "
            + ("truncated by max_ncalls; " if not converged else "")
            + f"posterior ESS = {ess:.0f}. logz and posterior intervals are "
            "unreliable -- raise max_ncalls or adjust the sampler settings.",
            stacklevel=2,
        )

    return FitResult(
        model=spec.name,
        logz=float(result["logz"]),
        logzerr=float(result["logzerr"]),
        samples=samples,
        settings=settings.as_dict(),
        meta=dict(spec.meta),
        ncall=ncall,
        runtime_s=runtime,
        ess=ess,
        converged=converged,
    )


def fit_family(
    family: ModelFamily,
    t,
    y,
    yerr,
    settings: SamplerSettings = SamplerSettings(),
    **kwargs,
) -> dict[str, FitResult]:
    """Fit every member of a model family to the same data."""
    return {
        name: run_nested(spec, t, y, yerr, settings=settings, **kwargs)
        for name, spec in family.members.items()
    }


def log10_bayes_factors(results: dict[str, FitResult]) -> dict[str, float]:
    """log10 Bayes factors for all ordered model pairs (numerator/denominator)."""
    ln10 = np.log(10.0)
    out = {}
    names = list(results)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            out[f"{a}/{b}"] = (results[a].logz - results[b].logz) / ln10
    return out


def save_result(result: FitResult, path: str) -> None:
    with open(path, "w") as f:
        json.dump(result.as_dict(), f)


def load_result(path: str) -> FitResult:
    with open(path) as f:
        d = json.load(f)
    return FitResult(**d)
