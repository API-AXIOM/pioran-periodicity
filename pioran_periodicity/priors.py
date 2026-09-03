"""Prior distributions and prior-transform construction for nested sampling.

Conventions (deliberate, see comparison_reports/simulation_code_bug_review.md):

* Every logarithmic parameter is base-10 and named ``log10_<quantity>``
  (fix M7/B3-class: one log base everywhere, frequencies always reported as
  frequencies).
* A prior is attached to a named parameter once; models that share a noise
  component share the *same* ``Parameter`` objects, so a null model and its
  periodic alternative can never disagree about the prior of a shared
  parameter (fix M1/B4).
* Distributions with singular edges (period -> 0, error scale -> 0) are not
  representable here by accident: ``Uniform`` validates lo < hi and the model
  factories enforce positive lower bounds for periods and error scales
  (fix M5/S1-real, S2-real).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
from scipy.stats import norm as _scipy_norm

__all__ = [
    "Uniform",
    "Normal",
    "LogUniform",
    "LogNormal",
    "ConditionalUniform",
    "ProcessRelativeNormal",
    "Parameter",
    "PriorTransform",
]


class Distribution:
    """Base class: maps u in [0, 1] to a parameter value."""

    def transform(self, u: float, previous: dict[str, float]) -> float:
        raise NotImplementedError

    def describe(self) -> str:
        raise NotImplementedError


@dataclass(frozen=True)
class Uniform(Distribution):
    lo: float
    hi: float

    def __post_init__(self):
        if not self.lo < self.hi:
            raise ValueError(f"Uniform requires lo < hi, got ({self.lo}, {self.hi})")

    def transform(self, u, previous):
        return self.lo + u * (self.hi - self.lo)

    def describe(self):
        return f"Uniform({self.lo}, {self.hi})"


@dataclass(frozen=True)
class Normal(Distribution):
    mu: float
    sigma: float

    def __post_init__(self):
        if self.sigma <= 0:
            raise ValueError("Normal requires sigma > 0")

    def transform(self, u, previous):
        return self.mu + self.sigma * _scipy_norm.ppf(u)

    def describe(self):
        return f"Normal({self.mu}, {self.sigma})"


@dataclass(frozen=True)
class LogUniform(Distribution):
    """Uniform in log10 between two positive bounds; returns the linear value."""

    lo: float
    hi: float

    def __post_init__(self):
        if not (0 < self.lo < self.hi):
            raise ValueError("LogUniform requires 0 < lo < hi")

    def transform(self, u, previous):
        llo, lhi = np.log10(self.lo), np.log10(self.hi)
        return 10.0 ** (llo + u * (lhi - llo))

    def describe(self):
        return f"LogUniform({self.lo}, {self.hi})"


@dataclass(frozen=True)
class LogNormal(Distribution):
    """Normal in log10, positive support; returns the linear value.

    ``mu_log10=0`` gives a prior median of exactly 1 -- the convention used
    for per-band amplitude scales pinned to 1 at the reference band (see
    multiband module), so a non-reference band's amplitude prior is centered
    on "same variability amplitude as the reference band" by default.
    """

    mu_log10: float
    sigma_log10: float

    def __post_init__(self):
        if self.sigma_log10 <= 0:
            raise ValueError("LogNormal requires sigma_log10 > 0")

    def transform(self, u, previous):
        return 10.0 ** (self.mu_log10 + self.sigma_log10 * _scipy_norm.ppf(u))

    def describe(self):
        return f"LogNormal(mu_log10={self.mu_log10}, sigma_log10={self.sigma_log10})"


@dataclass(frozen=True)
class ProcessRelativeNormal(Distribution):
    """Normal(0, fraction * sigma_process), with sigma_process taken from an
    EARLIER sampled parameter rather than fixed in advance.

    Used for the sine coefficients: the physically meaningful quantity is the
    periodic amplitude *relative to the object's own stochastic variability*,
    f = A / sigma, not an absolute magnitude. Quasar variability spans
    sigma ~ 0.03-0.5 mag across a survey, so a fixed absolute scale is
    strongly informative for a quiet object and nearly vacuous for a variable
    one -- which silently varies the Occam factor, and hence the effective
    detection threshold, from object to object, so one null calibration would
    not transfer across the sample.

    This is legitimate hierarchical modelling, p(A | sigma) p(sigma), and NOT
    the M2/M6 defect of setting a prior from the data: ``variance_param`` is a
    *sampled* parameter, never ``var(y)``. It must be declared before this one
    (``build_family`` puts noise parameters first).

    With independent ProcessRelativeNormal priors on the two sine
    coefficients, the induced prior on the amplitude ratio f is Rayleigh with
    scale ``fraction`` and the phase is uniform.
    """

    fraction: float
    variance_param: str = "log10_variance"

    def __post_init__(self):
        if self.fraction <= 0:
            raise ValueError("ProcessRelativeNormal requires fraction > 0")

    def transform(self, u, previous):
        if self.variance_param not in previous:
            raise KeyError(
                f"ProcessRelativeNormal needs {self.variance_param!r} to be "
                f"transformed first; available: {list(previous)}. Noise "
                f"parameters are declared before mean parameters, so this "
                f"means the noise model does not expose a process variance "
                f"(CARMA does not -- give it an absolute "
                f"sine_amplitude_scale instead)."
            )
        sigma_process = np.sqrt(10.0 ** previous[self.variance_param])
        return self.fraction * sigma_process * _scipy_norm.ppf(u)

    def describe(self):
        return (
            f"Normal(0, {self.fraction} * sqrt(10**{self.variance_param}))"
        )


@dataclass(frozen=True)
class ConditionalUniform(Distribution):
    """Uniform whose bounds may depend on previously transformed parameters.

    ``lo``/``hi`` are either floats or names of parameters that appear
    *earlier* in the same PriorTransform. Used for the red-noise constraint
    alpha_high ~ Uniform(alpha_low, alpha_max), applied identically in every
    model that contains the noise component (fix M1/B4: the periodic and
    non-periodic model use the very same conditional prior, and
    alpha_high >= alpha_low holds everywhere -- no inverted bends).
    """

    lo: float | str
    hi: float | str

    def _resolve(self, bound, previous):
        if isinstance(bound, str):
            if bound not in previous:
                raise KeyError(
                    f"ConditionalUniform bound '{bound}' must be transformed "
                    f"before this parameter; available: {list(previous)}"
                )
            return previous[bound]
        return bound

    def transform(self, u, previous):
        lo = self._resolve(self.lo, previous)
        hi = self._resolve(self.hi, previous)
        return lo + u * (hi - lo)

    def describe(self):
        return f"Uniform({self.lo}, {self.hi})"


@dataclass(frozen=True)
class Parameter:
    """A named model parameter with its prior."""

    name: str
    prior: Distribution
    latex: str = ""

    def describe(self) -> str:
        return f"{self.name} ~ {self.prior.describe()}"


@dataclass
class PriorTransform:
    """Ordered set of parameters exposing an ultranest prior transform.

    The transform resolves conditional bounds in declaration order, so a
    ConditionalUniform may only reference parameters declared before it.
    """

    parameters: Sequence[Parameter] = field(default_factory=list)

    @property
    def names(self) -> list[str]:
        return [p.name for p in self.parameters]

    @property
    def ndim(self) -> int:
        return len(self.parameters)

    def __call__(self, cube: np.ndarray) -> np.ndarray:
        params = np.array(cube, dtype=float, copy=True)
        previous: dict[str, float] = {}
        for i, p in enumerate(self.parameters):
            params[i] = p.prior.transform(float(cube[i]), previous)
            previous[p.name] = params[i]
        return params

    def as_dict(self, values: np.ndarray) -> dict[str, float]:
        return dict(zip(self.names, (float(v) for v in values)))

    def describe(self) -> list[str]:
        return [p.describe() for p in self.parameters]


def make_transform_fn(prior: PriorTransform) -> Callable[[np.ndarray], np.ndarray]:
    """Return a plain function for samplers that reject callables with state."""

    def transform(cube: np.ndarray) -> np.ndarray:
        return prior(cube)

    return transform
