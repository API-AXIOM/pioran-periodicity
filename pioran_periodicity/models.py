"""Model factories: noise model x {none, sine, linear, sine+linear}.

The central design rule (fix M1/B4): a noise model's parameters -- including
their priors -- are defined ONCE and shared verbatim by the non-periodic
model and every periodic/trend alternative built from it. Bayes factors
between the members of a :class:`ModelFamily` therefore compare models that
differ *only* by their mean function.

Prior choices live in :class:`PriorConfig` so that different applications
(simulation study vs. real sources) can use different priors without
touching model code. Amplitude-like priors must be chosen from unit
conventions known *before* looking at an individual light curve -- never
from statistics of the fitted data (fix M2/B2: no data peeking; see also
M6 on calibrating sine-amplitude priors to the expected variability scale).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

from .kernels import (
    FrequencyBand,
    carma_kernel,
    drw_kernel,
    gp_log_likelihood,
    obpl_kernel,
)
from .means import combine_means, linear_mean, sine_mean
from .priors import ConditionalUniform, Normal, Parameter, PriorTransform, Uniform

__all__ = ["PriorConfig", "ModelSpec", "ModelFamily", "build_family"]


# ---------------------------------------------------------------------------
# Prior configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PriorConfig:
    """Application-specific prior settings (simulations vs. real data).

    All log quantities are base-10 (fix M7). Defaults are the package's
    recommendations for the thesis simulation study; real-data analyses
    should construct their own instance.
    """

    # process variance: total integrated power, in the (data units)^2 of the
    # application. M6: choose the range from the known/expected variability
    # scale, NOT from var(y) of the light curve being fitted.
    log10_variance: tuple[float, float] = (-4.0, 1.0)

    # bend frequency (1/time-unit; year^-1 in the thesis setups)
    log10_fbend: tuple[float, float] = (-3.0, 2.0)

    # OBPL slopes; alpha_high is conditional on alpha_low (fix M1/B4)
    alpha_low: tuple[float, float] = (0.0, 2.0)
    alpha_high_max: float = 4.0

    # sine mean: A1, A2 ~ Normal(0, sine_amplitude_scale); Rayleigh marginal
    # amplitude. Set the scale to the expected variability amplitude (M6).
    sine_amplitude_scale: float = 0.5
    # period range; lower bound must be > 0 (fix M5)
    period: tuple[float, float] = (0.05, 5.0)

    # linear-trend mean
    slope: tuple[float, float] = (-2.0, 2.0)
    intercept: tuple[float, float] = (-2.0, 2.0)

    # optional error-bar rescale nu; None disables the parameter.
    # Lower bound must be > 0 (fix S2-real: nu = 0 collapses the GP diagonal).
    err_scale: tuple[float, float] | None = None

    # CARMA coefficient priors (base-10 logs)
    log10_carma_alpha: tuple[float, float] = (-3.0, 3.0)
    log10_carma_beta: tuple[float, float] = (-6.5, 6.5)
    log10_carma_sigma: tuple[float, float] = (-1.3, 2.3)

    def __post_init__(self):
        if self.period[0] <= 0:
            raise ValueError("period lower bound must be > 0 (fix M5)")
        if self.err_scale is not None and self.err_scale[0] <= 0:
            raise ValueError("err_scale lower bound must be > 0 (fix S2-real)")
        if self.sine_amplitude_scale <= 0:
            raise ValueError("sine_amplitude_scale must be > 0")


# ---------------------------------------------------------------------------
# Model specification
# ---------------------------------------------------------------------------


@dataclass
class ModelSpec:
    """A single model: parameters + log-likelihood builder.

    ``loglike(params_dict, t, y, yerr)`` returns the GP log-likelihood.
    Use with inference.run_nested, which wires in the data and adds the
    non-finite guard.
    """

    name: str
    prior: PriorTransform
    loglike: Callable[[dict, np.ndarray, np.ndarray, np.ndarray], float]
    meta: dict = field(default_factory=dict)

    @property
    def param_names(self) -> list[str]:
        return self.prior.names

    def describe(self) -> str:
        lines = [f"Model {self.name}:"]
        lines += [f"  {d}" for d in self.prior.describe()]
        return "\n".join(lines)


@dataclass
class ModelFamily:
    """A noise model and its mean-function variants, with shared priors."""

    noise: str
    members: dict[str, ModelSpec]

    def __getitem__(self, key) -> ModelSpec:
        return self.members[key]

    def names(self) -> list[str]:
        return list(self.members)


# ---------------------------------------------------------------------------
# Internal builders
# ---------------------------------------------------------------------------


def _noise_parameters(
    noise: str, cfg: PriorConfig, carma_order: tuple[int, int]
) -> list[Parameter]:
    """Noise-model parameters, defined once and shared by all variants."""
    if noise == "drw":
        return [
            Parameter("log10_variance", Uniform(*cfg.log10_variance)),
            Parameter("log10_fbend", Uniform(*cfg.log10_fbend)),
        ]
    if noise == "obpl":
        return [
            Parameter("alpha_low", Uniform(*cfg.alpha_low)),
            # conditional: alpha_high ~ U(alpha_low, alpha_high_max) in EVERY
            # variant (fix M1/B4)
            Parameter(
                "alpha_high", ConditionalUniform("alpha_low", cfg.alpha_high_max)
            ),
            Parameter("log10_fbend", Uniform(*cfg.log10_fbend)),
            Parameter("log10_variance", Uniform(*cfg.log10_variance)),
        ]
    if noise == "carma":
        p, q = carma_order
        params = [
            Parameter(f"log10_alpha{i}", Uniform(*cfg.log10_carma_alpha))
            for i in range(p)
        ]
        params += [
            Parameter(f"log10_beta{j + 1}", Uniform(*cfg.log10_carma_beta))
            for j in range(q)
        ]
        params += [Parameter("log10_sigma", Uniform(*cfg.log10_carma_sigma))]
        return params
    raise ValueError(f"unknown noise model '{noise}'")


def _mean_parameters(variant: str, cfg: PriorConfig) -> list[Parameter]:
    params: list[Parameter] = []
    if "sine" in variant:
        s = cfg.sine_amplitude_scale
        params += [
            Parameter("A1", Normal(0.0, s)),
            Parameter("A2", Normal(0.0, s)),
            Parameter("period", Uniform(*cfg.period)),
        ]
    if "linear" in variant:
        params += [
            Parameter("slope", Uniform(*cfg.slope)),
            Parameter("intercept", Uniform(*cfg.intercept)),
        ]
    return params


def _kernel_builder(
    noise: str,
    carma_order: tuple[int, int],
    band: FrequencyBand | None,
    n_components: int,
    basis_function: str,
):
    if noise == "drw":

        def build(p):
            return drw_kernel(p["log10_variance"], p["log10_fbend"])

    elif noise == "obpl":
        if band is None:
            raise ValueError("OBPL models need a FrequencyBand")

        def build(p):
            return obpl_kernel(
                p["log10_variance"],
                p["alpha_low"],
                p["log10_fbend"],
                p["alpha_high"],
                band,
                n_components=n_components,
                basis_function=basis_function,
                check_density=False,
            )  # checked once at build_family

    elif noise == "carma":
        cp, cq = carma_order

        def build(p):
            log10_alphas = [p[f"log10_alpha{i}"] for i in range(cp)]
            log10_betas = [p[f"log10_beta{j + 1}"] for j in range(cq)]
            return carma_kernel(cp, cq, log10_alphas, log10_betas, p["log10_sigma"])

    else:
        raise ValueError(f"unknown noise model '{noise}'")
    return build


def _mean_builder(variant: str):
    if variant == "plain":
        return lambda p: None
    if variant == "sine":
        return lambda p: (lambda t: sine_mean(t, p["A1"], p["A2"], p["period"]))
    if variant == "linear":
        return lambda p: (lambda t: linear_mean(t, p["slope"], p["intercept"]))
    if variant == "sine+linear":
        return lambda p: combine_means(
            lambda t: sine_mean(t, p["A1"], p["A2"], p["period"]),
            lambda t: linear_mean(t, p["slope"], p["intercept"]),
        )
    raise ValueError(f"unknown variant '{variant}'")


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------


def build_family(
    noise: str,
    cfg: PriorConfig,
    variants: Sequence[str] = ("plain", "sine"),
    band: FrequencyBand | None = None,
    n_components: int = 20,
    basis_function: str = "SHO",
    carma_order: tuple[int, int] = (2, 1),
) -> ModelFamily:
    """Build a noise model and its mean-function variants.

    Parameters
    ----------
    noise : "drw" | "obpl" | "carma"
    cfg : PriorConfig
    variants : subset of {"plain", "sine", "linear", "sine+linear"}
    band : FrequencyBand, required for OBPL (fix M3: explicit, robust band)
    n_components, basis_function : OBPL approximation settings
    carma_order : (p, q) for CARMA

    All variants share the SAME noise Parameter objects (fix M1/B4).
    """
    if noise == "obpl" and band is not None:
        band.check_density(n_components)  # single upfront density check (M3)

    noise_params = _noise_parameters(noise, cfg, carma_order)
    kernel_of = _kernel_builder(noise, carma_order, band, n_components, basis_function)

    err_param = (
        [Parameter("err_scale", Uniform(*cfg.err_scale))]
        if cfg.err_scale is not None
        else []
    )

    members: dict[str, ModelSpec] = {}
    for variant in variants:
        params = list(noise_params) + _mean_parameters(variant, cfg) + list(err_param)
        prior = PriorTransform(params)
        mean_of = _mean_builder(variant)

        def loglike(pdict, t, y, yerr, _kernel_of=kernel_of, _mean_of=mean_of):
            kernel = _kernel_of(pdict)
            mean_func = _mean_of(pdict)
            return gp_log_likelihood(
                kernel,
                t,
                y,
                yerr,
                mean_func=mean_func,
                err_scale=pdict.get("err_scale", 1.0),
            )

        name = noise if variant == "plain" else f"{noise}+{variant}"
        members[name] = ModelSpec(
            name=name,
            prior=prior,
            loglike=loglike,
            meta={
                "noise": noise,
                "variant": variant,
                "n_components": n_components if noise == "obpl" else None,
                "basis_function": basis_function if noise == "obpl" else None,
                "band": (
                    None
                    if band is None
                    else {
                        "f_min": band.f_min,
                        "f_max": band.f_max,
                        "S_low": band.S_low,
                        "S_high": band.S_high,
                    }
                ),
                "carma_order": carma_order if noise == "carma" else None,
            },
        )
    return ModelFamily(noise=noise, members=members)
