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
    gp_log_likelihood_multiband,
    obpl_kernel,
)
from .means import combine_means, linear_mean, sine_mean
from .multiband import EFFECTIVE_WAVELENGTHS
from .priors import (
    ConditionalUniform,
    LogNormal,
    LogUniform,
    Normal,
    Parameter,
    PriorTransform,
    ProcessRelativeNormal,
    Uniform,
)

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

    # Sine mean. A_cos, A_sin are COEFFICIENTS, not the amplitude, and are
    # unrelated to the scenario CSV's `A1` column (the injected amplitude) --
    # see means.py on naming (fix MB2). The induced prior on the amplitude
    # sqrt(A_cos^2 + A_sin^2) is Rayleigh.
    #
    # Two mutually exclusive parametrisations, preferring the first:
    #   sine_amplitude_fraction -- HIERARCHICAL: A ~ f * sigma_process, with
    #     sigma_process a sampled parameter. The prior is then on the
    #     dimensionless ratio f = A/sigma, which is the physically meaningful
    #     quantity and is scale-free across a survey spanning sigma ~
    #     0.03-0.5 mag. Requires a noise model exposing log10_variance.
    #   sine_amplitude_scale -- ABSOLUTE, in data units. Required for CARMA,
    #     whose process variance is not a sampled parameter. Used as the
    #     fallback whenever the relative form cannot be built.
    # At least one must be set. Default f = 1.2 gives a Rayleigh(1.2) prior on
    # f: median 1.4, 90th pct 2.6, 99th 3.6 -- bracketing PG 1302-102's
    # measured f = 2.3 without weight at f > 5, where a periodic signal would
    # dominate the light curve.
    sine_amplitude_fraction: float | None = 1.2
    sine_amplitude_scale: float | None = 0.5
    # period range in YEARS; sampled LOG-uniformly (period is a scale
    # parameter, so equal weight per octave, and the Occam contribution is
    # the single number log(hi/lo)). Lower bound must be > 0 (fix M5).
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

    # Sine colour index beta_sine ~ Normal(0, sine_colour_scale), fitted only
    # when build_family(fit_sine_colour=True). It frees the PERIODIC
    # component's per-band amplitude c_b = (lambda_b/lambda_ref)**-beta_sine
    # from the red noise's a_b. beta_sine = 0 is achromatic; beta_sine equal
    # to the noise's colour index is what red-noise leakage looks like, so
    # the posterior's position between those is the discriminant. Scale 1.0
    # is broad enough to cover both and either sign.
    sine_colour_scale: float = 1.0

    # multi-band (non-reference-band) priors: a_b ~ LogNormal(0, sigma),
    # mu_b ~ Normal(0, scale). Reference band's a_ref=1, mu_ref=0 are pinned,
    # not fit (see multiband.BandEncoding, models._band_parameters).
    band_log_amp_sigma: float = 0.3
    band_mu_scale: float = 0.5

    def __post_init__(self):
        if self.period[0] <= 0:
            raise ValueError("period lower bound must be > 0 (fix M5)")
        if self.err_scale is not None and self.err_scale[0] <= 0:
            raise ValueError("err_scale lower bound must be > 0 (fix S2-real)")
        if self.sine_amplitude_fraction is None and self.sine_amplitude_scale is None:
            raise ValueError(
                "set sine_amplitude_fraction (relative, preferred) or "
                "sine_amplitude_scale (absolute); both are None"
            )
        if (
            self.sine_amplitude_fraction is not None
            and self.sine_amplitude_fraction <= 0
        ):
            raise ValueError("sine_amplitude_fraction must be > 0")
        if self.sine_amplitude_scale is not None and self.sine_amplitude_scale <= 0:
            raise ValueError("sine_amplitude_scale must be > 0")
        if self.sine_colour_scale <= 0:
            raise ValueError("sine_colour_scale must be > 0")


# ---------------------------------------------------------------------------
# Model specification
# ---------------------------------------------------------------------------


@dataclass
class ModelSpec:
    """A single model: parameters + log-likelihood builder.

    ``loglike(params_dict, t, y, yerr, band=None)`` returns the GP
    log-likelihood; ``band`` (integer per-point band codes) is only used
    when the family was built with ``photometric_bands`` set, otherwise it
    is accepted and ignored. Use with inference.run_nested, which wires in
    the data and adds the non-finite guard.
    """

    name: str
    prior: PriorTransform
    loglike: Callable[..., float]
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


def _uses_relative_sine_amplitude(cfg: PriorConfig, noise: str) -> bool:
    """Whether the hierarchical (relative) sine-amplitude prior applies.

    Single source of truth: both the prior construction and the ``meta``
    record derive from this, so the recorded parametrisation can never
    disagree with the one actually built.
    """
    return cfg.sine_amplitude_fraction is not None and noise != "carma"


def _sine_amplitude_prior_label(cfg: PriorConfig, noise: str, variant: str):
    if "sine" not in variant:
        return None
    if _uses_relative_sine_amplitude(cfg, noise):
        return f"relative:f={cfg.sine_amplitude_fraction}"
    return f"absolute:{cfg.sine_amplitude_scale}"


def _mean_parameters(
    variant: str, cfg: PriorConfig, noise: str = "drw"
) -> list[Parameter]:
    """Mean-function parameters.

    ``noise`` selects the sine-amplitude parametrisation: every noise model
    except CARMA exposes ``log10_variance``, so the hierarchical
    ``sine_amplitude_fraction`` prior can be used; CARMA's process variance
    is a nonlinear function of its AR/MA coefficients rather than a sampled
    parameter, so it falls back to the absolute ``sine_amplitude_scale``.
    """
    params: list[Parameter] = []
    if "sine" in variant:
        relative = _uses_relative_sine_amplitude(cfg, noise)
        if relative:
            amp_prior = ProcessRelativeNormal(cfg.sine_amplitude_fraction)
        else:
            if cfg.sine_amplitude_scale is None:
                raise ValueError(
                    f"noise model {noise!r} cannot use the relative sine "
                    f"amplitude prior (no sampled process variance); set an "
                    f"absolute sine_amplitude_scale on the PriorConfig"
                )
            amp_prior = Normal(0.0, cfg.sine_amplitude_scale)
        params += [
            Parameter("A_cos", amp_prior),
            Parameter("A_sin", amp_prior),
            Parameter("period", LogUniform(*cfg.period)),
        ]
    if "linear" in variant:
        params += [
            Parameter("slope", Uniform(*cfg.slope)),
            Parameter("intercept", Uniform(*cfg.intercept)),
        ]
    return params


def _band_parameters(
    photometric_bands: Sequence[str] | None,
    cfg: PriorConfig,
    fit_band_means: bool = True,
) -> list[Parameter]:
    """Per-band amplitude/mean parameters for non-reference bands only.

    ``photometric_bands`` must be the non-reference band names in the same
    order as ``multiband.BandEncoding.others`` -- the reference band's own
    amplitude/mean are pinned (a_ref=1, mu_ref=0), not fit. Returns [] when
    ``photometric_bands`` is None/empty (default), so single-band models are
    completely unaffected.

    ``fit_band_means=False`` drops the ``mu_b`` parameters, fixing every
    band's offset to 0 and halving the parameters multi-band adds. Valid
    only when the data really is centred per band -- true by construction
    for injected simulation campaigns (``run_sim.py`` injects no band
    offsets unless asked), NOT generally true for real photometry, where
    genuine colour offsets exist and dropping mu_b would push them into the
    residuals. Exists because sampler cost grows superlinearly in
    dimension: for 6-band LSST this is ndim 12 -> 7 (drw).
    """
    if not photometric_bands:
        return []
    params: list[Parameter] = []
    for b in photometric_bands:
        params.append(Parameter(f"a_{b}", LogNormal(0.0, cfg.band_log_amp_sigma)))
        if fit_band_means:
            params.append(Parameter(f"mu_{b}", Normal(0.0, cfg.band_mu_scale)))
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
        return lambda p: (
            lambda t: sine_mean(t, p["A_cos"], p["A_sin"], p["period"])
        )
    if variant == "linear":
        return lambda p: (lambda t: linear_mean(t, p["slope"], p["intercept"]))
    if variant == "sine+linear":
        return lambda p: combine_means(
            lambda t: sine_mean(t, p["A_cos"], p["A_sin"], p["period"]),
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
    photometric_bands: Sequence[str] | None = None,
    fit_band_means: bool = True,
    fit_sine_colour: bool = False,
    reference_band: str | None = None,
    survey: str = "lsst",
) -> ModelFamily:
    """Build a noise model and its mean-function variants.

    Parameters
    ----------
    noise : "drw" | "obpl" | "carma"
    cfg : PriorConfig
    variants : subset of {"plain", "sine", "linear", "sine+linear"}
    band : FrequencyBand, required for OBPL (fix M3: explicit, robust band).
        NOT the photometric band -- see ``photometric_bands`` below.
    n_components, basis_function : OBPL approximation settings
    carma_order : (p, q) for CARMA
    fit_sine_colour : add a single ``beta_sine`` parameter giving the
        PERIODIC component its own per-band amplitude
        ``c_b = (lambda_b/lambda_ref)**-beta_sine``, independent of the red
        noise's ``a_b``. Costs exactly one dimension regardless of how many
        bands there are, because the colour dependence is parametrised rather
        than free per band. Requires ``photometric_bands`` and
        ``reference_band``, and applies only to sine-bearing variants.
        Default False, which keeps ``c_b = a_b`` (the periodic component
        shares the noise's colour).
    reference_band, survey : needed only with ``fit_sine_colour``, to look up
        filter wavelengths in ``multiband.EFFECTIVE_WAVELENGTHS``.
    photometric_bands : non-reference photometric band names (e.g. ZTF/LSST
        filters), in ``multiband.BandEncoding.others`` order. None (default)
        builds an ordinary single-band model, unchanged from before this
        parameter existed. When given, every variant's loglike expects an
        additional ``band`` array argument (integer codes from
        ``BandEncoding.encode``) and fits per-band ``a_b``/``mu_b``
        alongside the shared noise/mean parameters.

    All variants share the SAME noise Parameter objects (fix M1/B4); when
    ``photometric_bands`` is given they also share the SAME band Parameter
    objects, for the same reason.
    """
    if noise == "obpl" and band is not None:
        band.check_density(n_components)  # single upfront density check (M3)

    # (n_other,) lambda_b / lambda_ref, precomputed so the likelihood does no
    # dict lookups per call. None unless the sine colour is actually fitted.
    sine_wl_ratios = None
    if fit_sine_colour:
        if not photometric_bands or reference_band is None:
            raise ValueError(
                "fit_sine_colour needs photometric_bands and reference_band"
            )
        table = EFFECTIVE_WAVELENGTHS.get(survey)
        if table is None:
            raise ValueError(
                f"no filter wavelengths for survey {survey!r}; "
                f"have {sorted(EFFECTIVE_WAVELENGTHS)}"
            )
        unknown = sorted(set(list(photometric_bands) + [reference_band]) - set(table))
        if unknown:
            raise ValueError(f"no wavelength for {survey} band(s) {unknown}")
        sine_wl_ratios = np.array(
            [table[b] / table[reference_band] for b in photometric_bands]
        )

    noise_params = _noise_parameters(noise, cfg, carma_order)
    kernel_of = _kernel_builder(noise, carma_order, band, n_components, basis_function)
    band_params = _band_parameters(photometric_bands, cfg, fit_band_means)

    err_param = (
        [Parameter("err_scale", Uniform(*cfg.err_scale))]
        if cfg.err_scale is not None
        else []
    )

    members: dict[str, ModelSpec] = {}
    for variant in variants:
        params = (
            list(noise_params)
            + list(band_params)
            + _mean_parameters(variant, cfg, noise)
            + (
                [Parameter("beta_sine", Normal(0.0, cfg.sine_colour_scale))]
                if (sine_wl_ratios is not None and "sine" in variant)
                else []
            )
            + list(err_param)
        )
        prior = PriorTransform(params)
        mean_of = _mean_builder(variant)

        def loglike(
            pdict,
            t,
            y,
            yerr,
            band=None,
            _kernel_of=kernel_of,
            _mean_of=mean_of,
            _photometric_bands=photometric_bands,
            _sine_wl_ratios=sine_wl_ratios,
        ):
            kernel = _kernel_of(pdict)
            mean_func = _mean_of(pdict)
            err_scale = pdict.get("err_scale", 1.0)
            if band is None:
                return gp_log_likelihood(
                    kernel, t, y, yerr, mean_func=mean_func, err_scale=err_scale,
                )
            # (n_bands,), index 0 is the pinned reference band (a=1, mu=0)
            band_amp = np.array(
                [1.0] + [pdict[f"a_{b}"] for b in _photometric_bands]
            )
            band_mu = np.array(
                [0.0]
                + [pdict.get(f"mu_{b}", 0.0) for b in _photometric_bands]
            )
            # c_b for the periodic component; None => c_b = a_b
            band_mean_amp = None
            if _sine_wl_ratios is not None and "beta_sine" in pdict:
                band_mean_amp = np.concatenate(
                    ([1.0], _sine_wl_ratios ** (-pdict["beta_sine"]))
                )
            return gp_log_likelihood_multiband(
                kernel,
                t,
                y,
                yerr,
                band,
                band_amp,
                band_mu,
                mean_func=mean_func,
                err_scale=err_scale,
                band_mean_amp=band_mean_amp,
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
                "sine_amplitude_prior": _sine_amplitude_prior_label(
                    cfg, noise, variant
                ),
                "fit_sine_colour": bool(
                    sine_wl_ratios is not None and "sine" in variant
                ),
                "reference_band": reference_band,
                "photometric_bands": (
                    list(photometric_bands) if photometric_bands else None
                ),
                "fit_band_means": bool(fit_band_means) if photometric_bands else None,
            },
        )
    return ModelFamily(noise=noise, members=members)
