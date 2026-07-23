"""GP kernel builders on top of pioranpy / Pioran.jl.

Fixes baked in (see comparison_reports/simulation_code_bug_review.md and
original_code_bug_review.md):

* M2/B1/B2 -- the process variance is a *sampled parameter* everywhere.
  No kernel in this package ever looks at the data to normalise itself
  (the legacy ``estimate_variance=True, init_variance=var(y)`` construction
  is intentionally not available here; it lives only in the quarantined
  old code under ``workspace/utils``).
* M3/B5 -- the OBPL approximation band is controlled by ``FrequencyBand``,
  whose default upper edge uses the *median* sampling interval rather than
  the minimum; the component density per frequency decade is checked, and
  ``psd_approximation_error`` measures the accuracy of the basis expansion.
* M7 -- all log parameters are base-10; bend "frequencies" are frequencies.

Conventions:

* DRW: k(tau) = variance * exp(-2*pi*f_bend*|tau|). With this definition
  f_bend is the PSD bend frequency of the Lorentzian,
  P(f) ~ 1 / (1 + (f/f_bend)^2). NOTE: the legacy code used
  k ~ exp(-f_bend*|tau|), i.e. its "bend frequency" was 1/tau without the
  2*pi; converting legacy values: f_bend(new) = f_bend(legacy) / (2*pi).
* OBPL: SingleBendingPowerLaw(alpha_low, f_bend, alpha_high) approximated
  with n_components basis functions; ``variance`` is the total integrated
  power of the process (integral of the PSD from 0 to infinity), passed to
  Pioran's ``approx`` with ``is_integrated_power=False``.
* CARMA: reuses the validated construction (Pioran ``CARMA`` with norm=0.5
  and is_integrated_power=false, real AR roots handled explicitly for p=2),
  which reproduces the tinygp quasisep CARMA covariance.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pioranpy as pa

__all__ = [
    "FrequencyBand",
    "drw_kernel",
    "obpl_kernel",
    "carma_kernel",
    "psd_approximation_error",
    "gp_log_likelihood",
    "MIN_COMPONENTS_PER_DECADE",
]

# Below this basis-function density the SHO/DRWCelerite expansion of a bending
# power law develops visible ripple and slope bias (issue M3).
MIN_COMPONENTS_PER_DECADE = 1.5


def _to_jl_vector(lst):
    return pa.jl.Vector[pa.jl.Float64](list(lst))


# ---------------------------------------------------------------------------
# Frequency band for PSD approximations
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FrequencyBand:
    """Frequency range over which a PSD approximation must be accurate.

    ``f_min``/``f_max`` bracket the science band; Pioran extends the actual
    basis grid to [f_min/S_low, f_max*S_high]. The default constructor derives
    f_max from the *median* sampling interval (fix M3/B5): the minimum
    interval is a heavy-tailed random variable when time stamps can be
    minutes apart (intra-night pairs), and letting it set the band trades
    approximation accuracy in the science band for resolution at frequencies
    that carry no information.
    """

    f_min: float
    f_max: float
    S_low: float = 20.0
    S_high: float = 20.0

    def __post_init__(self):
        if not (0 < self.f_min < self.f_max):
            raise ValueError(
                f"need 0 < f_min < f_max, got ({self.f_min}, {self.f_max})"
            )
        if self.S_low < 1 or self.S_high < 1:
            raise ValueError("S_low and S_high must be >= 1")

    @classmethod
    def from_times(
        cls, t, f_max_method: str = "median", S_low: float = 20.0, S_high: float = 20.0
    ) -> "FrequencyBand":
        """Derive the band from a time array.

        f_min = 1/(t_max - t_min); f_max = 1/(2*dt) with dt given by
        ``f_max_method``: "median" (default, robust) or "min" (legacy
        behaviour of the original code -- only sensible when the sampling
        pattern guarantees a floor on the spacing).
        """
        t = np.sort(np.asarray(t, dtype=float))
        diffs = np.diff(t)
        diffs = diffs[diffs > 0]
        if len(diffs) == 0:
            raise ValueError("need at least two distinct time stamps")
        if f_max_method == "median":
            dt = float(np.median(diffs))
        elif f_max_method == "min":
            dt = float(np.min(diffs))
        else:
            raise ValueError(f"unknown f_max_method '{f_max_method}'")
        T = float(t[-1] - t[0])
        return cls(f_min=1.0 / T, f_max=1.0 / (2.0 * dt), S_low=S_low, S_high=S_high)

    @property
    def grid_decades(self) -> float:
        """Decades spanned by the full approximation grid [f0, fM]."""
        f0 = self.f_min / self.S_low
        fM = self.f_max * self.S_high
        return float(np.log10(fM / f0))

    def check_density(self, n_components: int, strict: bool = False) -> float:
        """Components per decade of the approximation grid (issue M3).

        Warns (or raises with strict=True) below MIN_COMPONENTS_PER_DECADE.
        """
        density = n_components / self.grid_decades
        if density < MIN_COMPONENTS_PER_DECADE:
            msg = (
                f"PSD approximation grid spans {self.grid_decades:.1f} decades "
                f"with {n_components} components ({density:.2f}/decade < "
                f"{MIN_COMPONENTS_PER_DECADE}); increase n_components or "
                f"narrow the band (issue M3)."
            )
            if strict:
                raise ValueError(msg)
            warnings.warn(msg, stacklevel=2)
        return density


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------


def drw_kernel(log10_variance: float, log10_fbend: float):
    """Damped-random-walk (Ornstein-Uhlenbeck) kernel.

    k(tau) = variance * exp(-2*pi*f_bend*|tau|), variance = 10**log10_variance,
    f_bend = 10**log10_fbend (PSD bend frequency of the Lorentzian).

    Pioran's Exp(A, alpha) is k(tau) = A/2 * exp(-alpha|tau|), hence A = 2*variance
    (the same x2 convention documented in the validated translation).
    """
    variance = 10.0 ** float(log10_variance)
    rate = 2.0 * np.pi * 10.0 ** float(log10_fbend)
    return pa.Exp(2.0 * variance, rate)


def obpl_kernel(
    log10_variance: float,
    alpha_low: float,
    log10_fbend: float,
    alpha_high: float,
    band: FrequencyBand,
    n_components: int = 20,
    basis_function: str = "SHO",
    check_density: bool = True,
):
    """One-bend power-law kernel via basis-function approximation.

    ``variance = 10**log10_variance`` is the total integrated power of the
    process (PSD integral from 0 to infinity) and is a *free, sampled*
    parameter -- fix M2/B2: the kernel never sees the data. Requires
    alpha_high >= alpha_low (enforce through the prior, fix M1/B4).
    """
    if alpha_high < alpha_low:
        raise ValueError(
            f"alpha_high ({alpha_high}) < alpha_low ({alpha_low}): not a "
            "red-noise bend; use a conditional prior (fix M1)."
        )
    if check_density:
        band.check_density(n_components)
    variance = 10.0 ** float(log10_variance)
    f_bend = 10.0 ** float(log10_fbend)
    psd = pa.SingleBendingPowerLaw(float(alpha_low), f_bend, float(alpha_high))
    return pa.approx(
        psd,
        band.f_min,
        band.f_max,
        int(n_components),
        variance,
        band.S_low,
        band.S_high,
        is_integrated_power=False,
        basis_function=basis_function,
    )


def carma_kernel(p: int, q: int, log10_alphas, log10_betas, log10_sigma: float):
    """CARMA(p, q) kernel matching the validated old-vs-new construction.

    Parameters are base-10 logs (fix M7); the covariance reproduces
    tinygp's quasisep CARMA with alpha = 10**log10_alphas and
    beta = [sigma, sigma*10**log10_betas...].
    """
    ar_coeffs = [10.0 ** float(a) for a in log10_alphas]
    sigma = 10.0 ** float(log10_sigma)
    beta_list = [sigma] + [sigma * 10.0 ** float(b) for b in log10_betas]

    if p == 2:
        roots = np.roots([1.0, ar_coeffs[1], ar_coeffs[0]])
        if np.all(np.abs(np.imag(roots)) < 1e-12):
            # real AR roots: Pioran's celerite conversion assumes complex
            # conjugate pairs, so build the celerite terms directly.
            def beta_poly(r):
                return sum(b * r**k for k, b in enumerate(beta_list))

            amps = []
            for k, rk in enumerate(roots):
                num = beta_poly(rk) * beta_poly(-rk)
                den = -2.0 * np.real(rk)
                for j, rj in enumerate(roots):
                    if j != k:
                        den *= (rj - rk) * (np.conj(rj) + rk)
                amps.append(np.real(num / den))
            a = [float(v) for v in amps]
            c = [float(-np.real(r)) for r in roots]
            zeros = [0.0] * len(a)
            return pa.Pioran.SumOfCelerite(
                _to_jl_vector(a),
                _to_jl_vector(zeros),
                _to_jl_vector(c),
                _to_jl_vector(zeros),
            )

    r_alpha = pa.quad2roots(ar_coeffs)
    return pa.CARMA(int(p), int(q), r_alpha, _to_jl_vector(beta_list), 0.5, False)


# ---------------------------------------------------------------------------
# Approximation accuracy diagnostic (fix M3/B5)
# ---------------------------------------------------------------------------


def psd_approximation_error(
    alpha_low: float,
    log10_fbend: float,
    alpha_high: float,
    band: FrequencyBand,
    n_components: int = 20,
    basis_function: str = "SHO",
    n_grid: int = 200,
) -> dict:
    """Relative error of the basis-function expansion of an OBPL PSD.

    Compares Pioran's approximated PSD against the analytic model over the
    science band [f_min, f_max] (both normalised at f_min so only the shape
    is compared). Returns max/median relative error -- the quantity the
    original pipeline never measured (issue M3/B5).
    """
    f_bend = 10.0 ** float(log10_fbend)
    psd = pa.SingleBendingPowerLaw(float(alpha_low), f_bend, float(alpha_high))
    f0 = band.f_min / band.S_low
    fM = band.f_max * band.S_high
    freqs = np.logspace(np.log10(band.f_min), np.log10(band.f_max), n_grid)

    approx_vals = np.asarray(
        pa.Pioran.approximated_psd(
            _to_jl_vector(freqs),
            psd,
            f0,
            fM,
            n_components=int(n_components),
            basis_function=basis_function,
        ),
        dtype=float,
    )
    # analytic single-bend power law, same form as Pioran's model:
    # P(f) = (f/f_bend)^{-alpha_low} / (1 + (f/f_bend)^{alpha_high - alpha_low})
    x = freqs / f_bend
    model_vals = x ** (-alpha_low) / (1.0 + x ** (alpha_high - alpha_low))

    approx_vals = approx_vals / approx_vals[0]
    model_vals = model_vals / model_vals[0]
    rel = np.abs(approx_vals - model_vals) / model_vals
    return {
        "max_rel_error": float(np.max(rel)),
        "median_rel_error": float(np.median(rel)),
        "n_components": int(n_components),
        "grid_decades": band.grid_decades,
        "components_per_decade": n_components / band.grid_decades,
    }


# ---------------------------------------------------------------------------
# GP log-likelihood
# ---------------------------------------------------------------------------


def gp_log_likelihood(
    kernel, t, y, yerr, mean_func=None, err_scale: float = 1.0
) -> float:
    """Gaussian-process log-likelihood with optional mean function.

    The mean is subtracted from the data and a zero-mean GP is evaluated
    (mathematically identical, validated against the old implementation).
    ``err_scale`` multiplies the reported uncertainties (nu in the thesis);
    must be > 0 -- the singular nu = 0 limit is rejected (fix S2-real).
    """
    if err_scale <= 0:
        raise ValueError("err_scale must be > 0")
    t = np.asarray(t, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    sigma2 = (err_scale * np.asarray(yerr, dtype=np.float64)) ** 2

    if mean_func is not None:
        y = y - np.asarray(mean_func(t), dtype=np.float64)

    gp = pa.ScalableGP(0.0, kernel)
    gp_cond = gp(t, sigma2)
    return float(pa.Pioran.logpdf(gp_cond, y))
