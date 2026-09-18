"""Shared reconstruction of a stored fit for posterior-predictive plotting.

Given a result file's ``meta`` dict and one parameter draw (posterior sample
or median), rebuild the kernel and mean function ``models.loglike`` used to
fit it, then condition a zero-mean GP on the residuals to get the
posterior-predictive curve in observed units.

Single band: ``mean_func = mu0 + shape_mean(t)``; the GP is zero-mean on
``y - mean_func(t)`` (mirrors ``models.loglike``, band=None branch).

Multi band: ``band_amp = [1.0, a_b...]``, ``band_mu = [mu0, mu_b...]``, the
reference band pinned at index 0. Residuals are
``r = (y - band_mu[band]) / band_amp[band] - m(t)`` -- the mean is
subtracted AFTER the division, in latent units (mirrors
``kernels.gp_log_likelihood_multiband``). Reversing those two steps is the
MB1 defect; a unit-amplitude test cannot detect it.

``err_scale`` is absent from every v2 result file; it defaults to 1.0
everywhere in this module. ``beta_sine`` (per-band sine colour) was never
fitted in v2 (``fit_sine_colour=False``); reconstructing it is not
implemented, so a params dict carrying it raises ``NotImplementedError``
rather than silently mispredicting.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pioranpy as pa

from pioran_periodicity.kernels import FrequencyBand, drw_kernel, obpl_kernel
from pioran_periodicity.means import sine_mean
from pioran_periodicity.multiband import BandEncoding

__all__ = [
    "is_multiband",
    "build_kernel_and_mean",
    "posterior_predictive",
]


def is_multiband(meta: dict) -> bool:
    """True when the fit carries per-band parameters for non-reference bands."""
    return bool(meta.get("photometric_bands"))


def build_kernel_and_mean(
    meta: dict, params: dict, band: FrequencyBand | None = None
) -> tuple[object, Callable | None]:
    """Reconstruct the kernel and mean function for one parameter draw.

    The mean includes ``mu0`` for single-band models (observed units) and
    excludes it for multi-band ones, where ``mu0`` belongs in
    ``band_mu[0]`` instead (see the module docstring and
    ``models.loglike``, models.py:555-566).
    """
    if "beta_sine" in params:
        raise NotImplementedError(
            "beta_sine (per-band sine colour) reconstruction is not "
            "implemented; no v2 fit used --fit-sine-colour"
        )
    noise, variant = meta["noise"], meta["variant"]
    if noise == "drw":
        kernel = drw_kernel(params["log10_variance"], params["log10_fbend"])
    elif noise == "obpl":
        kernel = obpl_kernel(
            params["log10_variance"],
            params["alpha_low"],
            params["log10_fbend"],
            params["alpha_high"],
            band,
            n_components=meta["n_components"],
            basis_function=meta["basis_function"],
            check_density=False,
        )
    else:
        raise ValueError(noise)

    # Sine coefficients were renamed A1/A2 -> A_cos/A_sin on 2026-09-03 (fix
    # MB2); older result files use the old keys.
    cos_key, sin_key = ("A_cos", "A_sin") if "A_cos" in params else ("A1", "A2")
    if "sine" in variant:

        def shape(t, p=params):
            return sine_mean(t, p[cos_key], p[sin_key], p["period"])

    else:
        shape = None

    # mu0 is the constant term of the single-band mean, but in the
    # multi-band model it is band_mu[0] in OBSERVED units and must stay out
    # of the latent-unit mean function (models.py:555-566).
    if is_multiband(meta):
        return kernel, shape
    mu0 = params.get("mu0", 0.0)
    if shape is None:
        return kernel, lambda t: np.full_like(np.asarray(t, dtype=float), mu0)
    return kernel, lambda t: mu0 + np.asarray(shape(t), dtype=float)


def _posterior_predictive_single(
    kernel,
    mean_func: Callable | None,
    t: np.ndarray,
    y: np.ndarray,
    yerr: np.ndarray,
    err_scale: float,
    t_grid: np.ndarray,
    need_std: bool,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Single-band posterior predictive (ported from paper/plot_fits.py)."""
    # shapes: t, y, yerr (n_points,); t_grid (n_grid,)
    sigma2 = (err_scale * yerr) ** 2
    y_resid = y - (mean_func(t) if mean_func is not None else 0.0)
    gp = pa.ScalableGP(0.0, kernel)
    gp_cond = gp(t, sigma2)
    fp = pa.posterior(gp_cond, y_resid)
    fp_grid = fp(t_grid)
    mu = np.asarray(pa.mean(fp_grid))  # (n_grid,)
    sd = None
    if need_std:
        sd = np.sqrt(np.diag(np.asarray(pa.cov(fp_grid))))
    if mean_func is not None:
        mu = mu + mean_func(t_grid)
    return mu, sd


def _posterior_predictive_multiband(
    meta: dict,
    kernel,
    mean_func: Callable | None,
    params: dict,
    t: np.ndarray,
    y: np.ndarray,
    yerr: np.ndarray,
    err_scale: float,
    t_grid: np.ndarray,
    band_labels: np.ndarray,
    need_std: bool,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Multi-band posterior predictive, in reference-band observed units.

    Mirrors ``kernels.gp_log_likelihood_multiband``: rescale into
    shared-latent units first, then subtract the mean function (which lives
    in latent units), condition a zero-mean GP, and map the predictive mean
    back to observed units in the reference band (a_ref=1, mu_ref=mu0).
    """
    bands = list(meta["photometric_bands"])
    # (n_bands,), index 0 is the pinned reference band (a=1, mu=mu0)
    band_amp = np.array([1.0] + [params[f"a_{b}"] for b in bands])
    band_mu = np.array(
        [params.get("mu0", 0.0)] + [params.get(f"mu_{b}", 0.0) for b in bands]
    )
    encoding = BandEncoding(reference=meta["reference_band"], others=tuple(bands))
    codes = encoding.encode(band_labels)  # (n_points,)

    # shapes: t, y, yerr, codes (n_points,); band_amp, band_mu (n_bands,)
    a = band_amp[codes]  # (n_points,)
    r = (y - band_mu[codes]) / a  # (n_points,), latent units
    if mean_func is not None:
        r = r - np.asarray(mean_func(t), dtype=float)

    gp = pa.ScalableGP(0.0, kernel)
    gp_cond = gp(t, (err_scale * yerr / a) ** 2)
    fp = pa.posterior(gp_cond, r)
    fp_grid = fp(t_grid)
    latent = np.asarray(pa.mean(fp_grid))  # (n_grid,), latent units
    sd = None
    if need_std:
        sd = np.sqrt(np.diag(np.asarray(pa.cov(fp_grid))))  # latent units
    if mean_func is not None:
        latent = latent + np.asarray(mean_func(t_grid), dtype=float)
    # reference band: a_ref = 1, mu_ref = mu0 = band_mu[0]
    mu = band_mu[0] + latent
    return mu, sd


def posterior_predictive(
    meta: dict,
    params: dict,
    t: np.ndarray,
    y: np.ndarray,
    yerr: np.ndarray,
    t_grid: np.ndarray,
    band_labels: np.ndarray | None = None,
    need_std: bool = False,
) -> tuple[np.ndarray, np.ndarray | None]:
    """GP posterior-predictive mean (and, optionally, std) on ``t_grid``.

    Returns the predictive curve in observed units, in the reference band
    when the fit is multi-band. ``err_scale`` defaults to 1.0 when absent
    from ``params`` (no v2 result stores it). ``need_std=False`` skips the
    O(N_grid^2) covariance computation.
    """
    band = None
    if meta.get("band") is not None:
        band = FrequencyBand(**meta["band"])
    kernel, mean_func = build_kernel_and_mean(meta, params, band=band)
    err_scale = params.get("err_scale", 1.0)

    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    yerr = np.asarray(yerr, dtype=float)
    t_grid = np.asarray(t_grid, dtype=float)

    if is_multiband(meta):
        if band_labels is None:
            raise ValueError("band_labels is required for multi-band fits")
        return _posterior_predictive_multiband(
            meta,
            kernel,
            mean_func,
            params,
            t,
            y,
            yerr,
            err_scale,
            t_grid,
            np.asarray(band_labels, dtype=object),
            need_std,
        )
    return _posterior_predictive_single(
        kernel, mean_func, t, y, yerr, err_scale, t_grid, need_std
    )
