"""Light-curve simulation and observation-window sampling.

Timmer & Koenig (1995) simulation via stingray, plus a seasonal
window-pattern sampler. Fixes baked in:

* S1 -- the simulated baseline must exceed the observed baseline by
  ``leakage_margin`` (default 10x, standard practice for red-noise
  leakage); violations raise unless explicitly overridden.
* S2 -- observation epochs are drawn as *distinct nights* (day-resolution),
  optionally with several exposures within a night. The legacy behaviour
  of drawing raw simulation-grid samples (which produced "nights" minutes
  apart and heavy-tailed minimum spacings) is not available here.
* The sinusoidal signal is injected as a deterministic mean added to the
  sampled fluxes; its phase relative to the observation window is
  randomised through the random window start (document convention).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pioran_periodicity.cadence import depth_to_fractional_error

__all__ = [
    "simulate_lightcurve",
    "sample_seasonal_pattern",
    "sample_real_cadence",
    "SimulatedLightCurve",
]

DAYS_PER_YEAR = 365.0
DAYS_PER_MONTH = 30.0


@dataclass
class SimulatedLightCurve:
    """Continuous simulated light curve (time in days, flux normalised)."""

    time: np.ndarray
    flux: np.ndarray
    dt_days: float


def simulate_lightcurve(
    psd_func,
    psd_params,
    n_samples: int = 2**21,
    dt_minutes: float = 10.0,
    mean: float = 1.0,
    rms: float = 0.15,
    seed: int = 1079,
) -> SimulatedLightCurve:
    """Simulate a red-noise light curve with the TK95 method (stingray).

    ``psd_func(freq, *psd_params)`` evaluated on the rFFT grid (DC removed).
    ``rms`` is the fractional rms (stingray convention: std = rms * mean).

    Red-noise leakage: frequencies below 1/(n_samples*dt) are absent from
    the simulation; callers sampling an observation baseline T_obs should
    keep n_samples*dt >= leakage_margin * T_obs (checked by
    :func:`sample_seasonal_pattern`, issue S1).
    """
    from stingray.simulator import simulator  # deferred heavy import

    dt_days = dt_minutes / (60.0 * 24.0)
    rng_seed = int(seed)
    np.random.seed(rng_seed)  # stingray uses the global numpy RNG
    sim = simulator.Simulator(
        N=int(n_samples), mean=float(mean), dt=dt_days, rms=float(rms)
    )
    freq = np.fft.rfftfreq(sim.N, d=sim.dt)[1:]
    spectrum = psd_func(freq, *psd_params)
    lc = sim.simulate(spectrum)
    return SimulatedLightCurve(
        time=np.asarray(lc.time, dtype=float),
        flux=np.asarray(lc.counts, dtype=float),
        dt_days=dt_days,
    )


def _check_leakage_margin(
    t_sim_days: float,
    t_obs_days: float,
    leakage_margin: float,
    enforce_leakage_margin: bool,
) -> None:
    """Shared S1 check (see module docstring) for both samplers below."""
    margin = t_sim_days / t_obs_days
    if margin < leakage_margin:
        msg = (
            f"simulated baseline is only {margin:.1f}x the observed "
            f"baseline (< {leakage_margin}x): low-frequency red-noise "
            f"leakage is under-represented (issue S1). Simulate a longer "
            f"light curve or lower leakage_margin explicitly."
        )
        if enforce_leakage_margin:
            raise ValueError(msg)
        import warnings

        warnings.warn(msg, stacklevel=3)


def sample_seasonal_pattern(
    lc: SimulatedLightCurve,
    n_windows: int = 9,
    nights_per_window: int = 7,
    window_period_months: float = 8.0,
    window_width_days: float = 10.0,
    obs_per_night: int = 1,
    night_window_hours: float = 8.0,
    data_loss_frac: float = 0.0,
    noise_sigma: float = 0.015,
    mean_signal=None,
    leakage_margin: float = 10.0,
    enforce_leakage_margin: bool = True,
    seed: int = 100,
):
    """Sample a seasonal observing pattern from a simulated light curve.

    Windows of ``window_width_days`` recur every ``window_period_months``;
    within each window, ``nights_per_window`` DISTINCT nights are chosen
    (fix S2), and ``obs_per_night`` samples are taken per chosen night,
    inside a night window of ``night_window_hours`` (so observations on
    consecutive nights are separated by at least
    24 - night_window_hours hours -- drawing epochs from the full day
    would defeat the distinct-night guarantee).
    Gaussian noise with ``noise_sigma`` is added; reported uncertainties
    equal ``noise_sigma``. ``mean_signal(t_years)``, if given, is added to
    the sampled fluxes (e.g. a sinusoid).

    Returns (t_years, flux, flux_err), time sorted, in years, NOT re-zeroed
    (the absolute offset randomises the signal phase relative to the
    window pattern).
    """
    rng = np.random.default_rng(int(seed))

    if not 0 < night_window_hours <= 24:
        raise ValueError("night_window_hours must be in (0, 24]")

    samples_per_day = int(round(1.0 / lc.dt_days))
    samples_per_night_window = max(1, int(samples_per_day * night_window_hours / 24.0))
    if obs_per_night > samples_per_night_window:
        raise ValueError(
            f"obs_per_night={obs_per_night} exceeds the "
            f"{samples_per_night_window} samples in a {night_window_hours} h "
            f"night window"
        )
    window_period_days = window_period_months * DAYS_PER_MONTH
    if window_period_days < window_width_days:
        raise ValueError("windows overlap: window_period < window_width")

    t_obs_days = (n_windows - 1) * window_period_days + window_width_days
    t_sim_days = lc.time[-1] - lc.time[0]
    _check_leakage_margin(t_sim_days, t_obs_days, leakage_margin, enforce_leakage_margin)

    n_days_total = int(len(lc.time) / samples_per_day)
    max_start_day = n_days_total - int(np.ceil(t_obs_days)) - 1
    if max_start_day <= 0:
        raise ValueError("observation pattern does not fit in the simulation")
    start_day = rng.integers(0, max_start_day)

    idx = []
    for w in range(n_windows):
        window_start_day = start_day + w * window_period_days
        # distinct nights within the window (fix S2)
        candidate_days = np.arange(
            int(window_start_day), int(window_start_day + window_width_days)
        )
        nights = rng.choice(candidate_days, size=nights_per_window, replace=False)
        for night in nights:
            night_start = night * samples_per_day
            offsets = rng.choice(
                samples_per_night_window, size=obs_per_night, replace=False
            )
            idx.extend(night_start + offsets)
    idx = np.array(sorted(idx), dtype=int)

    if data_loss_frac > 0:
        keep = len(idx) - int(len(idx) * data_loss_frac)
        idx = np.sort(rng.choice(idx, size=keep, replace=False))

    t_years = lc.time[idx] / DAYS_PER_YEAR
    flux = lc.flux[idx] + rng.normal(0.0, noise_sigma, size=len(idx))
    if mean_signal is not None:
        flux = flux + np.asarray(mean_signal(t_years), dtype=float)
    flux_err = np.full(len(idx), float(noise_sigma))
    return t_years, flux, flux_err


def sample_real_cadence(
    lc: SimulatedLightCurve,
    cadence,
    noise_model,
    ref_mag: float,
    mean_signal=None,
    leakage_margin: float = 10.0,
    enforce_leakage_margin: bool = True,
    seed: int = 100,
):
    """Sample a REAL survey cadence (ZTF or LSST, from
    :class:`pioran_periodicity.cadence.CadenceLibrary`) from a simulated
    light curve, instead of :func:`sample_seasonal_pattern`'s synthetic
    seasonal-window pattern.

    ``cadence`` is a single object's cadence DataFrame as returned by
    ``CadenceLibrary.get``/``.random`` -- columns ``mjd, band, mag, magerr,
    depth, seeing``, NaN wherever a column doesn't apply to that survey. The
    relative gap structure of the real epochs (``mjd - mjd.min()``) is
    preserved exactly; only the placement within the simulated buffer is
    randomised (mirroring ``sample_seasonal_pattern``'s ``start_day``), so
    an injected signal's phase relative to the real seasonal gaps is not
    fixed run to run.

    Noise is heteroscedastic and per-epoch: rows with a real ``depth``
    (LSST/OpSim) use :func:`pioran_periodicity.cadence.depth_to_fractional_error`;
    rows without (ZTF) use ``noise_model(band, ref_mag)`` -- the survey's
    fitted magerr(mag) relation evaluated at the object's fixed catalog
    magnitude ``ref_mag`` (this pipeline simulates one brightness level per
    object, not per-epoch photometry). The resulting fractional error is
    scaled by the realised light curve's own mean flux level.

    Returns (t_years, flux, flux_err), time sorted, in years, NOT re-zeroed
    -- same contract as :func:`sample_seasonal_pattern`.
    """
    rng = np.random.default_rng(int(seed))

    mjd = cadence["mjd"].to_numpy(dtype=float)
    band = cadence["band"].to_numpy()
    depth = cadence["depth"].to_numpy(dtype=float)
    order = np.argsort(mjd)
    mjd, band, depth = mjd[order], band[order], depth[order]

    offset_days = mjd - mjd[0]
    t_obs_days = float(offset_days[-1])
    t_sim_days = lc.time[-1] - lc.time[0]
    _check_leakage_margin(t_sim_days, t_obs_days, leakage_margin, enforce_leakage_margin)

    samples_per_day = int(round(1.0 / lc.dt_days))
    n_days_total = int(len(lc.time) / samples_per_day)
    max_start_day = n_days_total - int(np.ceil(t_obs_days)) - 1
    if max_start_day <= 0:
        raise ValueError("real cadence baseline does not fit in the simulation")
    start_day = rng.integers(0, max_start_day)

    # Nearest grid index for each real epoch (real timestamps don't fall
    # exactly on the dt_minutes-spaced simulation grid).
    idx = np.round((start_day + offset_days) * samples_per_day).astype(int)
    idx = np.clip(idx, 0, len(lc.time) - 1)

    t_years = lc.time[idx] / DAYS_PER_YEAR
    ref_flux = float(np.mean(lc.flux))

    frac_err = np.empty(len(idx), dtype=float)
    has_depth = np.isfinite(depth)
    if has_depth.any():
        frac_err[has_depth] = depth_to_fractional_error(ref_mag, depth[has_depth])
    if (~has_depth).any():
        if noise_model is None:
            raise ValueError(
                "cadence has rows without a real `depth` (non-LSST epochs) "
                "but no noise_model was given"
            )
        for b in np.unique(band[~has_depth]):
            sel = (~has_depth) & (band == b)
            frac_err[sel] = noise_model(b, np.full(int(sel.sum()), ref_mag))
    flux_err = frac_err * ref_flux

    flux = lc.flux[idx] + rng.normal(0.0, flux_err)
    if mean_signal is not None:
        flux = flux + np.asarray(mean_signal(t_years), dtype=float)
    return t_years, flux, flux_err
