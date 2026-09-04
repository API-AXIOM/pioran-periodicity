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
  sampled magnitudes; its phase relative to the observation window is
  randomised through the random window start (document convention).

Everything here is in MAGNITUDES, matching the real-data path
(``data.load_photometry_csv``, ``multiband.cadence_to_multiband_series``,
``run_realdata.py``), which has always fit magnitudes directly with no
mag-to-flux conversion. The simulator used to generate fractional FLUX and
convert the surveys' magnitude errors into it; that made the one quantity
the pipeline actually fits differ between the simulated and real branches
for no benefit -- the GP likelihood is unit-agnostic. See
:data:`FRACTIONAL_FLUX_TO_MAG` for converting amplitudes recorded under the
old convention.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pioran_periodicity.cadence import lsst_magnitude_error

__all__ = [
    "simulate_lightcurve",
    "sample_seasonal_pattern",
    "sample_real_cadence",
    "SimulatedLightCurve",
    "MAG_TO_FRACTIONAL_FLUX",
    "FRACTIONAL_FLUX_TO_MAG",
    "MAGNITUDE_UNITS",
]

DAYS_PER_YEAR = 365.0
DAYS_PER_MONTH = 30.0

# d(F)/F per magnitude: F = 10^(-0.4 m) => |dF/F| = 0.4 ln(10) |dm| = 0.921
# |dm|. Both survey noise prescriptions return MAGNITUDE errors and the light
# curves are now magnitudes too, so nothing in this module converts between
# the two any more; the constants remain for callers migrating amplitudes
# recorded under the old fractional-flux convention.
MAG_TO_FRACTIONAL_FLUX = 0.4 * np.log(10.0)

# 2.5/ln(10) = 1.0857. Multiply an amplitude expressed as a FRACTIONAL FLUX
# rms by this to get the magnitude rms describing the SAME physical
# variability: sigma_m = sigma_F/F / (0.4 ln 10). Campaign inputs calibrated
# in flux (`rms`, `noiseSIGMA`, the injected `A1` triads) were scaled by it
# when this module moved to magnitudes, so the physical amplitude -- and
# hence the dimensionless f = A/sigma the sine prior is written in -- is
# unchanged. Reading a flux-era 0.15 as "0.15 mag" instead would silently
# shrink the injected variability by 8%.
FRACTIONAL_FLUX_TO_MAG = 1.0 / MAG_TO_FRACTIONAL_FLUX

# Value of the `units` column that campaign config CSVs (and the `units`
# field of a cached light-curve .npz) must carry to be accepted by
# scripts/run_sim.py. It exists so that inputs prepared for the old
# fractional-flux simulator fail loudly instead of running 8% quiet.
MAGNITUDE_UNITS = "mag"


@dataclass
class SimulatedLightCurve:
    """Continuous simulated light curve: time in days, ``mag`` in MAGNITUDES.

    ``mag`` is the model magnitude about ``mean_mag`` (0 by default, i.e. a
    zero-mean magnitude deviation); the per-band zero point is applied later,
    in :func:`sample_real_cadence`. The attribute is deliberately not called
    ``flux`` any more: it is a different physical quantity, and a silent
    reinterpretation of the old name is exactly the failure this rename
    prevents.
    """

    time: np.ndarray
    mag: np.ndarray
    dt_days: float


def simulate_lightcurve(
    psd_func,
    psd_params,
    n_samples: int = 2**21,
    dt_minutes: float = 10.0,
    mean_mag: float = 0.0,
    sigma_mag: float = 0.15 * FRACTIONAL_FLUX_TO_MAG,
    seed: int = 1079,
) -> SimulatedLightCurve:
    """Simulate a red-noise MAGNITUDE light curve with TK95 (stingray).

    ``psd_func(freq, *psd_params)`` evaluated on the rFFT grid (DC removed).
    ``sigma_mag`` is the ABSOLUTE standard deviation of the returned
    magnitudes, in mag, and ``mean_mag`` their mean level. Contrast the
    previous signature, which took a ``mean`` flux level and a ``rms``
    *fractional* flux rms (std = rms * mean); the keywords were renamed
    rather than reinterpreted so that a caller still passing ``mean=1.0,
    rms=0.15`` raises a TypeError instead of quietly simulating 8% less
    variability than it did before (see :data:`FRACTIONAL_FLUX_TO_MAG`).

    Stingray only knows how to scale to a *fractional* rms, so the process is
    drawn at unit mean with fractional rms ``sigma_mag`` -- giving absolute
    std ``sigma_mag`` exactly -- and then recentred on ``mean_mag``. For a
    given ``seed`` this is the same realisation the flux version produced,
    rescaled and shifted; the PSD shape is untouched.

    Magnitudes are inverted relative to flux (brighter = smaller), so the
    magnitude series is, physically, the negative of the flux deviation. No
    sign flip is applied: the process is zero-mean and symmetric, so the
    flipped realisation is drawn from the identical distribution, and the
    injected sinusoid's sign is absorbed by its (randomised) phase.

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
        N=int(n_samples), mean=1.0, dt=dt_days, rms=float(sigma_mag)
    )
    freq = np.fft.rfftfreq(sim.N, d=sim.dt)[1:]
    spectrum = psd_func(freq, *psd_params)
    lc = sim.simulate(spectrum)
    # unit-mean series -> zero-mean magnitude deviation -> requested level
    mag = np.asarray(lc.counts, dtype=float) - 1.0 + float(mean_mag)
    return SimulatedLightCurve(
        time=np.asarray(lc.time, dtype=float),
        mag=mag,
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
    noise_sigma: float = 0.015 * FRACTIONAL_FLUX_TO_MAG,
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
    Gaussian noise with ``noise_sigma`` -- in MAGNITUDES, like everything
    else here -- is added; reported uncertainties equal ``noise_sigma``. This
    synthetic pattern has no survey noise model, so the uncertainty is
    homoscedastic by construction (unlike :func:`sample_real_cadence`).
    ``mean_signal(t_years)``, if given, is added to the sampled magnitudes
    (e.g. a sinusoid, amplitude in mag).

    Returns (t_years, mag, mag_err), time sorted, in years, NOT re-zeroed
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
    mag = lc.mag[idx] + rng.normal(0.0, noise_sigma, size=len(idx))
    if mean_signal is not None:
        mag = mag + np.asarray(mean_signal(t_years), dtype=float)
    mag_err = np.full(len(idx), float(noise_sigma))
    return t_years, mag, mag_err


def _band_reference_magnitudes(band, mag_real, ref_mag: float) -> dict:
    """Median REAL magnitude per photometric band, for the ZTF noise model.

    ``ref_mag`` is the object's r-band catalogue magnitude; using it for every
    band biases the non-r noise (quasars here sit ~0.37 mag fainter in g, which
    made g errors 23% too small). Where the cadence carries real photometry,
    each band gets its own zero point instead.

    Falls back to ``ref_mag`` for any band with no finite magnitude. Filters
    with ``np.isfinite`` rather than ``np.nanmedian``, which warns AND returns
    NaN on an all-NaN slice (defect MB3.4).
    """
    out = {}
    for b in np.unique(band):
        vals = mag_real[band == b]
        finite = vals[np.isfinite(vals)]
        out[b] = float(np.median(finite)) if finite.size else float(ref_mag)
    return out


def sample_real_cadence(
    lc: SimulatedLightCurve,
    cadence,
    noise_model,
    ref_mag: float,
    mean_signal=None,
    leakage_margin: float = 10.0,
    enforce_leakage_margin: bool = True,
    seed: int = 100,
    band_amp=None,
    band_mu=None,
    return_band: bool = False,
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

    Noise comes from one of two survey-specific prescriptions, chosen row by
    row on whether a real ``depth`` is present. Both natively return the same
    quantity -- a MAGNITUDE error -- which is now also the unit of the light
    curve, so it is reported as-is with no conversion or rescaling:

    * **LSST/OpSim** (finite ``depth``): the Ivezic et al. (2019) single-visit
      model via :func:`pioran_periodicity.cadence.lsst_magnitude_error`,
      evaluated at the visit's ``fiveSigmaDepth`` and the epoch's own
      noise-free simulated magnitude. Doubly heteroscedastic: depth varies
      epoch to epoch with observing conditions (21.0-25.2 across the library)
      AND the source's brightness varies. Includes the 0.005 mag systematic
      floor, which the previous approximation lacked entirely.
    * **ZTF** (no depth): the survey's ``magerr(mag)`` polynomial, fitted to
      real ZTF photometry, evaluated at each epoch's own NOISE-FREE SIMULATED
      magnitude. Returns a MAGNITUDE error, converted here with
      magnitude. Heteroscedastic, and deliberately so: in
      real ZTF photometry ``magerr`` is very nearly a deterministic function
      of the source's brightness at that epoch (within-object
      ``corr(mag, magerr)`` has median 0.998 over 1076 object-bands with >=50
      epochs), spanning a p90/p10 ratio of ~1.31. Applying the same relation
      to the simulated brightness sequence reproduces that structure.

      The per-band zero point is the median REAL magnitude in that band from
      the cadence library, not the scalar ``ref_mag``: ``ref_mag`` is the
      r-band catalogue magnitude, and feeding it to the g-band polynomial
      made g errors 23% too small (quasars here are ~0.37 mag fainter in g).
      Evaluating a convex ``magerr(mag)`` at a single mean magnitude also
      biases it low by Jensen's inequality -- together those made the old
      sigma 0.85x the object's real median magerr, and put it outside the
      object's own real p10-p90 range 68% of the time.

      The per-band zero point also sets where on the ``magerr(mag)`` /
      Ivezic curve the object sits: the simulated series is a magnitude
      DEVIATION about ``lc``'s mean level, and ``band_ref[b] + deviation``
      places it at the band's real brightness.

      The magnitude is taken from the noise-free model (latent process plus
      any injected periodic signal), NOT from the realised noisy magnitude
      -- a noise draw must never feed back into its own uncertainty. Note
      the consequence: an injected periodic signal does imprint on the error
      bars, because a brighter epoch genuinely has a smaller magnitude
      error. That is what real photometry does.

    The prescriptions differ because the surveys do (ZTF has real photometry
    but no published per-visit depth; the LSST cadence is an OpSim visit
    schedule with depths but no photometry), but both are magnitude errors on
    a magnitude light curve, which is what makes the ZTF-vs-LSST precision
    comparison meaningful. Neither feeds the realised variability back into
    the uncertainty.


    ``band_amp`` / ``band_mu`` ({band: value} dicts, default None) inject
    colour-dependent variability: ``y_b(t) = mu_b + a_b * x(t) + noise_b(t)``,
    the same shared-latent-process model
    ``kernels.gp_log_likelihood_multiband`` fits. That model is linear in the
    light-curve variable, so it carries over to magnitudes unchanged (a
    magnitude colour index is now the natural way to state it). ``a_b``
    scales the variability *about the light curve's mean level*, not the mean
    itself, and scales the injected periodic ``mean_signal`` too (matching
    how the fitted model treats ``mean_func``); the per-epoch photometric
    noise is NOT scaled, since it comes from the survey's depth, not from the
    source. Use
    :func:`pioran_periodicity.multiband.power_law_band_amplitudes` to build
    ``band_amp`` from a single colour index. Both default to None, which
    reproduces the identical-variability-in-every-band behaviour.

    Returns (t_years, mag, mag_err), time sorted, in years, NOT re-zeroed
    -- same contract as :func:`sample_seasonal_pattern`. With
    ``return_band=True`` returns (t_years, mag, mag_err, band) instead,
    where ``band`` is the (n_points,) array of per-epoch band labels; the
    default 3-tuple keeps every existing caller working unchanged.
    """
    rng = np.random.default_rng(int(seed))

    mjd = cadence["mjd"].to_numpy(dtype=float)
    band = cadence["band"].to_numpy()
    depth = cadence["depth"].to_numpy(dtype=float)
    if "mag" in getattr(cadence, "columns", ()):
        mag_real = cadence["mag"].to_numpy(dtype=float)
    else:
        mag_real = np.full(len(mjd), np.nan)
    order = np.argsort(mjd)
    mjd, band, depth, mag_real = mjd[order], band[order], depth[order], mag_real[order]

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
    # mean level of the simulated MAGNITUDE series (0 by default): the point
    # the per-band amplitudes scale about, and the origin of the deviation
    # that the per-band zero point is added to below.
    ref_level = float(np.mean(lc.mag))

    # (n_points,) per-epoch amplitude/offset; only built when colour
    # dependence was actually requested, so the no-band_amp path stays
    # identical to before this parameter existed.
    latent = lc.mag[idx]
    if band_amp is not None:
        missing = sorted(set(np.unique(band)) - set(band_amp))
        if missing:
            raise ValueError(f"band_amp has no entry for band(s) {missing}")
        a = np.array([band_amp[b] for b in band], dtype=float)
        # scale the VARIABILITY about the mean level, not the mean itself
        latent = ref_level + a * (latent - ref_level)
    else:
        a = None

    signal = None
    if mean_signal is not None:
        signal = np.asarray(mean_signal(t_years), dtype=float)
        # the fitted model divides mean_func by a_b along with the latent
        # process (see kernels.gp_log_likelihood_multiband), so the injected
        # periodic signal is scaled the same way -- injection and inference
        # assume the same thing about the periodic component
        signal = signal if a is None else a * signal
    offsets = None
    if band_mu is not None:
        missing = sorted(set(np.unique(band)) - set(band_mu))
        if missing:
            raise ValueError(f"band_mu has no entry for band(s) {missing}")
        offsets = np.array([band_mu[b] for b in band], dtype=float)

    # The source's own noise-free brightness, used ONLY to set the per-epoch
    # uncertainty -- never the realised noisy magnitude, which would feed a
    # noise draw back into its own error bar. ``band_mu`` is deliberately
    # excluded: it is a per-band offset the fitted model estimates, and the
    # band's real mean level is already carried by the per-band zero point
    # below.
    clean = latent if signal is None else latent + signal

    # Per-epoch APPARENT magnitude of the noise-free model: the band's real
    # zero point plus this epoch's deviation from the simulation's mean
    # level. Both surveys go through this same quantity; only the sigma(mag)
    # prescription differs. This is the whole of the former flux->magnitude
    # conversion -- an addition now, with no logarithm and so no need to
    # guard against a non-positive flux.
    band_ref = _band_reference_magnitudes(band, mag_real, ref_mag)
    epoch_mag = np.array([band_ref[b] for b in band], dtype=float) + (
        clean - ref_level
    )

    mag_err = np.empty(len(idx), dtype=float)
    has_depth = np.isfinite(depth)
    if has_depth.any():
        for b in np.unique(band[has_depth]):
            sel = has_depth & (band == b)
            mag_err[sel] = lsst_magnitude_error(epoch_mag[sel], depth[sel], band=b)
    if (~has_depth).any():
        if noise_model is None:
            raise ValueError(
                "cadence has rows without a real `depth` (non-LSST epochs) "
                "but no noise_model was given"
            )
        for b in np.unique(band[~has_depth]):
            sel = (~has_depth) & (band == b)
            mag_err[sel] = noise_model(b, epoch_mag[sel])

    # Both prescriptions return a MAGNITUDE error and the light curve is in
    # magnitudes, so `mag_err` is already the reported uncertainty: no
    # fractional-flux conversion, and no rescaling by the epoch's own
    # brightness (a magnitude error is absolute, not fractional). The noise
    # is Gaussian in MAGNITUDES, which is also what the surveys' quoted
    # magerr means and what the fitted likelihood assumes.
    mag = latent + rng.normal(0.0, mag_err)
    if signal is not None:
        mag = mag + signal
    if offsets is not None:
        mag = mag + offsets

    if return_band:
        return t_years, mag, mag_err, band
    return t_years, mag, mag_err
