"""Simulate light curves from a bending-power-law PSD and fit noise models.

For each row of a scenario config CSV (columns: ID, bendfreq, lowalpha,
highalpha, sharpness, rms, simSEED, sampleSEED; optionally period, A1 -- see
below), the observation cadence comes from EITHER of two mutually exclusive
column groups:

* synthetic seasonal window (the original path): NumofWINDOW,
  NightsperWINDOW, OBSperiod, WINDOWwidth, dataLOSSfrac, noiseSIGMA.
* a real survey cadence (2026-07-29): ``cadence_source`` (format
  ``<survey>:<object_id>``, e.g. ``ztf:SDSS_J075056.32+525640.9``) plus
  ``ref_mag`` (the object's fixed catalog magnitude, used to evaluate the
  survey's heteroscedastic noise model). Requires ``--cadence-library``.
  See ``pioran_periodicity.cadence.CadenceLibrary`` and
  ``pioran_periodicity.simulate.sample_real_cadence``.

A CSV mixes rows from either group; each row's own columns decide which path
it takes.

The steps for each row:
  1. simulate a light curve from the true bending-power-law PSD with the
     package simulator (seeds taken from the config CSV for reproducibility),
     adding a true sine signal if the row has ``period``/``A1`` columns with
     ``A1 != 0`` (amplitude ``A1`` directly, zero phase -- matches the
     confirmed thesis-text convention "A_sine = A1", NOT
     ``original/batch_run.py``'s unrelated ``0.15*A1`` scaling; see
     ``workspace/run_sim.py``'s ``SCENARIOS["3.7"]`` comment in the
     thesis-replication repo, which pins this down explicitly),
  2. cache it under ``<lc-dir>/<ID>.npz`` for future reuse,
  3. fit the requested models (DRW, CARMA, OBPL, each with/without a sine
     mean) with nested sampling and save each FitResult under
     ``<out-dir>/<ID>_<model>.json``.

A CSV without ``period``/``A1`` columns (or with ``A1 == 0``) simulates pure
red noise, as before -- this is what makes a config a genuine null/no-signal
scenario for FPR calibration.

The sine period prior is FIXED at ``PERIOD_PRIOR`` (0.2--8.0 YEARS) for every
scenario, null and signal alike, overridable only through the explicit
``--period-max`` flag. It used to be derived from the CSV's own ``period``
column, which silently gave null CSVs (empty column) an upper bound of 4.0 yr
and signal CSVs 8.0 yr: the false-positive rate was then calibrated under a
model with half the sine prior volume of the model used to measure detection
power, biasing every null Bayes factor by ~log(7.8/3.8) ~ 0.7 nat relative to
the signal runs. A Bayes factor is only comparable across runs sharing a
prior (defect MB3.1). Injected periods outside the prior are now a hard error
rather than a silent truncation.

Resumable: existing light-curve .npz files are reused, existing per-fit
JSONs are skipped. Split across workers with --stride / --worker.

    conda run -n <env> python scripts/run_sim.py --config-csv scenario.csv \
        --lc-dir lightcurves/3_4 --out-dir results/3_4 \
        --n-sims 100 --stride 4 --worker 0
"""

from __future__ import annotations

import argparse
import os
import warnings

import numpy as np
import pandas as pd

import pioran_periodicity as pp
from pioran_periodicity.cadence import CadenceLibrary
from pioran_periodicity.inference import SamplerSettings, run_nested, save_result
from pioran_periodicity.kernels import FrequencyBand, psd_approximation_error
from pioran_periodicity.means import sine_mean
from pioran_periodicity.multiband import BandEncoding, power_law_band_amplitudes
from pioran_periodicity.simulate import (
    sample_real_cadence,
    sample_seasonal_pattern,
    FRACTIONAL_FLUX_TO_MAG,
    MAGNITUDE_UNITS,
    simulate_lightcurve,
)

# Simulation grid constants
N_SAMPLES = 2**21  # TK95 realisation length (39.9 yr at 10 min) -- synthetic scenarios
# Real ZTF/LSST cadences span the full multi-year survey history (up to ~10 yr),
# not a hand-picked short window -- N_SAMPLES leaves only a ~4-5.6x leakage
# margin (S1) against that, under the 10x default. 2**23 (159.6 yr) clears 10x
# for both surveys (ZTF ~21.6x, LSST ~16.0x). Only used for cadence_source rows;
# synthetic scenarios keep N_SAMPLES so their cached .npz files stay valid.
N_SAMPLES_REAL_CADENCE = 2**23
DT_MINUTES = 10.0
PSD_NORM = 20.0

# OBPL approximation controls: explicit band from the light curve's median
# cadence; bump components until the PSD approximation is accurate.
OBPL_N_COMPONENTS = 20
OBPL_MAX_REL_ERROR = 0.05


# Prior configuration for the simulation study. Amplitude-like priors are set
# from the KNOWN simulation scale, never from the fitted data. No error-scale
# parameter: the simulation noise is known exactly. Only the sine period cap
# varies, and it is now declared PER SCENARIO CSV in a `period_max` column
# written by the builders -- see resolve_period_prior.
# Sine period prior, in YEARS (t is in years throughout this pipeline). The
# upper bound here is only the legacy fallback for CSVs predating that
# column; the campaign values are ZTF 6.0 and LSST WFD/synthetic 9.0, which
# track the surveys' different baselines.
PERIOD_PRIOR = (0.2, 8.0)


def make_cfg(period_max: float = PERIOD_PRIOR[1]) -> pp.PriorConfig:
    return pp.PriorConfig(
        # mag^2 now that the simulator generates magnitudes (it was flux^2).
        # The campaigns' truth, sigma = 0.163 mag, is log10 var = -1.58 --
        # interior to this range, as -1.65 was under the old convention, so
        # the bounds need no change.
        log10_variance=(-4.0, 1.0),
        log10_fbend=(-3.0, 2.0),
        alpha_low=(0.0, 2.0),
        alpha_high_max=4.0,
        # Relative (hierarchical) sine amplitude: the prior is on
        # f = A / sigma_process, not on an absolute amplitude, so it is
        # scale-free across objects AND unaffected by the move from flux to
        # magnitudes. The scale is kept only as the CARMA fallback -- CARMA
        # has no sampled process variance -- and equals the campaigns' known
        # simulated rms, so f = A1/scale there. Both numerator and
        # denominator are now magnitudes, so f is numerically unchanged.
        sine_amplitude_fraction=1.2,
        sine_amplitude_scale=0.15 * FRACTIONAL_FLUX_TO_MAG,
        period=(PERIOD_PRIOR[0], period_max),
        err_scale=None,
    )


def require_magnitude_units(df, path) -> None:
    """Refuse a config CSV that was not written for the magnitude simulator.

    The simulator generated fractional FLUX until 2026-09-04; ``rms``,
    ``noiseSIGMA`` and ``A1`` are now MAGNITUDES. The two conventions differ
    by only 2.5/ln(10) = 1.0857, so a stale CSV would run to completion and
    silently inject 8% less variability than its calibration assumed -- the
    kind of error that never announces itself. Requiring an explicit
    ``units`` column makes that impossible: old CSVs have none and stop here.
    """
    if "units" not in df.columns:
        raise ValueError(
            f"{path} has no `units` column. Config CSVs written before the "
            f"simulator moved to magnitudes hold fractional-FLUX `rms`, "
            f"`noiseSIGMA` and `A1`; multiply them by "
            f"{FRACTIONAL_FLUX_TO_MAG:.4f} (2.5/ln 10) to preserve the "
            f"physical amplitude and add units={MAGNITUDE_UNITS!r}, or "
            f"regenerate the CSV with the current scripts/make_*_csv.py."
        )
    found = set(df["units"].astype(str).unique())
    if found != {MAGNITUDE_UNITS}:
        raise ValueError(
            f"{path}: units must be {MAGNITUDE_UNITS!r} in every row, "
            f"found {sorted(found)}"
        )


def require_magnitude_lightcurve(npz, path) -> None:
    """Refuse a cached light curve that was simulated in fractional flux.

    A flux-era cache is indistinguishable from a valid one by shape or
    scale, so the ``units`` field is the only thing that tells them apart.
    Flux-era light curves cannot be rescaled into magnitudes after the fact,
    because their per-epoch uncertainties were derived in flux.

    Shared by :func:`simulate_or_load` and scripts that read ``lc_dir``
    directly (``check_multiband_per_band.py``) -- the check is three lines
    and was got wrong once when copy-pasted, so there is one copy.
    """
    cached = str(npz["units"]) if "units" in npz.files else ""
    if cached != MAGNITUDE_UNITS:
        raise ValueError(
            f"{path} was simulated in fractional flux (no "
            f"units={MAGNITUDE_UNITS!r} marker). Delete the cache directory "
            f"and re-simulate; flux-era light curves cannot be rescaled into "
            f"magnitudes after the fact, because their per-epoch "
            f"uncertainties were derived in flux."
        )


def resolve_period_prior(df, period_max):
    """The sine period prior's upper bound (years) for this scenario CSV.

    Precedence: an explicit ``--period-max`` wins; otherwise the value the
    CSV builder STAMPED in its ``period_max`` column; otherwise the legacy
    ``PERIOD_PRIOR`` default, with a warning. If both are present and they
    DISAGREE the run is refused -- silently preferring one of them is how a
    campaign ends up with a prior nobody recorded (MB3.1).

    The stamped column is not "deriving the prior from the data": it is a
    declared constant that the builder wrote down because it knows which
    survey the scenario targets (ZTF 6.0 / LSST WFD and synthetic 9.0). What
    MB3.1 forbade was inferring the bound from the injected ``period``
    values, which is exactly what the check below only VALIDATES against.

    NaN-safe by construction: null CSVs carry an all-NaN ``period`` column,
    and the finite mask leaves an empty array rather than relying on
    ``max``/``nanmax`` behaviour with NaN (MB3.4 -- ``np.nanmax`` of an
    all-NaN array warns and returns NaN, and the old ``max(4.0, nan)``
    returned 4.0 only because of Python's argument order).
    """
    stamped = None
    if "period_max" in getattr(df, "columns", ()):
        values = np.unique(np.asarray(df["period_max"], dtype=float))
        if values.size != 1:
            raise ValueError(
                f"scenario CSV has {values.size} distinct period_max values "
                f"({values}); one scenario file must carry one prior"
            )
        stamped = float(values[0])

    if period_max is None:
        if stamped is None:
            period_max = PERIOD_PRIOR[1]
            print(
                f"WARNING: scenario CSV has no period_max column; falling "
                f"back to the legacy default {period_max} yr. Rebuild the "
                f"CSV so the prior is recorded with the scenario."
            )
        else:
            period_max = stamped
    elif stamped is not None and float(period_max) != stamped:
        raise ValueError(
            f"--period-max {float(period_max)!r} contradicts the "
            f"period_max={stamped!r} stamped in the scenario CSV. Drop the "
            f"flag to use the stamped value, or rebuild the CSV."
        )

    period_max = float(period_max)
    if not np.isfinite(period_max) or period_max <= 0:
        raise ValueError(f"period_max must be finite and > 0, got {period_max}")
    if "period" in getattr(df, "columns", ()):
        injected = np.asarray(df["period"], dtype=float)
        injected = injected[np.isfinite(injected)]
        if injected.size and injected.max() >= period_max:
            raise ValueError(
                f"scenario injects periods up to {injected.max():g} yr, "
                f"outside the sine period prior (upper bound {period_max:g} "
                f"yr). Raise --period-max AND rerun the matching null "
                f"campaign with the same value, or the two are not "
                f"comparable (MB3.1)."
            )
    return period_max


MODEL_FILE_NAMES = {
    "drw": "drw",
    "drw+sine": "drw_sine",
    "carma": "carma",
    "carma+sine": "carma_sine",
    "obpl": "obpl",
    "obpl+sine": "obpl_sine",
}


def bend_pl(f, norm, f_bend, alpha_lo, alpha_hi, sharpness):
    """Bending power law used to generate the data (f in day^-1, alpha_hi
    negative for a falling high-frequency slope).

    At ``sharpness = 1`` this is exactly Pioran's ``SingleBendingPowerLaw``
    (sign convention alpha_model = -alpha_here), so injection and the fitted
    OBPL share one PSD family -- which is what the campaign CSVs now set.
    Pioran has no sharpness parameter, so any other value makes the OBPL
    model unable to represent the simulated knee (defect MB3.5).
    """
    return (norm * (f / f_bend) ** alpha_lo) / (
        1.0 + (f / f_bend) ** (sharpness * (alpha_lo - alpha_hi))
    ) ** (1.0 / sharpness)


def true_mean_signal(row):
    """The true periodic signal to inject, or None for a pure-red-noise
    (null) light curve.

    ``period``/``A1`` are optional CSV columns; a CSV that omits them, or
    sets ``A1 == 0``, simulates pure red noise -- this is what makes a
    config a genuine null scenario for FPR calibration. When present, ``A1``
    IS the true amplitude directly, in MAGNITUDES like the light curve
    (thesis-text convention "A_sine = A1",
    confirmed in workspace/run_sim.py's SCENARIOS["3.7"] comment in the
    thesis-replication repo -- NOT a fraction of rms; do not reintroduce an
    ``rms *`` scaling here without re-checking that source).
    """
    if "period" not in row.index or "A1" not in row.index:
        return None
    amplitude = float(row["A1"])
    if amplitude == 0.0:
        return None
    period = float(row["period"])
    return lambda t_years: sine_mean(t_years, 0.0, amplitude, period)


def has_real_cadence(row) -> bool:
    return "cadence_source" in row.index and pd.notna(row["cadence_source"])


def band_amp_beta(row):
    """Colour-dependence index for this row, or None for no colour dependence.

    ``band_amp_beta`` is an optional CSV column: per-band variability
    amplitudes are ``a_b = (lambda_b / lambda_ref) ** (-beta)``, so beta = 0
    means every band varies identically -- exactly what the simulator did
    before multi-band support. A CSV that omits the column, or leaves it
    blank, is simulated the old way and its cached light curves stay
    byte-identical (no ``band`` array saved, global-median centring).

    Note beta = 0 is NOT the same as omitting the column: beta = 0 still
    records band identities and centres on the reference band, giving a
    genuine multi-band null case to fit.
    """
    if "band_amp_beta" not in row.index or pd.isna(row["band_amp_beta"]):
        return None
    return float(row["band_amp_beta"])


def simulate_or_load(
    row, lc_dir, enforce_leakage_margin=True, cadence_lib=None, n_samples_override=None
):
    """Return (t_years, mag, mag_err) for one CSV row, simulating and caching
    the light curve on first use. Everything is in MAGNITUDES.

    ``row["cadence_source"]`` (format ``<survey>:<object_id>``), if present
    and non-null, samples a REAL survey cadence via ``cadence_lib`` instead
    of the synthetic seasonal-window pattern -- see module docstring.

    ``n_samples_override``, if given, forces the TK95 simulation length for
    EVERY row in this call, superseding the cadence_source-based default
    below -- for re-simulating a synthetic-window (non-cadence_source) CSV
    at a longer length to fix a leakage-margin violation (S1) that the
    default N_SAMPLES doesn't clear for that CSV's observed baseline (e.g.
    the original signal_case.csv/null_case.csv at NumofWINDOW=20).
    """
    lc_id = int(row["ID"])
    path = os.path.join(lc_dir, f"{lc_id}.npz")
    if os.path.exists(path):
        d = np.load(path)
        require_magnitude_lightcurve(d, path)
        # `band` is absent from light curves cached before multi-band support
        band = d["band"] if "band" in d.files else None
        return d["t"], d["y"], d["yerr"], band

    cadence_source = row["cadence_source"] if has_real_cadence(row) else None
    beta = None
    encoding = None
    if n_samples_override is not None:
        n_samples = n_samples_override
    else:
        n_samples = N_SAMPLES_REAL_CADENCE if cadence_source is not None else N_SAMPLES

    psd_params = [
        PSD_NORM,
        float(row["bendfreq"]),
        float(row["lowalpha"]),
        float(row["highalpha"]),
        float(row["sharpness"]),
    ]
    lc = simulate_lightcurve(
        bend_pl,
        psd_params,
        n_samples=n_samples,
        dt_minutes=DT_MINUTES,
        mean_mag=0.0,
        sigma_mag=float(row["rms"]),
        seed=int(row["simSEED"]),
    )

    if cadence_source is not None:
        if cadence_lib is None:
            raise ValueError(
                f"row {lc_id} has cadence_source={cadence_source!r} but no "
                f"--cadence-library was given"
            )
        survey, object_id = str(cadence_source).split(":", 1)
        cadence = cadence_lib.get(survey, object_id)
        beta = band_amp_beta(row)
        band_amp = None
        if beta is not None:
            encoding = BandEncoding.from_counts(cadence["band"].to_numpy(dtype=object))
            band_amp = power_law_band_amplitudes(
                encoding.names, beta, encoding.reference, survey=survey
            )
        out = sample_real_cadence(
            lc,
            cadence,
            noise_model=cadence_lib.noise_models.get(survey),
            ref_mag=float(row["ref_mag"]),
            mean_signal=true_mean_signal(row),
            leakage_margin=10.0,
            enforce_leakage_margin=enforce_leakage_margin,
            seed=int(row["sampleSEED"]),
            band_amp=band_amp,
            return_band=beta is not None,
        )
        band = out[3] if beta is not None else None
        t, y, yerr = out[0], out[1], out[2]
    else:
        t, y, yerr = sample_seasonal_pattern(
            lc,
            n_windows=int(row["NumofWINDOW"]),
            nights_per_window=int(row["NightsperWINDOW"]),
            window_period_months=float(row["OBSperiod"]),
            window_width_days=float(row["WINDOWwidth"]),
            obs_per_night=1,
            data_loss_frac=float(row["dataLOSSfrac"]),
            noise_sigma=float(row["noiseSIGMA"]),
            mean_signal=true_mean_signal(row),
            leakage_margin=10.0,
            enforce_leakage_margin=enforce_leakage_margin,
            seed=int(row["sampleSEED"]),
        )
        band = None
    t = t - t[0]
    if band is None:
        y = y - np.median(y)
    else:
        # centre on the REFERENCE band's own median, generalising the
        # single-band convention so the fitted model's pinned mu_ref = 0 is
        # valid by construction (see multiband.cadence_to_multiband_series)
        y = y - np.median(y[band == encoding.reference])

    os.makedirs(lc_dir, exist_ok=True)
    np.savez(
        path,
        t=t,
        y=y,
        yerr=yerr,
        highalpha=float(row["highalpha"]),
        bendfreq=float(row["bendfreq"]),
        true_period=float(row["period"]) if "period" in row.index else np.nan,
        true_A1=float(row["A1"]) if "A1" in row.index else 0.0,
        lowalpha=float(row["lowalpha"]),
        sharpness=float(row["sharpness"]),
        rms=float(row["rms"]),
        noiseSIGMA=float(row["noiseSIGMA"]) if "noiseSIGMA" in row.index else np.nan,
        units=MAGNITUDE_UNITS,
        cadence_source=str(cadence_source) if cadence_source is not None else "",
        simSEED=int(row["simSEED"]),
        sampleSEED=int(row["sampleSEED"]),
        n_points=len(t),
        band_amp_beta=beta if cadence_source is not None else np.nan,
        **({} if band is None else {"band": band.astype(str)}),
    )
    return t, y, yerr, band


def obpl_components(t, alpha_high_max=None):
    """Pick an OBPL component count that keeps the PSD approximation error
    below OBPL_MAX_REL_ERROR for this light curve's band.

    The error is checked at the WORST CASE the sampler can reach -- the top of
    the ``alpha_high`` prior -- not at a fixed mid-range slope. It previously
    tested a hard-coded ``alpha_high=2.5`` while the prior runs to 4.0, and
    that was actively violated, not merely untidy: at n=20 the error on a
    typical ZTF sampling (1055 points over 7.4 yr) is 1.18% at alpha=2.5 but
    **8.55% at alpha=4.0**, against a 5% budget. The steep-slope cells are
    exactly the scientifically interesting ones, so the model being fitted
    there was not the model we thought. Measured at n=20 across samplings:

    ==========================  =======  =======
    sampling                    a=2.5    a=4.0
    ==========================  =======  =======
    LSST WFD 690 pts / 9.9 yr     1.04%    4.20%
    ZTF     1055 pts / 7.4 yr     1.18%    8.55%
    sparse   200 pts / 9.9 yr     0.45%    1.27%
    synthetic 300 pts / 9.5 yr    0.47%    3.49%
    ==========================  =======  =======

    Expect some light curves to need n=30 (0.63% at alpha=4.0), which costs
    more per likelihood call.
    """
    if alpha_high_max is None:
        alpha_high_max = pp.PriorConfig().alpha_high_max
    band = FrequencyBand.from_times(t)
    n = OBPL_N_COMPONENTS
    while n <= 60:
        err = psd_approximation_error(
            0.5, 0.0, float(alpha_high_max), band, n_components=n
        )
        if err["max_rel_error"] <= OBPL_MAX_REL_ERROR:
            return band, n, err
        n += 10
    return band, n, err


def build_models(
    t,
    cfg,
    want=("drw", "carma", "obpl"),
    photometric_bands=None,
    fit_sine_colour=False,
    reference_band=None,
    survey="lsst",
):
    """Return {file_model_name: ModelSpec} for the requested noise families.

    ``photometric_bands`` (non-reference band names, ``BandEncoding.others``
    order) builds multi-band models with per-band ``a_b``/``mu_b``; None
    (default) builds ordinary single-band models, unchanged.

    ``fit_sine_colour`` adds ONE parameter, ``beta_sine``, giving the periodic
    component its own colour dependence independent of the red noise's. Costs
    a single dimension however many filters there are.
    """
    families = {}
    n_comp = None
    mb = {
        "photometric_bands": photometric_bands,
        "fit_sine_colour": fit_sine_colour,
        "reference_band": reference_band,
        "survey": survey,
    }
    if "drw" in want:
        families["drw"] = pp.build_family("drw", cfg, variants=("plain", "sine"), **mb)
    if "carma" in want:
        families["carma"] = pp.build_family(
            "carma", cfg, variants=("plain", "sine"), carma_order=(2, 1), **mb
        )
    if "obpl" in want:
        band, n_comp, _ = obpl_components(t)
        families["obpl"] = pp.build_family(
            "obpl", cfg, variants=("plain", "sine"), band=band, n_components=n_comp,
            **mb,
        )
    specs = {}
    for fam in families.values():
        for name, spec in fam.members.items():
            specs[MODEL_FILE_NAMES[name]] = spec
    return specs, n_comp


def fit_seed(lc_id, model_name):
    """Deterministic per-fit seed (reproducible resampling)."""
    return (lc_id * 131 + hash(model_name)) % (2**31)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--config-csv",
        required=True,
        help="scenario config CSV (columns documented in module docstring)",
    )
    ap.add_argument("--lc-dir", required=True, help="light-curve cache directory")
    ap.add_argument("--out-dir", required=True, help="fit-result output directory")
    ap.add_argument(
        "--n-sims", type=int, default=100, help="light curves per config group"
    )
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument(
        "--group-col",
        default="highalpha",
        help="CSV column to group by when limiting --n-sims per config",
    )
    ap.add_argument("--filter-col", default="lowalpha")
    ap.add_argument("--filter-value", type=float, default=0.0)
    ap.add_argument(
        "--models", default="all", help="comma list of drw,carma,obpl or 'all'"
    )
    ap.add_argument(
        "--multiband",
        action="store_true",
        help="fit the shared-latent-process multi-band model (per-band a_b, "
        "mu_b) instead of merging all bands into one series. Requires light "
        "curves simulated from a CSV with a band_amp_beta column. Adds 2 free "
        "parameters per non-reference band, which costs substantially more "
        "sampler calls -- see scripts/REMOTE_RUN.md before a large campaign",
    )
    ap.add_argument(
        "--enforce-leakage-margin",
        type=lambda s: s.lower() != "false",
        default=True,
        help="raise if simulated baseline is under leakage_margin=10x the "
        "observed span (S1); pass false to accept a shorter margin (as done "
        "for scenario 3.5 at NumofWINDOW=20, see changes_and_decisions.md)",
    )
    ap.add_argument(
        "--cadence-library",
        default=None,
        help="path to a CadenceLibrary.to_cache() directory; required if "
        "--config-csv has any non-null cadence_source rows (real ZTF/LSST "
        "cadences instead of the synthetic seasonal window)",
    )
    ap.add_argument(
        "--n-samples",
        type=int,
        default=None,
        help="override the TK95 simulation length (samples) for every row "
        "in this CSV; default: N_SAMPLES_REAL_CADENCE for cadence_source "
        "rows, N_SAMPLES otherwise. Use this to re-simulate a "
        "synthetic-window CSV at a longer length to fix a leakage-margin "
        "violation (S1) its default N_SAMPLES doesn't clear.",
    )
    ap.add_argument(
        "--max-ncalls",
        type=int,
        default=1_000_000,
        help="ultranest likelihood-call cap per fit. The default suffices for "
        "single-band and ZTF multi-band models, but NOT for 6-band LSST "
        "multi-band fits (12-17 dimensions), which need 2.7M-4.9M and are "
        "silently truncated at the default -- producing a logz that is not an "
        "evidence estimate. Use ~8000000 with --multiband on LSST cadences.",
    )
    ap.add_argument(
        "--fit-sine-colour",
        action="store_true",
        help="fit beta_sine: give the periodic component its own per-band "
        "amplitude c_b = (lambda_b/lambda_ref)**-beta_sine, independent of "
        "the red noise's a_b. Without it c_b = a_b, so the sine and the noise "
        "share a colour and multi-band data cannot distinguish a real signal "
        "from red-noise leakage. Costs ONE extra dimension regardless of the "
        "number of filters. Requires --multiband.",
    )
    ap.add_argument(
        "--checkpoint-dir",
        default=None,
        help="enable ultranest on-disk checkpointing under this directory "
        "(one subdirectory per lc_id/model). Makes a fit truncated by "
        "--max-ncalls resumable: rerunning with a larger cap continues the "
        "existing integration instead of restarting it. Costs disk, but for "
        "long multi-band fits it is the difference between extending a run "
        "and throwing it away.",
    )
    ap.add_argument(
        "--period-max",
        type=float,
        default=None,
        help="upper bound (years) of the sine period prior. Defaults to the "
        "value the CSV builder stamped in the scenario's period_max column "
        "(ZTF 6.0, LSST WFD and synthetic 9.0); passing it here is only for "
        "CSVs predating that column, and a value contradicting the column "
        "is an error. THE SAME VALUE MUST be used for a null campaign and "
        "the signal campaign it calibrates (MB3.1).",
    )
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    df = pd.read_csv(args.config_csv)
    require_magnitude_units(df, args.config_csv)

    needs_cadence_lib = "cadence_source" in df.columns and df["cadence_source"].notna().any()
    if needs_cadence_lib and not args.cadence_library:
        raise ValueError(
            f"{args.config_csv} has cadence_source rows but --cadence-library "
            f"was not given"
        )
    cadence_lib = CadenceLibrary.from_cache(args.cadence_library) if args.cadence_library else None

    period_max = resolve_period_prior(df, args.period_max)
    cfg = make_cfg(period_max)

    picked = (
        df[df[args.filter_col] == args.filter_value]
        .groupby(args.group_col, sort=True)
        .head(args.n_sims)
    )
    picked = picked.iloc[args.worker :: args.stride]

    want = args.models.split(",") if args.models != "all" else ["drw", "carma", "obpl"]
    want_files = set()
    for m in want:
        want_files |= {v for k, v in MODEL_FILE_NAMES.items() if k.split("+")[0] == m}

    settings_base = dict(
        min_num_live_points=400, frac_remain=0.01, max_ncalls=args.max_ncalls
    )

    for _, row in picked.iterrows():
        lc_id = int(row["ID"])
        todo = [
            fn
            for fn in want_files
            if not os.path.exists(os.path.join(args.out_dir, f"{lc_id}_{fn}.json"))
        ]
        if not todo:
            continue

        t, y, yerr, band_labels = simulate_or_load(
            row,
            args.lc_dir,
            enforce_leakage_margin=args.enforce_leakage_margin,
            cadence_lib=cadence_lib,
            n_samples_override=args.n_samples,
        )
        # multi-band fitting only when asked for AND the light curve carries
        # band identities (i.e. was simulated with a band_amp_beta column)
        encoding = band_code = photometric_bands = None
        if args.multiband:
            if band_labels is None:
                raise ValueError(
                    f"--multiband given but light curve {lc_id} has no band "
                    "labels; its CSV needs a band_amp_beta column and the "
                    "cached .npz must be re-simulated"
                )
            encoding = BandEncoding.from_counts(band_labels)
            band_code = encoding.encode(band_labels)
            photometric_bands = encoding.others

        need = {fn.split("_")[0] for fn in todo}
        specs, n_comp = build_models(
            t,
            cfg,
            want=need,
            photometric_bands=photometric_bands,
            fit_sine_colour=args.fit_sine_colour and photometric_bands is not None,
            reference_band=None if encoding is None else encoding.reference,
            survey=(
                str(row["cadence_source"]).split(":", 1)[0]
                if has_real_cadence(row)
                else "lsst"
            ),
        )

        for fn in todo:
            spec = specs[fn]
            out_path = os.path.join(args.out_dir, f"{lc_id}_{fn}.json")
            settings = SamplerSettings(seed=fit_seed(lc_id, fn), **settings_base)
            fit_log_dir = None
            if args.checkpoint_dir:
                fit_log_dir = os.path.join(args.checkpoint_dir, f"{lc_id}_{fn}")
                os.makedirs(fit_log_dir, exist_ok=True)
            with warnings.catch_warnings():
                # ultranest is noisy, but never swallow our own convergence
                # warning -- that suppression is why the 2026-08 LSST
                # multi-band truncation went unnoticed for a week
                warnings.simplefilter("ignore")
                warnings.filterwarnings("always", message=r".*posterior ESS.*")
                result = run_nested(
                    spec, t, y, yerr, settings=settings, show_status=False,
                    band=band_code, log_dir=fit_log_dir,
                )
            result.meta.update(
                highalpha=float(row.get("highalpha", np.nan)),
                lc_id=lc_id,
                n_points=len(t),
                obpl_n_components=n_comp,
                true_period=float(row["period"]) if "period" in row.index else None,
                true_A1=float(row["A1"]) if "A1" in row.index else 0.0,
                true_band_amp_beta=band_amp_beta(row),
                reference_band=None if encoding is None else encoding.reference,
                photometric_bands=(
                    None if photometric_bands is None else list(photometric_bands)
                ),
            )
            save_result(result, out_path)
            flag = "" if result.converged else " [UNCONVERGED]"
            print(
                f"{lc_id} {fn:10s} logZ={result.logz:8.2f} "
                f"ESS={result.ess:6.0f}{flag}",
                flush=True,
            )

    print("DONE", flush=True)


if __name__ == "__main__":
    main()
