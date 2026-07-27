"""Simulate light curves from a bending-power-law PSD and fit noise models.

For each row of a scenario config CSV (columns: ID, bendfreq, lowalpha,
highalpha, sharpness, rms, NumofWINDOW, NightsperWINDOW, OBSperiod,
WINDOWwidth, dataLOSSfrac, noiseSIGMA, simSEED, sampleSEED):
  1. simulate a light curve from the true bending-power-law PSD with the
     package simulator (seeds taken from the config CSV for reproducibility),
  2. cache it under ``<lc-dir>/<ID>.npz`` for future reuse,
  3. fit the requested models (DRW, CARMA, OBPL, each with/without a sine
     mean) with nested sampling and save each FitResult under
     ``<out-dir>/<ID>_<model>.json``.

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
from pioran_periodicity.inference import SamplerSettings, run_nested, save_result
from pioran_periodicity.kernels import FrequencyBand, psd_approximation_error
from pioran_periodicity.simulate import sample_seasonal_pattern, simulate_lightcurve

# Simulation grid constants
N_SAMPLES = 2**21  # TK95 realisation length (39.9 yr at 10 min)
DT_MINUTES = 10.0
PSD_NORM = 20.0

# OBPL approximation controls: explicit band from the light curve's median
# cadence; bump components until the PSD approximation is accurate.
OBPL_N_COMPONENTS = 20
OBPL_MAX_REL_ERROR = 0.05

# Prior configuration for the simulation study. Amplitude-like priors are set
# from the KNOWN simulation scale, never from the fitted data. No error-scale
# parameter: the simulation noise is known exactly.
CFG = pp.PriorConfig(
    log10_variance=(-4.0, 1.0),
    log10_fbend=(-3.0, 2.0),
    alpha_low=(0.0, 2.0),
    alpha_high_max=4.0,
    sine_amplitude_scale=0.15,
    period=(0.2, 4.0),
    err_scale=None,
)

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
    negative for a falling high-frequency slope)."""
    return (norm * (f / f_bend) ** alpha_lo) / (
        1.0 + (f / f_bend) ** (sharpness * (alpha_lo - alpha_hi))
    ) ** (1.0 / sharpness)


def simulate_or_load(row, lc_dir, enforce_leakage_margin=True):
    """Return (t_years, flux, err) for one CSV row, simulating and caching
    the light curve on first use."""
    lc_id = int(row["ID"])
    path = os.path.join(lc_dir, f"{lc_id}.npz")
    if os.path.exists(path):
        d = np.load(path)
        return d["t"], d["y"], d["yerr"]

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
        n_samples=N_SAMPLES,
        dt_minutes=DT_MINUTES,
        mean=1.0,
        rms=float(row["rms"]),
        seed=int(row["simSEED"]),
    )
    t, y, yerr = sample_seasonal_pattern(
        lc,
        n_windows=int(row["NumofWINDOW"]),
        nights_per_window=int(row["NightsperWINDOW"]),
        window_period_months=float(row["OBSperiod"]),
        window_width_days=float(row["WINDOWwidth"]),
        obs_per_night=1,
        data_loss_frac=float(row["dataLOSSfrac"]),
        noise_sigma=float(row["noiseSIGMA"]),
        mean_signal=None,
        leakage_margin=10.0,
        enforce_leakage_margin=enforce_leakage_margin,
        seed=int(row["sampleSEED"]),
    )
    t = t - t[0]
    y = y - np.median(y)

    os.makedirs(lc_dir, exist_ok=True)
    np.savez(
        path,
        t=t,
        y=y,
        yerr=yerr,
        highalpha=float(row["highalpha"]),
        bendfreq=float(row["bendfreq"]),
        lowalpha=float(row["lowalpha"]),
        sharpness=float(row["sharpness"]),
        rms=float(row["rms"]),
        noiseSIGMA=float(row["noiseSIGMA"]),
        simSEED=int(row["simSEED"]),
        sampleSEED=int(row["sampleSEED"]),
        n_points=len(t),
    )
    return t, y, yerr


def obpl_components(t):
    """Pick an OBPL component count that keeps the PSD approximation error
    below OBPL_MAX_REL_ERROR for this light curve's band."""
    band = FrequencyBand.from_times(t)
    n = OBPL_N_COMPONENTS
    while n <= 60:
        err = psd_approximation_error(0.5, 0.0, 2.5, band, n_components=n)
        if err["max_rel_error"] <= OBPL_MAX_REL_ERROR:
            return band, n, err
        n += 10
    return band, n, err


def build_models(t, want=("drw", "carma", "obpl")):
    """Return {file_model_name: ModelSpec} for the requested noise families."""
    families = {}
    n_comp = None
    if "drw" in want:
        families["drw"] = pp.build_family("drw", CFG, variants=("plain", "sine"))
    if "carma" in want:
        families["carma"] = pp.build_family(
            "carma", CFG, variants=("plain", "sine"), carma_order=(2, 1)
        )
    if "obpl" in want:
        band, n_comp, _ = obpl_components(t)
        families["obpl"] = pp.build_family(
            "obpl", CFG, variants=("plain", "sine"), band=band, n_components=n_comp
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
        "--enforce-leakage-margin",
        type=lambda s: s.lower() != "false",
        default=True,
        help="raise if simulated baseline is under leakage_margin=10x the "
        "observed span (S1); pass false to accept a shorter margin (as done "
        "for scenario 3.5 at NumofWINDOW=20, see changes_and_decisions.md)",
    )
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    df = pd.read_csv(args.config_csv)
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
        min_num_live_points=400, frac_remain=0.01, max_ncalls=1_000_000
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

        t, y, yerr = simulate_or_load(
            row, args.lc_dir, enforce_leakage_margin=args.enforce_leakage_margin
        )
        need = {fn.split("_")[0] for fn in todo}
        specs, n_comp = build_models(t, want=need)

        for fn in todo:
            spec = specs[fn]
            out_path = os.path.join(args.out_dir, f"{lc_id}_{fn}.json")
            settings = SamplerSettings(seed=fit_seed(lc_id, fn), **settings_base)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = run_nested(
                    spec, t, y, yerr, settings=settings, show_status=False
                )
            result.meta.update(
                highalpha=float(row.get("highalpha", np.nan)),
                lc_id=lc_id,
                n_points=len(t),
                obpl_n_components=n_comp,
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
