"""Refit already-simulated real-cadence light curves with the multi-band model.

Motivation: on LSST real-cadence DRW null simulations, refitting each band
independently showed most objects have ZERO bands that individually detect a
period, while the merged-band fit reports a decisive detection -- i.e. naive
band merging itself manufactures spurious structure (see
``summaries/per_band_drw_check.csv``). This script refits those same cached
light curves with the shared-latent-process multi-band model
(``pioran_periodicity.multiband``) and tabulates merged vs per-band vs
multi-band Bayes factors.

Band recovery: cached ``.npz`` light curves do not store per-epoch band
labels, but ``simulate.sample_real_cadence`` keeps *every* cadence row and
sorts by ``mjd``, so the i-th light-curve point is the i-th row of the
mjd-sorted cadence. This script asserts that length identity per object and
refuses to fit if it does not hold, so the mapping is exact rather than
approximate.

Priors and sampler settings are imported from ``run_sim.py`` so the new
Bayes factors are directly comparable with the campaign numbers they are
being checked against -- do not diverge them.

    conda run -n <env> --no-capture-output python \
        scripts/check_multiband_per_band.py \
        --check-csv  ~/work/data/quasar_cadences/summaries/per_band_drw_check.csv \
        --config-csv <cadences>/scenario_csvs/lsst_real_cadence_null_case.csv \
        --lc-dir <cadences>/simulations/lsst_real_cadence_null_case/lightcurves \
        --cadence-library ~/work/data/quasar_cadences/cadence_library \
        --out-dir ~/work/data/quasar_cadences/simulations/lsst_per_band_check/multiband
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from dataclasses import asdict

import numpy as np
import pandas as pd

from pioran_periodicity import (
    SamplerSettings,
    build_family,
    load_result,
    log10_bayes_factors,
    run_nested,
    save_result,
)
from pioran_periodicity.cadence import CadenceLibrary
from pioran_periodicity.multiband import BandEncoding

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_sim import (  # noqa: E402
    MODEL_FILE_NAMES,
    fit_seed,
    make_cfg,
    require_magnitude_lightcurve,
    require_magnitude_units,
)

MODELS = ("drw", "drw+sine")
SETTINGS_BASE = dict(min_num_live_points=400, frac_remain=0.01, max_ncalls=1_000_000)


def classify(b: float) -> str:
    """Same thresholds as scripts/aggregate_results.py."""
    return "detect" if b < -2 else ("refute" if b > 2 else "inconclusive")


def recover_bands(cadence: pd.DataFrame, n_points: int) -> np.ndarray:
    """Per-epoch band labels for a cached real-cadence light curve.

    ``cadence``: the object's full cadence DataFrame. Returns a (n_points,)
    array of band-label strings in light-curve order. Raises if the cadence
    and the light curve are not the same length, which is the only thing
    that could break the 1:1 mjd-sorted correspondence.
    """
    cad = cadence.sort_values("mjd", kind="stable")
    if len(cad) != n_points:
        raise ValueError(
            f"cadence has {len(cad)} rows but light curve has {n_points} "
            "points -- band identity cannot be recovered by position"
        )
    return cad["band"].to_numpy(dtype=object)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--check-csv", required=True, help="per_band_drw_check.csv")
    ap.add_argument("--config-csv", required=True, help="scenario CSV for these lc_ids")
    ap.add_argument("--lc-dir", required=True, help="cached .npz light-curve dir")
    ap.add_argument("--cadence-library", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument(
        "--reference-band",
        default=None,
        help="pin this band as reference (a=1, mu=0); default most-observed",
    )
    ap.add_argument("--period-max", type=float, default=4.0,
                    help="must match the original campaign's cfg (run_sim: 4.0)")
    ap.add_argument("--report", default="multiband_comparison.csv")
    ap.add_argument(
        "--max-ncalls",
        type=int,
        default=SETTINGS_BASE["max_ncalls"],
        help="the campaign default (1e6) was tuned at ndim 2-5; the multi-band "
        "model is ndim 12-15 and slice-samples 2*ndim steps per point, so it "
        "needs a larger budget to avoid truncating (ESS ~ 1)",
    )
    ap.add_argument(
        "--lc-ids",
        default=None,
        help="comma list restricting which lc_ids to fit (default: all in "
        "--check-csv); use for a pilot fit",
    )
    ap.add_argument(
        "--models",
        default=",".join(MODELS),
        help=f"comma list of {list(MODELS)}",
    )
    args = ap.parse_args()

    settings_base = dict(SETTINGS_BASE, max_ncalls=args.max_ncalls)
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    if set(models) - set(MODELS):
        ap.error(f"unknown models {sorted(set(models) - set(MODELS))}")

    chk = pd.read_csv(os.path.expanduser(args.check_csv))
    cfg_df = pd.read_csv(os.path.expanduser(args.config_csv))
    # This script reads the campaign's config CSV and cached light curves
    # directly rather than through simulate_or_load, so it must repeat both
    # of run_sim.py's unit guards. It fits with make_cfg's magnitude-
    # calibrated priors, so a flux-era input would produce per-band
    # diagnostics that are quietly 8% off rather than failing.
    require_magnitude_units(cfg_df, args.config_csv)
    lib = CadenceLibrary.from_cache(os.path.expanduser(args.cadence_library))
    out_dir = os.path.expanduser(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    cfg = make_cfg(args.period_max)  # identical priors to the campaign
    lc_ids = sorted(int(i) for i in chk["lc_id"].unique())
    if args.lc_ids:
        want_ids = {int(s) for s in args.lc_ids.split(",") if s.strip()}
        lc_ids = [i for i in lc_ids if i in want_ids]
    print(
        f"{len(lc_ids)} objects x {len(models)} models "
        f"(max_ncalls={args.max_ncalls:,})\n",
        flush=True,
    )

    rows = []
    for lc_id in lc_ids:
        lc_id = int(lc_id)  # pandas gives np.int64; keeps seeds/meta JSON-safe
        npz_path = os.path.join(os.path.expanduser(args.lc_dir), f"{lc_id}.npz")
        npz = np.load(npz_path)
        require_magnitude_lightcurve(npz, npz_path)
        t, y, yerr = npz["t"], npz["y"], npz["yerr"]

        crow = cfg_df.loc[cfg_df["ID"] == lc_id].iloc[0]
        survey, obj = str(crow["cadence_source"]).split(":", 1)
        labels = recover_bands(lib.get(survey, obj), len(t))
        enc = BandEncoding.from_counts(labels, reference=args.reference_band)
        band_code = enc.encode(labels)  # (n_points,), 0 = reference band

        family = build_family(
            "drw", cfg, variants=("plain", "sine"), photometric_bands=enc.others
        )
        results = {}
        for model in models:
            fname = f"{lc_id}_{MODEL_FILE_NAMES[model]}.json"
            path = os.path.join(out_dir, fname)
            if os.path.exists(path):
                try:
                    results[model] = load_result(path)
                except (ValueError, KeyError) as exc:
                    # a crash mid-write leaves a truncated JSON: refit it
                    print(f"  {lc_id} {model:9s} REFIT (unreadable: {exc})", flush=True)
                    os.remove(path)
                else:
                    print(
                        f"  {lc_id} {model:9s} (cached) logZ={results[model].logz:.2f}",
                        flush=True,
                    )
                    continue
            spec = family[model]
            settings = SamplerSettings(
                seed=int(fit_seed(lc_id, MODEL_FILE_NAMES[model])), **settings_base
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = run_nested(
                    spec, t, y, yerr, settings=settings, show_status=False,
                    band=band_code,
                )
            res.meta.update(
                lc_id=int(lc_id),
                cadence_source=str(crow["cadence_source"]),
                mode="multiband",
                reference_band=enc.reference,
                band_names=list(enc.names),
                band_counts={
                    b: int(np.sum(band_code == k)) for k, b in enumerate(enc.names)
                },
                n_points=int(len(t)),
                prior_config=asdict(cfg),
            )
            save_result(res, path)
            results[model] = res
            flag = "" if res.converged else " [UNCONVERGED]"
            print(
                f"  {lc_id} {model:9s} logZ={res.logz:9.2f}+/-{res.logzerr:.2f} "
                f"ndim={spec.prior.ndim} n={len(t)}{flag}",
                flush=True,
            )

        if not {"drw", "drw+sine"} <= set(results):
            continue  # pilot / partial --models run: no Bayes factor to report
        bf_mb = log10_bayes_factors(
            {"drw": results["drw"], "drw_sine": results["drw+sine"]}
        )["drw/drw_sine"]
        grp = chk[chk["lc_id"] == lc_id]
        rows.append(
            {
                "lc_id": lc_id,
                "object": obj,
                "n_points": len(t),
                "reference_band": enc.reference,
                "bf_merged": float(grp["bf_merged"].iloc[0]),
                "outcome_merged": classify(float(grp["bf_merged"].iloc[0])),
                "n_bands_detecting": int((grp["outcome"] == "detect").sum()),
                "n_bands": int(len(grp)),
                "bf_multiband": bf_mb,
                "outcome_multiband": classify(bf_mb),
                "converged": all(r.converged for r in results.values()),
            }
        )

    if not rows:
        print("\nno complete (drw, drw+sine) pairs -- no comparison written")
        return
    report = pd.DataFrame(rows)
    report_path = os.path.join(out_dir, args.report)
    report.to_csv(report_path, index=False)
    print("\n" + report.to_string(index=False))
    print(f"\nwrote {report_path}", flush=True)


if __name__ == "__main__":
    main()
