"""Multi-band GP fits of REAL survey photometry (ZTF/LSST cadence library).

Fits each object's interleaved multi-band light curve with the shared-latent-
process model ``y_b(t) = mu_b + a_b*x(t) + noise_b(t)`` (see
``pioran_periodicity.multiband``), instead of merging every band into one
series as ``run_sim.py``/``run_realdata.py`` do. One band (the most-observed,
overridable) is the pinned reference (``a_ref=1``, ``mu_ref=0``); every other
band gets a free amplitude ``a_b`` and colour offset ``mu_b``.

``--mode merged`` reruns the SAME objects, data and priors through the old
single-band path (all bands stacked, no per-band parameters) so the two can
be compared head to head; this is what Phase 4 of the multi-band plan needs.

Resumable: a fit whose output JSON already exists under
``<out-dir>/<object_id>_<model>.json`` is skipped. Rerun after a crash.

    conda run -n <env> --no-capture-output python scripts/fit_multiband.py \
        --cadence-library /path/to/cadence_cache --survey ztf \
        --object-ids ids.txt --out-dir results/multiband \
        --models drw,drw+sine

Amplitude-like priors (``--sine-amplitude-scale``, ``--band-log-amp-sigma``,
``--band-mu-scale``, ``--log10-variance``) are FIXED hyperparameters here,
never derived from the fitted data (models.py fix M2/B2). Their defaults are
recorded in ``<out-dir>/run_config.json`` -- confirm them before a production
run.
"""

from __future__ import annotations

import argparse
import json
import os
import warnings
from dataclasses import asdict

import numpy as np

import pioran_periodicity as pp
from pioran_periodicity import (
    PriorConfig,
    SamplerSettings,
    build_family,
    run_nested,
    save_result,
)
from pioran_periodicity.cadence import CadenceLibrary
from pioran_periodicity.multiband import cadence_to_multiband_series

MODEL_FILE_NAMES = {"drw": "drw", "drw+sine": "drw_sine"}


def make_cfg(args, period_max: float) -> PriorConfig:
    """Prior configuration for real magnitude photometry.

    Amplitude/offset scales are fixed CLI hyperparameters (never data-
    derived); only the sine period cap depends on the observed baseline,
    which is timing information, not an amplitude statistic.
    """
    return PriorConfig(
        log10_variance=tuple(args.log10_variance),
        log10_fbend=tuple(args.log10_fbend),
        sine_amplitude_scale=args.sine_amplitude_scale,
        period=(args.period_min, period_max),
        err_scale=tuple(args.err_scale) if args.err_scale else None,
        band_log_amp_sigma=args.band_log_amp_sigma,
        band_mu_scale=args.band_mu_scale,
    )


def fit_seed(object_id: str, model_name: str) -> int:
    """Deterministic per-fit seed (reproducible posterior resampling)."""
    return (abs(hash(object_id)) * 131 + abs(hash(model_name))) % (2**31)


def read_object_ids(spec: str, lib: CadenceLibrary, survey: str) -> list[str]:
    """Object ids from a file (one per line), a comma list, or 'all'."""
    if spec == "all":
        return lib.object_ids(survey)
    if os.path.exists(spec):
        with open(spec) as f:
            return [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]
    return [s.strip() for s in spec.split(",") if s.strip()]


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--cadence-library",
        required=True,
        help="path to a CadenceLibrary.to_cache() directory",
    )
    ap.add_argument("--survey", required=True, help="survey name, e.g. ztf or lsst")
    ap.add_argument(
        "--object-ids",
        default="all",
        help="file with one object id per line, a comma list, or 'all'",
    )
    ap.add_argument("--out-dir", required=True, help="fit-result output directory")
    ap.add_argument(
        "--models", default="drw,drw+sine", help="comma list of drw,drw+sine"
    )
    ap.add_argument(
        "--mode",
        choices=("multiband", "merged"),
        default="multiband",
        help="multiband: per-band a_b/mu_b (default). merged: old single-band "
        "path on the same data/priors, for head-to-head comparison",
    )
    ap.add_argument(
        "--reference-band",
        default=None,
        help="pin this band as the reference (a=1, mu=0); default is the "
        "object's most-observed band",
    )
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--worker", type=int, default=0)

    # --- priors (fixed hyperparameters; recorded in run_config.json) ---
    ap.add_argument("--log10-variance", type=float, nargs=2, default=(-4.0, 1.0))
    ap.add_argument("--log10-fbend", type=float, nargs=2, default=(-3.0, 2.0))
    ap.add_argument("--sine-amplitude-scale", type=float, default=0.15)
    ap.add_argument("--period-min", type=float, default=0.2, help="years")
    ap.add_argument(
        "--period-max",
        type=float,
        default=None,
        help="years; default is half the object's observed baseline",
    )
    ap.add_argument(
        "--err-scale",
        type=float,
        nargs=2,
        default=(0.05, 1.5),
        help="fitted error-scale prior for real photometry; pass no value to "
        "disable via --no-err-scale",
    )
    ap.add_argument("--no-err-scale", action="store_true")
    ap.add_argument("--band-log-amp-sigma", type=float, default=0.3)
    ap.add_argument("--band-mu-scale", type=float, default=0.5)

    # --- sampler ---
    ap.add_argument("--min-num-live-points", type=int, default=400)
    ap.add_argument("--max-ncalls", type=int, default=1_000_000)
    ap.add_argument("--frac-remain", type=float, default=0.01)
    ap.add_argument("--min-points", type=int, default=20, help="skip sparser objects")
    args = ap.parse_args()

    if args.no_err_scale:
        args.err_scale = None

    want = [m.strip() for m in args.models.split(",") if m.strip()]
    unknown = set(want) - set(MODEL_FILE_NAMES)
    if unknown:
        ap.error(f"unknown models {sorted(unknown)}; have {sorted(MODEL_FILE_NAMES)}")

    lib = CadenceLibrary.from_cache(args.cadence_library)
    object_ids = read_object_ids(args.object_ids, lib, args.survey)
    object_ids = object_ids[args.worker :: args.stride]

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "run_config.json"), "w") as f:
        json.dump(
            {
                "args": vars(args),
                "n_objects": len(object_ids),
                "package_version": pp.__version__,
            },
            f,
            indent=2,
            default=str,
        )

    print(f"{len(object_ids)} objects x {len(want)} models queued "
          f"[mode={args.mode}]\n", flush=True)

    for i, object_id in enumerate(object_ids, 1):
        todo = [
            m
            for m in want
            if not os.path.exists(
                os.path.join(args.out_dir, f"{object_id}_{MODEL_FILE_NAMES[m]}.json")
            )
        ]
        if not todo:
            continue

        cadence = lib.get(args.survey, object_id)
        try:
            t, y, yerr, band_code, enc = cadence_to_multiband_series(
                cadence, reference=args.reference_band
            )
        except ValueError as exc:
            print(f"[{i}] SKIP {object_id}: {exc}", flush=True)
            continue
        if len(t) < args.min_points:
            print(f"[{i}] SKIP {object_id}: {len(t)} points < {args.min_points}",
                  flush=True)
            continue

        baseline = float(t[-1] - t[0])  # years
        period_max = args.period_max if args.period_max else 0.5 * baseline
        if period_max <= args.period_min:
            print(f"[{i}] SKIP {object_id}: baseline {baseline:.2f} yr too short",
                  flush=True)
            continue
        cfg = make_cfg(args, period_max)

        if args.mode == "multiband":
            photometric_bands, band_arg = enc.others, band_code
        else:
            photometric_bands, band_arg = None, None
        family = build_family(
            "drw",
            cfg,
            variants=("plain", "sine"),
            photometric_bands=photometric_bands,
        )

        for model in todo:
            spec = family[model]
            out_path = os.path.join(
                args.out_dir, f"{object_id}_{MODEL_FILE_NAMES[model]}.json"
            )
            settings = SamplerSettings(
                seed=fit_seed(object_id, model),
                min_num_live_points=args.min_num_live_points,
                max_ncalls=args.max_ncalls,
                frac_remain=args.frac_remain,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = run_nested(
                    spec,
                    t,
                    y,
                    yerr,
                    settings=settings,
                    show_status=False,
                    band=band_arg,
                )
            result.meta.update(
                object_id=object_id,
                survey=args.survey,
                mode=args.mode,
                reference_band=enc.reference,
                band_names=list(enc.names),
                band_counts={
                    b: int(np.sum(band_code == k)) for k, b in enumerate(enc.names)
                },
                n_points=len(t),
                baseline_years=baseline,
                prior_config=asdict(cfg),
            )
            save_result(result, out_path)
            flag = "" if result.converged else " [UNCONVERGED]"
            print(
                f"[{i}/{len(object_ids)}] {object_id} {model:9s} "
                f"logZ={result.logz:9.2f}+/-{result.logzerr:.2f} "
                f"ndim={spec.prior.ndim} n={len(t)}{flag}",
                flush=True,
            )

    print("DONE", flush=True)


if __name__ == "__main__":
    main()
