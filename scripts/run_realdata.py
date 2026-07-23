"""Real-data re-analysis with the pioran_periodicity package.

Fits PG 1302-102 (CRTS, mag) and PG 1553+113 (Fermi-LAT, ln-flux) with DRW and
OBPL noise models against sine / linear / sine+linear mean alternatives via
nested sampling (ultranest).

Design notes: free (sampled) process variance everywhere, shared priors
across variants, robust median-spacing FrequencyBand with a measured PSD
approximation error, base-10 logs and f_bend reported as a frequency,
positive lower bounds on period and err_scale.

Resumable: a fit whose output JSON already exists under
<out-dir>/<source>/<model>.json is skipped. Rerun after a crash.

    conda run -n <env> --no-capture-output python scripts/run_realdata.py \
        --data-dir /path/to/AGNobsdata --out-dir results/realdata
"""

from __future__ import annotations

import argparse
import json
import os
import warnings

import pioran_periodicity as pp
from pioran_periodicity import (
    FrequencyBand,
    PriorConfig,
    SamplerSettings,
    build_family,
    psd_approximation_error,
    run_nested,
    save_result,
)
from pioran_periodicity.models import ModelSpec
from pioran_periodicity.priors import Parameter, PriorTransform, Uniform

# Representative OBPL parameter point for the approximation-accuracy check.
REP_POINT = dict(alpha_low=0.5, log10_fbend=0.0, alpha_high=2.5)
MAX_REL_TOL = 0.05

# ---------------------------------------------------------------------------
# Prior configs (fixed per source; documented in output)
# ---------------------------------------------------------------------------
CFG_1302 = PriorConfig(
    log10_variance=(-5.0, 0.5),  # mag^2; data rms ~0.1 mag -> var ~1e-2
    log10_fbend=(-2.0, 1.5),  # yr^-1
    alpha_low=(-0.25, 2.0),
    alpha_high_max=4.0,
    sine_amplitude_scale=0.08,
    period=(0.1, 6.67),
    err_scale=(0.05, 1.5),
)
CFG_1553 = PriorConfig(
    log10_variance=(-4.0, 1.0),
    log10_fbend=(-3.0, 1.0),
    alpha_low=(-0.25, 2.0),
    alpha_high_max=4.0,
    sine_amplitude_scale=0.3,
    period=(0.1, 8.0),
    slope=(-2.0, 2.0),
    intercept=(-2.0, 2.0),
    err_scale=(0.05, 1.5),
)


def _cfg_as_dict(cfg: PriorConfig) -> dict:
    from dataclasses import asdict

    return asdict(cfg)


def choose_n_components(
    band: FrequencyBand, start: int = 20, candidates=(20, 30, 40, 60, 80)
) -> tuple[int, dict]:
    """Smallest n_components whose OBPL approximation max_rel_error <= tol."""
    last = None
    for n in candidates:
        err = psd_approximation_error(
            REP_POINT["alpha_low"],
            REP_POINT["log10_fbend"],
            REP_POINT["alpha_high"],
            band,
            n_components=n,
        )
        last = err
        if err["max_rel_error"] <= MAX_REL_TOL:
            return n, err
    return candidates[-1], last


def make_ah24_spec(base_spec: ModelSpec, name: str) -> ModelSpec:
    """Copy an OBPL family member but replace the conditional alpha_high prior
    with an INDEPENDENT Uniform(2, 4) (alpha_high experiment only).
    """
    new_params = []
    for p in base_spec.prior.parameters:
        if p.name == "alpha_high":
            new_params.append(Parameter("alpha_high", Uniform(2.0, 4.0)))
        else:
            new_params.append(p)
    meta = dict(base_spec.meta)
    meta["alpha_high_prior"] = "Uniform(2.0, 4.0) [independent, experiment]"
    return ModelSpec(
        name=name,
        prior=PriorTransform(new_params),
        loglike=base_spec.loglike,
        meta=meta,
    )


def build_jobs(data_dir):
    """Return list of (source, model_name, spec, seed, extra_meta)."""
    jobs = []

    pg1302_path = os.path.join(data_dir, "graham2015data.csv")
    pg1553_path = os.path.join(data_dir, "PG1553_113_logbase.txt")

    # ---- PG 1302-102 ----
    t, y, yerr = pp.load_pg1302(path=pg1302_path)
    band = FrequencyBand.from_times(t)
    n1302, err1302 = choose_n_components(band)
    print(
        f"[PG1302] band f=[{band.f_min:.3g},{band.f_max:.3g}] "
        f"grid_decades={band.grid_decades:.2f}; OBPL n_components={n1302} "
        f"(max_rel_error={err1302['max_rel_error']:.4f})"
    )

    drw1302 = build_family("drw", CFG_1302, variants=("plain", "sine"))
    obpl1302 = build_family(
        "obpl", CFG_1302, variants=("plain", "sine"), band=band, n_components=n1302
    )

    seed = 101
    for name in ("drw", "drw+sine"):
        jobs.append(("PG1302", name, drw1302[name], seed, {}))
        seed += 1
    for name in ("obpl", "obpl+sine"):
        jobs.append(("PG1302", name, obpl1302[name], seed, {"approx_error": err1302}))
        seed += 1

    # alpha_high experiment (2 extra PG1302 OBPL fits with independent U(2,4))
    obpl_ah24 = make_ah24_spec(obpl1302["obpl"], "obpl_ah24")
    obplsine_ah24 = make_ah24_spec(obpl1302["obpl+sine"], "obpl+sine_ah24")
    jobs.append(
        (
            "PG1302",
            "obpl_ah24",
            obpl_ah24,
            301,
            {"approx_error": err1302, "experiment": "alpha_high~U(2,4)"},
        )
    )
    jobs.append(
        (
            "PG1302",
            "obpl+sine_ah24",
            obplsine_ah24,
            302,
            {"approx_error": err1302, "experiment": "alpha_high~U(2,4)"},
        )
    )

    # ---- PG 1553+113 ----
    t2, y2, e2 = pp.load_pg1553(path=pg1553_path)
    band2 = FrequencyBand.from_times(t2)
    n1553, err1553 = choose_n_components(band2)
    print(
        f"[PG1553] band f=[{band2.f_min:.3g},{band2.f_max:.3g}] "
        f"grid_decades={band2.grid_decades:.2f}; OBPL n_components={n1553} "
        f"(max_rel_error={err1553['max_rel_error']:.4f})"
    )

    variants4 = ("plain", "sine", "linear", "sine+linear")
    drw1553 = build_family("drw", CFG_1553, variants=variants4)
    obpl1553 = build_family(
        "obpl", CFG_1553, variants=variants4, band=band2, n_components=n1553
    )

    seed = 201
    for name in ("drw", "drw+sine", "drw+linear", "drw+sine+linear"):
        jobs.append(("PG1553", name, drw1553[name], seed, {}))
        seed += 1
    for name in ("obpl", "obpl+sine", "obpl+linear", "obpl+sine+linear"):
        jobs.append(("PG1553", name, obpl1553[name], seed, {"approx_error": err1553}))
        seed += 1

    data = {"PG1302": (t, y, yerr), "PG1553": (t2, y2, e2)}
    approx = {
        "PG1302": {
            "n_components": n1302,
            "error": err1302,
            "f_min": band.f_min,
            "f_max": band.f_max,
            "grid_decades": band.grid_decades,
        },
        "PG1553": {
            "n_components": n1553,
            "error": err1553,
            "f_min": band2.f_min,
            "f_max": band2.f_max,
            "grid_decades": band2.grid_decades,
        },
    }
    return jobs, data, approx


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--data-dir",
        required=True,
        help="directory containing graham2015data.csv and " "PG1553_113_logbase.txt",
    )
    ap.add_argument("--out-dir", required=True, help="fit-result output directory")
    args = ap.parse_args()

    warnings.simplefilter("ignore")
    jobs, data, approx = build_jobs(args.data_dir)

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "run_config.json"), "w") as f:
        json.dump(
            {
                "approx": approx,
                "prior_config_pg1302": _cfg_as_dict(CFG_1302),
                "prior_config_pg1553": _cfg_as_dict(CFG_1553),
                "package_version": pp.__version__,
            },
            f,
            indent=2,
        )

    print(f"\n{len(jobs)} fits queued.\n")
    for i, (source, name, spec, seed, extra) in enumerate(jobs, 1):
        out_dir = os.path.join(args.out_dir, source)
        os.makedirs(out_dir, exist_ok=True)
        fname = name.replace("+", "_") + ".json"
        out_path = os.path.join(out_dir, fname)
        if os.path.exists(out_path):
            print(f"[{i}/{len(jobs)}] SKIP {source}/{name} (exists)")
            continue

        t, y, yerr = data[source]
        settings = SamplerSettings(
            min_num_live_points=400, frac_remain=0.01, max_ncalls=1_000_000, seed=seed
        )
        print(
            f"[{i}/{len(jobs)}] RUN  {source}/{name}  seed={seed} "
            f"ndim={spec.prior.ndim}",
            flush=True,
        )
        result = run_nested(spec, t, y, yerr, settings=settings, show_status=False)
        result.meta.update(extra)
        result.meta["source"] = source
        result.meta["seed"] = seed
        save_result(result, out_path)
        hit_cap = result.ncall >= settings.max_ncalls
        print(
            f"        -> logZ={result.logz:.3f}+/-{result.logzerr:.3f} "
            f"ncall={result.ncall} runtime={result.runtime_s:.0f}s"
            f"{'  [HIT max_ncalls]' if hit_cap else ''}",
            flush=True,
        )

    print("\nAll queued fits processed.")


if __name__ == "__main__":
    main()
