"""Build a scenario config CSV for the real-cadence robustness study: the
SAME DRW-robustness grid as ``make_slope_robustness_csv.py`` (highalpha
sweep x period/A1 detection-power triads, or a null/no-signal variant), but
sampled onto REAL ZTF/LSST survey cadences instead of the synthetic
seasonal-window pattern -- checks whether the method's power/FPR
calibration (built entirely on synthetic cadences so far) holds up under
real, irregular sampling. Output is a drop-in ``--config-csv`` for
``run_sim.py`` (with ``--cadence-library`` also required at run time).

Reps are tied to a fixed, stratified pool of real objects (quartiles of
each object's ``n_epochs``, so sparse- and densely-sampled objects are both
represented), reused IDENTICALLY across every (highalpha, period, A1) cell:
rep r always uses the same real object's cadence, so the same
``--n-per-cell`` real cadences run through the entire grid. This makes the
output directly comparable, cell for cell, against ``signal_case.csv`` /
``null_case.csv`` (same highalpha sweep, same period/A1 triads, same
n_per_cell=20 default) -- the only thing that differs per row is
``cadence_source``/``ref_mag`` replacing the five synthetic-window columns.

``HIGHALPHA_DEFAULT`` and ``PERIOD_A1_DEFAULT`` are intentionally duplicated
from ``make_slope_robustness_csv.py`` rather than imported (small, stable
config; avoids a fragile sibling-script import that only works when scripts/
happens to be on sys.path).

    # null variant, 6 highalpha x 20 reps = 120 rows
    conda run -n <env> python scripts/make_real_cadence_csv.py \\
        --variant null --survey ztf \\
        --cadence-library ~/work/data/quasar_cadences/cadence_library \\
        --master-csv ~/work/data/quasar_cadences/ztf_data/master.csv \\
        --first-id 40000 --out ztf_real_cadence_null_case.csv

    # signal variant, 6 highalpha x 9 (period,A1) cells x 20 reps = 1080 rows
    conda run -n <env> python scripts/make_real_cadence_csv.py \\
        --variant signal --survey ztf \\
        --cadence-library ~/work/data/quasar_cadences/cadence_library \\
        --master-csv ~/work/data/quasar_cadences/ztf_data/master.csv \\
        --first-id 41000 --out ztf_real_cadence_signal_case.csv
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

HIGHALPHA_DEFAULT = "-2.0,-2.4,-2.8,-3.2,-3.6,-4.0"

# Same triads as make_slope_robustness_csv.py's PERIOD_A1_DEFAULT -- see that
# file's docstring for provenance. Reused here (not recalibrated for real
# cadences) so detection-power comparisons are apples to apples: any
# difference between this campaign and signal_case.csv is attributable to
# the cadence, not to a different amplitude grid.
PERIOD_A1_DEFAULT = [
    "1.25:0.015,0.12,0.24",
    "3.75:0.1125,0.24,0.3675",
    "7.5:0.1125,0.53,0.75",
]

FIXED_DEFAULTS = dict(
    lowalpha=-1.0,
    bendfreq=0.005479452054794521,
    sharpness=10.0,
    rms=0.15,
)

CSV_COLUMNS = [
    "ID", "simSEED", "sampleSEED", "rms", "bendfreq", "lowalpha", "highalpha",
    "sharpness", "period", "A1", "cadence_source", "ref_mag",
]


def parse_period_a1(specs: list[str]) -> dict[float, list[float]]:
    out = {}
    for spec in specs:
        period_str, a1_str = spec.split(":")
        out[float(period_str)] = [float(v) for v in a1_str.split(",")]
    return out


def pick_stratified_objects(master_csv: str, n_objects: int, seed: int) -> pd.DataFrame:
    """A fixed, reproducible pool of ``n_objects`` matched targets, spread
    evenly across quartiles of ``n_epochs`` (so sparse- and densely-sampled
    real cadences are both represented, not just whichever happens to sort
    first). Returns columns object_id, rmag, n_epochs, baseline_days."""
    master = pd.read_csv(master_csv)
    matched = master[master["matched"]].copy()
    if len(matched) < n_objects:
        raise ValueError(
            f"{master_csv} has only {len(matched)} matched targets, "
            f"need {n_objects}"
        )

    matched["_quartile"] = pd.qcut(matched["n_epochs"], 4, labels=False, duplicates="drop")
    quartiles = sorted(matched["_quartile"].unique())
    base, extra = divmod(n_objects, len(quartiles))

    rng = np.random.default_rng(seed)
    picked_ids = []
    for i, q in enumerate(quartiles):
        pool = matched.index[matched["_quartile"] == q]
        k = min(base + (1 if i < extra else 0), len(pool))
        picked_ids.extend(rng.choice(pool, size=k, replace=False))

    cols = ["object_id", "rmag", "n_epochs", "baseline_days"]
    return matched.loc[picked_ids, cols].reset_index(drop=True)


def build_rows(highalpha, period_a1, objects, first_id, seed, survey, fixed):
    """Generate rows. ``objects`` (a DataFrame from pick_stratified_objects)
    supplies both the rep count (len(objects)) and, via ``rep`` as the row
    index, WHICH real cadence each rep uses -- identically across every
    (highalpha, period, A1) cell, for direct cell-for-cell comparability
    against the synthetic-cadence campaigns."""
    n_per_cell = len(objects)
    rng = np.random.default_rng(seed)
    sim_seeds = rng.integers(1, 100_000, size=n_per_cell)
    sample_seeds = rng.integers(1, 100_000, size=n_per_cell)

    if period_a1 is None:
        cells = [(np.nan, 0.0)]
    else:
        cells = [(p, a1) for p, a1_list in period_a1.items() for a1 in a1_list]

    rows = []
    lc_id = first_id
    for ha in highalpha:
        for period, a1 in cells:
            for rep, obj in objects.iterrows():
                rows.append(
                    dict(
                        ID=lc_id,
                        simSEED=int(sim_seeds[rep]),
                        sampleSEED=int(sample_seeds[rep]),
                        highalpha=ha,
                        period=period,
                        A1=a1,
                        cadence_source=f"{survey}:{obj['object_id']}",
                        ref_mag=float(obj["rmag"]),
                        **fixed,
                    )
                )
                lc_id += 1
    return rows, lc_id


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--variant", required=True, choices=["null", "signal"])
    ap.add_argument("--survey", required=True, choices=["ztf", "lsst"])
    ap.add_argument(
        "--cadence-library", required=True,
        help="a CadenceLibrary.to_cache() directory (must contain this survey)",
    )
    ap.add_argument(
        "--master-csv", required=True,
        help="the survey's master.csv (ztf_data/ or lsst_data/), for rmag + "
        "n_epochs-quartile stratification",
    )
    ap.add_argument("--out", required=True, help="output CSV path")
    ap.add_argument(
        "--highalpha", default=HIGHALPHA_DEFAULT,
        help=f"comma-separated highalpha sweep (default: {HIGHALPHA_DEFAULT})",
    )
    ap.add_argument(
        "--period-a1", action="append", default=None,
        help='signal variant only, repeatable: "PERIOD:A1_1,A1_2,..." '
        f"(default: {PERIOD_A1_DEFAULT})",
    )
    ap.add_argument(
        "--n-per-cell", type=int, default=20,
        help="size of the stratified real-object pool = reps per cell "
        "(default 20, matching signal_case.csv/null_case.csv)",
    )
    ap.add_argument("--first-id", type=int, default=0)
    ap.add_argument("--seed", type=int, default=20260728)
    for col, default in FIXED_DEFAULTS.items():
        ap.add_argument(f"--{col}", type=type(default), default=default)
    args = ap.parse_args()

    # Fail loud: the library must actually have this survey before spending
    # time on stratification.
    from pioran_periodicity.cadence import CadenceLibrary

    lib = CadenceLibrary.from_cache(args.cadence_library)
    if args.survey not in lib.surveys():
        raise ValueError(
            f"{args.cadence_library} has surveys {lib.surveys()}, "
            f"not {args.survey!r}"
        )

    highalpha = [float(v) for v in args.highalpha.split(",")]
    fixed = {col: getattr(args, col) for col in FIXED_DEFAULTS}
    period_a1 = None if args.variant == "null" else parse_period_a1(
        args.period_a1 or PERIOD_A1_DEFAULT
    )

    objects = pick_stratified_objects(args.master_csv, args.n_per_cell, args.seed)
    missing = set(objects["object_id"]) - set(lib.object_ids(args.survey))
    if missing:
        raise ValueError(
            f"{len(missing)} objects from {args.master_csv} are not in the "
            f"cadence library (e.g. {sorted(missing)[:3]}) -- library may be stale"
        )

    rows, next_id = build_rows(
        highalpha, period_a1, objects, args.first_id, args.seed, args.survey, fixed
    )
    df = pd.DataFrame(rows)[CSV_COLUMNS]
    df.to_csv(args.out, index=False)

    n_cells = len(highalpha) * (
        1 if period_a1 is None else sum(len(v) for v in period_a1.values())
    )
    print(
        f"wrote {args.out}: {len(df)} rows ({args.variant} variant, "
        f"{len(highalpha)} highalpha x {n_cells // len(highalpha)} "
        f"(period,A1) cells x {len(objects)} reps, IDs {args.first_id}-{next_id - 1})"
    )
    print(
        f"object pool: n_epochs range {objects['n_epochs'].min()}-"
        f"{objects['n_epochs'].max()}, baseline_days range "
        f"{objects['baseline_days'].min():.0f}-{objects['baseline_days'].max():.0f}"
    )


if __name__ == "__main__":
    main()
