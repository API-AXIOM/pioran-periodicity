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

from pioran_periodicity.simulate import (
    FRACTIONAL_FLUX_TO_MAG,
    MAGNITUDE_UNITS,
    flux_amplitude_to_mag,
)

# Steepest value is -3.5, NOT -4.0. The fitted slope is alpha_high =
# -highalpha and the prior is alpha_high ~ U(alpha_low, 4.0), so a truth of
# 4.0 sits ON the boundary (17% of those fits piled above 3.9; the cell
# recovered 3.76 while every other cell recovered to within 0.1). The 4.0
# bound is itself the basis limit: with the SHO basis at n=20 components --
# what the campaigns used -- the PSD approximation error is 3% at
# alpha_high=4.0 but 35% at 4.5. Truths <= 3.5 stay interior to both the
# prior and the accurate region (1.8% at n=20). See defect MB3.2.
HIGHALPHA_DEFAULT = "-2.0,-2.3,-2.6,-2.9,-3.2,-3.5"

# Same triads as make_slope_robustness_csv.py's PERIOD_A1_DEFAULT -- see that
# file's docstring for provenance, including why they are recorded in
# fractional FLUX (the unit they were calibrated in) and converted once to
# the magnitudes the simulator now emits. Reused here (not recalibrated for
# real cadences) so detection-power comparisons are apples to apples: any
# difference between this campaign and signal_case.csv is attributable to
# the cadence, not to a different amplitude grid.
PERIOD_A1_FLUX = {
    1.25: [0.015, 0.12, 0.24],
    3.75: [0.1125, 0.24, 0.3675],
    7.5: [0.1125, 0.53, 0.75],
}
PERIOD_A1_DEFAULT = [
    f"{period}:" + ",".join(str(flux_amplitude_to_mag(a1)) for a1 in a1_list)
    for period, a1_list in PERIOD_A1_FLUX.items()
]

FIXED_DEFAULTS = dict(
    lowalpha=-1.0,
    bendfreq=0.005479452054794521,
    # Pioran's SingleBendingPowerLaw is P(f) = (f/fb)^-a1 / (1 +
    # (f/fb)^(a2-a1)): sharpness is hard-wired to 1, with no free parameter.
    # Simulating at sharpness=10 put a knee in the data the fitted OBPL
    # cannot represent -- a factor 1.87 (0.27 dex) PSD discrepancy AT the
    # bend frequency, which sits inside the science band. Injection and
    # inference now share one PSD family (defect MB3.5).
    sharpness=1.0,
    # MAGNITUDES (the simulator's unit since 2026-09-04), converted from the
    # fractional-flux 0.15 the campaigns were calibrated at so the physical
    # amplitude is unchanged: 0.1629 mag.
    rms=0.15 * FRACTIONAL_FLUX_TO_MAG,
)

CSV_COLUMNS = [
    # `units` marks rms/A1 as magnitudes; run_sim.py refuses a CSV without
    # it, so a flux-era config cannot be run by mistake.
    "ID", "units", "simSEED", "sampleSEED", "rms", "bendfreq", "lowalpha",
    "highalpha", "sharpness", "period", "A1", "cadence_source", "ref_mag",
]

# Optional multi-band colour column: run_sim.py's band_amp_beta(row) reads
# it if present (falls back to a merged single-band fit if absent), so
# adding it here is the only change needed to produce a --multiband-ready
# scenario CSV.
CSV_COLUMNS_MULTIBAND = CSV_COLUMNS + ["band_amp_beta"]


def parse_period_a1(specs: list[str]) -> dict[float, list[float]]:
    out = {}
    for spec in specs:
        period_str, a1_str = spec.split(":")
        out[float(period_str)] = [float(v) for v in a1_str.split(",")]
    return out


def pick_stratified_objects(
    master_csv: str, n_objects: int, seed: int, max_n_epochs: int | None = None
) -> pd.DataFrame:
    """A fixed, reproducible pool of ``n_objects`` matched targets, spread
    evenly across quartiles of ``n_epochs`` (so sparse- and densely-sampled
    real cadences are both represented, not just whichever happens to sort
    first). Returns columns object_id, rmag, n_epochs, baseline_days.

    ``max_n_epochs`` drops objects above that epoch count BEFORE stratifying,
    so the quartiles are computed within the surviving population rather than
    spending one whole quartile on the excluded tail. For LSST this is how
    you get a Wide-Fast-Deep-only pool: the deep-drilling fields are a
    separate population (47 objects, 690-47950 epochs) and the WFD ceiling is
    887, with NO object anywhere between 887 and 2360 -- so any cut in that
    gap (1000 is the natural one) separates them exactly. Mixing the two
    makes "LSST" a blend of survey strategies differing ~30x in sampling
    density, and lands ~45% of the campaign's cost on a handful of objects.
    """
    master = pd.read_csv(master_csv)
    matched = master[master["matched"]].copy()
    if max_n_epochs is not None:
        n_before = len(matched)
        matched = matched[matched["n_epochs"] <= max_n_epochs]
        print(
            f"--max-n-epochs {max_n_epochs}: kept {len(matched)} of "
            f"{n_before} matched targets"
        )
    if len(matched) < n_objects:
        raise ValueError(
            f"{master_csv} has only {len(matched)} matched targets"
            + (f" at n_epochs <= {max_n_epochs}" if max_n_epochs else "")
            + f", need {n_objects}"
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


def build_rows(highalpha, period_a1, objects, first_id, seed, survey, fixed,
                band_amp_beta=None):
    """Generate rows. ``objects`` (a DataFrame from pick_stratified_objects)
    supplies both the rep count (len(objects)) and, via ``rep`` as the row
    index, WHICH real cadence each rep uses -- identically across every
    (highalpha, period, A1) cell, for direct cell-for-cell comparability
    against the synthetic-cadence campaigns.

    ``band_amp_beta``, if given, is stamped as a constant column on every
    row (see CSV_COLUMNS_MULTIBAND)."""
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
                row = dict(
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
                if band_amp_beta is not None:
                    row["band_amp_beta"] = band_amp_beta
                rows.append(row)
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
        "with the period in YEARS and the amplitudes in MAGNITUDES "
        f"(default: {PERIOD_A1_DEFAULT})",
    )
    ap.add_argument(
        "--n-per-cell", type=int, default=20,
        help="size of the stratified real-object pool = reps per cell "
        "(default 20, matching signal_case.csv/null_case.csv)",
    )
    ap.add_argument("--first-id", type=int, default=0)
    ap.add_argument("--seed", type=int, default=20260728)
    ap.add_argument(
        "--max-n-epochs", type=int, default=None,
        help="drop objects with more than this many epochs from the pool "
        "before stratifying. For LSST use 1000 to get a Wide-Fast-Deep-only "
        "pool: WFD tops out at 887 epochs and the deep-drilling fields start "
        "at 2360, so any cut in that empty gap separates them exactly",
    )
    ap.add_argument(
        "--band-amp-beta", type=float, default=None,
        help="if given, stamp a constant band_amp_beta column (per-band "
        "variability colour index) on every row, producing a "
        "--multiband-ready CSV; omit for a merged single-band CSV",
    )
    for col, default in FIXED_DEFAULTS.items():
        ap.add_argument(
            f"--{col}", type=type(default), default=default,
            # `rms` is in MAGNITUDES; a flux-era --rms 0.15 would still be
            # stamped units="mag" and pass run_sim.py's guard while injecting
            # 8% less variability. See make_slope_robustness_csv.FIXED_HELP.
            help=(
                "process rms, in MAGNITUDES (flux-era 0.15 -> %(default).4f mag)"
                if col == "rms" else "(default: %(default)s)"
            ),
        )
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

    objects = pick_stratified_objects(
        args.master_csv, args.n_per_cell, args.seed, args.max_n_epochs
    )
    missing = set(objects["object_id"]) - set(lib.object_ids(args.survey))
    if missing:
        raise ValueError(
            f"{len(missing)} objects from {args.master_csv} are not in the "
            f"cadence library (e.g. {sorted(missing)[:3]}) -- library may be stale"
        )

    rows, next_id = build_rows(
        highalpha, period_a1, objects, args.first_id, args.seed, args.survey, fixed,
        band_amp_beta=args.band_amp_beta,
    )
    columns = CSV_COLUMNS_MULTIBAND if args.band_amp_beta is not None else CSV_COLUMNS
    df = pd.DataFrame(rows).assign(units=MAGNITUDE_UNITS)[columns]
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
