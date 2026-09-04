"""Build a scenario config CSV for the real-cadence robustness study: the
SAME DRW-robustness grid as ``make_slope_robustness_csv.py`` (highalpha
sweep x period/A1 detection-power triads, or a null/no-signal variant), but
sampled onto REAL ZTF/LSST survey cadences instead of the synthetic
seasonal-window pattern -- checks whether the method's power/FPR
calibration (built entirely on synthetic cadences so far) holds up under
real, irregular sampling. Output is a drop-in ``--config-csv`` for
``run_sim.py`` (with ``--cadence-library`` also required at run time).

Objects are drawn by SIMPLE RANDOM SAMPLING, FRESH IN EVERY CELL.  Both
halves of that matter and both were deliberate reversals of the earlier
design:

* **No stratification.** Balancing the pool across ``n_epochs`` quartiles
  (what this script used to do) makes the sample uniform in the stratifying
  variable, so the false-positive rate it estimates belongs to a synthetic
  re-weighted population that does not exist.  Simple random sampling
  estimates the FPR for "a randomly chosen AGN in this survey", which is
  the quantity the campaign exists to quote.  ``rmag`` and ``n_epochs`` are
  recorded per row and should be analysed as post-hoc COVARIATES instead --
  more informative, and free.
* **Fresh draws per cell.** Reusing one fixed pool across every cell pairs
  the cells, which cancels object-to-object scatter in cell-to-cell
  CONTRASTS -- but which object you draw explains 56-66% of the variance in
  log10 BF (ICC 0.655 ZTF / 0.557 LSST), so a fixed pool over-specifies the
  ABSOLUTE FPR and, decisively, makes the naive binomial SE a CONDITIONAL
  SE that understates population uncertainty.  Fresh draws make that naive
  SE approximately correct for the population.  The accepted cost is ~2.3-2.9x
  more reps for equal CONTRAST power (SE of a cell-to-cell mean difference
  at n=100: ZTF 0.0571 -> 0.0971, LSST 0.1833 -> 0.2755); the absolute FPR is
  the headline, and the staged 20/50/100 design absorbs it.

Rows within a cell are in random order, so any leading slice of a cell is
itself a simple random sample -- and ``--rep-start`` builds a
non-overlapping EXTENSION of an earlier stage (see ``build_rows``).

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
#
# FOUR points, not the earlier six: the Tier-1 design uses the SAME slope
# axis for the single-band and multi-band campaigns so the two modes are
# directly comparable, and the multi-band cost only affords four.
HIGHALPHA_DEFAULT = "-2.0,-2.5,-3.0,-3.5"

# Upper bound (years) of the sine period prior, per cadence -- stamped into
# every row as ``period_max`` so run_sim.py cannot be launched against the
# wrong one.  It is survey-dependent ON PURPOSE, because the surveys have
# different baselines: measured pool baselines are ZTF 6.07/7.40/7.59 yr
# (min/median/max) and LSST WFD 9.20/9.89/9.98, so a period near 7.5 yr is
# ~1.0 cycle or fewer for over half the ZTF pool.  The synthetic cadence
# (make_slope_robustness_csv.py, 9.53 yr baseline) shares LSST's 9.0.
#
# CONSEQUENCE, accepted deliberately: a LogUniform period prior contributes
# an Occam factor of log(hi/lo), so ZTF and LSST Bayes factors do NOT sit on
# a common prior and their FPRs are not a like-for-like comparison of the
# METHOD.  They are different surveys and that difference is real; what must
# stay matched is a signal campaign and the null campaign that calibrates it
# (MB3.1), which share a survey and therefore share this value.
P_MAX_BY_SURVEY = {"ztf": 6.0, "lsst": 9.0}

# LSST WFD-only screen.  The deep-drilling fields are a separate population
# (47 objects, 690-47950 epochs) while WFD tops out at 887 and the DDFs
# start at 2360, so any cut in that empty gap separates them exactly.
# Leaving them in makes "LSST" a blend of two survey strategies differing
# ~30x in sampling density and lands ~45% of the campaign cost on a handful
# of objects -- one of them XMMC_149.82382+2.22786, 47,950 epochs, which
# crashed twice at ~135 h CPU each.  ZTF needs no screen (master max 5014).
MAX_N_EPOCHS_BY_SURVEY = {"ztf": None, "lsst": 1000}

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
    # `period_max` is the sine period prior's upper bound in years; run_sim.py
    # reads it so the prior travels WITH the scenario instead of depending on
    # a launch flag being remembered (the MB3.1 failure mode).
    "highalpha", "sharpness", "period", "A1", "cadence_source", "ref_mag",
    "period_max", "n_epochs", "dec",
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


DAYS_PER_YEAR = 365.25


def eligible_objects(
    master_csv: str,
    max_n_epochs: int | None = None,
    min_baseline_years: float | None = None,
) -> pd.DataFrame:
    """The POPULATION a campaign samples from: every matched target in
    ``master_csv``, screened to ``n_epochs <= max_n_epochs`` and to
    ``baseline_days >= min_baseline_years``.

    No stratification and no sub-selection beyond those screens -- the draw
    itself happens per cell in ``build_rows``.  Returns columns object_id,
    rmag, n_epochs, baseline_days, dec.

    ``min_baseline_years`` defaults (in ``main``) to the scenario's
    ``period_max``, which keeps the campaign honest about its own design
    rule: P_max is chosen at the pool's SHORTEST baseline so that every
    fit's period prior is supported by that object's own data.  That rule
    used to be enforced by hand, when the pool was a pre-filtered list of
    100; drawing at random from the whole master file quietly reintroduced
    objects the prior overruns (ZTF baselines reach down to 3.81 yr against
    a 6.0 yr P_max -- 1 object of 443; LSST WFD is unaffected, its minimum
    being 9.12 yr against 9.0).  Small exposure, but it is the difference
    between a stated design rule and an enforced one.
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
    if min_baseline_years is not None:
        n_before = len(matched)
        matched = matched[
            matched["baseline_days"] >= min_baseline_years * DAYS_PER_YEAR
        ]
        if len(matched) < n_before:
            print(
                f"min baseline {min_baseline_years} yr: dropped "
                f"{n_before - len(matched)} of {n_before} objects whose "
                f"baseline is shorter than the period prior"
            )
    if matched.empty:
        raise ValueError(f"{master_csv} has no matched targets to draw from")
    cols = ["object_id", "rmag", "n_epochs", "baseline_days", "dec"]
    return matched[cols].reset_index(drop=True)


def build_rows(highalpha, period_a1, population, n_per_cell, first_id, seed,
               survey, fixed, period_max, rep_start=0, band_amp_beta=None):
    """Generate rows, drawing ``n_per_cell`` objects INDEPENDENTLY IN EVERY
    (highalpha, period, A1) cell by simple random sampling without
    replacement -- see the module docstring for why the old fixed stratified
    pool was abandoned.

    Each cell gets its own RNG stream, spawned from ``seed`` via
    ``SeedSequence``, and draws a full permutation of the population which is
    then sliced ``[rep_start : rep_start + n_per_cell]``.  Two consequences,
    both wanted:

    * Rows within a cell are in random order, so a leading slice of a cell is
      itself a simple random sample of the population (this is what makes a
      20-rep stage-1 subset of a 100-rep CSV legitimate).
    * ``rep_start`` builds a non-overlapping EXTENSION of an earlier stage:
      pilot ``n_per_cell=20, rep_start=0``, then ``n_per_cell=80,
      rep_start=20`` with the SAME ``seed`` and a fresh ``--first-id`` scales
      20 -> 100 reps/cell without re-drawing -- or re-fitting -- any object the
      pilot already used, and without any object appearing twice in a cell.
      This mirrors ``make_slope_robustness_csv.build_rows``.
    """
    n_total = rep_start + n_per_cell
    if n_total > len(population):
        raise ValueError(
            f"need {n_total} distinct objects per cell (rep_start "
            f"{rep_start} + n_per_cell {n_per_cell}) but the screened "
            f"population has only {len(population)}"
        )

    if period_a1 is None:
        cells = [(np.nan, 0.0)]
    else:
        cells = [(p, a1) for p, a1_list in period_a1.items() for a1 in a1_list]
    grid = [(ha, period, a1) for ha in highalpha for period, a1 in cells]

    # One independent, reproducible stream per cell. Spawning (rather than
    # one shared rng consumed in grid order) keeps a cell's draw identical
    # no matter how many other cells the CSV happens to contain, so adding a
    # slope to the sweep does not silently redraw the existing ones.
    streams = np.random.SeedSequence(seed).spawn(len(grid))

    rows = []
    lc_id = first_id
    for (ha, period, a1), stream in zip(grid, streams):
        rng = np.random.default_rng(stream)
        picked = rng.permutation(len(population))[rep_start:n_total]
        sim_seeds = rng.integers(1, 100_000, size=n_total)[rep_start:]
        sample_seeds = rng.integers(1, 100_000, size=n_total)[rep_start:]
        for rep, idx in enumerate(picked):
            obj = population.iloc[idx]
            row = dict(
                ID=lc_id,
                simSEED=int(sim_seeds[rep]),
                sampleSEED=int(sample_seeds[rep]),
                highalpha=ha,
                period=period,
                A1=a1,
                cadence_source=f"{survey}:{obj['object_id']}",
                ref_mag=float(obj["rmag"]),
                # Recorded so the FPR can be analysed against sampling
                # density post hoc -- the covariate route that replaces
                # stratifying on it.
                n_epochs=int(obj["n_epochs"]),
                # `dec` is recorded because the cadence LIBRARY itself was
                # built by declination-stratified sampling of the parent
                # quasar catalogue (tutorials/Get_Quasar_Cadences.ipynb), so
                # the population we now sample simply-at-random is DEC
                # BALANCED, not a random sample of the survey footprint. To
                # the extent FPR varies with declination -- plausible via
                # airmass, depth and visit count -- an FPR quoted as "a
                # randomly chosen AGN in this survey" inherits that
                # imbalance. Recording dec makes the size of the effect
                # measurable after the fact instead of merely arguable.
                dec=float(obj["dec"]),
                period_max=float(period_max),
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
        help="the survey's master.csv (ztf_data/ or lsst_data/); supplies "
        "the population of objects to sample, plus rmag and n_epochs",
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
        help="reps per cell -- objects are drawn fresh in each cell, so this "
        "is a sample size, not a pool size (default 20, matching "
        "signal_case.csv/null_case.csv)",
    )
    ap.add_argument(
        "--rep-start", type=int, default=0,
        help="build a non-overlapping EXTENSION of an earlier stage instead "
        "of a fresh campaign: reuse that stage's --seed, set this to its "
        "--n-per-cell, and give a fresh --first-id (e.g. 20 -> 100 reps/cell "
        "is --n-per-cell 80 --rep-start 20)",
    )
    ap.add_argument("--first-id", type=int, default=0)
    ap.add_argument("--seed", type=int, default=20260728)
    ap.add_argument(
        "--max-n-epochs", type=int, default=None,
        help="drop objects with more than this many epochs from the sampled "
        "population. Defaults per survey to "
        f"{MAX_N_EPOCHS_BY_SURVEY} -- 1000 gives a Wide-Fast-Deep-only LSST "
        "population (WFD tops out at 887 epochs, the deep-drilling fields "
        "start at 2360). Pass 0 to disable the screen entirely",
    )
    ap.add_argument(
        "--min-baseline-years", type=float, default=None,
        help="drop objects whose baseline is shorter than this. Defaults to "
        "the scenario's period_max, so no fit gets a period prior its own "
        "data cannot support. Pass 0 to disable",
    )
    ap.add_argument(
        "--period-max", type=float, default=None,
        help="upper bound (years) of the sine period prior, stamped into "
        f"every row. Defaults per survey to {P_MAX_BY_SURVEY}; a signal "
        "campaign and the null campaign that calibrates it MUST share it",
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
    # time building the population.
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

    max_n_epochs = args.max_n_epochs
    if max_n_epochs is None:
        max_n_epochs = MAX_N_EPOCHS_BY_SURVEY[args.survey]
    elif max_n_epochs == 0:
        max_n_epochs = None
    period_max = args.period_max
    if period_max is None:
        period_max = P_MAX_BY_SURVEY[args.survey]

    population = eligible_objects(
        args.master_csv, max_n_epochs,
        min_baseline_years=(
            period_max if args.min_baseline_years is None
            else (args.min_baseline_years or None)
        ),
    )
    # Screen against the cadence library BEFORE drawing, so a stale library
    # shrinks the population rather than failing the run at fit time.
    known = set(lib.object_ids(args.survey))
    n_before = len(population)
    population = population[population["object_id"].isin(known)]
    if len(population) < n_before:
        print(
            f"dropped {n_before - len(population)} of {n_before} objects "
            f"absent from the cadence library"
        )
    if population.empty:
        raise ValueError(
            f"no object in {args.master_csv} is present in the "
            f"{args.survey} cadence library -- library may be stale"
        )
    population = population.reset_index(drop=True)

    rows, next_id = build_rows(
        highalpha, period_a1, population, args.n_per_cell, args.first_id,
        args.seed, args.survey, fixed, period_max,
        rep_start=args.rep_start, band_amp_beta=args.band_amp_beta,
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
        f"(period,A1) cells x {args.n_per_cell} reps drawn fresh per cell, "
        f"IDs {args.first_id}-{next_id - 1}, period_max {period_max} yr)"
    )
    used = df["cadence_source"].nunique()
    print(
        f"sampled population: {len(population)} objects, n_epochs range "
        f"{population['n_epochs'].min()}-{population['n_epochs'].max()}, "
        f"baseline_days range {population['baseline_days'].min():.0f}-"
        f"{population['baseline_days'].max():.0f}; "
        f"{used} distinct objects used across {n_cells} cells"
    )


if __name__ == "__main__":
    main()
