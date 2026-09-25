"""Build scenario CSVs for the multi-band colour-dependence campaign.

Two blocks per survey, both fit with the shared-latent-process multi-band
model (``run_sim.py --multiband``):

* NULL (``A1 = 0``)  -> false-positive rate vs the colour index
  ``band_amp_beta`` and the red-noise slope ``highalpha``.
* SIGNAL (``A1 > 0``) -> detection power vs beta, at amplitudes bracketing
  the 50% crossing measured in the existing merged campaigns (period
  3.75 yr: A1=0.24 gives ~0.06 power, A1=0.3675 gives 0.31-0.85 depending
  on slope, so 0.24/0.3675/0.53 straddles it).

``band_amp_beta`` sets per-band variability amplitude
``a_b = (lambda_b/lambda_ref)^(-beta)``, grounded in real quasar
structure-function colour trends: beta=0.35 gives an LSST u/y amplitude
contrast of 1.41, beta=0.7 gives 1.97. beta=0 is the control (identical
variability in every band -- what Phase 4 already probed).

NOTE beta=0 is NOT the same as omitting the column: it still records band
identities and centres on the reference band, so it is a genuine multi-band
null rather than the legacy single-band path.

**Sampling**: objects are drawn by simple random sampling, FRESH IN EVERY
(block, beta, highalpha, A1) cell, exactly as ``make_real_cadence_csv.py``
does -- see that module's docstring for the full argument.

This REPLACES an explicitly paired design (one fixed stratified pool reused
at every cell, so object-to-object scatter cancelled in beta contrasts).
That pairing is genuinely lost, and it was the reason 50 sims per cell was
argued to be informative about a colour effect: the SE of a cell-to-cell
difference rises ~1.5-1.7x, i.e. ~2.3-2.9x more reps for equal contrast
power. It is given up on purpose, because the campaign's headline output is
the ABSOLUTE false-positive rate and a fixed pool makes that number, and
especially its error bar, conditional on one draw of objects. Read beta
contrasts here as needing the staged rep counts, not 50.

    conda run -n <env> python scripts/make_multiband_csv.py \
        --survey lsst --master-csv <...>/lsst_data/master.csv \
        --out-dir <dir> --n-sims 50
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

from pioran_periodicity.simulate import MAGNITUDE_UNITS, flux_amplitude_to_mag

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_real_cadence_csv import (  # noqa: E402
    FIXED_DEFAULTS,
    MAX_N_EPOCHS_BY_SURVEY,
    P_MAX_BY_SURVEY,
    eligible_objects,
)

# Campaign axes (2026-08-18 design discussion).
# Fixed at the LSST community value (2026-09-08): multi-band AGN simulations
# for Rubin use sigma_DRW ~ lambda^-0.479 at the filter effective wavelengths
# (MacLeod et al. 2010), so this matches other LSST forecasts. Our own
# within-object measurement on 401 paired ZTF g/r light curves gives
# beta = 0.718 [0.637, 0.805]; the two differ because MacLeod's is a
# cross-object regression while ours is within-object. Kept as a single value,
# not a grid: the intrinsic object-to-object spread in beta is unresolved
# (observed sd 1.01 vs 1.47 expected from estimator noise alone).
# A {0, 0.479, 0.70} sensitivity arm is deferred to after the baseline runs.
BETAS = (0.479,)
# -3.5 rather than -4.0: a fitted alpha_high of 4.0 sits exactly on the
# prior bound, which is itself the SHO/n=20 basis-accuracy limit (MB3.2).
# The SAME four-point axis as the single-band campaigns
# (make_real_cadence_csv.HIGHALPHA_DEFAULT), so single-band and multi-band
# FPRs are directly comparable slope for slope -- the earlier design used
# two slopes here against six there.
NULL_HIGHALPHA = (-2.0, -2.5, -3.0, -3.5)
SIGNAL_HIGHALPHA = (-3.5,)
# Recorded in the fractional FLUX they were calibrated in, converted once to
# the magnitudes the simulator now emits -- see make_slope_robustness_csv.py.
SIGNAL_A1_FLUX = (0.24, 0.3675, 0.53)
SIGNAL_A1 = tuple(flux_amplitude_to_mag(a1) for a1 in SIGNAL_A1_FLUX)
SIGNAL_PERIOD = 3.75  # yr; well inside the prior and the best-mapped axis

CSV_COLUMNS = [
    # `units` marks rms/A1 as magnitudes; run_sim.py refuses a CSV without it.
    "ID", "units", "simSEED", "sampleSEED", "rms", "bendfreq", "lowalpha",
    "highalpha", "sharpness", "period", "A1", "cadence_source", "ref_mag",
    "period_max", "n_epochs", "dec", "band_amp_beta", "block",
]


def build_rows(population, survey, first_id, seed, n_per_cell, period_max,
               rep_start=0, bendfreq=None):
    """Rows for every (block, beta, highalpha, A1) cell, drawing
    ``n_per_cell`` objects independently in each one.

    Same mechanics as ``make_real_cadence_csv.build_rows``: one spawned RNG
    stream per cell, a permutation sliced ``[rep_start : rep_start +
    n_per_cell]`` so ``rep_start`` yields a non-overlapping extension of an
    earlier stage, and rows in random order within a cell.
    """
    cells = []
    for beta in BETAS:
        for ha in NULL_HIGHALPHA:
            cells.append(("null", beta, ha, np.nan, 0.0))
    for beta in BETAS:
        for ha in SIGNAL_HIGHALPHA:
            for a1 in SIGNAL_A1:
                cells.append(("signal", beta, ha, SIGNAL_PERIOD, a1))

    n_total = rep_start + n_per_cell
    if n_total > len(population):
        raise ValueError(
            f"need {n_total} distinct objects per cell but the screened "
            f"population has only {len(population)}"
        )
    streams = np.random.SeedSequence(seed).spawn(len(cells))

    rows = []
    lc_id = first_id
    for (block, beta, ha, period, a1), stream in zip(cells, streams):
        rng = np.random.default_rng(stream)
        picked = rng.permutation(len(population))[rep_start:n_total]
        sim_seeds = rng.integers(1, 100_000, size=n_total)[rep_start:]
        sample_seeds = rng.integers(1, 100_000, size=n_total)[rep_start:]
        for rep, idx in enumerate(picked):
            obj = population.iloc[idx]
            rows.append(dict(
                ID=lc_id,
                simSEED=int(sim_seeds[rep]),
                sampleSEED=int(sample_seeds[rep]),
                highalpha=ha,
                period=period,
                A1=a1,
                cadence_source=f"{survey}:{obj['object_id']}",
                ref_mag=float(obj["rmag"]),
                n_epochs=int(obj["n_epochs"]),
                dec=float(obj["dec"]),  # see make_real_cadence_csv
                period_max=float(period_max),
                band_amp_beta=beta,
                block=block,
                **{**FIXED_DEFAULTS,
                   **({} if bendfreq is None else {"bendfreq": bendfreq})},
            ))
            lc_id += 1
    return rows


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--survey", required=True, choices=("ztf", "lsst"))
    ap.add_argument("--master-csv", required=True,
                    help="survey master.csv (supplies object_id, rmag, n_epochs)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-sims", type=int, default=50,
                    help="light curves per cell (50 -> FPR +/-3.1%% at p=0.05)")
    ap.add_argument("--id-start", type=int, default=None,
                    help="first lc_id; default 70000 (ztf) / 80000 (lsst), "
                         "chosen to avoid colliding with existing campaign ids")
    ap.add_argument("--seed", type=int, default=20260818)
    ap.add_argument("--rep-start", type=int, default=0,
                    help="build a non-overlapping EXTENSION of an earlier "
                         "stage; reuse that stage's --seed and --n-sims and "
                         "give a fresh --id-start")
    ap.add_argument("--max-n-epochs", type=int, default=None,
                    help="screen the sampled population; defaults per survey "
                         f"to {MAX_N_EPOCHS_BY_SURVEY}. Pass 0 to disable")
    ap.add_argument("--min-baseline-years", type=float, default=None,
                    help="drop objects with a baseline shorter than this; "
                         "defaults to the scenario's period_max. 0 disables")
    ap.add_argument("--bendfreq", type=float, default=None,
                    help="PSD bend frequency in 1/DAY, stamped on every row; "
                         "defaults to make_real_cadence_csv.FIXED_DEFAULTS "
                         f"({FIXED_DEFAULTS['bendfreq']:.9f} = 0.35/yr)")
    ap.add_argument("--period-max", type=float, default=None,
                    help="sine period prior upper bound (yr), stamped on "
                         f"every row; defaults per survey to {P_MAX_BY_SURVEY}")
    args = ap.parse_args()

    id_start = args.id_start
    if id_start is None:
        id_start = 70000 if args.survey == "ztf" else 80000

    max_n_epochs = args.max_n_epochs
    if max_n_epochs is None:
        max_n_epochs = MAX_N_EPOCHS_BY_SURVEY[args.survey]
    elif max_n_epochs == 0:
        max_n_epochs = None
    period_max = args.period_max
    if period_max is None:
        period_max = P_MAX_BY_SURVEY[args.survey]

    population = eligible_objects(
        os.path.expanduser(args.master_csv), max_n_epochs,
        min_baseline_years=(
            period_max if args.min_baseline_years is None
            else (args.min_baseline_years or None)
        ),
    )
    rows = build_rows(
        population, args.survey, id_start, args.seed, args.n_sims,
        period_max, rep_start=args.rep_start, bendfreq=args.bendfreq,
    )
    df = pd.DataFrame(rows).assign(units=MAGNITUDE_UNITS)[CSV_COLUMNS]

    out_dir = os.path.expanduser(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"{args.survey}_multiband_campaign.csv")
    df.to_csv(out, index=False)

    n_null = int((df["block"] == "null").sum())
    n_sig = int((df["block"] == "signal").sum())
    print(f"wrote {out}")
    print(f"  {len(df)} light curves: {n_null} null + {n_sig} signal")
    print(f"  {len(df) // args.n_sims} cells x {args.n_sims} sims")
    print(f"  beta            : {sorted(df['band_amp_beta'].unique())}")
    print(f"  null highalpha  : {sorted(df[df.block == 'null'].highalpha.unique())}")
    print(f"  signal A1       : {sorted(df[df.block == 'signal'].A1.unique())}")
    print(f"  ids             : {df.ID.min()}-{df.ID.max()}")
    print(f"  population      : {len(population)} objects "
          f"(period_max {period_max} yr, max_n_epochs {max_n_epochs})")
    print(f"  cadence objects : {df['cadence_source'].nunique()} distinct, "
          f"drawn fresh per cell")


if __name__ == "__main__":
    main()
