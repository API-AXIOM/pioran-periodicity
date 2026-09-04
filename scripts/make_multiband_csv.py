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
flux in every band -- what Phase 4 already probed).

NOTE beta=0 is NOT the same as omitting the column: it still records band
identities and centres on the reference band, so it is a genuine multi-band
null rather than the legacy single-band path.

**Paired design**: the same real cadence object and the same simulation
seeds are reused at every (beta, highalpha, A1) cell, exactly as
``make_real_cadence_csv.py`` does. Object-to-object scatter therefore
cancels when comparing across beta -- which is what makes 50 sims per cell
informative about a colour effect.

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

from pioran_periodicity.simulate import FRACTIONAL_FLUX_TO_MAG, MAGNITUDE_UNITS

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_real_cadence_csv import (  # noqa: E402
    FIXED_DEFAULTS,
    pick_stratified_objects,
)

# Campaign axes (2026-08-18 design discussion).
BETAS = (0.0, 0.35, 0.7)
# -3.5 rather than -4.0: a fitted alpha_high of 4.0 sits exactly on the
# prior bound, which is itself the SHO/n=20 basis-accuracy limit (MB3.2).
NULL_HIGHALPHA = (-2.0, -3.5)
SIGNAL_HIGHALPHA = (-3.5,)
# Recorded in the fractional FLUX they were calibrated in, converted once to
# the magnitudes the simulator now emits -- see make_slope_robustness_csv.py.
SIGNAL_A1_FLUX = (0.24, 0.3675, 0.53)
SIGNAL_A1 = tuple(a1 * FRACTIONAL_FLUX_TO_MAG for a1 in SIGNAL_A1_FLUX)
SIGNAL_PERIOD = 3.75  # yr; well inside the prior and the best-mapped axis

CSV_COLUMNS = [
    # `units` marks rms/A1 as magnitudes; run_sim.py refuses a CSV without it.
    "ID", "units", "simSEED", "sampleSEED", "rms", "bendfreq", "lowalpha",
    "highalpha", "sharpness", "period", "A1", "cadence_source", "ref_mag",
    "band_amp_beta", "block",
]


def build_rows(objects, survey, first_id, seed):
    """Rows for every (block, beta, highalpha, A1) cell x every object.

    ``objects`` (from pick_stratified_objects) fixes both the reps per cell
    and WHICH cadence each rep uses; the seeds are drawn once per rep and
    reused across all cells, so cells differ only in the campaign axes.
    """
    n_per_cell = len(objects)
    rng = np.random.default_rng(seed)
    sim_seeds = rng.integers(1, 100_000, size=n_per_cell)
    sample_seeds = rng.integers(1, 100_000, size=n_per_cell)

    cells = []
    for beta in BETAS:
        for ha in NULL_HIGHALPHA:
            cells.append(("null", beta, ha, np.nan, 0.0))
    for beta in BETAS:
        for ha in SIGNAL_HIGHALPHA:
            for a1 in SIGNAL_A1:
                cells.append(("signal", beta, ha, SIGNAL_PERIOD, a1))

    rows = []
    lc_id = first_id
    for block, beta, ha, period, a1 in cells:
        for rep, obj in objects.iterrows():
            rows.append(dict(
                ID=lc_id,
                simSEED=int(sim_seeds[rep]),
                sampleSEED=int(sample_seeds[rep]),
                highalpha=ha,
                period=period,
                A1=a1,
                cadence_source=f"{survey}:{obj['object_id']}",
                ref_mag=float(obj["rmag"]),
                band_amp_beta=beta,
                block=block,
                **FIXED_DEFAULTS,
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
    args = ap.parse_args()

    id_start = args.id_start
    if id_start is None:
        id_start = 70000 if args.survey == "ztf" else 80000

    objects = pick_stratified_objects(
        os.path.expanduser(args.master_csv), args.n_sims, args.seed
    )
    rows = build_rows(objects, args.survey, id_start, args.seed)
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
    print(f"  cadence objects : {objects['object_id'].nunique()} (paired across cells)")


if __name__ == "__main__":
    main()
