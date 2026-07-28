"""Build a scenario config CSV for the DRW-robustness sensitivity study:
crosses a swept PSD high-frequency slope (``highalpha``) with either no true
signal (a null/false-positive-rate scenario) or a small set of period-
specific injected-sine amplitudes (a detection-power scenario), at a fixed
observing cadence. Output is a drop-in ``--config-csv`` for ``run_sim.py``.

Background: this experiment tests how badly the standard DRW noise-model
assumption performs when the true process is a steeper bending power law
(OBPL) -- i.e. how the false-positive rate (null variant) and detection
power (signal variant) degrade as ``highalpha`` moves away from -2 (the
DRW-consistent slope). See ``run_sim.py``'s module docstring for the
``A1``-is-the-true-amplitude-directly convention this relies on.

Amplitude triads (signal variant): detectability against this PSD is
strongly period-dependent, so a single global amplitude can't represent
"low / at detection limit / comfortably above" at every period -- the
default triads below were empirically calibrated per period (a dedicated
calibration batch at each period, DRW-only, ~20 reps/amplitude) rather than
guessed; override with --period-a1 if recalibrating for a different PSD or
cadence.

    # null (no true signal) variant, 6 highalpha x 20 reps = 120 rows
    conda run -n <env> python scripts/make_slope_robustness_csv.py \\
        --variant null --n-per-cell 20 --out null_case.csv

    # signal (detection-power) variant, default period-specific triads,
    # 6 highalpha x 9 (period,A1) cells x 20 reps = 1080 rows
    conda run -n <env> python scripts/make_slope_robustness_csv.py \\
        --variant signal --n-per-cell 20 --first-id 30000 --out signal_case.csv

    # custom amplitude triad, e.g. recalibrated for a different cadence
    conda run -n <env> python scripts/make_slope_robustness_csv.py \\
        --variant signal --period-a1 "2.0:0.05,0.2,0.4" --out custom.csv

    # scale a completed 20-rep pilot up to 100 reps/cell: a non-overlapping
    # extension CSV (--rep-start = the pilot's --n-per-cell, a fresh
    # --first-id past the pilot's ID range, the SAME --seed) that can share
    # the pilot's --lc-dir/--out-dir without touching its rows/IDs
    conda run -n <env> python scripts/make_slope_robustness_csv.py \\
        --variant signal --n-per-cell 80 --rep-start 20 --first-id 31080 \\
        --out signal_case_extra.csv
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

HIGHALPHA_DEFAULT = "-2.0,-2.4,-2.8,-3.2,-3.6,-4.0"

# Period-specific (low, at-limit ~50% detect, comfortably-above ~90%+ detect)
# amplitude triads for the fixed cadence below (NumofWINDOW=20, rms=0.15).
# period=1.25: calibrated directly (a 6-point, n=20/cell DRW-only batch at
#   highalpha=-3.0 found 0% up to A1=0.075, 35% at 0.095, 45% at 0.1125;
#   0.12 interpolates the ~50% point, 0.24 matches the saturating point of
#   the historical validated reference run at this period).
# period=3.75, 7.5: taken directly from that same historical reference run's
#   own (period, A1) grid at highalpha=-3.0 (real grid points where
#   available; the period=7.5 at-limit point, 0.53, interpolates between
#   that grid's 0.495@38% and 0.6225@86%).
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
    noiseSIGMA=0.015,
    NightsperWINDOW=15,
    OBSperiod=6,
    WINDOWwidth=60,
    dataLOSSfrac=0.2,
    NumofWINDOW=20,
)

CSV_COLUMNS = [
    "ID",
    "simSEED",
    "sampleSEED",
    "rms",
    "noiseSIGMA",
    "bendfreq",
    "lowalpha",
    "highalpha",
    "sharpness",
    "period",
    "A1",
    "NightsperWINDOW",
    "OBSperiod",
    "WINDOWwidth",
    "dataLOSSfrac",
    "NumofWINDOW",
]


def parse_period_a1(specs: list[str]) -> dict[float, list[float]]:
    """``["1.25:0.1,0.2", "3.75:0.05,0.3"]`` -> ``{1.25: [0.1, 0.2], ...}``"""
    out = {}
    for spec in specs:
        period_str, a1_str = spec.split(":")
        out[float(period_str)] = [float(v) for v in a1_str.split(",")]
    return out


def build_rows(highalpha, period_a1, n_per_cell, rep_start, first_id, seed, fixed):
    """Generate rows for reps [rep_start, rep_start + n_per_cell) of each
    cell. ``rep_start=0`` builds a fresh pilot; a later call with the same
    ``seed`` and ``rep_start`` set to the pilot's ``n_per_cell`` (and a
    fresh, non-overlapping ``first_id``) builds a non-overlapping extension
    that scales the pilot up -- e.g. pilot ``n_per_cell=20, rep_start=0``,
    then extension ``n_per_cell=80, rep_start=20`` for 20 -> 100 reps/cell.
    Note this does not reproduce a from-scratch 100-rep run's seeds
    bit-for-bit (two sequential rng draws of size 100 land at a different
    stream position than one draw of size 100), but that's fine: the
    extension never touches the pilot's rows/IDs, so it only needs its own
    reps to be reproducible and non-overlapping, which they are.
    """
    rng = np.random.default_rng(seed)
    n_total = rep_start + n_per_cell
    sim_seeds = rng.integers(1, 100_000, size=n_total)[rep_start:]
    sample_seeds = rng.integers(1, 100_000, size=n_total)[rep_start:]

    rows = []
    lc_id = first_id
    for ha in highalpha:
        if period_a1 is None:
            for rep in range(n_per_cell):
                rows.append(
                    dict(
                        ID=lc_id,
                        simSEED=int(sim_seeds[rep]),
                        sampleSEED=int(sample_seeds[rep]),
                        highalpha=ha,
                        period=np.nan,
                        A1=0.0,
                        **fixed,
                    )
                )
                lc_id += 1
        else:
            for period, a1_list in period_a1.items():
                for a1 in a1_list:
                    for rep in range(n_per_cell):
                        rows.append(
                            dict(
                                ID=lc_id,
                                simSEED=int(sim_seeds[rep]),
                                sampleSEED=int(sample_seeds[rep]),
                                highalpha=ha,
                                period=period,
                                A1=a1,
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
    ap.add_argument("--out", required=True, help="output CSV path")
    ap.add_argument(
        "--highalpha",
        default=HIGHALPHA_DEFAULT,
        help=f"comma-separated highalpha sweep (default: {HIGHALPHA_DEFAULT})",
    )
    ap.add_argument(
        "--period-a1",
        action="append",
        default=None,
        help='signal variant only, repeatable: "PERIOD:A1_1,A1_2,..." '
        f"(default: {PERIOD_A1_DEFAULT})",
    )
    ap.add_argument(
        "--n-per-cell", type=int, default=20, help="new reps per cell in THIS CSV"
    )
    ap.add_argument(
        "--rep-start",
        type=int,
        default=0,
        help="0 for a fresh pilot; set to a prior run's --n-per-cell (with a "
        "fresh, non-overlapping --first-id and the SAME --seed) to generate "
        "a non-overlapping extension that scales that pilot up",
    )
    ap.add_argument("--first-id", type=int, default=0)
    ap.add_argument("--seed", type=int, default=20260728)
    for col, default in FIXED_DEFAULTS.items():
        ap.add_argument(f"--{col}", type=type(default), default=default)
    args = ap.parse_args()

    highalpha = [float(v) for v in args.highalpha.split(",")]
    fixed = {col: getattr(args, col) for col in FIXED_DEFAULTS}

    if args.variant == "null":
        period_a1 = None
    else:
        period_a1 = parse_period_a1(args.period_a1 or PERIOD_A1_DEFAULT)

    rows, next_id = build_rows(
        highalpha,
        period_a1,
        args.n_per_cell,
        args.rep_start,
        args.first_id,
        args.seed,
        fixed,
    )
    df = pd.DataFrame(rows)[CSV_COLUMNS]
    df.to_csv(args.out, index=False)

    n_cells = len(highalpha) * (
        1 if period_a1 is None else sum(len(v) for v in period_a1.values())
    )
    print(
        f"wrote {args.out}: {len(df)} rows ({args.variant} variant, "
        f"{len(highalpha)} highalpha x {n_cells // len(highalpha)} "
        f"(period,A1) cells x {args.n_per_cell} reps, "
        f"IDs {args.first_id}-{next_id - 1})"
    )


if __name__ == "__main__":
    main()
