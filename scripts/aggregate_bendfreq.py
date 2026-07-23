"""Bend-frequency estimation aggregation.

Summarises the posterior of log_bend_freq (= log10 f_bend, f_bend in yr^-1)
across a set of simulated light curves via the "average histogram" method:
resample one posterior value per light curve, histogram, repeat N times,
take the median and 16/84th-percentile counts per bin.

Models: DRW and OBPL (the red-noise models; both carry log_bend_freq).

    conda run -n <env> python scripts/aggregate_bendfreq.py \
        --results-dir results/3_4 --config-csv scenario.csv \
        --group-col highalpha --true-log-fbend 0.0 --out summary.json
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

MODELS = ["drw", "obpl"]
BINS = np.linspace(-3.0, 3.0, 25)  # log10 f_bend [yr^-1]
CENTERS = 0.5 * (BINS[:-1] + BINS[1:])
N_ITER = 5000
RNG = np.random.default_rng(0)


def posteriors(results_dir, ids, model):
    out = []
    for i in ids:
        p = os.path.join(results_dir, f"{i}_{model}.json")
        if not os.path.exists(p):
            continue
        s = np.asarray(json.load(open(p))["samples"]["log10_fbend"], float)
        if s.size:
            out.append(s)
    return out


def avg_histogram(posts):
    """Average-histogram: one draw per LC, histogram, repeat; median+/-1sigma."""
    counts = np.empty((N_ITER, len(CENTERS)))
    for k in range(N_ITER):
        draws = np.array([p[RNG.integers(len(p))] for p in posts])
        counts[k], _ = np.histogram(draws, bins=BINS)
    med = np.percentile(counts, 50, axis=0)
    lo = np.percentile(counts, 16, axis=0)
    hi = np.percentile(counts, 84, axis=0)
    pooled = np.concatenate([RNG.choice(p, size=min(len(p), 2000)) for p in posts])
    est = np.percentile(pooled, [16, 50, 84])
    return med, lo, hi, est


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--results-dir",
        required=True,
        help="directory of <ID>_<model>.json FitResult files",
    )
    ap.add_argument("--config-csv", required=True)
    ap.add_argument(
        "--group-col", required=True, help="CSV column to group light curves by"
    )
    ap.add_argument("--filter-col", default=None)
    ap.add_argument("--filter-value", type=float, default=None)
    ap.add_argument("--true-log-fbend", type=float, default=0.0)
    ap.add_argument("--out", required=True, help="output summary JSON path")
    args = ap.parse_args()

    df = pd.read_csv(args.config_csv)
    if args.filter_col is not None:
        df = df[df[args.filter_col] == args.filter_value]

    key_values = {
        "true_log_fbend": args.true_log_fbend,
        "units": "log10 f_bend [yr^-1]",
        "bin_centers": [round(c, 3) for c in CENTERS],
        "configs": {},
    }
    summary_rows = []
    for gval, sub in df.groupby(args.group_col, sort=True):
        ids = sorted(int(x) for x in sub["ID"])
        cfg_out = {}
        for model in MODELS:
            posts = posteriors(args.results_dir, ids, model)
            if not posts:
                continue
            med, lo, hi, est = avg_histogram(posts)
            cfg_out[model] = {
                "n_lightcurves": len(posts),
                "est_log_fbend_median": round(float(est[1]), 3),
                "est_log_fbend_16_84": [
                    round(float(est[0]), 3),
                    round(float(est[2]), 3),
                ],
                "bias_vs_true": round(float(est[1] - args.true_log_fbend), 3),
                "width_16_84": round(float(est[2] - est[0]), 3),
                "hist_median_counts": [round(float(x), 2) for x in med],
                "hist_1sigma_lo": [round(float(x), 2) for x in lo],
                "hist_1sigma_hi": [round(float(x), 2) for x in hi],
            }
        key_values["configs"][f"{args.group_col}={gval:g}"] = cfg_out
        summary_rows.append((f"{args.group_col}={gval:g}", cfg_out))

    with open(args.out, "w") as f:
        json.dump(
            {
                "method": "average histogram of log_bend_freq posteriors",
                "true_log_fbend": args.true_log_fbend,
                "n_iter": N_ITER,
                "key_values": key_values,
            },
            f,
            indent=1,
        )
    print("written", args.out)

    for gname, cfg_out in summary_rows:
        cells = []
        for m in MODELS:
            if m in cfg_out:
                o = cfg_out[m]
                cells.append(
                    f"{m}: median={o['est_log_fbend_median']:+.2f} "
                    f"bias={o['bias_vs_true']:+.2f} width={o['width_16_84']:.2f} "
                    f"(n={o['n_lightcurves']})"
                )
        print(f"  {gname:16s} " + " | ".join(cells))


if __name__ == "__main__":
    main()
