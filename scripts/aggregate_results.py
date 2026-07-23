"""Aggregate a scenario's per-light-curve FitResult JSONs into a summary
JSON of log10 Bayes factors.

For every (scenario, configuration, model pair) compute
log10 B = (logZ_rednoise - logZ_periodic) / ln(10) per light curve and
classify with the Kass-Raftery thresholds:
  log10 B < -2  -> 'detect'   (decisive evidence for periodicity)
  -2 .. +2      -> 'inconclusive'
  log10 B > +2  -> 'refute'   (decisive evidence refuting periodicity)

    conda run -n <env> python scripts/aggregate_results.py \
        --results-dir results/3_4 --config-csv scenario.csv \
        --group-cols highalpha --out summary.json
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd

PAIRS = {
    "DRW": ("drw", "drw_sine"),
    "CARMA21": ("carma", "carma_sine"),
    "OBPL": ("obpl", "obpl_sine"),
}


def classify(b):
    return "detect" if b < -2 else ("refute" if b > 2 else "inconclusive")


def logz(results_dir, lc_id, model):
    path = os.path.join(results_dir, f"{lc_id}_{model}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)["logz"]


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--results-dir",
        required=True,
        help="directory of <ID>_<model>.json FitResult files",
    )
    ap.add_argument(
        "--config-csv",
        required=True,
        help="scenario config CSV (must contain an ID column)",
    )
    ap.add_argument(
        "--group-cols",
        default="",
        help="comma-separated config CSV columns to group by",
    )
    ap.add_argument("--out", required=True, help="output summary JSON path")
    args = ap.parse_args()

    group_cols = [c for c in args.group_cols.split(",") if c]
    df = pd.read_csv(args.config_csv).set_index("ID")
    ids = sorted(
        {
            int(f.split("_")[0])
            for f in os.listdir(args.results_dir)
            if f.endswith(".json") and f[0].isdigit()
        }
    )

    per_cfg = defaultdict(lambda: defaultdict(list))
    for lc_id in ids:
        row = df.loc[float(lc_id)]
        key = ", ".join(f"{c}={row[c]:g}" for c in group_cols) or "all"
        for pair, (base, sine) in PAIRS.items():
            z0, z1 = logz(args.results_dir, lc_id, base), logz(
                args.results_dir, lc_id, sine
            )
            if z0 is None or z1 is None:
                continue
            per_cfg[key][pair].append((z0 - z1) / np.log(10))

    table = {}
    for key in sorted(per_cfg):
        table[key] = {}
        for pair, bfs in per_cfg[key].items():
            bfs = np.array(bfs)
            table[key][pair] = {
                "n": len(bfs),
                "log10_BF_mean": float(bfs.mean()),
                "log10_BF_values": [round(float(b), 3) for b in bfs],
                "outcomes": {
                    c: int(sum(classify(b) == c for b in bfs))
                    for c in ("detect", "inconclusive", "refute")
                },
            }

    with open(args.out, "w") as f:
        json.dump(
            {
                "table": table,
                "thresholds": "log10B<-2 detect | +-2 inconclusive | >2 refute",
            },
            f,
            indent=1,
        )
    print(f"written {args.out}")

    for key, pairs in table.items():
        cells = []
        for pair in PAIRS:
            if pair in pairs:
                p = pairs[pair]
                o = p["outcomes"]
                cells.append(
                    f"{pair}: {p['log10_BF_mean']:+.2f} "
                    f"(d{o['detect']}/i{o['inconclusive']}/r{o['refute']})"
                )
        print(f"  {key:42s} " + " | ".join(cells))


if __name__ == "__main__":
    main()
