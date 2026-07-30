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

``--config-csv`` is optional: every ``run_sim.py`` FitResult JSON already
carries its own ``meta`` (``highalpha``, and ``true_period``/``true_A1`` when
a signal was injected). Omit ``--config-csv`` to group directly off that
``meta`` instead of a config CSV -- the only reliable option when a results
directory was assembled from multiple pilot/extension CSVs whose ID ranges
you don't want to reconcile by hand.
"""

from __future__ import annotations

import argparse
import glob
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

# meta key -> group-column name, for the --config-csv-less fallback
META_GROUP_COLS = {"highalpha": "highalpha", "true_period": "period", "true_A1": "A1"}


def classify(b):
    return "detect" if b < -2 else ("refute" if b > 2 else "inconclusive")


def logz(results_dir, lc_id, model):
    path = os.path.join(results_dir, f"{lc_id}_{model}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)["logz"]


def meta_for(results_dir, lc_id):
    """Read ``meta`` off whichever fit result exists first for ``lc_id``."""
    for f in sorted(glob.glob(os.path.join(results_dir, f"{lc_id}_*.json"))):
        with open(f) as fh:
            return json.load(fh)["meta"]
    raise FileNotFoundError(f"no result JSON for id {lc_id} in {results_dir}")


def row_lookup(results_dir, ids, config_csv, group_cols):
    """Return ``{lc_id: {col: value}}`` for ``group_cols``, either from a
    config CSV (if given) or from each result's own ``meta``.
    """
    if config_csv is not None:
        df = pd.read_csv(config_csv).set_index("ID")
        return {lc_id: df.loc[float(lc_id)] for lc_id in ids}
    rows = {}
    for lc_id in ids:
        meta = meta_for(results_dir, lc_id)
        row = {}
        for meta_key, col in META_GROUP_COLS.items():
            if col in group_cols and meta_key in meta and meta[meta_key] is not None:
                row[col] = meta[meta_key]
        rows[lc_id] = row
    return rows


def build_table(results_dir, group_cols=(), config_csv=None):
    """Aggregate ``<ID>_<model>.json`` FitResults into the summary table
    described in the module docstring. ``group_cols`` come from
    ``config_csv`` if given, else from each result's own ``meta`` (see
    ``row_lookup``).
    """
    ids = sorted(
        {
            int(f.split("_")[0])
            for f in os.listdir(results_dir)
            if f.endswith(".json") and f[0].isdigit()
        }
    )
    rows = row_lookup(results_dir, ids, config_csv, group_cols)

    per_cfg = defaultdict(lambda: defaultdict(list))
    for lc_id in ids:
        row = rows[lc_id]
        key = ", ".join(f"{c}={row[c]:g}" for c in group_cols if c in row) or "all"
        for pair, (base, sine) in PAIRS.items():
            z0, z1 = logz(results_dir, lc_id, base), logz(results_dir, lc_id, sine)
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
    return table


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
        default=None,
        help="scenario config CSV (must contain an ID column); omit to "
        "group directly off each result's own meta instead",
    )
    ap.add_argument(
        "--group-cols",
        default="",
        help="comma-separated columns to group by",
    )
    ap.add_argument("--out", required=True, help="output summary JSON path")
    args = ap.parse_args()

    group_cols = [c for c in args.group_cols.split(",") if c]
    table = build_table(args.results_dir, group_cols, args.config_csv)

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
