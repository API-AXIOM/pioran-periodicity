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

Convergence and ESS gating
---------------------------
A fit truncated by ``max_ncalls`` has not finished integrating: its ``logz``
is wherever the integration happened to be when the cap fired, and its
posterior has typically collapsed onto a handful of live points. Such a fit
records ``converged: false`` (and an ``ess`` far below the sample count).

Because the periodic model carries the extra parameters, it is the one that
gets truncated first, which biases ``logz_rednoise - logz_periodic``
*upwards* and silently suppresses detections. Both members of a pair must
therefore be trustworthy for their Bayes factor to mean anything, so by
default this script **drops any pair where either fit failed to converge**
and reports the retained fraction per cell. Pass ``--keep-unconverged`` to
restore the old unfiltered behaviour (for diagnosing a run, not for
quoting results).

Independently, this script also **drops any pair where either fit's posterior
ESS falls below a floor (default 200.0)**. A fit can be flagged converged and
still have ESS=1 if it stopped on a likelihood blow-up below ``max_ncalls``; in
this case the evidence value is unreliable regardless of the convergence flag.
The ESS floor is chosen to sit in the gap between known pathological cases
(ESS=1) and the healthy population. This gate is always applied (independent
of ``--keep-unconverged``) and is configurable via ``--min-ess``.
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


def fit_summary(results_dir, lc_id, model):
    """Return ``(logz, converged, ess)`` for one fit, or ``None`` if absent.

    ``converged`` defaults to True for legacy result files written before the
    flag existed -- those predate the multi-band models and never approached
    ``max_ncalls``, so treating them as converged preserves their behaviour.

    ``ess`` defaults to np.inf for legacy result files written before the ess
    key existed, so they always pass the ESS floor gate.
    """
    path = os.path.join(results_dir, f"{lc_id}_{model}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        d = json.load(f)
    return d["logz"], bool(d.get("converged", True)), float(d.get("ess", np.inf))


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


def build_table(
    results_dir, group_cols=(), config_csv=None, keep_unconverged=False, min_ess=200.0
):
    """Aggregate ``<ID>_<model>.json`` FitResults into the summary table
    described in the module docstring. ``group_cols`` come from
    ``config_csv`` if given, else from each result's own ``meta`` (see
    ``row_lookup``).

    Unless ``keep_unconverged``, pairs where either fit has
    ``converged: false`` are excluded; each cell reports ``n_dropped`` and
    ``converged_frac`` so the loss is always visible alongside the numbers.

    Pairs where either fit has ESS below ``min_ess`` are also excluded,
    even if both fits are marked converged; such fits truncated below
    max_ncalls may have stopped on a likelihood blow-up with ESS=1 and
    unreliable evidence.
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
    dropped = defaultdict(lambda: defaultdict(int))
    low_ess_pairs = defaultdict(lambda: defaultdict(int))
    low_ess_excluded = defaultdict(lambda: defaultdict(list))
    for lc_id in ids:
        row = rows[lc_id]
        key = ", ".join(f"{c}={row[c]:g}" for c in group_cols if c in row) or "all"
        for pair, (base, sine) in PAIRS.items():
            f0 = fit_summary(results_dir, lc_id, base)
            f1 = fit_summary(results_dir, lc_id, sine)
            if f0 is None or f1 is None:
                continue
            (z0, ok0, e0), (z1, ok1, e1) = f0, f1
            if not (ok0 and ok1) and not keep_unconverged:
                dropped[key][pair] += 1
                continue
            low = [
                {"lc_id": lc_id, "model": m, "ess": e}
                for m, e in ((base, e0), (sine, e1))
                if e < min_ess
            ]
            if low:
                low_ess_pairs[key][pair] += 1
                low_ess_excluded[key][pair].extend(low)
                continue
            per_cfg[key][pair].append((lc_id, (z0 - z1) / np.log(10)))

    table = {}
    for key in sorted(set(per_cfg) | set(dropped) | set(low_ess_pairs)):
        table[key] = {}
        for pair in PAIRS:
            entries = per_cfg[key].get(pair, [])
            ids_list = [lc_id for lc_id, _ in entries]
            bfs = np.array([b for _, b in entries])
            n_drop = dropped[key][pair]
            n_drop_ess = low_ess_pairs[key][pair]
            if not len(bfs) and not n_drop and not n_drop_ess:
                continue
            total = len(bfs) + n_drop + n_drop_ess
            table[key][pair] = {
                "n": len(bfs),
                "n_dropped_unconverged": n_drop,
                "n_dropped_low_ess": n_drop_ess,
                "excluded": low_ess_excluded[key][pair],
                "lc_ids": ids_list,
                "converged_frac": round(len(bfs) / total, 3),
                "log10_BF_mean": float(bfs.mean()) if len(bfs) else None,
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
    ap.add_argument(
        "--keep-unconverged",
        action="store_true",
        help="include pairs whose fits were truncated by max_ncalls. Their "
        "logz is not an evidence estimate and the resulting Bayes factors are "
        "biased towards 'refute' -- for diagnosis only, never for results",
    )
    ap.add_argument(
        "--min-ess",
        type=float,
        default=200.0,
        help="drop any pair where either fit's posterior ESS falls below "
        "this floor. A fit can be flagged converged and still have ESS=1 "
        "(v2 lsst_single 310335_obpl stopped below max_ncalls on a "
        "likelihood blow-up), and its logz is then not an evidence "
        "estimate. 200 sits in the empty gap between the 4 bad v2 fits "
        "(ESS=1) and the healthy population (1st percentile 1552).",
    )
    args = ap.parse_args()

    group_cols = [c for c in args.group_cols.split(",") if c]
    table = build_table(
        args.results_dir,
        group_cols,
        args.config_csv,
        args.keep_unconverged,
        args.min_ess,
    )

    with open(args.out, "w") as f:
        json.dump(
            {
                "table": table,
                "thresholds": "log10B<-2 detect | +-2 inconclusive | >2 refute",
                "unconverged_pairs": "included" if args.keep_unconverged else "dropped",
                "min_ess": args.min_ess,
            },
            f,
            indent=1,
        )
    print(f"written {args.out}")
    if args.keep_unconverged:
        print(
            "  WARNING: --keep-unconverged -- Bayes factors below include "
            "truncated fits and are biased towards 'refute'"
        )

    kept = sum(p["n"] for pairs in table.values() for p in pairs.values())
    drop_unconverged = sum(
        p["n_dropped_unconverged"] for pairs in table.values() for p in pairs.values()
    )
    drop_ess = sum(
        p["n_dropped_low_ess"] for pairs in table.values() for p in pairs.values()
    )
    drop_total = drop_unconverged + drop_ess
    for key, pairs in table.items():
        cells = []
        for pair in PAIRS:
            if pair in pairs:
                p = pairs[pair]
                o = p["outcomes"]
                m = p["log10_BF_mean"]
                mean = "  n/a " if m is None else f"{m:+.2f}"
                n_total = p["n"] + p["n_dropped_unconverged"] + p["n_dropped_low_ess"]
                cells.append(
                    f"{pair}: {mean} "
                    f"(d{o['detect']}/i{o['inconclusive']}/r{o['refute']}) "
                    f"n={p['n']}/{n_total}"
                )
        print(f"  {key:42s} " + " | ".join(cells))

    if drop_total:
        msg = f"\n  DROPPED {drop_total} of {kept + drop_total} pairs "
        msg += f"({100 * drop_total / (kept + drop_total):.1f}%): "
        msgs = []
        if drop_unconverged:
            msgs.append(f"{drop_unconverged} unconverged (truncated by max_ncalls)")
        if drop_ess:
            msgs.append(f"{drop_ess} below ESS floor ({args.min_ess})")
        msg += " + ".join(msgs)
        msg += (
            ". Pairs must be trustworthy for their Bayes factor to mean "
            "anything -- a cell with few retained pairs cannot support a "
            "rate estimate."
        )
        print(msg)


if __name__ == "__main__":
    main()
