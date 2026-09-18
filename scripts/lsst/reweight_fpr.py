"""Post-stratify a campaign's false-positive rate onto an unstratified parent.

A campaign drawn from a declination-stratified target list does not, on its
face, estimate a population rate. This script re-expresses one as

    FPR_pop = sum_h W_h p_h

with ``p_h`` the rate measured in stratum ``h`` and ``W_h`` that stratum's
share of the parent from ``fetch_parent.py``.

Strata are cut on OpSim VISIT COUNT, not declination. Sampling density is the
causal driver of the false-positive rate -- declination carries no additional
signal once density is in the model -- and visit count is the one density
measure known for every parent object without simulating it. On the v2 LSST
null it reproduces the campaign's own dense/sparse labels 399/400 at a single
400-visit cut.

Fits are gated twice: the usual ``converged`` flag (a fit truncated by
``max_ncalls`` has not finished integrating) AND an ESS floor. The second
gate matters -- a nested-sampling run can report ``converged: true`` with an
effective sample size of 1 and a ``logz`` wrong by many orders of magnitude,
which no convergence flag catches.

    conda run -n <env> python scripts/lsst/reweight_fpr.py \
        --results-dir <campaign>/results --config-csv v2_lsst_n400_final.csv \
        --library-master lsst_data/master.csv \
        --visit-map visit_counts_nside64.npy --parent parent_lsst.csv \
        --edges 400 --edges 200,400,700
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

PAIRS = {"DRW": ("drw", "drw_sine"), "OBPL": ("obpl", "obpl_sine")}
LN10 = np.log(10.0)


def load_pairs(results_dir: Path, pair: str, ess_min: float) -> pd.DataFrame:
    """-> one row per light curve: log10 B, detect flag, slope."""
    noise, periodic = PAIRS[pair]
    rows, dropped = [], {"unconverged": 0, "low_ess": 0}
    for f in glob.glob(str(results_dir / f"*_{noise}.json")):
        lc_id = int(os.path.basename(f).split("_")[0])
        mate = results_dir / f"{lc_id}_{periodic}.json"
        if not mate.exists():
            continue
        a, b = json.load(open(f)), json.load(open(mate))
        if not (a["converged"] and b["converged"]):
            dropped["unconverged"] += 1
            continue
        if min(a["ess"], b["ess"]) < ess_min:
            dropped["low_ess"] += 1
            continue
        rows.append(dict(ID=lc_id, bf=(a["logz"] - b["logz"]) / LN10,
                         highalpha=a["meta"]["highalpha"]))
    print(f"  {pair}: {len(rows)} pairs retained, dropped {dropped}")
    return pd.DataFrame(rows)


def attach_visits(df: pd.DataFrame, config_csv: Path, library_master: Path,
                  visit_map: Path, nside: int) -> pd.DataFrame:
    """Join each light curve to the visit count at its cadence object."""
    import healpy as hp

    cfg = pd.read_csv(config_csv)[["ID", "cadence_source"]]
    lib = pd.read_csv(library_master)
    lib = lib.set_index("lsst:" + lib["object_id"].astype(str))

    df = df.merge(cfg, on="ID", how="left")
    df = df.assign(ra=df["cadence_source"].map(lib["ra"]),
                   dec=df["cadence_source"].map(lib["dec"]))
    if df["ra"].isna().any():
        raise SystemExit(f"{int(df['ra'].isna().sum())} light curves not matched "
                         "to the cadence library")

    counts = np.load(visit_map)
    df["n_visits"] = counts[hp.ang2pix(nside, df["ra"].to_numpy(),
                                       df["dec"].to_numpy(), lonlat=True)]
    return df


def reweight(df: pd.DataFrame, parent_visits: np.ndarray, edges: Sequence[float],
             threshold: float, nboot: int, rng: np.random.Generator) -> pd.DataFrame:
    """Post-stratified FPR per slope cell, with a parametric bootstrap CI."""
    w = np.bincount(np.digitize(parent_visits, edges),
                    minlength=len(edges) + 1).astype(float)
    w /= w.sum()                                             # (n_strata,)
    df = df.assign(h=np.digitize(df["n_visits"].to_numpy(), edges),
                   detect=df["bf"] < threshold)

    out = []
    for ha, g in df.groupby("highalpha"):
        p = g.groupby("h")["detect"].mean()
        n = g.groupby("h")["detect"].count()
        present = [h for h in range(len(w)) if w[h] > 0 and h in p.index]
        norm = float(sum(w[h] for h in present))
        est = float(sum(w[h] * p[h] for h in present) / norm)

        boot = np.empty(nboot)
        for i in range(nboot):
            boot[i] = sum(w[h] * rng.binomial(int(n[h]), p[h]) / n[h]
                          for h in present) / norm
        lo, hi = np.percentile(boot, [2.5, 97.5])
        out.append(dict(highalpha=ha,
                        n=len(g),
                        naive=100 * g["detect"].mean(),
                        reweighted=100 * est,
                        lo=100 * lo, hi=100 * hi,
                        empty_strata=int((w > 0).sum() - len(present))))
    return pd.DataFrame(out).sort_values("highalpha", ascending=False)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--results-dir", type=Path, required=True)
    ap.add_argument("--config-csv", type=Path, required=True)
    ap.add_argument("--library-master", type=Path, required=True)
    ap.add_argument("--visit-map", type=Path, required=True)
    ap.add_argument("--parent", type=Path, required=True)
    ap.add_argument("--pair", default="DRW", choices=sorted(PAIRS))
    ap.add_argument("--edges", action="append", default=None,
                    help="comma-separated visit-count cuts; repeat for several "
                         "stratifications (default: 400)")
    ap.add_argument("--threshold", type=float, default=-2.0,
                    help="log10 B below which a null counts as a detection")
    ap.add_argument("--ess-min", type=float, default=50.0)
    ap.add_argument("--nside", type=int, default=64)
    ap.add_argument("--nboot", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=20260912)
    ap.add_argument("--out", type=Path, default=None, help="optional CSV of results")
    args = ap.parse_args()

    df = load_pairs(args.results_dir, args.pair, args.ess_min)
    df = attach_visits(df, args.config_csv, args.library_master,
                       args.visit_map, args.nside)
    parent = pd.read_csv(args.parent)["n_visits"].to_numpy()
    print(f"parent n={len(parent)}, campaign n={len(df)}")

    rng = np.random.default_rng(args.seed)
    frames = []
    for spec in (args.edges or ["400"]):
        edges = [float(x) for x in spec.split(",")]
        w = np.bincount(np.digitize(parent, edges), minlength=len(edges) + 1)
        share = np.bincount(np.digitize(df["n_visits"].to_numpy(), edges),
                            minlength=len(edges) + 1)
        print(f"\n=== strata at {edges} ===")
        print("  parent weights  ", np.round(w / w.sum(), 4))
        print("  campaign shares ", np.round(share / share.sum(), 4))
        print("  campaign counts ", share)
        r = reweight(df, parent, edges, args.threshold, args.nboot, rng)
        print(r.to_string(index=False, float_format=lambda x: f"{x:6.1f}"))
        frames.append(r.assign(edges=spec))

    if args.out is not None:
        pd.concat(frames).to_csv(args.out, index=False)
        print("\nwritten", args.out)


if __name__ == "__main__":
    main()
