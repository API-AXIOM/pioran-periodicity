"""Scan a directory of run_sim.py/run_realdata.py FitResult JSONs and report
which fits did not converge.

A fit is flagged if `converged` is False (truncated by max_ncalls before its
termination criterion was met) or if `ess` (posterior effective sample size)
falls below --min-ess -- a collapsed run can have converged=True but
ess ~ 1, which still means the posterior samples are unusable.

    conda run -n <env> python scripts/check_convergence.py \
        --results-dir results/pilot_slope_power [--min-ess 50]
"""

from __future__ import annotations

import argparse
import json
import os


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
        "--min-ess",
        type=float,
        default=20.0,
        help="flag fits with posterior ESS below this, even if converged=True",
    )
    args = ap.parse_args()

    files = sorted(
        f for f in os.listdir(args.results_dir) if f.endswith(".json")
    )
    if not files:
        print(f"no .json files found in {args.results_dir}")
        return

    n_ok = 0
    flagged = []
    for fname in files:
        path = os.path.join(args.results_dir, fname)
        with open(path) as f:
            d = json.load(f)
        converged = d.get("converged", True)
        ess = d.get("ess", float("nan"))
        ncall = d.get("ncall")
        reasons = []
        if not converged:
            reasons.append("truncated by max_ncalls")
        if ess == ess and ess < args.min_ess:  # ess==ess filters NaN
            reasons.append(f"low ESS ({ess:.1f} < {args.min_ess:g})")
        if reasons:
            flagged.append((fname, d.get("model"), ncall, ess, reasons))
        else:
            n_ok += 1

    print(f"{args.results_dir}: {len(files)} fits, {n_ok} clean, "
          f"{len(flagged)} flagged")
    for fname, model, ncall, ess, reasons in flagged:
        print(f"  {fname:35s} model={model!s:12s} ncall={ncall!s:>8s} "
              f"ess={ess:8.1f}  " + "; ".join(reasons))


if __name__ == "__main__":
    main()
