"""Apply the Tier-1 null campaign's PRE-REGISTERED stopping rule to a stage's
results, and print the exact command that builds the next stage.

The staged 20 -> 50 -> 100 design costs the same as going straight to 100,
because ``--rep-start`` extensions never re-draw or re-fit an object a cell
already used. Its only real risk is stopping *because a result looked good*,
so the rule is fixed in advance and encoded here rather than applied by eye:

    stop at n=20   if the steepest cell's FPR is >= 30% AND the shallowest
                   cell's is ~0, with non-overlapping Wilson intervals
                   (the slope trend is already established);
    stop at n=50   if the steepest cell's FPR is >= 20%;
    go to n=100    whenever EVERY cell is under 10% -- n=50 cannot separate
                   1% from 3% (the Wilson interval on 2/50 is [0.4%, 10.5%]).

The "all cells under 10%" clause is checked FIRST and wins outright: a
uniformly low FPR is precisely the case the smaller stages cannot resolve.

Input is a summary JSON from ``aggregate_results.py`` grouped on the slope
axis, e.g.

    conda run -n <env> python scripts/aggregate_results.py \\
        --results-dir <dir> --group-cols highalpha --out summary.json
    conda run -n <env> python scripts/campaign_stage_decision.py \\
        --summary summary.json --pair drw --stage 20

Convergence gate
----------------
A stage decision made on truncated fits is worthless: 89% of one earlier
multi-band LSST campaign was cut short by ``max_ncalls``, and the "FPR is
fixed" conclusion drawn from it was an artifact. This script therefore
REFUSES to return a verdict when a cell's converged fraction is below
``--min-converged-frac`` (default 0.9), and says so per cell.
"""

from __future__ import annotations

import argparse
import json
import math
import sys

Z95 = 1.959963984540054

# The rule's thresholds, in one place so the tests pin the same numbers the
# script applies.
STOP_AT_20_STEEPEST = 0.30
STOP_AT_50_STEEPEST = 0.20
ALL_CELLS_LOW = 0.10
STAGES = (20, 50, 100)


def wilson(k: int, n: int, z: float = Z95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Used rather than the normal approximation because the campaign's whole
    point is resolving rates near 0, where k/n +/- 1.96*sqrt(p(1-p)/n) gives
    a degenerate [0, 0] interval at k=0 and can run negative elsewhere.
    """
    if n <= 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def cells_from_summary(summary: dict, pair: str) -> list[dict]:
    """One record per slope cell: FPR = detections / n for the null campaign.

    A 'detect' in a NULL scenario is by construction a false positive, so
    the aggregator's outcome counts are the FPR numerator directly.
    """
    table = summary["table"] if "table" in summary else summary
    # aggregate_results.py keys its pairs "DRW"/"CARMA21"/"OBPL"; accept any
    # casing rather than silently reporting an empty campaign.
    available = sorted({k for pairs in table.values() for k in pairs})
    match = {k.lower(): k for k in available}.get(pair.lower())
    if match is None:
        raise SystemExit(
            f"no model pair {pair!r} in this summary; it has: "
            f"{', '.join(available) or '(none)'}"
        )
    cells = []
    for key, pairs in table.items():
        if match not in pairs:
            continue
        entry = pairs[match]
        n = entry["n"]
        k = entry["outcomes"]["detect"]
        slope = None
        for part in key.split(","):
            name, _, value = part.strip().partition("=")
            if name == "highalpha":
                slope = float(value)
        lo, hi = wilson(k, n)
        cells.append({
            "cell": key,
            "highalpha": slope,
            "n": n,
            "detections": k,
            "fpr": (k / n) if n else float("nan"),
            "ci95": (lo, hi),
            "converged_frac": entry.get("converged_frac", 1.0),
            "n_dropped_unconverged": entry.get("n_dropped_unconverged", 0),
        })
    cells.sort(key=lambda c: (c["highalpha"] is None, c["highalpha"]))
    return cells


def decide(cells: list[dict], stage: int) -> dict:
    """The pre-registered rule. Returns a verdict dict; never consults
    anything but the numbers and the stage."""
    if not cells:
        return {"verdict": "NO DATA", "reason": "no cells matched this pair"}

    fprs = [c["fpr"] for c in cells]
    steepest = cells[0]   # most negative highalpha
    shallowest = cells[-1]

    if max(fprs) < ALL_CELLS_LOW:
        return {
            "verdict": "EXTEND",
            "next_stage": 100,
            "reason": (
                f"every cell is under {ALL_CELLS_LOW:.0%} (max "
                f"{max(fprs):.1%}); n={stage} cannot resolve rates this low, "
                f"so the rule goes straight to 100"
            ),
        }

    if stage <= 20:
        separated = steepest["ci95"][0] > shallowest["ci95"][1]
        if steepest["fpr"] >= STOP_AT_20_STEEPEST and separated:
            return {
                "verdict": "STOP",
                "reason": (
                    f"steepest cell {steepest['fpr']:.1%} >= "
                    f"{STOP_AT_20_STEEPEST:.0%} and its Wilson interval "
                    f"clears the shallowest cell's -- the slope trend is "
                    f"established at n={stage}"
                ),
            }
        why = []
        if steepest["fpr"] < STOP_AT_20_STEEPEST:
            why.append(
                f"steepest cell {steepest['fpr']:.1%} < {STOP_AT_20_STEEPEST:.0%}"
            )
        if not separated:
            why.append(
                f"Wilson intervals overlap (steepest {steepest['ci95'][0]:.1%}"
                f"-{steepest['ci95'][1]:.1%} vs shallowest "
                f"{shallowest['ci95'][0]:.1%}-{shallowest['ci95'][1]:.1%})"
            )
        return {"verdict": "EXTEND", "next_stage": 50, "reason": "; ".join(why)}

    if stage <= 50:
        if steepest["fpr"] >= STOP_AT_50_STEEPEST:
            return {
                "verdict": "STOP",
                "reason": (
                    f"steepest cell {steepest['fpr']:.1%} >= "
                    f"{STOP_AT_50_STEEPEST:.0%} at n={stage}"
                ),
            }
        return {
            "verdict": "EXTEND",
            "next_stage": 100,
            "reason": (
                f"steepest cell {steepest['fpr']:.1%} < "
                f"{STOP_AT_50_STEEPEST:.0%}"
            ),
        }

    return {
        "verdict": "STOP",
        "reason": f"n={stage} is the final stage; the rule prescribes no extension",
    }


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--summary", required=True,
                    help="summary JSON from aggregate_results.py "
                         "(--group-cols highalpha)")
    ap.add_argument("--pair", default="DRW",
                    help="model pair to decide on, as keyed by "
                         "aggregate_results.py: DRW, CARMA21 or OBPL "
                         "(case-insensitive; default: %(default)s)")
    ap.add_argument("--stage", type=int, required=True,
                    help="reps per cell in the stage just finished (20/50/100)")
    ap.add_argument("--min-converged-frac", type=float, default=0.9,
                    help="refuse a verdict if any cell falls below this "
                         "converged fraction (default: %(default)s)")
    ap.add_argument("--builder-cmd", default=None,
                    help="the make_*_csv.py command that built this stage; "
                         "echoed back with --rep-start/--n-per-cell filled in "
                         "for the extension")
    args = ap.parse_args()

    with open(args.summary) as fh:
        summary = json.load(fh)
    cells = cells_from_summary(summary, args.pair)

    print(f"pair={args.pair}  stage n={args.stage}")
    print(f"{'cell':<24}{'det/n':>10}{'FPR':>9}{'95% Wilson':>18}{'conv':>8}")
    for c in cells:
        lo, hi = c["ci95"]
        print(
            f"{c['cell']:<24}{c['detections']:>5}/{c['n']:<4}"
            f"{c['fpr']:>8.1%}{lo:>9.1%}-{hi:<8.1%}{c['converged_frac']:>7.0%}"
        )

    bad = [c for c in cells if c["converged_frac"] < args.min_converged_frac]
    if bad:
        print(
            f"\nNO VERDICT: {len(bad)} cell(s) below "
            f"{args.min_converged_frac:.0%} converged "
            f"({', '.join(c['cell'] for c in bad)}). A stage decision made on "
            f"truncated fits is not a decision -- raise --max-ncalls and "
            f"finish those fits first."
        )
        return 2

    verdict = decide(cells, args.stage)
    print(f"\n{verdict['verdict']}: {verdict['reason']}")
    if verdict["verdict"] == "EXTEND":
        nxt = verdict["next_stage"]
        print(
            f"\nExtend {args.stage} -> {nxt} reps/cell: rebuild with the SAME "
            f"--seed, a fresh --first-id, and\n"
            f"    --n-per-cell {nxt - args.stage} --rep-start {args.stage}\n"
            f"then fit the new CSV into the SAME results dir "
            f"(already-fit IDs are skipped, not redone)."
        )
        if args.builder_cmd:
            print(f"\n    {args.builder_cmd} \\\n"
                  f"        --n-per-cell {nxt - args.stage} "
                  f"--rep-start {args.stage}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
