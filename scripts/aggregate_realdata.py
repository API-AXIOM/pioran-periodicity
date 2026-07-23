"""Aggregate real-data FitResult JSONs into a summary JSON + markdown report.

Reads every FitResult JSON under <results-dir>/<source>/, computes posterior
medians and 1-sigma (16/84 percentile) intervals and log10 Bayes factors for
the standard model pairs, and writes a summary JSON and a markdown report.

Robust to partial completion: only models with a result file are aggregated;
missing models are reported.

    conda run -n <env> python scripts/aggregate_realdata.py \
        --results-dir results/realdata --out summary.json --summary-md summary.md
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

import pioran_periodicity as pp
from pioran_periodicity.inference import load_result

LN10 = np.log(10.0)

EXPECTED = {
    "PG1302": ["drw", "drw+sine", "obpl", "obpl+sine", "obpl_ah24", "obpl+sine_ah24"],
    "PG1553": [
        "drw",
        "drw+sine",
        "drw+linear",
        "drw+sine+linear",
        "obpl",
        "obpl+sine",
        "obpl+linear",
        "obpl+sine+linear",
    ],
}

BF_PAIRS = [
    ("PG1302", "drw+sine", "drw"),
    ("PG1302", "obpl+sine", "obpl"),
    ("PG1302", "obpl+sine", "drw+sine"),
    ("PG1302", "obpl", "drw"),
    ("PG1302", "obpl+sine_ah24", "obpl_ah24"),
    ("PG1302", "obpl+sine", "obpl+sine_ah24"),
    ("PG1553", "drw+sine", "drw"),
    ("PG1553", "obpl+sine", "obpl"),
    ("PG1553", "obpl+sine", "drw+sine"),
    ("PG1553", "drw+sine+linear", "drw+linear"),
    ("PG1553", "obpl+sine+linear", "obpl+linear"),
    ("PG1553", "drw+linear", "drw"),
    ("PG1553", "obpl+linear", "obpl"),
]


def _fname(model: str) -> str:
    return model.replace("+", "_") + ".json"


def load_all(results_dir):
    out = {}
    for source, models in EXPECTED.items():
        out[source] = {}
        for m in models:
            p = os.path.join(results_dir, source, _fname(m))
            if os.path.exists(p):
                out[source][m] = load_result(p)
    return out


def summarise_posteriors(res):
    med, sig = {}, {}
    for name, vals in res.samples.items():
        arr = np.asarray(vals, dtype=float)
        med[name] = float(np.median(arr))
        lo, hi = np.percentile(arr, [16, 84])
        sig[name] = [float(lo), float(hi)]
    return med, sig


def _verdict(log10bf, label):
    a = abs(log10bf)
    if a < 0.5:
        strength = "inconclusive"
    elif a < 1.0:
        strength = "substantial"
    elif a < 2.0:
        strength = "strong"
    else:
        strength = "decisive"
    fav = "periodic" if log10bf > 0 else "non-periodic"
    return f"{label}: log10 BF={log10bf:+.2f} -> {strength} evidence for {fav} model"


def build_summary(all_res, results_dir):
    logz = {}
    model_results = {}
    for source, models in all_res.items():
        for m, res in models.items():
            med, sig = summarise_posteriors(res)
            logz[(source, m)] = res.logz
            key = f"{source.lower()}_{m.replace('+', '_')}"
            model_results[key] = {
                "logz": res.logz,
                "logzerr": res.logzerr,
                "posterior_medians": med,
                "posterior_1sigma": sig,
                "ncall": res.ncall,
                "runtime_s": res.runtime_s,
                "hit_max_ncalls": res.ncall >= res.settings.get("max_ncalls", np.inf),
                "seed": res.meta.get("seed"),
                "n_components": res.meta.get("n_components"),
            }

    def bf(source, a, b):
        if (source, a) in logz and (source, b) in logz:
            return (logz[(source, a)] - logz[(source, b)]) / LN10
        return None

    test_statistics = {}
    for source, a, b in BF_PAIRS:
        v = bf(source, a, b)
        if v is not None:
            key = f"{source.lower()}_{a.replace('+', '_')}_vs_{b.replace('+', '_')}"
            test_statistics[key] = {
                "statistic": v,
                "p_value": None,
                "conclusion": _verdict(v, f"{a} vs {b} ({source})"),
            }

    cfg_path = os.path.join(results_dir, "run_config.json")
    metadata = {"package_version": pp.__version__}
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            metadata["run_config"] = json.load(f)

    summary = {
        "model_results": model_results,
        "test_statistics": test_statistics,
        "metadata": metadata,
    }

    missing = {}
    for source, models in EXPECTED.items():
        miss = [m for m in models if m not in all_res.get(source, {})]
        if miss:
            missing[source] = miss
    if missing:
        summary["missing"] = missing

    return summary, logz


def write_summary_md(all_res, logz, path):
    L = ["# Real-data re-analysis summary\n"]
    L.append(
        "Nested sampling (ultranest) fits of DRW / OBPL noise models "
        "against periodic (sine) / linear mean alternatives.\n"
    )

    for source in EXPECTED:
        if source not in all_res:
            continue
        L.append(f"\n## logZ per model -- {source}\n")
        L.append("| model | logZ | +/- | ncall | hit cap | runtime (s) |")
        L.append("|---|---|---|---|---|---|")
        for m in EXPECTED[source]:
            if m in all_res[source]:
                r = all_res[source][m]
                cap = "yes" if r.ncall >= r.settings.get("max_ncalls", 1e18) else ""
                L.append(
                    f"| {m} | {r.logz:.3f} | {r.logzerr:.3f} | "
                    f"{r.ncall} | {cap} | {r.runtime_s:.0f} |"
                )
            else:
                L.append(f"| {m} | (missing) | | | | |")

    L.append("\n## log10 Bayes factors (numerator / denominator)\n")
    L.append("Positive => evidence favours the numerator model.\n")
    L.append("| source | comparison | log10 BF |")
    L.append("|---|---|---|")
    for s, a, b in BF_PAIRS:
        if (s, a) in logz and (s, b) in logz:
            v = (logz[(s, a)] - logz[(s, b)]) / LN10
            L.append(f"| {s} | {a} / {b} | {v:+.2f} |")

    for source in EXPECTED:
        if source not in all_res:
            continue
        L.append(f"\n## Posterior medians +/- 1sigma (16/84 pct) -- {source}\n")
        for m in EXPECTED[source]:
            if m not in all_res[source]:
                continue
            r = all_res[source][m]
            med, sig = summarise_posteriors(r)
            L.append(f"\n**{m}** (logZ={r.logz:.2f}):\n")
            L.append("| param | median | 16% | 84% |")
            L.append("|---|---|---|---|")
            for p in med:
                L.append(
                    f"| {p} | {med[p]:+.3f} | {sig[p][0]:+.3f} | {sig[p][1]:+.3f} |"
                )

    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--results-dir",
        required=True,
        help="directory containing <source>/<model>.json FitResults",
    )
    ap.add_argument("--out", required=True, help="output summary JSON path")
    ap.add_argument("--summary-md", default=None, help="optional markdown report path")
    args = ap.parse_args()

    all_res = load_all(args.results_dir)
    summary, logz = build_summary(all_res, args.results_dir)

    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print("Wrote", args.out)

    if args.summary_md:
        write_summary_md(all_res, logz, args.summary_md)
        print("Wrote", args.summary_md)

    for source, miss in summary.get("missing", {}).items():
        print(f"MISSING {source}: {miss}")


if __name__ == "__main__":
    main()
