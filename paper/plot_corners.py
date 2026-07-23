"""Corner plots of the posterior distributions for real-data fits.

Reads only the stored FitResult JSONs (equal-weight resampled posterior
samples), so no Julia/pioranpy call is involved.

Diagonal titles show median with +1sigma/-1sigma (16/84 percentiles).
The figure title records logZ, its uncertainty, the posterior ESS and the
convergence flag -- a low-ESS/unconverged fit's contours are not
trustworthy and are labelled as such.

    conda run -n <env> python paper/plot_corners.py \
        --results-dir results/realdata --out-dir plots/corner
    conda run -n <env> python paper/plot_corners.py \
        --results-dir results/realdata --out-dir plots/corner \
        --source PG1302 --model obpl_sine --force
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import corner
import matplotlib.pyplot as plt
import numpy as np

# Display labels for the package's parameter names (all logs base 10).
LABELS = {
    "log10_variance": r"$\log_{10}\sigma^2$",
    "log10_fbend": r"$\log_{10} f_\mathrm{bend}$",
    "alpha_low": r"$\alpha_\mathrm{low}$",
    "alpha_high": r"$\alpha_\mathrm{high}$",
    "A1": r"$A_1$",
    "A2": r"$A_2$",
    "period": r"$T_\mathrm{period}$",
    "err_scale": r"$\nu$",
    "slope": r"$m$",
    "intercept": r"$b$",
}

SOURCE_TITLES = {
    "PG1302": "PG 1302$-$102",
    "PG1553": "PG 1553$+$113",
}

MODEL_TITLES = {
    "drw": "DRW",
    "drw_sine": "DRW + sine",
    "drw_linear": "DRW + linear",
    "drw_sine_linear": "DRW + sine + linear",
    "obpl": "OBPL",
    "obpl_sine": "OBPL + sine",
    "obpl_linear": "OBPL + linear",
    "obpl_sine_linear": "OBPL + sine + linear",
    "obpl_ah24": r"OBPL [$\alpha_\mathrm{high}\sim U(2,4)$]",
    "obpl_sine_ah24": r"OBPL + sine [$\alpha_\mathrm{high}\sim U(2,4)$]",
}


def make_corner(path, out_dir, force=False):
    source = os.path.basename(os.path.dirname(path))
    model = os.path.basename(path).replace(".json", "")
    out_path = os.path.join(out_dir, f"{source}_{model}_corner.png")
    if os.path.exists(out_path) and not force:
        print(f"skip {out_path} (exists)")
        return

    with open(path) as f:
        result = json.load(f)

    samples = result["samples"]
    names = list(samples.keys())
    data = np.column_stack([np.asarray(samples[n], dtype=float) for n in names])
    labels = [LABELS.get(n, n) for n in names]

    # Drop any parameter with zero spread: corner cannot draw a histogram
    # for a delta function (would only happen for a collapsed posterior).
    spread = data.std(axis=0)
    keep = spread > 0
    if not keep.all():
        dropped = [n for n, k in zip(names, keep) if not k]
        print(
            f"  WARNING {source}/{model}: dropping degenerate parameter(s) "
            f"{dropped} (zero posterior spread)"
        )
        data = data[:, keep]
        labels = [lbl for lbl, k in zip(labels, keep) if k]

    fig = corner.corner(
        data,
        labels=labels,
        bins=40,
        show_titles=True,
        quantiles=[0.16, 0.5, 0.84],
        title_quantiles=[0.16, 0.5, 0.84],
        title_kwargs={"fontsize": 9},
        label_kwargs={"fontsize": 12},
        plot_datapoints=False,
        fill_contours=True,
        levels=(1 - np.exp(-0.5), 1 - np.exp(-2.0)),  # 1 and 2 sigma (2-D)
        color="tab:blue",
    )

    ess = result.get("ess", None)
    max_ncalls = result.get("settings", {}).get("max_ncalls")
    converged = result.get("converged")
    if converged is None and max_ncalls is not None:
        converged = result.get("ncall", 0) < max_ncalls

    if ess is None or not np.isfinite(ess):
        ess_txt = "ESS n/a"
    else:
        ess_txt = f"ESS = {ess:.0f}"
    flag = "" if converged else "   [UNCONVERGED -- contours unreliable]"
    fig.suptitle(
        f"{SOURCE_TITLES.get(source, source)} -- {MODEL_TITLES.get(model, model)}\n"
        f"$\\log Z = {result['logz']:.2f} \\pm {result['logzerr']:.2f}$, "
        f"{ess_txt}{flag}",
        fontsize=13,
        y=1.02,
    )

    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}  ({data.shape[1]} params, {ess_txt})")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--results-dir",
        required=True,
        help="directory containing <source>/<model>.json FitResults",
    )
    ap.add_argument("--out-dir", required=True, help="output directory for PNGs")
    ap.add_argument("--source", default=None, help="only this source")
    ap.add_argument("--model", default=None, help="only this model")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    pattern = os.path.join(
        args.results_dir, args.source or "*", f"{args.model or '*'}.json"
    )
    paths = sorted(p for p in glob.glob(pattern) if not p.endswith("run_config.json"))
    if not paths:
        raise SystemExit(f"no fit JSONs matched {pattern}")

    for path in paths:
        make_corner(path, args.out_dir, force=args.force)
    print(f"\n{len(paths)} corner plots processed -> {args.out_dir}/")


if __name__ == "__main__":
    main()
