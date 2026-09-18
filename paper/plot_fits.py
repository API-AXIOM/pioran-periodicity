"""Posterior-predictive plots for real-data fits: data as points with error
bars, the GP posterior-predictive median curve with a +-1 sigma band, and a
handful of random posterior draws filling the observational gaps on a
regular time grid.

For the +sine panels, red curves additionally show the periodic MEAN
function alone -- posterior draws (thin) and the median-parameter sinusoid
(solid) -- so one can see what periodic signals the model infers.

One figure per source per noise family:
  <source>_drw_panels.png    : DRW (top), DRW+sine (bottom)
  <source>_obpl_panels.png   : OBPL (top), OBPL+sine (bottom)

    conda run -n <env> python paper/plot_fits.py \
        --data-dir /path/to/AGNobsdata \
        --results-dir results/realdata --out-dir plots
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import pioran_periodicity as pp
from pioran_periodicity.means import sine_mean
from pioran_periodicity.predict import posterior_predictive

N_GRID = 400
N_DRAWS = 10
SEED = 0

PANEL_GROUPS = {
    "drw": ["drw", "drw_sine"],
    "obpl": ["obpl", "obpl_sine"],
}
PANEL_TITLES = {
    "drw": "DRW",
    "drw_sine": "DRW + sine",
    "obpl": "OBPL",
    "obpl_sine": "OBPL + sine",
}


def sources(data_dir):
    return {
        "PG1302": dict(
            loader=lambda: pp.load_pg1302(
                path=os.path.join(data_dir, "graham2015data.csv")
            ),
            ylabel="Magnitude - median",
            title="PG 1302$-$102 (CRTS)",
        ),
        "PG1553": dict(
            loader=lambda: pp.load_pg1553(
                path=os.path.join(data_dir, "PG1553_113_logbase.txt")
            ),
            ylabel="ln(Flux) - median",
            title="PG 1553$+$113 (Fermi-LAT)",
        ),
    }


def load_fit(results_dir, source, model):
    with open(os.path.join(results_dir, source, f"{model}.json")) as f:
        return json.load(f)


def fit_curves(results_dir, source, model, t, y, yerr, rng):
    """Median curve (+-1 sigma) and N_DRAWS posterior-draw curves on a grid."""
    result = load_fit(results_dir, source, model)
    meta = result["meta"]
    samples = result["samples"]
    names = list(samples.keys())
    n_samples = len(samples[names[0]])

    t_grid = np.linspace(t.min(), t.max(), N_GRID)

    # err_scale is absent from every v2 result; posterior_predictive
    # defaults it to 1.0 internally (pioran_periodicity/predict.py).
    median_params = {name: float(np.median(samples[name])) for name in names}
    med_mu, med_sd = posterior_predictive(
        meta, median_params, t, y, yerr, t_grid, need_std=True
    )

    idx = rng.choice(n_samples, size=N_DRAWS, replace=False)
    draw_curves = []
    for i in idx:
        p = {name: float(samples[name][i]) for name in names}
        mu_i, _ = posterior_predictive(meta, p, t, y, yerr, t_grid, need_std=False)
        draw_curves.append(mu_i)

    # Sine coefficients were renamed A1/A2 -> A_cos/A_sin on 2026-09-03
    # (fix MB2). Result files written before then use the old keys, and
    # single-band results from before the rename remain scientifically
    # valid, so both spellings must load.
    cos_key, sin_key = ("A_cos", "A_sin") if "A_cos" in samples else ("A1", "A2")

    sine_curves = None
    sine_median = None
    if "sine" in meta["variant"]:
        sine_curves = [
            sine_mean(
                t_grid,
                float(samples[cos_key][i]),
                float(samples[sin_key][i]),
                float(samples["period"][i]),
            )
            for i in idx
        ]
        sine_median = sine_mean(
            t_grid,
            median_params[cos_key],
            median_params[sin_key],
            median_params["period"],
        )

    return (
        t_grid,
        med_mu,
        med_sd,
        draw_curves,
        sine_curves,
        sine_median,
        result["logz"],
        result["logzerr"],
    )


def plot_panel(
    ax,
    t,
    y,
    yerr,
    t_grid,
    med_mu,
    med_sd,
    draw_curves,
    sine_curves,
    sine_median,
    logz,
    logzerr,
    title,
):
    ax.fill_between(
        t_grid,
        med_mu - med_sd,
        med_mu + med_sd,
        color="gold",
        alpha=0.45,
        zorder=1,
        label=r"posterior median $\pm1\sigma$",
    )
    for i, curve in enumerate(draw_curves):
        ax.plot(
            t_grid,
            curve,
            color="tab:blue",
            lw=0.8,
            alpha=0.5,
            zorder=2,
            label="posterior draws (GP + mean)" if i == 0 else None,
        )
    if sine_curves is not None:
        for i, curve in enumerate(sine_curves):
            ax.plot(
                t_grid,
                curve,
                color="tab:red",
                lw=0.8,
                alpha=0.5,
                zorder=3,
                label="periodic mean draws" if i == 0 else None,
            )
        ax.plot(
            t_grid,
            sine_median,
            color="darkred",
            lw=1.6,
            zorder=4,
            label="periodic mean (median)",
        )
    ax.plot(t_grid, med_mu, color="k", lw=1.3, zorder=5, label="posterior median")
    ax.errorbar(
        t,
        y,
        yerr=yerr,
        fmt=".",
        color="black",
        ms=4,
        elinewidth=0.6,
        capsize=0,
        zorder=6,
        label="data",
    )
    ax.set_title(f"{title}   ($\\log Z = {logz:.2f}\\pm{logzerr:.2f}$)", fontsize=11)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper right", ncol=2, framealpha=0.85)


def make_figure(results_dir, out_dir, source, group, cfg, rng):
    t, y, yerr = cfg["loader"]()
    models = PANEL_GROUPS[group]

    fig, axes = plt.subplots(len(models), 1, figsize=(11, 8), sharex=True)
    for ax, model in zip(axes, models):
        t_grid, med_mu, med_sd, draws, sine_draws, sine_median, logz, logzerr = (
            fit_curves(results_dir, source, model, t, y, yerr, rng)
        )
        plot_panel(
            ax,
            t,
            y,
            yerr,
            t_grid,
            med_mu,
            med_sd,
            draws,
            sine_draws,
            sine_median,
            logz,
            logzerr,
            PANEL_TITLES[model],
        )
        ax.set_ylabel(cfg["ylabel"], fontsize=10)
    axes[-1].set_xlabel("Time (years)", fontsize=11)
    fig.suptitle(
        f"{cfg['title']} -- {group.upper()} family "
        f"(pioran_periodicity v{pp.__version__})",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{source}_{group}_panels.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--data-dir",
        required=True,
        help="directory containing graham2015data.csv and " "PG1553_113_logbase.txt",
    )
    ap.add_argument(
        "--results-dir",
        required=True,
        help="directory containing <source>/<model>.json FitResults",
    )
    ap.add_argument("--out-dir", required=True, help="output directory for PNGs")
    ap.add_argument(
        "--source",
        choices=["PG1302", "PG1553"],
        default=None,
        help="only this source (default: all)",
    )
    ap.add_argument(
        "--group",
        choices=list(PANEL_GROUPS),
        default=None,
        help="only this panel group (default: all)",
    )
    ap.add_argument(
        "--force", action="store_true", help="regenerate even if the PNG already exists"
    )
    args = ap.parse_args()

    all_sources = sources(args.data_dir)
    wanted_sources = [args.source] if args.source else list(all_sources)
    groups = [args.group] if args.group else list(PANEL_GROUPS)

    for source in wanted_sources:
        cfg = all_sources[source]
        for group in groups:
            out_path = os.path.join(args.out_dir, f"{source}_{group}_panels.png")
            if os.path.exists(out_path) and not args.force:
                print(f"skip {out_path} (exists)")
                continue
            rng = np.random.default_rng(SEED)  # identical draw indices across panels
            make_figure(args.results_dir, args.out_dir, source, group, cfg, rng)


if __name__ == "__main__":
    main()
