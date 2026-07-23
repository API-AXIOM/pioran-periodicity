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
import pioranpy as pa

import pioran_periodicity as pp
from pioran_periodicity.kernels import (  # noqa: E501
    FrequencyBand,
    drw_kernel,
    obpl_kernel,
)
from pioran_periodicity.means import sine_mean

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


def build_kernel_and_mean(model_meta, params, band=None):
    """Reconstruct the kernel and mean function for one parameter draw."""
    noise = model_meta["noise"]
    variant = model_meta["variant"]

    if noise == "drw":
        kernel = drw_kernel(params["log10_variance"], params["log10_fbend"])
    elif noise == "obpl":
        kernel = obpl_kernel(
            params["log10_variance"],
            params["alpha_low"],
            params["log10_fbend"],
            params["alpha_high"],
            band,
            n_components=model_meta["n_components"],
            basis_function=model_meta["basis_function"],
            check_density=False,
        )
    else:
        raise ValueError(noise)

    if "sine" in variant:

        def mean_func(t, p=params):
            return sine_mean(t, p["A1"], p["A2"], p["period"])

    else:
        mean_func = None
    return kernel, mean_func


def predict(kernel, mean_func, t, y, yerr, err_scale, t_grid, need_std=False):
    """GP posterior-predictive mean (and, optionally, std) on t_grid.

    ``need_std=False`` skips the O(N_grid^2) covariance-matrix computation
    (only needed once, for the median curve's band).
    """
    sigma2 = (err_scale * yerr) ** 2
    y_resid = y - (mean_func(t) if mean_func is not None else 0.0)
    gp = pa.ScalableGP(0.0, kernel)
    gp_cond = gp(t, sigma2)
    fp = pa.posterior(gp_cond, y_resid)
    fp_grid = fp(t_grid)
    mu = np.asarray(pa.mean(fp_grid))
    sd = None
    if need_std:
        sd = np.sqrt(np.diag(np.asarray(pa.cov(fp_grid))))
    if mean_func is not None:
        mu = mu + mean_func(t_grid)
    return mu, sd


def fit_curves(results_dir, source, model, t, y, yerr, rng):
    """Median curve (+-1 sigma) and N_DRAWS posterior-draw curves on a grid."""
    result = load_fit(results_dir, source, model)
    meta = result["meta"]
    samples = result["samples"]
    names = list(samples.keys())
    n_samples = len(samples[names[0]])

    band = None
    if meta.get("band") is not None:
        band = FrequencyBand(**meta["band"])

    t_grid = np.linspace(t.min(), t.max(), N_GRID)

    median_params = {name: float(np.median(samples[name])) for name in names}
    kernel, mean_func = build_kernel_and_mean(meta, median_params, band=band)
    med_mu, med_sd = predict(
        kernel, mean_func, t, y, yerr, median_params["err_scale"], t_grid, need_std=True
    )

    idx = rng.choice(n_samples, size=N_DRAWS, replace=False)
    draw_curves = []
    for i in idx:
        p = {name: float(samples[name][i]) for name in names}
        kernel_i, mean_func_i = build_kernel_and_mean(meta, p, band=band)
        mu_i, _ = predict(
            kernel_i, mean_func_i, t, y, yerr, p["err_scale"], t_grid, need_std=False
        )
        draw_curves.append(mu_i)

    sine_curves = None
    sine_median = None
    if "sine" in meta["variant"]:
        sine_curves = [
            sine_mean(
                t_grid,
                float(samples["A1"][i]),
                float(samples["A2"][i]),
                float(samples["period"][i]),
            )
            for i in idx
        ]
        sine_median = sine_mean(
            t_grid, median_params["A1"], median_params["A2"], median_params["period"]
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
