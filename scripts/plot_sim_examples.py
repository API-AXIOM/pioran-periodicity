"""Per-light-curve diagnostic figures for the v2 simulation campaign.

Two figures per selected light curve:

  <id>_panels.png : 2x2 grid, one panel per model in ``MODELS`` -- data with
                     error bars (coloured by band for multi-band light
                     curves), the GP posterior-predictive median +-1 sigma
                     band, 10 posterior-draw curves, and for the ``+sine``
                     models the periodic mean alone (draws + median),
                     mirroring ``paper/plot_fits.py``'s ``plot_panel``.
  <id>_corner.png : 2x2 grid of corner plots (one per model) plus a text box
                     giving each model's logZ +- error and both pairs'
                     log10 Bayes factor.

This module is a library of plotting functions (``load_lightcurve``,
``make_panels_figure``, ``make_corner_figure``); a later script selects
which light curve IDs to plot and drives them -- there is no CLI here.

``make_panels_figure`` reimplements the panel body locally rather than
importing ``paper/plot_fits.py``'s ``plot_panel``: that function always
draws data points in a single colour and this figure additionally needs to
colour them per band for multi-band light curves, so the two panels are not
drop-in compatible. The colour palette, curve styles and legend labels are
kept identical to ``plot_panel`` so the figures read the same way.
``make_corner_figure`` does reuse ``paper/plot_corners.py``'s ``LABELS``
dict and its zero-posterior-spread guard, since those need no adaptation.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import corner  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "paper"))
from plot_corners import LABELS  # noqa: E402

# pioran_periodicity/__init__.py imports matplotlib.pyplot before pulling in
# pioranpy (see its module docstring on the DYLD_LIBRARY_PATH/libexpat
# clash), so this import must come after matplotlib.use("Agg") + pyplot
# above, matching paper/plot_fits.py's ordering.
from pioran_periodicity.means import sine_mean  # noqa: E402
from pioran_periodicity.predict import posterior_predictive  # noqa: E402

MODELS = ("drw", "drw_sine", "obpl", "obpl_sine")

PANEL_TITLES = {
    "drw": "DRW",
    "drw_sine": "DRW + sine",
    "obpl": "OBPL",
    "obpl_sine": "OBPL + sine",
}

N_GRID = 400
N_DRAWS = 10

# Colours for per-band data points, keyed by LSST/Rubin ugrizy filter name
# (ZTF's g/r/i is a strict subset). Every LSST object in the campaign pool
# carries all six filters (283/283 objects queried from the cadence library
# have u, g, r, i, z, y; median and max band count are both 6), so fewer
# than 6 entries here would silently draw two different filters in the same
# colour. Values are 6 of the 8 Okabe-Ito colour-blind-safe categorical
# colours (Wong, Nature Methods 8:441, 2011; the community standard for
# colour-blind-safe qualitative palettes) -- chosen over the conventional
# ugrizy plotting colours (which are not colour-blind-safe) per the
# colour-blind-safe > pretty preference. Black and yellow are dropped: black
# is reserved for the single-band fallback below and for other plot
# elements (posterior median line, axis text), and yellow reads poorly
# against the "gold" posterior-band fill already used in the same panel.
BAND_COLORS = {
    "u": "#0072B2",  # blue
    "g": "#009E73",  # bluish green
    "r": "#D55E00",  # vermillion
    "i": "#CC79A7",  # reddish purple
    "z": "#E69F00",  # orange
    "y": "#56B4E9",  # sky blue
}


def load_lightcurve(lc_dir: str, lc_id: int) -> dict:
    """Load one simulated light curve's arrays from ``<lc_dir>/<lc_id>.npz``.

    Returns ``{"t", "y", "yerr", "band"}``; ``band`` is ``None`` when the
    npz has no ``band`` array (synthetic and single-band arms).
    """
    path = os.path.join(lc_dir, f"{lc_id}.npz")
    with np.load(path, allow_pickle=True) as npz:
        band = np.asarray(npz["band"]) if "band" in npz.files else None
        return {
            "t": np.asarray(npz["t"], dtype=float),
            "y": np.asarray(npz["y"], dtype=float),
            "yerr": np.asarray(npz["yerr"], dtype=float),
            "band": band,
        }


def _load_fit(results_dir: str, lc_id: int, model: str) -> dict:
    path = os.path.join(results_dir, f"{lc_id}_{model}.json")
    with open(path) as f:
        return json.load(f)


def _plot_data_points(
    ax, t: np.ndarray, y: np.ndarray, yerr: np.ndarray, band: Optional[np.ndarray]
) -> None:
    """Error-bar the data, one colour per band (black for single-band)."""
    if band is None:
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
        return
    # shapes: t, y, yerr, band all (n_points,)
    bands = sorted(set(band.tolist()))
    unknown = [b for b in bands if b not in BAND_COLORS]
    if unknown:
        raise ValueError(
            f"no colour defined for band(s) {unknown}; BAND_COLORS only "
            f"covers {sorted(BAND_COLORS)} (LSST ugrizy). Silently reusing "
            "a colour would draw two different bands identically -- extend "
            "BAND_COLORS instead."
        )
    for b in bands:
        mask = band == b  # (n_points,) boolean
        ax.errorbar(
            t[mask],
            y[mask],
            yerr=yerr[mask],
            fmt=".",
            color=BAND_COLORS[b],
            ms=4,
            elinewidth=0.6,
            capsize=0,
            zorder=6,
            label=f"data ({b})",
        )


def _plot_panel(
    ax,
    t: np.ndarray,
    y: np.ndarray,
    yerr: np.ndarray,
    band: Optional[np.ndarray],
    t_grid: np.ndarray,
    med_mu: np.ndarray,
    med_sd: np.ndarray,
    draw_curves: list,
    sine_curves: Optional[list],
    sine_median: Optional[np.ndarray],
    logz: float,
    logzerr: float,
    title: str,
) -> None:
    """One model panel: data, posterior median +-1 sigma, draws, sine mean.

    Visual vocabulary matches paper/plot_fits.py's plot_panel (same colours
    and legend labels); this version additionally colours data points by
    band (see _plot_data_points) for multi-band light curves.
    """
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
    _plot_data_points(ax, t, y, yerr, band)
    ax.set_title(f"{title}   ($\\log Z = {logz:.2f}\\pm{logzerr:.2f}$)", fontsize=11)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper right", ncol=2, framealpha=0.85)


def make_panels_figure(
    results_dir: str,
    lc_dir: str,
    lc_id: int,
    out_path: str,
    caption: str,
    rng: np.random.Generator,
) -> None:
    """2x2 grid of posterior-predictive panels, one per model in MODELS."""
    lc = load_lightcurve(lc_dir, lc_id)
    t, y, yerr, band = lc["t"], lc["y"], lc["yerr"], lc["band"]
    t_grid = np.linspace(t.min(), t.max(), N_GRID)  # (N_GRID,)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
    for ax, model in zip(axes.flat, MODELS):
        result = _load_fit(results_dir, lc_id, model)
        meta = result["meta"]
        samples = result["samples"]
        names = list(samples.keys())
        n_samples = len(samples[names[0]])

        # err_scale is absent from every v2 result; posterior_predictive
        # defaults it to 1.0 internally.
        median_params = {name: float(np.median(samples[name])) for name in names}
        med_mu, med_sd = posterior_predictive(
            meta,
            median_params,
            t,
            y,
            yerr,
            t_grid,
            band_labels=band,
            need_std=True,
        )

        n_draws = min(N_DRAWS, n_samples)
        idx = rng.choice(n_samples, size=n_draws, replace=False)
        draw_curves = []
        for i in idx:
            p = {name: float(samples[name][i]) for name in names}
            mu_i, _ = posterior_predictive(
                meta,
                p,
                t,
                y,
                yerr,
                t_grid,
                band_labels=band,
                need_std=False,
            )
            draw_curves.append(mu_i)

        # A1/A2 -> A_cos/A_sin rename (fix MB2, 2026-09-03); legacy result
        # files use the old keys.
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

        _plot_panel(
            ax,
            t,
            y,
            yerr,
            band,
            t_grid,
            med_mu,
            med_sd,
            draw_curves,
            sine_curves,
            sine_median,
            result["logz"],
            result["logzerr"],
            PANEL_TITLES[model],
        )

    for ax in axes[-1, :]:
        ax.set_xlabel("Time (years)", fontsize=11)
    for ax in axes[:, 0]:
        ax.set_ylabel("Magnitude - median", fontsize=10)
    fig.suptitle(f"light curve {lc_id}\n{caption}", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def make_corner_figure(
    results_dir: str, lc_id: int, out_path: str, caption: str
) -> None:
    """2x2 grid of corner plots (one per model) plus an evidence text box."""
    fig = plt.figure(figsize=(16, 15))
    # A dedicated bottom strip for the text box, separate from the 2x2 corner
    # grid: corner panels with more than a handful of parameters (e.g. OBPL's
    # alpha_low/alpha_high or multi-band a_b/mu_b) otherwise grow tall enough
    # to overlap a text box placed directly on `fig`.
    grid_fig, text_fig = fig.subfigures(2, 1, height_ratios=[0.92, 0.08])
    subfigs = grid_fig.subfigures(2, 2)

    logz = {}
    logzerr = {}
    for subfig, model in zip(subfigs.flat, MODELS):
        result = _load_fit(results_dir, lc_id, model)
        logz[model] = float(result["logz"])
        logzerr[model] = float(result["logzerr"])
        samples = result["samples"]
        names = list(samples.keys())
        # (n_samples, n_params)
        data = np.column_stack([np.asarray(samples[n], dtype=float) for n in names])
        labels = [LABELS.get(n, n) for n in names]

        # Drop any parameter with zero spread: corner cannot draw a
        # histogram for a delta function (mirrors paper/plot_corners.py).
        spread = data.std(axis=0)  # (n_params,)
        keep = spread > 0
        if not keep.all():
            data = data[:, keep]
            labels = [lbl for lbl, k in zip(labels, keep) if k]

        corner.corner(
            data,
            labels=labels,
            bins=40,
            show_titles=True,
            quantiles=[0.16, 0.5, 0.84],
            title_quantiles=[0.16, 0.5, 0.84],
            title_kwargs={"fontsize": 8},
            label_kwargs={"fontsize": 10},
            plot_datapoints=False,
            fill_contours=True,
            levels=(1 - np.exp(-0.5), 1 - np.exp(-2.0)),  # 1 and 2 sigma (2-D)
            color="tab:blue",
            fig=subfig,
        )
        subfig.suptitle(PANEL_TITLES[model], fontsize=12)

    log10_b_drw = (logz["drw"] - logz["drw_sine"]) / np.log(10)
    log10_b_obpl = (logz["obpl"] - logz["obpl_sine"]) / np.log(10)

    lines = [
        f"logZ  {model:<11s}= {logz[model]:.2f} +- {logzerr[model]:.2f}"
        for model in MODELS
    ]
    lines.append(
        f"log10 B (DRW)  = {log10_b_drw:<10.2f}log10 B (OBPL) = {log10_b_obpl:.2f}"
    )
    text_fig.text(
        0.02, 0.5, "\n".join(lines), fontsize=10, family="monospace", va="center"
    )
    fig.suptitle(f"light curve {lc_id}\n{caption}", fontsize=14)

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
