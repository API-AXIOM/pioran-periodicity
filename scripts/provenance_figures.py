"""Regenerate the data-provenance figures for the v2 report set.

No code survives for these figures in either repo -- they were made ad hoc in
2026-08-14 (see comparison_reports/real_cadence_report.tex for what they show)
and the simulator has changed since (in particular the LSST noise model,
pioran_periodicity/cadence.py's Ivezic et al. 2019 form, replaced an earlier
fractional-flux approximation on 2026-09-something -- see that function's
docstring for the old model's error). This script recomputes the
model-implied halves from the current code; the measured CSVs
(ztf_real_magerr.csv, real_lsst_alert_photometry.csv) are real photometry and
are reused as-is.

    conda run -n periodicity313 python scripts/provenance_figures.py \
        --out-root ~/work/data/quasar_cadences/results
"""

from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pioran_periodicity.cadence import lsst_magnitude_error
from pioran_periodicity.visualization import apply_style, save_figure

SUMMARIES = os.path.expanduser("~/work/data/quasar_cadences/summaries")
ZTF_MAGERR_CSV = os.path.join(SUMMARIES, "ztf_real_magerr.csv")
LSST_ALERT_CSV = os.path.join(SUMMARIES, "real_lsst_alert_photometry.csv")

MAG_GRID = np.linspace(17.0, 25.0, 200)
LSST_BANDS = ["u", "g", "r", "i", "z", "y"]
# representative LSST WFD depths (median fiveSigmaDepth per band from the
# cadence library), used to evaluate the model-implied error curve
LSST_MEDIAN_DEPTH = {
    "u": 23.29, "g": 24.54, "r": 23.93, "i": 23.45, "z": 22.94, "y": 21.94,
}


def _lsst_implied_curve(band: str) -> np.ndarray:
    """Model-implied LSST magnitude error vs magnitude, at that band's
    median survey depth. (n_grid,) -- one point per MAG_GRID entry."""
    depth = np.full_like(MAG_GRID, LSST_MEDIAN_DEPTH[band])
    return lsst_magnitude_error(MAG_GRID, depth, band=band)


def _binned_median(mag: np.ndarray, err: np.ndarray, edges: np.ndarray):
    """Median err per mag bin, plus 10th/90th percentile shading bounds."""
    idx = np.digitize(mag, edges) - 1
    med = np.full(len(edges) - 1, np.nan)
    lo = np.full(len(edges) - 1, np.nan)
    hi = np.full(len(edges) - 1, np.nan)
    for i in range(len(edges) - 1):
        vals = err[idx == i]
        if len(vals) >= 5:
            med[i] = np.median(vals)
            lo[i] = np.percentile(vals, 10)
            hi[i] = np.percentile(vals, 90)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, med, lo, hi


def plot_photometric_error_comparison(ztf: pd.DataFrame) -> plt.Figure:
    """ZTF real measured error vs LSST model-implied error, pooled across
    bands. Two-panel: (left) ZTF, (right) LSST model curves per band."""
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    edges = np.linspace(15.5, 21.5, 25)
    c, med, lo, hi = _binned_median(ztf["mag"].to_numpy(), ztf["magerr"].to_numpy(), edges)
    ax = axes[0]
    ax.fill_between(c, lo, hi, alpha=0.25, color="tab:blue")
    ax.plot(c, med, color="tab:blue", lw=1.6)
    ax.set_xlabel("ZTF magnitude")
    ax.set_ylabel("magerr (measured)")
    ax.set_title("ZTF: real measured error")
    ax.grid(alpha=0.3)

    ax = axes[1]
    for band in LSST_BANDS:
        ax.plot(MAG_GRID, _lsst_implied_curve(band), lw=1.4, label=band)
    ax.set_xlabel("LSST magnitude")
    ax.set_ylabel("magerr (model-implied, Ivezic+2019)")
    ax.set_title("LSST: simulator's noise model")
    ax.legend(fontsize=8, ncol=3)
    ax.grid(alpha=0.3)

    fig.suptitle("Photometric error vs. magnitude: simulation inputs", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return fig


def plot_photometric_error_comparison_threeway(
    ztf: pd.DataFrame, lsst_alert: pd.DataFrame
) -> plt.Figure:
    """Adds real LSST alert-stream photometry (external validation) to the
    ZTF-vs-LSST-model comparison, pooled across bands."""
    apply_style()
    fig, ax = plt.subplots(figsize=(7, 5))

    edges = np.linspace(15.5, 21.5, 25)
    c, med, lo, hi = _binned_median(ztf["mag"].to_numpy(), ztf["magerr"].to_numpy(), edges)
    ax.fill_between(c, lo, hi, alpha=0.2, color="tab:blue")
    ax.plot(c, med, color="tab:blue", lw=1.6, label="ZTF real measured")

    # model-implied curve, pooled: median across bands at each mag
    pooled_model = np.median(
        [_lsst_implied_curve(b) for b in LSST_BANDS], axis=0
    )
    ax.plot(MAG_GRID, pooled_model, color="tab:orange", lw=1.6,
             label="LSST model-implied (Ivezic+2019)")

    edges2 = np.linspace(17.0, 25.0, 33)
    mag_a = lsst_alert["mag"].to_numpy()
    err_a = lsst_alert["magerr"].to_numpy()
    c2, med2, lo2, hi2 = _binned_median(mag_a, err_a, edges2)
    ax.fill_between(c2, lo2, hi2, alpha=0.2, color="tab:green")
    ax.plot(c2, med2, color="tab:green", lw=1.6, label="LSST real alert-stream (ALeRCE)")

    ax.set_xlabel("magnitude")
    ax.set_ylabel("magerr, median with 10-90th pctile shading")
    ax.set_title("Photometric error: simulation inputs vs. external validation")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_photometric_error_by_band(ztf: pd.DataFrame, lsst_alert: pd.DataFrame) -> plt.Figure:
    """Same threeway comparison, split by band: one panel per LSST filter,
    each overlaid with the pooled ZTF curve for reference."""
    apply_style()
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True, sharey=True)

    edges = np.linspace(15.5, 21.5, 25)
    c, med, _, _ = _binned_median(ztf["mag"].to_numpy(), ztf["magerr"].to_numpy(), edges)

    for ax, band in zip(axes.flat, LSST_BANDS):
        ax.plot(c, med, color="tab:blue", lw=1.0, alpha=0.6, label="ZTF (pooled)")
        ax.plot(MAG_GRID, _lsst_implied_curve(band), color="tab:orange", lw=1.6,
                 label="LSST model")
        sub = lsst_alert[lsst_alert["band_name"] == band]
        if len(sub) >= 20:
            edges2 = np.linspace(17.0, 25.0, 25)
            c2, med2, lo2, hi2 = _binned_median(
                sub["mag"].to_numpy(), sub["magerr"].to_numpy(), edges2
            )
            ax.fill_between(c2, lo2, hi2, alpha=0.2, color="tab:green")
            ax.plot(c2, med2, color="tab:green", lw=1.4, label="LSST real (ALeRCE)")
        n_alert = len(sub)
        flag = "  [u/y least trustworthy]" if band in ("u", "y") else ""
        ax.set_title(f"{band}-band (n_real={n_alert}){flag}", fontsize=10)
        ax.grid(alpha=0.3)
        if ax is axes.flat[0]:
            ax.legend(fontsize=7)

    for ax in axes[-1]:
        ax.set_xlabel("magnitude")
    for ax in axes[:, 0]:
        ax.set_ylabel("magerr")
    fig.suptitle("Photometric error by band", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def plot_photometric_error_ratio_by_band(lsst_alert: pd.DataFrame) -> plt.Figure:
    """Ratio of real (alert-stream) to synthetic (model-implied) median
    error, per band and magnitude bin. >1: real noisier than modeled."""
    apply_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    edges = np.linspace(17.0, 25.0, 25)
    centers = 0.5 * (edges[:-1] + edges[1:])
    for band in LSST_BANDS:
        sub = lsst_alert[lsst_alert["band_name"] == band]
        if len(sub) < 20:
            continue
        _, med_real, _, _ = _binned_median(sub["mag"].to_numpy(), sub["magerr"].to_numpy(), edges)
        depth = np.full(len(centers), LSST_MEDIAN_DEPTH[band])
        model = lsst_magnitude_error(centers, depth, band=band)
        ratio = med_real / model
        ax.plot(centers, ratio, marker="o", ms=3, lw=1.2, label=band)

    ax.axhline(1.0, color="gray", lw=1, ls="--")
    ax.set_xlabel("magnitude")
    ax.set_ylabel("real / model-implied median error")
    ax.set_title("Photometric error ratio: real alert-stream / simulator model")
    ax.legend(fontsize=8, ncol=3)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def _spectral_window(t: np.ndarray, freq: np.ndarray) -> np.ndarray:
    """Spectral window |sum_j exp(-2pi i f t_j)|^2 / N^2 -- the standard
    definition (NOT a Lomb-Scargle periodogram of constant data, which is
    undefined: LombScargle divides by var(y), and constant y has var=0).
    ``t``: (n,), ``freq``: (n_freq,). Returns (n_freq,)."""
    phase = -2j * np.pi * np.outer(freq, t)  # (n_freq, n)
    return np.abs(np.exp(phase).sum(axis=1)) ** 2 / len(t) ** 2


def plot_window_function_examples() -> plt.Figure:
    """Spectral window (unit-weight sampling times) for four representative
    real cadences -- two ZTF, two LSST -- WITH nightly binning applied,
    since simulate.bin_nightly entered the simulator after the original
    figure was made."""
    from pioran_periodicity.cadence import CadenceLibrary
    from pioran_periodicity.simulate import bin_nightly

    apply_style()
    lib_root = os.path.expanduser("~/work/data/quasar_cadences/cadence_library")
    lib = CadenceLibrary.from_cache(lib_root)

    examples = []
    for survey in ("ztf", "lsst"):
        ids = lib.object_ids(survey)
        rng = np.random.default_rng(0)
        picks = rng.choice(ids, size=min(2, len(ids)), replace=False)
        for oid in picks:
            cad = lib.get(survey, oid)
            t_years = (cad["mjd"].to_numpy() - cad["mjd"].min()) / 365.25
            band = cad["band"].to_numpy() if "band" in cad else None
            t_binned, _, _, _ = bin_nightly(t_years, np.zeros_like(t_years), np.ones_like(t_years), band=band)
            examples.append((f"{survey}:{oid}", t_binned))

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, (label, t) in zip(axes.flat, examples):
        freq = np.linspace(0.02, 4.0, 3000)  # cycles/year
        power = _spectral_window(t, freq)
        ax.plot(freq, power, lw=0.8, color="tab:blue")
        ax.axvline(2.0, color="gray", ls=":", lw=1, label="1 yr alias" if ax is axes.flat[0] else None)
        ax.axvline(1.0, color="gray", ls="--", lw=1, label="half-year alias" if ax is axes.flat[0] else None)
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("frequency (cycles/yr)")
        ax.set_ylabel("window power")
        ax.grid(alpha=0.3)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle("Spectral window functions (nightly-binned cadences)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


FIGURES = {
    "photometric_error_comparison": lambda ztf, alert: plot_photometric_error_comparison(ztf),
    "photometric_error_comparison_threeway": plot_photometric_error_comparison_threeway,
    "photometric_error_by_band": plot_photometric_error_by_band,
    "photometric_error_ratio_by_band": lambda ztf, alert: plot_photometric_error_ratio_by_band(alert),
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-root", default=os.path.expanduser("~/work/data/quasar_cadences/results"))
    ap.add_argument("--only", default=None, help="comma list of figure names, default: all")
    args = ap.parse_args()

    out_dir = os.path.join(args.out_root, "provenance")
    os.makedirs(out_dir, exist_ok=True)

    ztf = pd.read_csv(ZTF_MAGERR_CSV)
    alert = pd.read_csv(LSST_ALERT_CSV)

    only = set(args.only.split(",")) if args.only else None

    for name, fn in FIGURES.items():
        if only and name not in only and "window_function_examples" not in only:
            continue
        fig = fn(ztf, alert)
        save_figure(fig, os.path.join(out_dir, f"{name}.png"))
        print(f"wrote {name}.png")

    if only is None or "window_function_examples" in only:
        fig = plot_window_function_examples()
        save_figure(fig, os.path.join(out_dir, "window_function_examples.png"))
        print("wrote window_function_examples.png")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
