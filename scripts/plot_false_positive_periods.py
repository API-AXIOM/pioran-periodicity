"""Histogram of posterior-median periods for DRW false positives.

A "DRW false positive" is a light curve where drw_sine was preferred over
drw (log10 B < -2) despite the light curve being a pure null (no injected
periodicity) -- i.e. what period does red-noise leakage get mistaken for.
Reads each campaign's summary.json (written by analyse_campaign.py) for the
false-positive lc_ids, then the drw_sine result JSON of each for its
posterior-median period.

    conda run -n periodicity313 python scripts/plot_false_positive_periods.py \
        --out-root ~/work/data/quasar_cadences/results
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pioran_periodicity.visualization import apply_style, parse_key, save_figure

DATA_ROOT = os.path.expanduser("~/work/data/quasar_cadences/v2")
# period prior is LogUniform(0.2, 9.0) yr for every v2 campaign; linear bins
# (axis is linear) over that range
PERIOD_BINS = np.linspace(0.2, 9.0, 25)
# bendfreq = 0.35/yr for every v2 null campaign (see [[bend-frequency-decision]]
# in project memory); marks the PSD's characteristic timescale on each panel
BEND_TIMESCALE_YR = 1.0 / 0.35


def _median_periods(results_dir: str, lc_ids: list[int]) -> np.ndarray:
    """Posterior-median period for each lc_id's drw_sine fit. (len(lc_ids),)"""
    periods = []
    for lc_id in lc_ids:
        with open(os.path.join(results_dir, f"{lc_id}_drw_sine.json")) as f:
            samples = json.load(f)["samples"]
        periods.append(float(np.median(samples["period"])))
    return np.asarray(periods)


def _drw_false_positives(summary_path: str) -> dict[str, list[int]]:
    """{cell_key: [lc_id, ...]} for every cell with a DRW log10B < -2 fit."""
    with open(summary_path) as f:
        table = json.load(f)["table"]
    out = {}
    for cell, pairs in table.items():
        drw = pairs.get("DRW")
        if drw is None:
            continue
        fps = [
            lc_id
            for lc_id, bf in zip(drw["lc_ids"], drw["log10_BF_values"])
            if bf < -2
        ]
        if fps:
            out[cell] = fps
    return out


def plot_lsst_single(out_root: str) -> None:
    results_dir = os.path.join(DATA_ROOT, "lsst_single", "results")
    fps = _drw_false_positives(os.path.join(out_root, "lsst_single", "summary.json"))
    cells = sorted(fps, key=lambda c: parse_key(c)["highalpha"])
    if not cells:
        print("lsst_single: no DRW false positives, skipping")
        return

    apply_style()
    fig, axes = plt.subplots(1, len(cells), figsize=(4 * len(cells), 4), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, cell in zip(axes, cells):
        periods = _median_periods(results_dir, fps[cell])
        ax.hist(periods, bins=PERIOD_BINS, color="tab:blue", edgecolor="white")
        ax.axvline(BEND_TIMESCALE_YR, color="gray", ls="--", lw=1.2,
                   label="1/bendfreq" if ax is axes[0] else None)
        ax.set_title(f"{cell}\n(n={len(periods)})", fontsize=10)
        ax.set_xlabel("posterior median period (yr)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("count")
    axes[0].legend(fontsize=8)
    fig.suptitle("LSST single: DRW false-positive periods, by slope", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    out_path = os.path.join(out_root, "lsst_single", "drw_false_positive_periods.png")
    save_figure(fig, out_path)
    print(f"wrote {out_path}")


def plot_synthetic(out_root: str) -> None:
    results_dir = os.path.join(DATA_ROOT, "synthetic", "results")
    fps = _drw_false_positives(os.path.join(out_root, "synthetic", "summary.json"))
    if not fps:
        print("synthetic: no DRW false positives, skipping")
        return

    snr_values = sorted({parse_key(c)["target_snr"] for c in fps})
    highalpha_values = sorted({parse_key(c)["highalpha"] for c in fps})
    colors = plt.cm.viridis(np.linspace(0, 1, len(highalpha_values)))
    color_by_alpha = dict(zip(highalpha_values, colors))

    apply_style()
    fig, axes = plt.subplots(1, len(snr_values), figsize=(4.5 * len(snr_values), 4), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, snr in zip(axes, snr_values):
        cells_here = [c for c in fps if parse_key(c)["target_snr"] == snr]
        for cell in sorted(cells_here, key=lambda c: parse_key(c)["highalpha"]):
            ha = parse_key(cell)["highalpha"]
            periods = _median_periods(results_dir, fps[cell])
            ax.hist(
                periods, bins=PERIOD_BINS, histtype="step", lw=1.8,
                color=color_by_alpha[ha], label=f"highalpha={ha:g} (n={len(periods)})",
            )
        ax.axvline(BEND_TIMESCALE_YR, color="gray", ls="--", lw=1.2,
                   label="1/bendfreq" if ax is axes[0] else None)
        ax.set_title(f"target_snr={snr:g}", fontsize=10)
        ax.set_xlabel("posterior median period (yr)")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("count")
    fig.suptitle("Synthetic: DRW false-positive periods, by SNR and slope", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    out_path = os.path.join(out_root, "synthetic", "drw_false_positive_periods.png")
    save_figure(fig, out_path)
    print(f"wrote {out_path}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-root", default=os.path.expanduser("~/work/data/quasar_cadences/results"))
    args = ap.parse_args()
    plot_lsst_single(args.out_root)
    plot_synthetic(args.out_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
