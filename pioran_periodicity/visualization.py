"""Plots for scenario Bayes-factor summary JSONs (``scripts/aggregate_results.py``
output).

Reads the ``{"table": {key: {family: {n, log10_BF_mean, log10_BF_values,
outcomes}}}}`` format produced by ``aggregate_results.py``. Keys are the
``group_cols`` string it builds, e.g. ``"highalpha=-2.4"`` or ``"period=1.25,
A1=0.24"`` -- ``parse_key`` below inverts that for any number of columns, so
these functions work for whatever a scenario's config CSV sweeps, not just
the columns used in the thesis scenarios (3.4-3.7).

Six plot kinds. The first four were ported from a throwaway prototype
(``workspace/plot_bf_prototypes.py`` in the thesis-replication repo) after
picking the most illustrative subset for the paper; the last two were added
for the DRW-robustness sensitivity study (see ``paper/plot_drw_robustness.py``):

* ``plot_strip``           -- per-configuration log10 BF scatter+box, with
                               optional Kass-Raftery decision-zone shading.
* ``plot_power_curves``    -- P(detect) vs. one swept variable, one line per
                               value of a second swept variable.
* ``plot_fpr_calibration`` -- false-positive rate vs. decision threshold,
                               one line per configuration (null scenarios,
                               i.e. no true periodic signal).
* ``plot_roc``             -- TPR vs. FPR sweeping the decision threshold,
                               built from an explicit null/alternative value
                               pair the caller assembles (there is no single
                               correct way to pool a null sample across
                               scenarios, so this module does not guess).
* ``plot_detection_rate``  -- P(detect) vs. one swept variable at the fixed
                               Kass-Raftery threshold, with binomial standard
                               error bars and no series dimension -- the
                               single-curve sibling of ``plot_power_curves``
                               (e.g. a false-positive-rate curve from a null
                               scenario's own table, no threshold sweep).
* ``plot_matched_roc``     -- ROC curves where the null and alternative
                               samples are drawn from *different* tables but
                               paired by a shared swept column (e.g. a
                               null_case and a signal_case table both swept
                               over the same ``highalpha`` grid) rather than
                               a single null held fixed across series.

    conda run -n <env> python -c "
    from pioran_periodicity.visualization import load_summary, plot_strip
    table = load_summary('summary_3_4.json')
    fig = plot_strip(table, 'highalpha', xlabel='high-frequency PSD slope')
    fig.savefig('strip_3_4.png', dpi=150, bbox_inches='tight')
    "
"""

from __future__ import annotations

import json
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

FAMILIES = ["DRW", "CARMA21", "OBPL"]

# dataviz reference palette: categorical slots 1/2/3 (blue/green/magenta),
# diverging blue<->red pair, neutral gray midpoint, chart chrome.
FAMILY_COLOR = {"DRW": "#2a78d6", "CARMA21": "#008300", "OBPL": "#e87ba4"}
DIV_BLUE, DIV_RED, DIV_GRAY = "#2a78d6", "#e34948", "#f0efec"
MUTED = "#898781"
GRID = "#e1e0d9"

_STYLE = {
    "axes.edgecolor": "#c3c2b7",
    "axes.labelcolor": "#0b0b0b",
    "xtick.color": "#52514e",
    "ytick.color": "#52514e",
    "grid.color": GRID,
    "font.size": 9,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
}


def apply_style() -> None:
    """Apply the shared chart style (call once, e.g. at script start)."""
    plt.rcParams.update(_STYLE)


def load_summary(path: str) -> dict:
    """Load an ``aggregate_results.py`` summary JSON and return its table."""
    with open(path) as f:
        return json.load(f)["table"]


def parse_key(key: str) -> dict[str, float]:
    """Invert an ``aggregate_results.py`` group key, e.g. ``"period=1.25,
    A1=0.24"`` -> ``{"period": 1.25, "A1": 0.24}``. ``"all"`` (no group
    columns) -> ``{}``.
    """
    if key == "all":
        return {}
    out = {}
    for part in key.split(", "):
        col, val = part.split("=")
        out[col] = float(val)
    return out


def sweep_values(table: dict, col: str) -> list[float]:
    """Sorted distinct values a config column takes across ``table`` keys."""
    return sorted({parse_key(k)[col] for k in table})


def key_for(table: dict, **cols: float) -> str:
    """The single table key whose parsed columns match ``cols`` exactly."""
    matches = [
        k for k in table if all(parse_key(k).get(c) == v for c, v in cols.items())
    ]
    if len(matches) != 1:
        raise KeyError(f"expected exactly one key matching {cols}, found {matches}")
    return matches[0]


def filter_table(table: dict, predicate) -> dict:
    """Subset a summary table to keys whose parsed columns satisfy
    ``predicate(parsed_key: dict[str, float]) -> bool``. Useful for pinning
    one swept column (e.g. an amplitude tier that isn't the same value
    across every other group column) before handing the result to a plot
    function that expects a simpler grid.
    """
    return {k: v for k, v in table.items() if predicate(parse_key(k))}


def roc_from_bf(
    null_vals: np.ndarray, alt_vals: np.ndarray, n_thresholds: int = 300
) -> tuple[np.ndarray, np.ndarray]:
    """FPR/TPR curves sweeping a threshold on log10 BF, low is "detect".

    ``null_vals``: log10 BF from configurations with no true periodic signal.
    ``alt_vals``: log10 BF from configurations with a true periodic signal.
    """
    lo = min(null_vals.min(), alt_vals.min())
    hi = max(null_vals.max(), alt_vals.max())
    thresholds = np.linspace(lo, hi, n_thresholds)
    fpr = np.array([np.mean(null_vals < t) for t in thresholds])
    tpr = np.array([np.mean(alt_vals < t) for t in thresholds])
    return fpr, tpr


def _family_panels(families, figsize, sharey=True, sharex=False):
    fig, axes = plt.subplots(
        1, len(families), figsize=figsize, sharey=sharey, sharex=sharex
    )
    if len(families) == 1:
        axes = [axes]
    return fig, axes


def plot_strip(
    table: dict,
    group_col: str,
    xlabel: str | None = None,
    families: Sequence[str] = FAMILIES,
    family_colors: dict = FAMILY_COLOR,
    shade_zones: bool = True,
    thresholds: tuple[float, float] = (-2.0, 2.0),
    title: str | None = None,
    seed: int = 0,
):
    """Per-configuration scatter + boxplot of log10 BF, one panel per family.

    ``thresholds`` are the Kass-Raftery decision boundaries (log10 B < lo ->
    "detect", > hi -> "refute"); shaded when ``shade_zones`` is True.
    """
    lo, hi = thresholds
    xs = sweep_values(table, group_col)
    fig, axes = _family_panels(families, figsize=(11, 3.2))
    rng = np.random.default_rng(seed)

    all_vals = np.concatenate(
        [table[k][fam]["log10_BF_values"] for k in table for fam in families]
    )
    pad = 0.08 * (all_vals.max() - all_vals.min())
    ylim = (all_vals.min() - pad, all_vals.max() + pad)

    for ax, fam in zip(axes, families):
        box_data = []
        for i, x in enumerate(xs):
            key = key_for(table, **{group_col: x})
            vals = np.array(table[key][fam]["log10_BF_values"])
            box_data.append(vals)
            jitter = rng.uniform(-0.15, 0.15, size=len(vals))
            ax.scatter(
                np.full(len(vals), i) + jitter,
                vals,
                s=6,
                color=family_colors[fam],
                alpha=0.35,
                linewidths=0,
            )
        bp = ax.boxplot(
            box_data,
            positions=range(len(xs)),
            widths=0.5,
            showfliers=False,
            patch_artist=True,
        )
        for patch in bp["boxes"]:
            patch.set_facecolor("white")
            patch.set_edgecolor("#0b0b0b")
            patch.set_linewidth(0.9)
        for med in bp["medians"]:
            med.set_color("#0b0b0b")
        ax.set_xticks(range(len(xs)))
        ax.set_xticklabels([f"{x:g}" for x in xs], fontsize=8)
        ax.set_title(fam, color=family_colors[fam], fontsize=10)
        ax.grid(axis="y", lw=0.5)
        ax.axhline(0, color=MUTED, lw=0.6)
        if shade_zones:
            ax.set_ylim(*ylim)
            if ylim[0] < lo:
                ax.axhspan(ylim[0], lo, color=DIV_RED, alpha=0.08, zorder=0)
                ax.axhline(lo, color=DIV_RED, ls="--", lw=0.8, alpha=0.6)
            if ylim[1] > hi:
                ax.axhspan(hi, ylim[1], color=DIV_BLUE, alpha=0.08, zorder=0)
                ax.axhline(hi, color=DIV_BLUE, ls="--", lw=0.8, alpha=0.6)
    axes[0].set_ylabel(r"$\log_{10} B$ (null vs. sine)")
    fig.supxlabel(xlabel or group_col, fontsize=9)
    fig.suptitle(title or f"raw log10 BF vs. {group_col}", fontsize=10)
    return fig


def plot_power_curves(
    table: dict,
    x_col: str,
    series_col: str,
    families: Sequence[str] = FAMILIES,
    family_colors: dict = FAMILY_COLOR,
    x_label: str | None = None,
    series_label: str | None = None,
    title: str | None = None,
    cmap_name: str = "Blues",
    invert_xaxis: bool = False,
):
    """P(detect) vs. ``x_col``, one line per value of ``series_col``.

    ``x_col``'s values are plotted in ascending numeric order by default;
    set ``invert_xaxis=True`` when descending reads more naturally (e.g. a
    slope that "steepens" as it becomes more negative).
    """
    x_vals = sweep_values(table, x_col)
    series_vals = sweep_values(table, series_col)
    cmap = plt.get_cmap(cmap_name)
    fig, axes = _family_panels(families, figsize=(11, 3.4))
    for ax, fam in zip(axes, families):
        for i, s in enumerate(series_vals):
            rates = []
            for x in x_vals:
                key = key_for(table, **{x_col: x, series_col: s})
                stats = table[key][fam]
                rates.append(stats["outcomes"]["detect"] / stats["n"])
            color = cmap(0.3 + 0.6 * i / max(len(series_vals) - 1, 1))
            ax.plot(
                x_vals, rates, marker="o", ms=3.5, lw=1.3, color=color, label=f"{s:g}"
            )
        ax.set_xlabel(x_label or x_col)
        ax.set_title(fam, color=family_colors[fam], fontsize=10)
        ax.grid(lw=0.5)
        ax.set_ylim(-0.02, 1.02)
        if invert_xaxis:
            ax.invert_xaxis()
    axes[0].set_ylabel("P(detect)")
    axes[-1].legend(
        title=series_label or series_col,
        frameon=False,
        fontsize=6.5,
        title_fontsize=7,
        loc="lower right",
    )
    fig.suptitle(title or f"detection power vs. {x_col} by {series_col}", fontsize=10)
    return fig


def plot_fpr_calibration(
    table: dict,
    group_col: str,
    families: Sequence[str] = FAMILIES,
    family_colors: dict = FAMILY_COLOR,
    xlabel: str | None = None,
    threshold_line: float = -2.0,
    title: str | None = None,
    cmap_name: str = "Blues",
    n_thresholds: int = 300,
):
    """False-positive rate vs. decision threshold, one line per configuration.

    Intended for scenarios with no true periodic signal (the "null" table);
    every log10 BF below the threshold is a false "detect".
    """
    xs = sweep_values(table, group_col)
    cmap = plt.get_cmap(cmap_name)
    fig, axes = _family_panels(families, figsize=(11, 3.4), sharex=True)
    for ax, fam in zip(axes, families):
        allvals = np.concatenate([table[k][fam]["log10_BF_values"] for k in table])
        thr = np.linspace(allvals.min(), allvals.max(), n_thresholds)
        for i, x in enumerate(xs):
            key = key_for(table, **{group_col: x})
            vals = np.array(table[key][fam]["log10_BF_values"])
            fpr = np.array([np.mean(vals < t) for t in thr])
            color = cmap(0.3 + 0.6 * i / max(len(xs) - 1, 1))
            ax.plot(thr, fpr, color=color, lw=1.3, label=f"{x:g}")
        ax.axvline(threshold_line, color=DIV_RED, ls="--", lw=0.8, alpha=0.6)
        ax.set_title(fam, color=family_colors[fam], fontsize=10)
        ax.set_xlabel(r"threshold on $\log_{10} B$")
        ax.grid(lw=0.5)
    axes[0].set_ylabel("false-positive rate (no true signal)")
    axes[-1].legend(
        title=xlabel or group_col,
        frameon=False,
        fontsize=6.5,
        title_fontsize=7,
        loc="upper left",
    )
    fig.suptitle(title or f"false-positive calibration vs. {group_col}", fontsize=10)
    return fig


def plot_roc(
    null_values: dict[str, np.ndarray],
    table_alt: dict,
    series_col: str,
    fixed_col: str,
    fixed_value: float,
    families: Sequence[str] = FAMILIES,
    family_colors: dict = FAMILY_COLOR,
    series_label: str | None = None,
    null_label: str = "null",
    title: str | None = None,
    cmap_name: str = "Blues",
    n_thresholds: int = 300,
):
    """ROC curves (TPR vs. FPR, threshold-swept) per family, one line per
    value of ``series_col`` in the alternative-hypothesis table, holding
    ``fixed_col`` at ``fixed_value``.

    ``null_values``: ``{family: array of log10 BF}`` from configurations
    with no true signal, already selected/pooled by the caller -- e.g. the
    single matching null configuration, or an explicit, documented pooling
    across nearby null configurations if the null and alternative scenarios
    don't share a sweep grid. This function does not choose that pooling
    for you.
    """
    series_vals = sweep_values(table_alt, series_col)
    cmap = plt.get_cmap(cmap_name)
    fig, axes = _family_panels(families, figsize=(11, 3.8), sharex=True, sharey=True)
    for ax, fam in zip(axes, families):
        null_vals = null_values[fam]
        for i, s in enumerate(series_vals):
            key = key_for(table_alt, **{series_col: s, fixed_col: fixed_value})
            alt_vals = np.array(table_alt[key][fam]["log10_BF_values"])
            fpr, tpr = roc_from_bf(null_vals, alt_vals, n_thresholds=n_thresholds)
            color = cmap(0.3 + 0.6 * i / max(len(series_vals) - 1, 1))
            ax.plot(fpr, tpr, color=color, lw=1.3, label=f"{s:g}")
        ax.plot([0, 1], [0, 1], color=MUTED, lw=0.8, ls="--")
        ax.set_title(fam, color=family_colors[fam], fontsize=10)
        ax.set_xlabel(f"FPR ({null_label})", fontsize=8)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.grid(lw=0.5)
    axes[0].set_ylabel(f"TPR ({fixed_col}={fixed_value:g})")
    axes[-1].legend(
        title=series_label or series_col,
        frameon=False,
        fontsize=6.5,
        title_fontsize=7,
        loc="lower right",
    )
    fig.suptitle(
        title
        or f"ROC: {null_label} vs. {series_col} sweep at {fixed_col}="
        f"{fixed_value:g}",
        fontsize=9.5,
    )
    return fig


def plot_detection_rate(
    table: dict,
    x_col: str,
    families: Sequence[str] = FAMILIES,
    family_colors: dict = FAMILY_COLOR,
    x_label: str | None = None,
    ylabel: str = "P(detect)",
    title: str | None = None,
    color: str | None = None,
    invert_xaxis: bool = False,
):
    """P(detect) vs. ``x_col`` at the table's own Kass-Raftery threshold,
    one panel per family, with binomial standard-error bars. No series
    dimension -- use ``plot_power_curves`` when a second swept variable
    should become separate lines.

    ``x_col``'s values are plotted in ascending numeric order by default;
    set ``invert_xaxis=True`` when descending reads more naturally (e.g. a
    slope that "steepens" as it becomes more negative).
    """
    x_vals = sweep_values(table, x_col)
    fig, axes = _family_panels(families, figsize=(11, 3.2))
    for ax, fam in zip(axes, families):
        rates, errs = [], []
        for x in x_vals:
            key = key_for(table, **{x_col: x})
            stats = table[key][fam]
            n = stats["n"]
            p = stats["outcomes"]["detect"] / n
            rates.append(p)
            errs.append(np.sqrt(p * (1 - p) / n) if n > 0 else 0.0)
        ax.errorbar(
            x_vals,
            rates,
            yerr=errs,
            marker="o",
            ms=4,
            lw=1.6,
            capsize=3,
            color=color or family_colors[fam],
        )
        ax.set_xlabel(x_label or x_col)
        ax.set_title(fam, color=family_colors[fam], fontsize=10)
        ax.grid(lw=0.5)
        ax.set_ylim(-0.02, 1.02)
        if invert_xaxis:
            ax.invert_xaxis()
    axes[0].set_ylabel(ylabel)
    fig.suptitle(title or f"{ylabel} vs. {x_col}", fontsize=10)
    return fig


def plot_matched_roc(
    null_table: dict,
    alt_table: dict,
    match_col: str,
    match_values: Sequence[float],
    families: Sequence[str] = FAMILIES,
    family_colors: dict = FAMILY_COLOR,
    alt_filter: dict[str, float] | None = None,
    match_label: str | None = None,
    title: str | None = None,
    cmap_name: str = "Blues",
    n_thresholds: int = 300,
):
    """ROC curves (TPR vs. FPR, threshold-swept) pairing a null and an
    alternative table by a shared swept column, one line per value in
    ``match_values``.

    Unlike ``plot_roc`` (one null curve reused across several alternative
    series), here *both* the null and the alternative sample change
    together at each ``match_col`` value -- e.g. a null_case table and a
    signal_case table both swept over the same ``highalpha`` grid, so the
    false-positive baseline is always the matching-slope null rather than a
    single pooled one.

    ``alt_filter``: optional ``{col: value}`` constraints applied to
    ``alt_table`` keys in addition to ``match_col`` (e.g. pin an amplitude
    tier); any other columns in ``alt_table``'s keys are pooled together.
    """
    cmap = plt.get_cmap(cmap_name)
    fig, axes = _family_panels(families, figsize=(11, 3.8), sharex=True, sharey=True)
    for ax, fam in zip(axes, families):
        for i, m in enumerate(match_values):
            null_key = key_for(null_table, **{match_col: m})
            null_vals = np.array(null_table[null_key][fam]["log10_BF_values"])
            constraints = {match_col: m, **(alt_filter or {})}
            alt_keys = [
                k
                for k in alt_table
                if all(parse_key(k).get(c) == v for c, v in constraints.items())
            ]
            alt_vals = np.concatenate(
                [np.array(alt_table[k][fam]["log10_BF_values"]) for k in alt_keys]
            )
            fpr, tpr = roc_from_bf(null_vals, alt_vals, n_thresholds=n_thresholds)
            color = cmap(0.3 + 0.6 * i / max(len(match_values) - 1, 1))
            ax.plot(fpr, tpr, color=color, lw=1.3, label=f"{m:g}")
        ax.plot([0, 1], [0, 1], color=MUTED, lw=0.8, ls="--")
        ax.set_title(fam, color=family_colors[fam], fontsize=10)
        ax.set_xlabel("FPR", fontsize=8)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.grid(lw=0.5)
    axes[0].set_ylabel("TPR")
    axes[-1].legend(
        title=match_label or match_col,
        frameon=False,
        fontsize=6.5,
        title_fontsize=7,
        loc="lower right",
    )
    fig.suptitle(title or f"ROC, matched by {match_col}", fontsize=9.5)
    return fig


def save_figure(fig, path: str, dpi: int = 150) -> None:
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
