"""Driver: turn one v2 simulation campaign into publishable output.

For a campaign (one of the five in ``CAMPAIGNS``) this script:

1. aggregates every ``<id>_<model>.json`` FitResult into a Bayes-factor
   summary table (``aggregate_results.build_table``, ESS-gated at
   ``ESS_FLOOR``);
2. writes that table as ``summary.json`` and a flat ``fpr_table.csv``;
3. draws the strip and false-positive-calibration figures
   (``pioran_periodicity.visualization``), restricted to the ``DRW``/
   ``OBPL`` families -- these campaigns have no CARMA fits;
4. for every retained configuration cell, picks two example light curves
   (one "interesting", one "boring", see ``select_examples``) and drives
   the panels/corner figures for each (``plot_sim_examples``);
5. records every pick, its class and any fallback substitution in
   ``manifest.json``, alongside the selection seed and the ESS floor. A
   cell that ``build_table`` reports with zero usable light curves in
   every class (all fits gated out by convergence/ESS) has no examples to
   pick; that cell is skipped with a printed WARNING and recorded under
   ``manifest.json``'s ``skipped_cells`` rather than aborting the run.

Output layout under ``--out-root/<campaign>/``::

    summary.json          -- {"table": ..., "min_ess": ..., ...}
    fpr_table.csv          -- cell, pair, n, n_dropped_*, detect/inconclusive/
                              refute counts, fpr
    strip*.png             -- one per pinned value of any leading group_col
                              (e.g. one per target_snr for the synthetic arm)
    fpr_calibration*.png   -- same pinning
    examples/*_panels.png
    examples/*_corner.png
    manifest.json          -- picks, classes, fallbacks, selection_seed,
                              min_ess

``lsst_multi`` has no local results yet; ``CAMPAIGNS["lsst_multi"]`` still
carries a config entry so ``--campaign all`` picks it up automatically once
the run lands, but until then it is skipped (results_dir absent) rather
than failing the whole invocation.

    conda run -n <env> python scripts/analyse_campaign.py --campaign all
    conda run -n <env> python scripts/analyse_campaign.py --campaign ztf_single \\
        --out-root ~/work/data/quasar_cadences/results
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import zlib
from dataclasses import dataclass
from typing import Optional

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from pioran_periodicity.visualization import (  # noqa: E402
    apply_style,
    filter_table,
    parse_key,
    plot_fpr_calibration,
    plot_strip,
    save_figure,
)

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))


def _load_module(name: str, filename: str):
    """Load a sibling script module by path (it is a script, not an
    installed package, so a plain ``import`` would not find it)."""
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(_SCRIPTS_DIR, filename)
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_aggregate_results = _load_module("aggregate_results", "aggregate_results.py")
build_table = _aggregate_results.build_table

# plot_sim_examples.py lives alongside this script; a sys.path insert (the
# pattern tests/test_aggregate_results.py uses) resolves the plain import.
sys.path.insert(0, _SCRIPTS_DIR)
from plot_sim_examples import make_corner_figure, make_panels_figure  # noqa: E402

FAMILIES = ("DRW", "OBPL")  # no CARMA in any v2 campaign

SELECTION_SEED = 20260917
DETECT, REFUTE = -2.0, 2.0
ESS_FLOOR = 200.0

# classify_lightcurve's fallback search order when the class select_examples
# wants is empty for a cell.
FALLBACK_ORDER = ("clear_non_detection", "inconclusive", "false_positive")

DATA_ROOT = os.path.expanduser("~/work/data/quasar_cadences/v2")
SIMS_ROOT = os.path.expanduser(
    "~/work/repositories/pioran_periodicity_ai/workspace/v2_sims"
)
DEFAULT_OUT_ROOT = os.path.expanduser("~/work/data/quasar_cadences/results")


@dataclass(frozen=True)
class CampaignConfig:
    """One campaign's inputs and grouping.

    ``cells``: the explicit cell restriction, as the exact
    ``aggregate_results.build_table`` key strings to keep (e.g.
    ``"target_snr=8, highalpha=-3.5"``), or ``None`` to keep every cell
    ``build_table`` produces.
    """

    results_dir: str
    lc_dir: str
    config_csv: str
    group_cols: tuple[str, ...]
    cells: Optional[list[str]]
    multiband: bool


def _synthetic_cells() -> list[str]:
    """The synthetic arm's restriction: target_snr in {1, 8, 15} x
    highalpha in {-2.0, -3.5} (6 of its full 5x4 grid)."""
    return [
        f"target_snr={snr:g}, highalpha={ha:g}"
        for snr in (1, 8, 15)
        for ha in (-2.0, -3.5)
    ]


CAMPAIGNS: dict[str, CampaignConfig] = {
    "synthetic": CampaignConfig(
        results_dir=os.path.join(DATA_ROOT, "synthetic", "results"),
        lc_dir=os.path.join(DATA_ROOT, "synthetic", "lightcurves"),
        config_csv=os.path.join(SIMS_ROOT, "v2_synth_n2000_final.csv"),
        group_cols=("target_snr", "highalpha"),
        cells=_synthetic_cells(),
        multiband=False,
    ),
    "ztf_single": CampaignConfig(
        results_dir=os.path.join(DATA_ROOT, "ztf_single", "results"),
        lc_dir=os.path.join(DATA_ROOT, "ztf_single", "lightcurves"),
        config_csv=os.path.join(SIMS_ROOT, "v2_ztf_n400_final.csv"),
        group_cols=("highalpha",),
        cells=None,
        multiband=False,
    ),
    "lsst_single": CampaignConfig(
        results_dir=os.path.join(DATA_ROOT, "lsst_single", "results"),
        lc_dir=os.path.join(DATA_ROOT, "lsst_single", "lightcurves"),
        config_csv=os.path.join(SIMS_ROOT, "v2_lsst_n400_final.csv"),
        group_cols=("highalpha",),
        cells=None,
        multiband=False,
    ),
    "ztf_multi": CampaignConfig(
        results_dir=os.path.join(DATA_ROOT, "ztf_multi", "results"),
        lc_dir=os.path.join(DATA_ROOT, "ztf_multi", "lightcurves"),
        config_csv=os.path.join(SIMS_ROOT, "v2_ztf_multiband_null.csv"),
        group_cols=("highalpha",),
        cells=None,
        multiband=True,
    ),
    "lsst_multi": CampaignConfig(
        results_dir=os.path.join(DATA_ROOT, "lsst_multi", "results"),
        lc_dir=os.path.join(DATA_ROOT, "lsst_multi", "lightcurves"),
        config_csv=os.path.join(SIMS_ROOT, "v2_lsst_multiband_null.csv"),
        group_cols=("highalpha",),
        cells=None,
        multiband=True,
    ),
}


def classify_lightcurve(bf_by_pair: dict[str, float]) -> str:
    """Kass-Raftery classification from a light curve's log10 BF per pair.

    ``"false_positive"``: either pair gives decisive evidence for
    periodicity (log10 B < ``DETECT``) -- a false alarm, since none of
    these campaigns' light curves carry a true periodic signal in the
    pairs this script classifies.
    ``"clear_non_detection"``: *both* pairs decisively refute periodicity
    (log10 B > ``REFUTE``).
    Otherwise ``"inconclusive"``.
    """
    values = list(bf_by_pair.values())
    if any(b < DETECT for b in values):
        return "false_positive"
    if all(b > REFUTE for b in values):
        return "clear_non_detection"
    return "inconclusive"


def _bf_by_lc_id(cell: dict) -> dict[int, dict[str, float]]:
    """Invert a ``build_table`` cell (``{pair: {lc_ids, log10_BF_values}}``)
    into ``{lc_id: {pair: bf}}``, keeping only light curves that have a
    value for every pair present in the cell -- ``classify_lightcurve``
    needs all pairs to apply the either/both rule correctly, and a light
    curve missing one pair (e.g. dropped there for low ESS) cannot be
    classified.
    """
    pairs = list(cell)
    bf_by_id: dict[int, dict[str, float]] = {}
    for pair in pairs:
        block = cell[pair]
        for lc_id, bf in zip(block["lc_ids"], block["log10_BF_values"]):
            bf_by_id.setdefault(lc_id, {})[pair] = float(bf)
    return {lc_id: bfs for lc_id, bfs in bf_by_id.items() if len(bfs) == len(pairs)}


def select_examples(
    table_cell: dict, cell_key: str, rng: np.random.Generator
) -> list[dict]:
    """Pick two example light curves from one ``build_table`` cell.

    If the cell holds >=1 false positive: draw one false positive and one
    clear non-detection. Otherwise: one clear non-detection and one
    inconclusive. The draw within a class is uniform at random (not by
    extremity), via ``rng.choice``. An empty class falls back to the
    nearest available class in ``FALLBACK_ORDER`` and records the
    substitution.

    Returns two ``{"lc_id", "klass", "log10_bf", "fallback"}`` dicts.
    """
    bf_by_id = _bf_by_lc_id(table_cell)
    if not bf_by_id:
        raise ValueError(f"cell '{cell_key}' has no light curve with all pairs")

    by_class: dict[str, list[int]] = {}
    for lc_id, bfs in bf_by_id.items():
        by_class.setdefault(classify_lightcurve(bfs), []).append(lc_id)

    wanted = (
        ["false_positive", "clear_non_detection"]
        if by_class.get("false_positive")
        else ["clear_non_detection", "inconclusive"]
    )

    picks = []
    for klass in wanted:
        actual_klass, fallback = klass, None
        candidates = by_class.get(klass, [])
        if not candidates:
            for alt in FALLBACK_ORDER:
                if by_class.get(alt):
                    actual_klass, fallback, candidates = alt, klass, by_class[alt]
                    break
            else:
                raise ValueError(f"cell '{cell_key}' has no light curve in any class")
        lc_id = int(rng.choice(sorted(candidates)))
        picks.append(
            {
                "lc_id": lc_id,
                "klass": actual_klass,
                "log10_bf": dict(bf_by_id[lc_id]),
                "fallback": fallback,
            }
        )
    return picks


def _selection_rng(cell_key: str) -> np.random.Generator:
    """A generator seeded from ``SELECTION_SEED`` combined with a stable
    hash of ``cell_key`` -- reproducible and independent of the order
    cells happen to be processed in. Uses ``zlib.crc32`` rather than the
    builtin ``hash()``, which is randomised per process by
    ``PYTHONHASHSEED`` and would make picks irreproducible across runs.
    """
    seed = (SELECTION_SEED + zlib.crc32(cell_key.encode("utf-8"))) % (2**32)
    return np.random.default_rng(seed)


def _slug(cell_key: str) -> str:
    return cell_key.replace(", ", "_").replace("=", "")


def _restrict(table: dict, cells: Optional[list[str]]) -> dict:
    if cells is None:
        return table
    missing = [c for c in cells if c not in table]
    if missing:
        print(f"  WARNING: configured cells not in table, skipping: {missing}")
    return {c: table[c] for c in cells if c in table}


def _iter_subtables(table: dict, group_cols: tuple[str, ...]):
    """Yield ``(filename_suffix, subtable, plot_group_col)``.

    A single group column is plotted directly. With more than one (only
    the synthetic arm's ``target_snr, highalpha``), every column but the
    last is pinned and one subtable is yielded per combination of pinned
    values, so ``plot_strip``/``plot_fpr_calibration`` -- which expect a
    single swept column -- see a simple grid each time.
    """
    if not group_cols:
        raise ValueError("group_cols must be non-empty")
    if len(group_cols) == 1:
        yield "", table, group_cols[0]
        return
    plot_col = group_cols[-1]
    pin_cols = group_cols[:-1]
    pin_values = sorted({tuple(parse_key(k)[c] for c in pin_cols) for k in table})
    for vals in pin_values:

        def _pred(parsed, cols=pin_cols, vs=vals):
            return all(parsed.get(c) == v for c, v in zip(cols, vs))

        sub = filter_table(table, _pred)
        suffix = "_" + "_".join(f"{c}{v:g}" for c, v in zip(pin_cols, vals))
        yield suffix, sub, plot_col


def _write_fpr_table(table: dict, out_path: str) -> None:
    """Write the flat cell/pair Bayes-factor summary as a CSV.

    The ``fpr`` column is literally ``detect / n``: a genuine
    false-positive rate for a no-true-signal cell (every v2 campaign in
    ``CAMPAIGNS`` today), but would read as a detection rate rather than
    an FPR if this driver is ever pointed at a signal-bearing campaign.
    """
    rows = []
    for cell_key in sorted(table):
        for pair in FAMILIES:
            if pair not in table[cell_key]:
                continue
            block = table[cell_key][pair]
            n = block["n"]
            outcomes = block["outcomes"]
            fpr = outcomes["detect"] / n if n > 0 else float("nan")
            rows.append(
                {
                    "cell": cell_key,
                    "pair": pair,
                    "n": n,
                    "n_dropped_unconverged": block["n_dropped_unconverged"],
                    "n_dropped_low_ess": block["n_dropped_low_ess"],
                    "detect": outcomes["detect"],
                    "inconclusive": outcomes["inconclusive"],
                    "refute": outcomes["refute"],
                    "fpr": fpr,
                }
            )
    pd.DataFrame(rows).to_csv(out_path, index=False)


def run_campaign(name: str, cfg: CampaignConfig, out_root: str) -> Optional[dict]:
    """Aggregate, plot and select examples for one campaign.

    Returns the manifest dict, or ``None`` (and prints a skip notice)
    when ``cfg.results_dir`` does not exist yet -- the ``lsst_multi`` case
    before that run lands.
    """
    if not os.path.isdir(cfg.results_dir):
        print(f"[{name}] SKIP: results_dir not found: {cfg.results_dir}")
        return None

    out_dir = os.path.join(out_root, name)
    examples_dir = os.path.join(out_dir, "examples")
    os.makedirs(examples_dir, exist_ok=True)

    # cfg.cells restricts ONLY which cells get example light curves plotted
    # (that cap exists purely to bound the number of per-light-curve figures
    # -- see the module docstring). Every numeric/summary output -- the
    # summary JSON, the FPR table, and the strip/calibration figures -- uses
    # the FULL table, unrestricted, so no cell's science is hidden from the
    # aggregate view even when it has no example figures of its own.
    table_full = build_table(
        cfg.results_dir,
        group_cols=list(cfg.group_cols),
        config_csv=cfg.config_csv,
        min_ess=ESS_FLOOR,
    )
    table = _restrict(table_full, cfg.cells)

    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(
            {
                "table": table_full,
                "thresholds": "log10B<-2 detect | +-2 inconclusive | >2 refute",
                "min_ess": ESS_FLOOR,
                "group_cols": list(cfg.group_cols),
                "cells_restricted_for_examples": cfg.cells,
            },
            f,
            indent=1,
        )

    _write_fpr_table(table_full, os.path.join(out_dir, "fpr_table.csv"))

    apply_style()
    for suffix, sub, group_col in _iter_subtables(table_full, cfg.group_cols):
        fig = plot_strip(sub, group_col, families=FAMILIES)
        save_figure(fig, os.path.join(out_dir, f"strip{suffix}.png"))
        fig = plot_fpr_calibration(sub, group_col, families=FAMILIES)
        save_figure(fig, os.path.join(out_dir, f"fpr_calibration{suffix}.png"))

    manifest_cells: dict[str, list[dict]] = {}
    skipped_cells: list[dict] = []
    for cell_key in sorted(table):
        rng = _selection_rng(cell_key)
        try:
            picks = select_examples(table[cell_key], cell_key, rng)
        except ValueError as exc:
            # build_table can emit a pair block with n=0 (every fit in the
            # cell gated out by convergence/ESS) rather than omitting the
            # cell -- select_examples then has no light curve to classify
            # in any class. Skip only this cell's examples, not the whole
            # campaign (and not everything queued after it under
            # --campaign all): the rest of this campaign's table, figures
            # and other cells' examples are still worth having.
            print(
                f"[{name}] WARNING: no example light curves for cell "
                f"'{cell_key}': {exc}"
            )
            manifest_cells[cell_key] = []
            skipped_cells.append({"cell": cell_key, "reason": str(exc)})
            continue
        slug = _slug(cell_key)
        pick_records = []
        for pick in picks:
            lc_id = pick["lc_id"]
            panels_path = os.path.join(
                examples_dir, f"{slug}_{pick['klass']}_{lc_id}_panels.png"
            )
            corner_path = os.path.join(
                examples_dir, f"{slug}_{pick['klass']}_{lc_id}_corner.png"
            )
            caption = f"{name} | {cell_key} | {pick['klass']}"
            if pick["fallback"]:
                caption += f" (fallback from {pick['fallback']})"
            make_panels_figure(
                cfg.results_dir, cfg.lc_dir, lc_id, panels_path, caption, rng
            )
            make_corner_figure(cfg.results_dir, lc_id, corner_path, caption)
            pick_records.append(
                {
                    **pick,
                    "panels_png": os.path.relpath(panels_path, out_dir),
                    "corner_png": os.path.relpath(corner_path, out_dir),
                }
            )
        manifest_cells[cell_key] = pick_records

    manifest = {
        "campaign": name,
        "results_dir": cfg.results_dir,
        "lc_dir": cfg.lc_dir,
        "config_csv": cfg.config_csv,
        "group_cols": list(cfg.group_cols),
        "multiband": cfg.multiband,
        "selection_seed": SELECTION_SEED,
        "min_ess": ESS_FLOOR,
        "cells": manifest_cells,
        "skipped_cells": skipped_cells,
    }
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)

    print(f"[{name}] wrote {out_dir}")
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--campaign",
        default="all",
        choices=list(CAMPAIGNS) + ["all"],
        help="campaign name, or 'all' to process every entry in CAMPAIGNS",
    )
    ap.add_argument(
        "--out-root",
        default=DEFAULT_OUT_ROOT,
        help="output root; each campaign writes to <out-root>/<campaign>/",
    )
    args = ap.parse_args()

    names = list(CAMPAIGNS) if args.campaign == "all" else [args.campaign]
    for name in names:
        run_campaign(name, CAMPAIGNS[name], args.out_root)


if __name__ == "__main__":
    main()
