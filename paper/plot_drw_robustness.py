"""Figures for the DRW-robustness sensitivity study: false-positive rate and
detection power of a DRW+sine vs. DRW model comparison as the true PSD's
high-frequency slope (``highalpha``) steepens away from -2 (DRW-consistent),
plus ROC curves pairing each null and alternative sample at the same slope.

Reads two ``run_sim.py`` results directories directly (``null_case``: no
true signal, swept over ``highalpha``; ``signal_case``: sine injected,
swept over ``highalpha`` x ``period`` x ``A1``) and groups by each result's
own ``meta`` via ``aggregate_results.build_table`` -- no config CSV needed,
which matters here since the results directories were assembled from
several pilot/extension CSVs with non-contiguous ID ranges. Only the DRW
model family exists in this study (no CARMA/OBPL fits), so every plot call
below pins ``families=["DRW"]``.

    conda run -n <env> python paper/plot_drw_robustness.py \
        --null-dir /path/to/simulations/null_case/results \
        --signal-dir /path/to/simulations/signal_case/results \
        --out-dir paper/figures
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from aggregate_results import build_table  # noqa: E402

from pioran_periodicity.visualization import (  # noqa: E402
    apply_style,
    filter_table,
    parse_key,
    plot_detection_rate,
    plot_matched_roc,
    plot_power_curves,
    save_figure,
    sweep_values,
)

FAMILIES = ["DRW"]

# period-specific (low, at-limit, comfortably-above) amplitude triads used
# by make_slope_robustness_csv.py's --variant signal default; the
# "comfortably above" tier (max A1 per period) is what the ROC figure pools.
ROC_MATCH_VALUES = [-2.0, -3.2, -4.0]


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--null-dir", required=True, help="null_case run_sim.py results dir"
    )
    ap.add_argument(
        "--signal-dir", required=True, help="signal_case run_sim.py results dir"
    )
    ap.add_argument("--out-dir", required=True, help="output directory for PNGs")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    apply_style()

    null_table = build_table(args.null_dir, group_cols=["highalpha"])
    signal_table = build_table(
        args.signal_dir, group_cols=["highalpha", "period", "A1"]
    )

    # --- false-positive rate vs. highalpha (null_case) ---
    fig = plot_detection_rate(
        null_table,
        "highalpha",
        families=FAMILIES,
        x_label="true highalpha",
        ylabel="false-positive rate",
        title="FPR vs. true PSD slope (null_case, DRW vs. DRW+sine)",
        invert_xaxis=True,
    )
    save_figure(fig, os.path.join(args.out_dir, "drw_robustness_fpr.png"))

    # --- detection power vs. highalpha, one figure per period ---
    for period in sweep_values(signal_table, "period"):
        sub = filter_table(signal_table, lambda p, per=period: p["period"] == per)
        fig = plot_power_curves(
            sub,
            x_col="highalpha",
            series_col="A1",
            families=FAMILIES,
            x_label="true highalpha",
            series_label="A1",
            title=f"Detection power vs. true PSD slope (period={period:g} yr)",
            invert_xaxis=True,
        )
        save_figure(
            fig,
            os.path.join(args.out_dir, f"drw_robustness_power_period{period:g}.png"),
        )

    # --- ROC: comfortably-above amplitude tier vs. matching-highalpha null ---
    parsed_keys = [parse_key(k) for k in signal_table]
    comfortable_a1 = {
        period: max(p["A1"] for p in parsed_keys if p.get("period") == period)
        for period in sweep_values(signal_table, "period")
    }
    comfortable_table = filter_table(
        signal_table,
        lambda p: p.get("A1") == comfortable_a1.get(p.get("period")),
    )
    fig = plot_matched_roc(
        null_table,
        comfortable_table,
        match_col="highalpha",
        match_values=ROC_MATCH_VALUES,
        families=FAMILIES,
        match_label="highalpha",
        title="ROC: comfortably-above amplitude tier vs. matching-highalpha null",
    )
    save_figure(fig, os.path.join(args.out_dir, "drw_robustness_roc.png"))

    print(f"wrote figures to {args.out_dir}")


if __name__ == "__main__":
    main()
