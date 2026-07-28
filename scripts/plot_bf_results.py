"""CLI driver for ``pioran_periodicity.visualization`` -- turns
``aggregate_results.py`` summary JSONs into the strip / power-curve /
FPR-calibration / ROC figures picked for the paper.

Each subcommand maps to one plotting function; see
``pioran_periodicity/visualization.py`` for the underlying logic and
docstrings (in particular ``plot_roc``'s null-value note -- the ``roc``
subcommand's ``--null-filter`` implements one explicit pooling choice, not
the only valid one).

    conda run -n <env> python scripts/plot_bf_results.py strip \\
        --summary summary_3_4.json --group-col highalpha \\
        --xlabel "high-frequency PSD slope" --out strip_3_4.png

    conda run -n <env> python scripts/plot_bf_results.py power-curves \\
        --summary summary_3_7.json --x-col A1 --series-col period \\
        --out power_curves_3_7.png

    conda run -n <env> python scripts/plot_bf_results.py fpr-calibration \\
        --summary summary_3_4.json --group-col highalpha \\
        --out fpr_calibration_3_4.png

    conda run -n <env> python scripts/plot_bf_results.py roc \\
        --null-summary summary_3_4.json --null-group-col highalpha \\
        --null-filter -2.8,-3.2 \\
        --alt-summary summary_3_7.json --series-col period \\
        --fixed-col A1 --fixed-value 0.3675 --out roc_period_A1_0.3675.png
"""

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")

import numpy as np

from pioran_periodicity.visualization import (
    FAMILIES,
    apply_style,
    load_summary,
    plot_fpr_calibration,
    plot_power_curves,
    plot_roc,
    plot_strip,
    save_figure,
)


def cmd_strip(args):
    table = load_summary(args.summary)
    fig = plot_strip(
        table,
        args.group_col,
        xlabel=args.xlabel,
        shade_zones=not args.no_zones,
        families=args.families,
    )
    save_figure(fig, args.out)
    print(f"wrote {args.out}")


def cmd_power_curves(args):
    table = load_summary(args.summary)
    fig = plot_power_curves(
        table,
        args.x_col,
        args.series_col,
        x_label=args.x_label,
        series_label=args.series_label,
        families=args.families,
    )
    save_figure(fig, args.out)
    print(f"wrote {args.out}")


def cmd_fpr_calibration(args):
    table = load_summary(args.summary)
    fig = plot_fpr_calibration(
        table,
        args.group_col,
        xlabel=args.xlabel,
        threshold_line=args.threshold,
        families=args.families,
    )
    save_figure(fig, args.out)
    print(f"wrote {args.out}")


def cmd_roc(args):
    table_null = load_summary(args.null_summary)
    table_alt = load_summary(args.alt_summary)
    null_filter = (
        [float(v) for v in args.null_filter.split(",")] if args.null_filter else None
    )

    null_values = {}
    for fam in args.families:
        keys = [
            k
            for k in table_null
            if null_filter is None
            or _key_col_value(k, args.null_group_col) in null_filter
        ]
        null_values[fam] = np.concatenate(
            [table_null[k][fam]["log10_BF_values"] for k in keys]
        )

    fig = plot_roc(
        null_values,
        table_alt,
        args.series_col,
        args.fixed_col,
        args.fixed_value,
        families=args.families,
        null_label=args.null_label,
    )
    save_figure(fig, args.out)
    print(f"wrote {args.out}")


def _key_col_value(key, col):
    from pioran_periodicity.visualization import parse_key

    return parse_key(key)[col]


def add_common(ap, default_out):
    ap.add_argument("--out", default=default_out, help="output PNG path")
    ap.add_argument(
        "--families",
        default=",".join(FAMILIES),
        type=lambda s: s.split(","),
        help=f"comma-separated model families (default: {','.join(FAMILIES)})",
    )


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("strip", help="scatter+box of log10 BF per configuration")
    p.add_argument("--summary", required=True, help="aggregate_results.py summary JSON")
    p.add_argument("--group-col", required=True, help="config CSV column swept")
    p.add_argument("--xlabel", default=None)
    p.add_argument(
        "--no-zones", action="store_true", help="disable decision-zone shading"
    )
    add_common(p, "strip.png")
    p.set_defaults(func=cmd_strip)

    p = sub.add_parser(
        "power-curves", help="P(detect) vs. x_col, one line per series_col"
    )
    p.add_argument("--summary", required=True)
    p.add_argument("--x-col", required=True)
    p.add_argument("--series-col", required=True)
    p.add_argument("--x-label", default=None)
    p.add_argument("--series-label", default=None)
    add_common(p, "power_curves.png")
    p.set_defaults(func=cmd_power_curves)

    p = sub.add_parser("fpr-calibration", help="false-positive rate vs. threshold")
    p.add_argument("--summary", required=True, help="a no-true-signal scenario summary")
    p.add_argument("--group-col", required=True)
    p.add_argument("--xlabel", default=None)
    p.add_argument("--threshold", type=float, default=-2.0)
    add_common(p, "fpr_calibration.png")
    p.set_defaults(func=cmd_fpr_calibration)

    p = sub.add_parser("roc", help="TPR vs. FPR, threshold-swept")
    p.add_argument(
        "--null-summary", required=True, help="a no-true-signal scenario summary"
    )
    p.add_argument("--null-group-col", required=True)
    p.add_argument(
        "--null-filter",
        default=None,
        help="comma-separated group-col values to pool for the null "
        "sample (default: use all keys in --null-summary)",
    )
    p.add_argument("--null-label", default="null")
    p.add_argument(
        "--alt-summary", required=True, help="a true-signal scenario summary"
    )
    p.add_argument(
        "--series-col", required=True, help="alt-table column, one line each"
    )
    p.add_argument("--fixed-col", required=True, help="alt-table column held fixed")
    p.add_argument("--fixed-value", required=True, type=float)
    add_common(p, "roc.png")
    p.set_defaults(func=cmd_roc)

    args = ap.parse_args()
    apply_style()
    args.func(args)


if __name__ == "__main__":
    main()
