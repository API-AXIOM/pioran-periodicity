"""Tests for pioran_periodicity.visualization against a small synthetic
summary table shaped like aggregate_results.py output.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from pioran_periodicity.visualization import (
    filter_table,
    key_for,
    parse_key,
    plot_detection_rate,
    plot_fpr_calibration,
    plot_matched_roc,
    plot_power_curves,
    plot_roc,
    plot_strip,
    roc_from_bf,
    sweep_values,
)

FAMILIES = ["DRW", "CARMA21"]


def _cell(rng, mean, n=40):
    vals = rng.normal(mean, 1.0, n)
    outcomes = {
        "detect": int(np.sum(vals < -2)),
        "refute": int(np.sum(vals > 2)),
    }
    outcomes["inconclusive"] = n - outcomes["detect"] - outcomes["refute"]
    return {
        "n": n,
        "log10_BF_mean": float(vals.mean()),
        "log10_BF_values": [round(float(v), 3) for v in vals],
        "outcomes": outcomes,
    }


@pytest.fixture
def table_1d():
    rng = np.random.default_rng(0)
    xs = [-4.0, -3.0, -2.0]
    return {
        f"highalpha={x:g}": {fam: _cell(rng, -x - 1) for fam in FAMILIES} for x in xs
    }


@pytest.fixture
def table_2d():
    rng = np.random.default_rng(1)
    periods, a1s = [1.25, 2.5], [0.25, 0.5]
    table = {}
    for p in periods:
        for a1 in a1s:
            key = f"period={p:g}, A1={a1:g}"
            table[key] = {fam: _cell(rng, -3.0 * a1) for fam in FAMILIES}
    return table


def test_parse_key_roundtrip():
    assert parse_key("highalpha=-2.4") == {"highalpha": -2.4}
    assert parse_key("period=1.25, A1=0.24") == {"period": 1.25, "A1": 0.24}
    assert parse_key("all") == {}


def test_sweep_values_and_key_for(table_1d):
    assert sweep_values(table_1d, "highalpha") == [-4.0, -3.0, -2.0]
    key = key_for(table_1d, highalpha=-3.0)
    assert key == "highalpha=-3"


def test_key_for_ambiguous_raises():
    table = {"a=1, b=1": {}, "a=1, b=2": {}}
    with pytest.raises(KeyError):
        key_for(table, a=1)


def test_roc_from_bf_perfect_separation():
    null_vals = np.full(50, 5.0)
    alt_vals = np.full(50, -5.0)
    fpr, tpr = roc_from_bf(null_vals, alt_vals, n_thresholds=50)
    assert fpr[0] == pytest.approx(0.0)
    assert tpr[-1] == pytest.approx(1.0)
    assert np.all(np.diff(fpr) >= 0)
    assert np.all(np.diff(tpr) >= 0)


def test_plot_strip_smoke(table_1d):
    fig = plot_strip(table_1d, "highalpha", families=FAMILIES)
    assert len(fig.axes) == len(FAMILIES)


def test_plot_power_curves_smoke(table_2d):
    fig = plot_power_curves(table_2d, "A1", "period", families=FAMILIES)
    assert len(fig.axes) == len(FAMILIES)


def test_plot_fpr_calibration_smoke(table_1d):
    fig = plot_fpr_calibration(table_1d, "highalpha", families=FAMILIES)
    assert len(fig.axes) == len(FAMILIES)


def test_plot_roc_smoke(table_1d, table_2d):
    null_values = {
        fam: np.array(table_1d["highalpha=-2"][fam]["log10_BF_values"])
        for fam in FAMILIES
    }
    fig = plot_roc(null_values, table_2d, "period", "A1", 0.25, families=FAMILIES)
    assert len(fig.axes) == len(FAMILIES)


def test_filter_table(table_2d):
    sub = filter_table(table_2d, lambda p: p["period"] == 1.25)
    assert sweep_values(sub, "period") == [1.25]
    assert sweep_values(sub, "A1") == [0.25, 0.5]


def test_plot_detection_rate_smoke(table_1d):
    fig = plot_detection_rate(table_1d, "highalpha", families=FAMILIES)
    assert len(fig.axes) == len(FAMILIES)


@pytest.fixture
def table_ha_period():
    """Two periods swept over the same highalpha grid as table_1d, shaped
    like a signal_case table paired against table_1d as the matching null.
    """
    rng = np.random.default_rng(2)
    xs = [-4.0, -3.0, -2.0]
    periods = [1.25, 3.75]
    table = {}
    for x in xs:
        for p in periods:
            key = f"highalpha={x:g}, period={p:g}"
            table[key] = {fam: _cell(rng, -x - 4) for fam in FAMILIES}
    return table


def test_plot_matched_roc_smoke(table_1d, table_ha_period):
    fig = plot_matched_roc(
        table_1d,
        table_ha_period,
        match_col="highalpha",
        match_values=[-4.0, -2.0],
        families=FAMILIES,
        alt_filter={"period": 1.25},
    )
    assert len(fig.axes) == len(FAMILIES)
