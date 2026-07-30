"""Tests for scripts/aggregate_results.py's build_table, including the
--config-csv-less fallback that groups directly off each result's own meta
(needed when a results dir mixes multiple pilot/extension config CSVs).
"""

from __future__ import annotations

import json
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from aggregate_results import build_table  # noqa: E402


def _write_result(results_dir, lc_id, model, logz, meta):
    with open(os.path.join(results_dir, f"{lc_id}_{model}.json"), "w") as f:
        json.dump({"model": model, "logz": logz, "meta": meta}, f)


@pytest.fixture
def results_dir_null(tmp_path):
    d = tmp_path / "null_case"
    d.mkdir()
    for lc_id, ha in [(1, -2.0), (2, -2.0), (3, -4.0)]:
        meta = {"highalpha": ha, "lc_id": lc_id}
        _write_result(d, lc_id, "drw", 100.0, meta)
        _write_result(d, lc_id, "drw_sine", 95.0, meta)
    return str(d)


@pytest.fixture
def results_dir_signal(tmp_path):
    d = tmp_path / "signal_case"
    d.mkdir()
    rows = [
        (10, -2.0, 1.25, 0.1),
        (11, -4.0, 1.25, 0.1),
        (12, -4.0, 3.75, 0.2),
    ]
    for lc_id, ha, period, a1 in rows:
        meta = {
            "highalpha": ha,
            "true_period": period,
            "true_A1": a1,
            "lc_id": lc_id,
        }
        _write_result(d, lc_id, "drw", 100.0, meta)
        _write_result(d, lc_id, "drw_sine", 110.0, meta)
    return str(d)


def test_build_table_from_meta_null(results_dir_null):
    table = build_table(results_dir_null, group_cols=["highalpha"])
    assert set(table) == {"highalpha=-2", "highalpha=-4"}
    assert table["highalpha=-2"]["DRW"]["n"] == 2
    assert table["highalpha=-4"]["DRW"]["n"] == 1
    # logz_drw - logz_drw_sine = 100 - 95 = 5 > 0 -> refute, not detect
    assert table["highalpha=-2"]["DRW"]["outcomes"]["refute"] == 2


def test_build_table_from_meta_signal(results_dir_signal):
    table = build_table(results_dir_signal, group_cols=["highalpha", "period", "A1"])
    assert set(table) == {
        "highalpha=-2, period=1.25, A1=0.1",
        "highalpha=-4, period=1.25, A1=0.1",
        "highalpha=-4, period=3.75, A1=0.2",
    }
    for key in table:
        # logz_drw - logz_drw_sine = 100 - 110 = -10 -> decisive detect
        assert table[key]["DRW"]["outcomes"]["detect"] == 1


def test_build_table_from_config_csv_still_works(tmp_path):
    d = tmp_path / "results"
    d.mkdir()
    meta = {"highalpha": -3.0, "lc_id": 1}
    _write_result(d, 1, "drw", 100.0, meta)
    _write_result(d, 1, "drw_sine", 100.0, meta)
    csv_path = tmp_path / "config.csv"
    pd.DataFrame({"ID": [1], "highalpha": [-3.0]}).to_csv(csv_path, index=False)

    table = build_table(str(d), group_cols=["highalpha"], config_csv=str(csv_path))
    assert set(table) == {"highalpha=-3"}
    assert table["highalpha=-3"]["DRW"]["outcomes"]["inconclusive"] == 1
