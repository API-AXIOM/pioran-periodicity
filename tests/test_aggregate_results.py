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


def _write_result(results_dir, lc_id, model, logz, meta, converged=None, ess=None):
    """``converged=None``/``ess=None`` write a legacy-style JSON with no such key."""
    payload = {"model": model, "logz": logz, "meta": meta}
    if converged is not None:
        payload["converged"] = converged
    if ess is not None:
        payload["ess"] = ess
    with open(os.path.join(results_dir, f"{lc_id}_{model}.json"), "w") as f:
        json.dump(payload, f)


@pytest.fixture
def results_dir_null(tmp_path):
    d = tmp_path / "null_case"
    d.mkdir()
    for lc_id, ha in [(1, -2.0), (2, -2.0), (3, -4.0)]:
        meta = {"highalpha": ha, "lc_id": lc_id}
        _write_result(d, lc_id, "drw", 100.0, meta, ess=None)
        _write_result(d, lc_id, "drw_sine", 95.0, meta, ess=None)
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
        _write_result(d, lc_id, "drw", 100.0, meta, ess=None)
        _write_result(d, lc_id, "drw_sine", 110.0, meta, ess=None)
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
    _write_result(d, 1, "drw", 100.0, meta, ess=None)
    _write_result(d, 1, "drw_sine", 100.0, meta, ess=None)
    csv_path = tmp_path / "config.csv"
    pd.DataFrame({"ID": [1], "highalpha": [-3.0]}).to_csv(csv_path, index=False)

    table = build_table(str(d), group_cols=["highalpha"], config_csv=str(csv_path))
    assert set(table) == {"highalpha=-3"}
    assert table["highalpha=-3"]["DRW"]["outcomes"]["inconclusive"] == 1


@pytest.fixture
def results_dir_mixed_convergence(tmp_path):
    """Three pairs at one config: both converged, sine truncated, base
    truncated. Only the first should survive the default gating.
    """
    d = tmp_path / "mixed"
    d.mkdir()
    meta = {"highalpha": -4.0}
    _write_result(d, 1, "drw", 100.0, meta, converged=True, ess=None)
    _write_result(d, 1, "drw_sine", 110.0, meta, converged=True, ess=None)
    _write_result(d, 2, "drw", 100.0, meta, converged=True, ess=None)
    _write_result(d, 2, "drw_sine", 10.0, meta, converged=False, ess=None)
    _write_result(d, 3, "drw", 100.0, meta, converged=False, ess=None)
    _write_result(d, 3, "drw_sine", 110.0, meta, converged=True, ess=None)
    return str(d)


def test_unconverged_pairs_dropped_by_default(results_dir_mixed_convergence):
    table = build_table(results_dir_mixed_convergence, group_cols=["highalpha"])
    cell = table["highalpha=-4"]["DRW"]
    assert cell["n"] == 1
    assert cell["n_dropped_unconverged"] == 2
    assert cell["converged_frac"] == pytest.approx(1 / 3, abs=1e-3)
    # the surviving pair is the genuine detection; the truncated sine fit
    # (logz 10 vs 100 -> log10 BF +39) would have flipped it to 'refute'
    assert cell["outcomes"] == {"detect": 1, "inconclusive": 0, "refute": 0}


def test_keep_unconverged_restores_old_behaviour(results_dir_mixed_convergence):
    table = build_table(
        results_dir_mixed_convergence, group_cols=["highalpha"], keep_unconverged=True
    )
    cell = table["highalpha=-4"]["DRW"]
    assert cell["n"] == 3
    assert cell["n_dropped_unconverged"] == 0
    # the truncated fit contributes a large spurious positive Bayes factor
    assert cell["outcomes"]["refute"] == 1
    assert cell["outcomes"]["detect"] == 2


def test_cell_with_no_surviving_pairs_is_reported_not_hidden(tmp_path):
    """A cell where everything was truncated must still appear, with n=0 --
    silently dropping it would look like the config was never run.
    """
    d = tmp_path / "allbad"
    d.mkdir()
    meta = {"highalpha": -4.0}
    _write_result(d, 1, "drw", 100.0, meta, converged=False, ess=None)
    _write_result(d, 1, "drw_sine", 10.0, meta, converged=False, ess=None)

    table = build_table(str(d), group_cols=["highalpha"])
    cell = table["highalpha=-4"]["DRW"]
    assert cell["n"] == 0
    assert cell["n_dropped_unconverged"] == 1
    assert cell["converged_frac"] == 0.0
    assert cell["log10_BF_mean"] is None


def test_legacy_results_without_converged_key_are_kept(results_dir_null):
    """Pre-multiband result files have no 'converged' field and must not be
    silently discarded as unconverged.
    """
    table = build_table(results_dir_null, group_cols=["highalpha"])
    assert table["highalpha=-2"]["DRW"]["n"] == 2
    assert table["highalpha=-2"]["DRW"]["n_dropped_unconverged"] == 0


def test_low_ess_pair_is_dropped(tmp_path):
    """A fit with ESS below the floor poisons its pair, exactly as an
    unconverged fit does -- even when it is flagged converged, which is the
    v2 lsst_single 310335_obpl case."""
    d = tmp_path / "ess_case"
    d.mkdir()
    meta = {"highalpha": -2.0}
    _write_result(d, 1, "drw", 100.0, meta, converged=True, ess=2000.0)
    _write_result(d, 1, "drw_sine", 95.0, meta, converged=True, ess=2000.0)
    _write_result(d, 2, "drw", 100.0, meta, converged=True, ess=1.0)
    _write_result(d, 2, "drw_sine", 95.0, meta, converged=True, ess=2000.0)

    table = build_table(str(d), group_cols=["highalpha"], min_ess=200.0)
    cell = table["highalpha=-2"]["DRW"]

    assert cell["n"] == 1
    assert cell["n_dropped_low_ess"] == 1
    assert cell["excluded"] == [{"lc_id": 2, "model": "drw", "ess": 1.0}]


def test_missing_ess_key_is_kept(tmp_path):
    """Legacy results predate the ess key and must not be dropped."""
    d = tmp_path / "legacy_case"
    d.mkdir()
    meta = {"highalpha": -2.0}
    _write_result(d, 1, "drw", 100.0, meta)
    _write_result(d, 1, "drw_sine", 95.0, meta)

    table = build_table(str(d), group_cols=["highalpha"], min_ess=200.0)
    assert table["highalpha=-2"]["DRW"]["n"] == 1


def test_lc_ids_stay_aligned_with_bf_values_after_a_drop(tmp_path):
    """Task 4 selects example light curves by id, so the ids must remain
    index-aligned with the Bayes factors once a pair has been dropped."""
    d = tmp_path / "align_case"
    d.mkdir()
    meta = {"highalpha": -2.0}
    for lc_id, logz_sine in [(1, 95.0), (2, 90.0), (3, 85.0)]:
        _write_result(d, lc_id, "drw", 100.0, meta, converged=True, ess=2000.0)
        _write_result(
            d,
            lc_id,
            "drw_sine",
            logz_sine,
            meta,
            converged=True,
            ess=1.0 if lc_id == 2 else 2000.0,
        )

    cell = build_table(str(d), group_cols=["highalpha"], min_ess=200.0)["highalpha=-2"][
        "DRW"
    ]

    assert cell["lc_ids"] == [1, 3]
    assert len(cell["log10_BF_values"]) == 2
    # id 1 has the smaller logz gap, so the smaller log10 BF of the two
    assert cell["log10_BF_values"][0] < cell["log10_BF_values"][1]


def test_both_fits_below_ess_floor_counts_as_one_dropped_pair(tmp_path):
    """When both members of a pair fall below the ESS floor, it counts as
    ONE dropped pair (not two). The excluded list tracks both failing fits,
    but converged_frac reflects one dropped pair."""
    d = tmp_path / "both_bad"
    d.mkdir()
    meta = {"highalpha": -2.0}
    # LC 1: both above floor, retained
    _write_result(d, 1, "drw", 100.0, meta, converged=True, ess=2000.0)
    _write_result(d, 1, "drw_sine", 95.0, meta, converged=True, ess=2000.0)
    # LC 2: BOTH below floor (1.0), dropped as one pair
    _write_result(d, 2, "drw", 100.0, meta, converged=True, ess=1.0)
    _write_result(d, 2, "drw_sine", 95.0, meta, converged=True, ess=1.0)
    # LC 3: both above floor, retained
    _write_result(d, 3, "drw", 100.0, meta, converged=True, ess=2000.0)
    _write_result(d, 3, "drw_sine", 95.0, meta, converged=True, ess=2000.0)

    table = build_table(str(d), group_cols=["highalpha"], min_ess=200.0)
    cell = table["highalpha=-2"]["DRW"]

    # Only LC 1 and 3 retained
    assert cell["n"] == 2
    # One dropped pair (LC 2), but two excluded entries (both fits)
    assert cell["n_dropped_low_ess"] == 1
    assert len(cell["excluded"]) == 2
    # Both excluded entries should reference LC 2
    assert cell["excluded"][0]["lc_id"] == 2
    assert cell["excluded"][1]["lc_id"] == 2
    assert cell["excluded"][0]["model"] == "drw"
    assert cell["excluded"][1]["model"] == "drw_sine"
    # converged_frac reflects 1 dropped pair, not 2
    # total = 2 retained + 1 dropped = 3
    assert cell["converged_frac"] == pytest.approx(2 / 3, abs=1e-3)
