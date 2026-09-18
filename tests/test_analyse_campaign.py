"""Tests for scripts/analyse_campaign.py: the campaign driver's
example-selection rule (verbatim from the task brief) plus integration
coverage for CAMPAIGNS and run_campaign against small tmp_path fixtures --
never the real v2 results directories.
"""

from __future__ import annotations

import json
import os
import sys
import zlib

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from analyse_campaign import (  # noqa: E402
    CAMPAIGNS,
    SELECTION_SEED,
    CampaignConfig,
    _selection_rng,
    classify_lightcurve,
    run_campaign,
    select_examples,
)
import plot_sim_examples  # noqa: E402


def _table_cell(bfs_by_id):
    """Build the aggregate_results-shaped cell from {lc_id: (drw_bf, obpl_bf)}."""
    ids = sorted(bfs_by_id)
    return {
        "DRW": {"lc_ids": ids, "log10_BF_values": [bfs_by_id[i][0] for i in ids]},
        "OBPL": {"lc_ids": ids, "log10_BF_values": [bfs_by_id[i][1] for i in ids]},
    }


def test_classify_uses_either_pair_for_detection():
    assert classify_lightcurve({"DRW": -3.0, "OBPL": 0.5}) == "false_positive"
    assert classify_lightcurve({"DRW": 0.5, "OBPL": -3.0}) == "false_positive"


def test_classify_requires_both_pairs_for_clear_non_detection():
    assert classify_lightcurve({"DRW": 3.0, "OBPL": 3.0}) == "clear_non_detection"
    assert classify_lightcurve({"DRW": 3.0, "OBPL": 0.5}) == "inconclusive"


def test_cell_with_false_positive_picks_one_of_each():
    cell = _table_cell(
        {
            1: (-3.0, 0.0),
            2: (-4.0, 0.0),  # false positives
            3: (3.0, 3.0),
            4: (4.0, 5.0),  # clear non-detections
            5: (0.5, 0.5),  # inconclusive
        }
    )
    picks = select_examples(cell, "highalpha=-2", np.random.default_rng(0))
    assert {p["klass"] for p in picks} == {"false_positive", "clear_non_detection"}
    assert len(picks) == 2


def test_cell_without_false_positive_picks_non_detection_and_inconclusive():
    cell = _table_cell({1: (3.0, 3.0), 2: (4.0, 4.0), 3: (0.5, 0.5)})
    picks = select_examples(cell, "highalpha=-3.5", np.random.default_rng(0))
    assert {p["klass"] for p in picks} == {"clear_non_detection", "inconclusive"}


def test_selection_is_reproducible_and_not_extremal():
    """Same seed -> same picks; and across seeds the pick is not always the
    most extreme member, which is what distinguishes random from argmin."""
    cell = _table_cell({i: (-3.0 - i, 0.0) for i in range(1, 6)} | {9: (3.0, 3.0)})
    key = "highalpha=-2"
    first = select_examples(cell, key, np.random.default_rng(0))
    again = select_examples(cell, key, np.random.default_rng(0))
    assert [p["lc_id"] for p in first] == [p["lc_id"] for p in again]

    chosen = {
        select_examples(cell, key, np.random.default_rng(s))[0]["lc_id"]
        for s in range(20)
    }
    assert len(chosen) > 1


def test_empty_class_records_fallback():
    """Neither clear_non_detection nor inconclusive is available, so the
    second pick must fall all the way through FALLBACK_ORDER
    (clear_non_detection -> inconclusive -> false_positive) to
    false_positive, and record which class was originally wanted."""
    cell = _table_cell({1: (-3.0, 0.0), 2: (-4.0, 0.0)})  # no non-detections
    picks = select_examples(cell, "highalpha=-2", np.random.default_rng(0))
    assert len(picks) == 2

    no_fallback = [p for p in picks if p["fallback"] is None]
    with_fallback = [p for p in picks if p["fallback"] is not None]
    assert len(no_fallback) == 1 and no_fallback[0]["klass"] == "false_positive"
    assert len(with_fallback) == 1
    assert with_fallback[0]["fallback"] == "clear_non_detection"
    assert with_fallback[0]["klass"] == "false_positive"


def test_fallback_walks_order_not_straight_to_last_resort():
    """When inconclusive IS available but clear_non_detection is not, the
    fallback must land on inconclusive (the next entry in FALLBACK_ORDER),
    not jump straight to false_positive -- proving the order is actually
    walked, not just "grab whatever class happens to be non-empty"."""
    cell = _table_cell({1: (-3.0, 0.0), 2: (0.5, 0.5)})  # fp + inconclusive
    picks = select_examples(cell, "highalpha=-2", np.random.default_rng(0))
    fallback_pick = next(p for p in picks if p["fallback"] is not None)
    assert fallback_pick["fallback"] == "clear_non_detection"
    assert fallback_pick["klass"] == "inconclusive"


def test_select_examples_raises_on_a_wholly_empty_cell():
    """A cell build_table can legitimately emit: every fit gated out by
    convergence/ESS, so both pairs report n=0 rather than the cell being
    omitted entirely. select_examples cannot pick anything from it and
    raises -- run_campaign is what must catch this (see
    test_run_campaign_skips_cell_with_no_examples_available), not silently
    return something meaningless.
    """
    empty_cell = {
        "DRW": {"lc_ids": [], "log10_BF_values": []},
        "OBPL": {"lc_ids": [], "log10_BF_values": []},
    }
    with pytest.raises(ValueError):
        select_examples(empty_cell, "highalpha=-2", np.random.default_rng(0))


def test_selection_rng_seed_is_a_pure_function_of_seed_and_key_not_hash():
    """_selection_rng must derive its seed from SELECTION_SEED and a
    *stable* hash of cell_key (zlib.crc32), never Python's builtin
    hash(), which is salted per-process by PYTHONHASHSEED. A regression
    to hash() would still pass every other test here (they all run in one
    process, where hash() is internally consistent) -- catch it by
    computing the expected seed independently, via the crc32 formula the
    docstring promises, and checking _selection_rng's generator produces
    the identical stream as a generator built directly from that seed.
    Any hash()-based implementation would, overwhelmingly, produce a
    different integer than crc32 for the same string, so this comparison
    fails immediately if the implementation drifts back to hash().
    """
    for key in ("highalpha=-2", "target_snr=8, highalpha=-3.5", "all"):
        expected_seed = (SELECTION_SEED + zlib.crc32(key.encode("utf-8"))) % (2**32)
        expected_rng = np.random.default_rng(expected_seed)
        actual_rng = _selection_rng(key)
        np.testing.assert_array_equal(
            expected_rng.standard_normal(50), actual_rng.standard_normal(50)
        )


# --------------------------------------------------------------------------
# CAMPAIGNS structure
# --------------------------------------------------------------------------


def test_campaigns_has_five_entries():
    assert set(CAMPAIGNS) == {
        "synthetic",
        "ztf_single",
        "lsst_single",
        "ztf_multi",
        "lsst_multi",
    }


def test_synthetic_is_restricted_to_six_cells():
    cfg = CAMPAIGNS["synthetic"]
    assert cfg.group_cols == ("target_snr", "highalpha")
    assert cfg.cells is not None
    assert len(cfg.cells) == 6
    assert set(cfg.cells) == {
        f"target_snr={snr:g}, highalpha={ha:g}"
        for snr in (1, 8, 15)
        for ha in (-2.0, -3.5)
    }


@pytest.mark.parametrize(
    "name", ["ztf_single", "lsst_single", "ztf_multi", "lsst_multi"]
)
def test_real_cadence_arms_use_all_slope_cells(name):
    cfg = CAMPAIGNS[name]
    assert cfg.group_cols == ("highalpha",)
    assert cfg.cells is None


def test_multiband_flag_matches_campaign_kind():
    for name in ("synthetic", "ztf_single", "lsst_single"):
        assert CAMPAIGNS[name].multiband is False
    for name in ("ztf_multi", "lsst_multi"):
        assert CAMPAIGNS[name].multiband is True


def test_campaign_paths_are_absolute():
    for cfg in CAMPAIGNS.values():
        assert os.path.isabs(cfg.results_dir)
        assert os.path.isabs(cfg.lc_dir)
        assert os.path.isabs(cfg.config_csv)


# --------------------------------------------------------------------------
# run_campaign integration, against tiny tmp_path fixtures
# --------------------------------------------------------------------------


def _write_fit(
    results_dir, lc_id, model, logz, highalpha, seed, converged=True, ess=2000.0
):
    rng = np.random.default_rng(seed)
    n = 40
    samples = {
        "log10_variance": rng.normal(-1, 0.1, n).tolist(),
        "log10_fbend": rng.normal(-1, 0.1, n).tolist(),
        "mu0": rng.normal(0, 0.05, n).tolist(),
    }
    if "sine" in model:
        samples |= {
            "A_cos": rng.normal(0, 0.1, n).tolist(),
            "A_sin": rng.normal(0, 0.1, n).tolist(),
            "period": rng.uniform(1, 3, n).tolist(),
        }
    payload = {
        "model": model,
        "logz": logz,
        "logzerr": 0.2,
        "samples": samples,
        "ess": ess,
        "converged": converged,
        "meta": {
            "highalpha": highalpha,
            "noise": "drw" if model.startswith("drw") else "obpl",
            "variant": "sine" if "sine" in model else "plain",
            "band": None,
            "photometric_bands": [],
            "reference_band": "r",
            "n_components": None,
            "basis_function": None,
        },
    }
    with open(os.path.join(results_dir, f"{lc_id}_{model}.json"), "w") as f:
        json.dump(payload, f)


def _write_lightcurve(lc_dir, lc_id, seed, n=20):
    rng = np.random.default_rng(seed)
    t = np.sort(rng.uniform(0, 5, n)) + np.arange(n) * 1e-9
    np.savez(
        os.path.join(lc_dir, f"{lc_id}.npz"),
        t=t,
        y=rng.normal(0, 0.2, n),
        yerr=np.full(n, 0.05),
    )


@pytest.fixture(autouse=True)
def _fake_posterior_predictive(monkeypatch):
    """Panels figures otherwise drive a real GP/Julia posterior-predictive
    call for every draw; this fixture swaps in a trivial stand-in so the
    integration test below stays fast and Julia-free, matching
    test_plot_sim_examples.py's test_panels_figure_plumbing pattern.
    """

    def _fake(meta, params, t, y, yerr, t_grid, band_labels=None, need_std=False):
        mu = np.full_like(np.asarray(t_grid, dtype=float), params.get("mu0", 0.0))
        sd = np.full_like(mu, 0.1) if need_std else None
        return mu, sd

    monkeypatch.setattr(plot_sim_examples, "posterior_predictive", _fake)


@pytest.fixture
def small_campaign(tmp_path):
    """A two-cell (highalpha=-2 / -3.5) campaign, grouped off each result's
    own meta (no config_csv). Cell -2 has a false positive (lc 101) and a
    clear non-detection (lc 102); cell -3.5 has only a clear non-detection
    (lc 201) and an inconclusive light curve (lc 202) -- exercising both
    branches of select_examples through the real driver.
    """
    results_dir = tmp_path / "results"
    lc_dir = tmp_path / "lightcurves"
    results_dir.mkdir()
    lc_dir.mkdir()

    # (lc_id, highalpha, drw_base, drw_sine, obpl_base, obpl_sine)
    rows = [
        (101, -2.0, 100.0, 110.0, 100.0, 100.0),  # DRW false positive
        (102, -2.0, 110.0, 100.0, 110.0, 100.0),  # both pairs refute
        (201, -3.5, 110.0, 100.0, 110.0, 100.0),  # both pairs refute
        (202, -3.5, 100.0, 100.0, 100.0, 100.0),  # both pairs inconclusive
    ]
    seed = 0
    for lc_id, ha, drw_b, drw_s, obpl_b, obpl_s in rows:
        _write_fit(str(results_dir), lc_id, "drw", drw_b, ha, seed)
        seed += 1
        _write_fit(str(results_dir), lc_id, "drw_sine", drw_s, ha, seed)
        seed += 1
        _write_fit(str(results_dir), lc_id, "obpl", obpl_b, ha, seed)
        seed += 1
        _write_fit(str(results_dir), lc_id, "obpl_sine", obpl_s, ha, seed)
        seed += 1
        _write_lightcurve(str(lc_dir), lc_id, seed)
        seed += 1

    cfg = CampaignConfig(
        results_dir=str(results_dir),
        lc_dir=str(lc_dir),
        config_csv=None,
        group_cols=("highalpha",),
        cells=None,
        multiband=False,
    )
    return cfg


def test_run_campaign_writes_expected_outputs(tmp_path, small_campaign):
    out_root = tmp_path / "out"
    manifest = run_campaign("test_campaign", small_campaign, str(out_root))

    assert manifest is not None
    out_dir = out_root / "test_campaign"
    assert (out_dir / "summary.json").exists()
    assert (out_dir / "fpr_table.csv").exists()
    assert (out_dir / "manifest.json").exists()
    assert (out_dir / "strip.png").exists()
    assert (out_dir / "fpr_calibration.png").exists()

    # summary.json's table has both cells
    with open(out_dir / "summary.json") as f:
        summary = json.load(f)
    assert set(summary["table"]) == {"highalpha=-2", "highalpha=-3.5"}
    assert summary["min_ess"] == 200.0

    # fpr_table.csv has the documented columns
    fpr_df = pd.read_csv(out_dir / "fpr_table.csv")
    assert list(fpr_df.columns) == [
        "cell",
        "pair",
        "n",
        "n_dropped_unconverged",
        "n_dropped_low_ess",
        "detect",
        "inconclusive",
        "refute",
        "fpr",
    ]
    assert set(fpr_df["cell"]) == {"highalpha=-2", "highalpha=-3.5"}

    # manifest: selection rule played out as designed for each cell
    assert manifest["selection_seed"] == 20260917
    assert manifest["min_ess"] == 200.0
    cells = manifest["cells"]
    assert set(cells) == {"highalpha=-2", "highalpha=-3.5"}

    klasses_m2 = {p["klass"] for p in cells["highalpha=-2"]}
    assert klasses_m2 == {"false_positive", "clear_non_detection"}
    assert all(p["fallback"] is None for p in cells["highalpha=-2"])

    klasses_m35 = {p["klass"] for p in cells["highalpha=-3.5"]}
    assert klasses_m35 == {"clear_non_detection", "inconclusive"}
    assert all(p["fallback"] is None for p in cells["highalpha=-3.5"])

    # every referenced figure file actually exists
    for picks in cells.values():
        for pick in picks:
            assert (out_dir / pick["panels_png"]).exists()
            assert (out_dir / pick["corner_png"]).exists()


@pytest.fixture
def campaign_with_an_empty_cell(tmp_path):
    """Like small_campaign, but with a third cell (highalpha=-4) where
    every fit is unconverged in both pairs -- build_table then reports
    that cell with n=0 lc_ids for DRW and OBPL rather than omitting it,
    which is exactly the shape select_examples cannot classify anything
    from (see test_select_examples_raises_on_a_wholly_empty_cell).
    """
    results_dir = tmp_path / "results"
    lc_dir = tmp_path / "lightcurves"
    results_dir.mkdir()
    lc_dir.mkdir()

    # (lc_id, highalpha, drw_base, drw_sine, obpl_base, obpl_sine)
    rows = [
        (101, -2.0, 100.0, 110.0, 100.0, 100.0),  # DRW false positive
        (102, -2.0, 110.0, 100.0, 110.0, 100.0),  # both pairs refute
        (201, -3.5, 110.0, 100.0, 110.0, 100.0),  # both pairs refute
        (202, -3.5, 100.0, 100.0, 100.0, 100.0),  # both pairs inconclusive
    ]
    seed = 0
    for lc_id, ha, drw_b, drw_s, obpl_b, obpl_s in rows:
        _write_fit(str(results_dir), lc_id, "drw", drw_b, ha, seed)
        seed += 1
        _write_fit(str(results_dir), lc_id, "drw_sine", drw_s, ha, seed)
        seed += 1
        _write_fit(str(results_dir), lc_id, "obpl", obpl_b, ha, seed)
        seed += 1
        _write_fit(str(results_dir), lc_id, "obpl_sine", obpl_s, ha, seed)
        seed += 1
        _write_lightcurve(str(lc_dir), lc_id, seed)
        seed += 1

    # lc 301 at highalpha=-4: every fit unconverged -> both pairs dropped,
    # build_table reports the cell with n=0 rather than omitting it.
    for model in ("drw", "drw_sine", "obpl", "obpl_sine"):
        _write_fit(str(results_dir), 301, model, 100.0, -4.0, seed, converged=False)
        seed += 1

    cfg = CampaignConfig(
        results_dir=str(results_dir),
        lc_dir=str(lc_dir),
        config_csv=None,
        group_cols=("highalpha",),
        cells=None,
        multiband=False,
    )
    return cfg


def test_run_campaign_skips_cell_with_no_examples_available(
    tmp_path, campaign_with_an_empty_cell
):
    """The reviewer-reproduced crash: select_examples raises ValueError on
    a cell with n=0 in every pair, and that must not abort the whole
    campaign (nor, by extension, anything queued after it under
    --campaign all). The other two cells must still get their normal
    examples, and the empty cell's skip must be recorded in the manifest.
    """
    out_root = tmp_path / "out"
    manifest = run_campaign(
        "empty_cell_campaign", campaign_with_an_empty_cell, str(out_root)
    )

    assert manifest is not None  # the process did not abort
    out_dir = out_root / "empty_cell_campaign"
    assert (out_dir / "manifest.json").exists()

    cells = manifest["cells"]
    assert set(cells) == {"highalpha=-2", "highalpha=-3.5", "highalpha=-4"}

    # the empty cell got no picks, and is recorded as skipped
    assert cells["highalpha=-4"] == []
    assert manifest["skipped_cells"] == [
        {
            "cell": "highalpha=-4",
            "reason": "cell 'highalpha=-4' has no light curve with all pairs",
        }
    ]

    # the other two cells were entirely unaffected: normal picks, files exist
    assert {p["klass"] for p in cells["highalpha=-2"]} == {
        "false_positive",
        "clear_non_detection",
    }
    assert {p["klass"] for p in cells["highalpha=-3.5"]} == {
        "clear_non_detection",
        "inconclusive",
    }
    for cell_key in ("highalpha=-2", "highalpha=-3.5"):
        for pick in cells[cell_key]:
            assert (out_dir / pick["panels_png"]).exists()
            assert (out_dir / pick["corner_png"]).exists()

    # summary/fpr_table/manifest.json were all still written for the whole
    # campaign, not abandoned partway through
    with open(out_dir / "summary.json") as f:
        summary = json.load(f)
    assert set(summary["table"]) == {"highalpha=-2", "highalpha=-3.5", "highalpha=-4"}
    fpr_df = pd.read_csv(out_dir / "fpr_table.csv")
    empty_rows = fpr_df[fpr_df["cell"] == "highalpha=-4"]
    assert (empty_rows["n"] == 0).all()


def test_run_campaign_skips_missing_results_dir(tmp_path):
    cfg = CampaignConfig(
        results_dir=str(tmp_path / "does_not_exist"),
        lc_dir=str(tmp_path / "lc_does_not_exist"),
        config_csv=None,
        group_cols=("highalpha",),
        cells=None,
        multiband=False,
    )
    out_root = tmp_path / "out"
    manifest = run_campaign("missing_campaign", cfg, str(out_root))
    assert manifest is None
    assert not (out_root / "missing_campaign").exists()
