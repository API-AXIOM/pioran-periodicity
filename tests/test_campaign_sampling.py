"""The Tier-1 campaign sampling design, enforced.

These tests pin decisions that are invisible in the output CSV if they go
wrong -- a stratified pool and a simple random one produce files of exactly
the same shape -- and that have already been got wrong once each.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))


def _module(name):
    return pytest.importorskip(name, reason="scripts/ not importable")


@pytest.fixture
def master(tmp_path):
    """A master.csv with a wide, deliberately lopsided n_epochs distribution:
    900 sparse objects and 100 dense ones, plus 5 deep-drilling-like monsters
    above any sane screen. Stratifying on n_epochs would over-represent the
    dense tail ~9x relative to its true frequency, which is what the
    no-stratification test detects.
    """
    rng = np.random.default_rng(0)
    n_epochs = np.concatenate([
        rng.integers(50, 300, 900),
        rng.integers(700, 890, 100),
        rng.integers(2400, 48000, 5),
    ])
    df = pd.DataFrame({
        "object_id": [f"OBJ{i:05d}" for i in range(len(n_epochs))],
        "rmag": rng.uniform(17.0, 21.0, len(n_epochs)),
        "n_epochs": n_epochs,
        "baseline_days": rng.uniform(3300.0, 3650.0, len(n_epochs)),
        "dec": rng.uniform(-60.0, 15.0, len(n_epochs)),
        "matched": True,
    })
    path = tmp_path / "master.csv"
    df.to_csv(path, index=False)
    return str(path)


class TestPopulationScreen:
    def test_eligible_objects_keeps_the_whole_matched_population(self, master):
        R = _module("make_real_cadence_csv")
        pop = R.eligible_objects(master)
        assert len(pop) == 1005

    def test_max_n_epochs_removes_the_deep_drilling_tail(self, master):
        R = _module("make_real_cadence_csv")
        pop = R.eligible_objects(master, max_n_epochs=1000)
        assert len(pop) == 1000
        assert pop["n_epochs"].max() <= 1000

    def test_lsst_screens_by_default_and_ztf_does_not(self):
        """The poison object (XMMC_149.82382+2.22786, 47,950 epochs, ~135 h
        CPU per crash) reached a campaign because the screen was opt-in."""
        R = _module("make_real_cadence_csv")
        assert R.MAX_N_EPOCHS_BY_SURVEY["lsst"] == 1000
        assert R.MAX_N_EPOCHS_BY_SURVEY["ztf"] is None


class TestSamplingIsSimpleRandomAndFreshPerCell:
    def _rows(self, R, master, n_per_cell=25, rep_start=0, seed=7):
        pop = R.eligible_objects(master, max_n_epochs=1000)
        rows, _ = R.build_rows(
            highalpha=[-2.0, -2.5, -3.0, -3.5], period_a1=None,
            population=pop, n_per_cell=n_per_cell, first_id=0, seed=seed,
            survey="lsst", fixed={}, period_max=9.0, rep_start=rep_start,
        )
        return pd.DataFrame(rows)

    def test_cells_draw_different_objects(self, master):
        """The old design reused ONE pool identically in every cell, which
        made the naive binomial SE on an FPR a conditional SE."""
        R = _module("make_real_cadence_csv")
        df = self._rows(R, master)
        per_cell = [
            set(g["cadence_source"]) for _, g in df.groupby("highalpha")
        ]
        for i in range(len(per_cell)):
            for j in range(i + 1, len(per_cell)):
                assert per_cell[i] != per_cell[j]

    def test_no_object_repeats_within_a_cell(self, master):
        R = _module("make_real_cadence_csv")
        df = self._rows(R, master)
        for _, g in df.groupby("highalpha"):
            assert g["cadence_source"].is_unique

    def test_draws_are_not_stratified_on_n_epochs(self, master):
        """Simple random sampling reproduces the population's own n_epochs
        distribution; quartile stratification would force ~25% of draws into
        the dense tail, which is only ~10% of the population."""
        R = _module("make_real_cadence_csv")
        pop = R.eligible_objects(master, max_n_epochs=1000)
        truth = float((pop["n_epochs"] > 700).mean())
        rows, _ = R.build_rows(
            highalpha=[-2.0], period_a1=None, population=pop,
            n_per_cell=400, first_id=0, seed=11, survey="lsst", fixed={},
            period_max=9.0,
        )
        drawn = float((pd.DataFrame(rows)["n_epochs"] > 700).mean())
        # 400 draws without replacement from 1000; a quartile-stratified
        # sample would sit at ~0.25, far outside this band.
        assert abs(drawn - truth) < 0.05
        assert drawn < 0.18

    def test_seed_makes_the_draw_reproducible(self, master):
        R = _module("make_real_cadence_csv")
        a = self._rows(R, master, seed=3)
        b = self._rows(R, master, seed=3)
        pd.testing.assert_frame_equal(a, b)

    def test_adding_a_slope_does_not_redraw_existing_cells(self, master):
        """Per-cell spawned streams, not one stream consumed in grid order:
        extending the sweep must not silently change the cells already run."""
        R = _module("make_real_cadence_csv")
        pop = R.eligible_objects(master, max_n_epochs=1000)
        kw = dict(period_a1=None, population=pop, n_per_cell=10, first_id=0,
                  seed=5, survey="lsst", fixed={}, period_max=9.0)
        four, _ = R.build_rows(highalpha=[-2.0, -2.5, -3.0, -3.5], **kw)
        five, _ = R.build_rows(highalpha=[-2.0, -2.5, -3.0, -3.5, -4.0], **kw)
        shared = pd.DataFrame(four)["cadence_source"]
        assert list(shared) == list(pd.DataFrame(five)["cadence_source"][:len(shared)])


class TestStagedExtension:
    def test_rep_start_extension_never_refits_a_stage1_object(self, master):
        """20 -> 50 -> 100 staging must not re-draw an object a cell already
        used, or the extra reps are not independent draws."""
        R = _module("make_real_cadence_csv")
        pop = R.eligible_objects(master, max_n_epochs=1000)
        kw = dict(highalpha=[-2.0, -3.5], period_a1=None, population=pop,
                  seed=42, survey="lsst", fixed={}, period_max=9.0)
        stage1, _ = R.build_rows(n_per_cell=20, first_id=0, rep_start=0, **kw)
        stage2, _ = R.build_rows(n_per_cell=30, first_id=1000, rep_start=20, **kw)
        s1, s2 = pd.DataFrame(stage1), pd.DataFrame(stage2)
        for ha in (-2.0, -3.5):
            a = set(s1[s1["highalpha"] == ha]["cadence_source"])
            b = set(s2[s2["highalpha"] == ha]["cadence_source"])
            assert len(a) == 20 and len(b) == 30
            assert not (a & b)

    def test_a_leading_slice_of_a_cell_is_a_valid_random_subset(self, master):
        """Rows are in random order within a cell, which is what makes the
        `_n20.csv` stage-1 convention (first 20 rows of each block) sound."""
        R = _module("make_real_cadence_csv")
        pop = R.eligible_objects(master, max_n_epochs=1000)
        kw = dict(highalpha=[-2.0], period_a1=None, population=pop,
                  first_id=0, seed=9, survey="lsst", fixed={}, period_max=9.0)
        full, _ = R.build_rows(n_per_cell=50, rep_start=0, **kw)
        head, _ = R.build_rows(n_per_cell=20, rep_start=0, **kw)
        full_ids = [r["cadence_source"] for r in full][:20]
        assert full_ids == [r["cadence_source"] for r in head]

    def test_extension_beyond_the_population_is_refused(self, master):
        R = _module("make_real_cadence_csv")
        pop = R.eligible_objects(master, max_n_epochs=1000)
        with pytest.raises(ValueError, match="only 1000"):
            R.build_rows(
                highalpha=[-2.0], period_a1=None, population=pop,
                n_per_cell=900, first_id=0, seed=1, survey="lsst", fixed={},
                period_max=9.0, rep_start=200,
            )


class TestPeriodPriorTravelsWithTheScenario:
    def test_per_survey_bounds(self):
        R = _module("make_real_cadence_csv")
        S = _module("make_slope_robustness_csv")
        assert R.P_MAX_BY_SURVEY == {"ztf": 6.0, "lsst": 9.0}
        # The synthetic cadence's 9.53 yr baseline sits with LSST WFD's, not
        # with ZTF's 6.07-7.59.
        assert S.PERIOD_MAX_DEFAULT == R.P_MAX_BY_SURVEY["lsst"]

    def test_stamped_column_is_used_when_no_flag_is_given(self):
        run_sim = _module("run_sim")
        df = pd.DataFrame({"period": [np.nan] * 3, "period_max": [6.0] * 3})
        assert run_sim.resolve_period_prior(df, None) == 6.0

    def test_flag_contradicting_the_column_is_refused(self):
        run_sim = _module("run_sim")
        df = pd.DataFrame({"period": [np.nan] * 3, "period_max": [6.0] * 3})
        with pytest.raises(ValueError, match="contradicts"):
            run_sim.resolve_period_prior(df, 9.0)

    def test_flag_agreeing_with_the_column_is_fine(self):
        run_sim = _module("run_sim")
        df = pd.DataFrame({"period": [np.nan] * 3, "period_max": [6.0] * 3})
        assert run_sim.resolve_period_prior(df, 6.0) == 6.0

    def test_a_scenario_may_not_mix_two_priors(self):
        run_sim = _module("run_sim")
        df = pd.DataFrame({"period": [np.nan] * 2, "period_max": [6.0, 9.0]})
        with pytest.raises(ValueError, match="distinct period_max"):
            run_sim.resolve_period_prior(df, None)

    def test_legacy_csv_without_the_column_still_runs(self):
        run_sim = _module("run_sim")
        df = pd.DataFrame({"period": [np.nan] * 2})
        assert run_sim.resolve_period_prior(df, None) == run_sim.PERIOD_PRIOR[1]

    def test_injected_period_outside_the_stamped_prior_is_refused(self):
        """MB3.1's original check must survive the new precedence logic."""
        run_sim = _module("run_sim")
        df = pd.DataFrame({"period": [7.5, 7.5], "period_max": [6.0, 6.0]})
        with pytest.raises(ValueError, match="outside the sine period prior"):
            run_sim.resolve_period_prior(df, None)


class TestSlopeAxisIsSharedAcrossCampaigns:
    def test_all_three_builders_use_the_same_four_slopes(self):
        R = _module("make_real_cadence_csv")
        S = _module("make_slope_robustness_csv")
        M = _module("make_multiband_csv")
        expected = (-2.0, -2.5, -3.0, -3.5)
        assert tuple(float(v) for v in R.HIGHALPHA_DEFAULT.split(",")) == expected
        assert tuple(float(v) for v in S.HIGHALPHA_DEFAULT.split(",")) == expected
        assert tuple(M.NULL_HIGHALPHA) == expected


def _summary(cells, pair="DRW", converged=1.0):
    """cells: {highalpha: (detections, n)} -> an aggregate_results-shaped dict."""
    table = {}
    for ha, (k, n) in cells.items():
        table[f"highalpha={ha:g}"] = {
            pair: {
                "n": n,
                "n_dropped_unconverged": 0,
                "converged_frac": converged,
                "outcomes": {"detect": k, "inconclusive": n - k, "refute": 0},
            }
        }
    return {"table": table}


class TestStoppingRule:
    """The rule is pre-registered; these tests are what stops it drifting to
    fit whatever the first stage happened to show."""

    def _decide(self, cells, stage, pair="DRW"):
        D = _module("campaign_stage_decision")
        return D.decide(D.cells_from_summary(_summary(cells), pair), stage)

    def test_wilson_interval_is_not_degenerate_at_zero(self):
        D = _module("campaign_stage_decision")
        lo, hi = D.wilson(0, 50)
        assert lo == pytest.approx(0.0, abs=1e-12) and 0.0 < hi < 0.15
        # 2/50. NOTE the design note quotes [0.4%, 10.5%] here, which is the
        # Clopper-Pearson exact interval; Wilson is the slightly wider
        # [1.1%, 13.5%]. Either supports the rule's point -- that n=50 cannot
        # separate 1% from 3% -- and Wilson is used because it behaves at k=0.
        lo, hi = D.wilson(2, 50)
        assert round(lo * 100, 1) == 1.1 and round(hi * 100, 1) == 13.5

    def test_all_cells_low_goes_straight_to_100(self):
        v = self._decide({-3.5: (2, 50), -3.0: (1, 50), -2.0: (0, 50)}, 50)
        assert v["verdict"] == "EXTEND" and v["next_stage"] == 100

    def test_all_cells_low_wins_even_at_stage_20(self):
        """n=20 cannot resolve a uniformly low FPR, so this clause is checked
        before the stage-20 stop condition."""
        v = self._decide({-3.5: (1, 20), -2.0: (0, 20)}, 20)
        assert v["verdict"] == "EXTEND" and v["next_stage"] == 100

    def test_stage20_stops_on_a_strong_separated_trend(self):
        v = self._decide({-3.5: (12, 20), -2.0: (0, 20)}, 20)
        assert v["verdict"] == "STOP"

    def test_stage20_extends_when_intervals_overlap(self):
        """A high steepest cell is NOT enough on its own -- at n=20 the
        intervals must actually separate."""
        v = self._decide({-3.5: (7, 20), -2.0: (4, 20)}, 20)
        assert v["verdict"] == "EXTEND" and v["next_stage"] == 50

    def test_stage20_extends_when_steepest_is_below_30_percent(self):
        v = self._decide({-3.5: (4, 20), -2.0: (0, 20)}, 20)
        assert v["verdict"] == "EXTEND" and v["next_stage"] == 50

    def test_stage50_stops_at_20_percent(self):
        v = self._decide({-3.5: (10, 50), -2.0: (0, 50)}, 50)
        assert v["verdict"] == "STOP"

    def test_stage50_extends_below_20_percent(self):
        v = self._decide({-3.5: (7, 50), -2.0: (0, 50)}, 50)
        assert v["verdict"] == "EXTEND" and v["next_stage"] == 100

    def test_stage100_is_terminal(self):
        v = self._decide({-3.5: (30, 100), -2.0: (2, 100)}, 100)
        assert v["verdict"] == "STOP"

    def test_stage_100_stops_even_when_every_cell_is_low(self):
        """Regression: the 'all cells under 10%' clause was checked before the
        terminal-stage check, so a completed n=100 campaign was told to
        'extend to 100' -- advice that cannot be acted on."""
        v = self._decide({-3.5: (2, 100), -3.0: (2, 100), -2.0: (0, 100)}, 100)
        assert v["verdict"] == "STOP"
        assert "final stage" in v["reason"]

    def test_stage_100_stops_when_all_cells_are_zero(self):
        v = self._decide({-3.5: (0, 100), -2.0: (0, 100)}, 100)
        assert v["verdict"] == "STOP"

    def test_steepest_cell_is_the_most_negative_slope(self):
        D = _module("campaign_stage_decision")
        cells = D.cells_from_summary(
            _summary({-2.0: (0, 20), -3.5: (12, 20), -3.0: (5, 20)}), "drw"
        )
        assert cells[0]["highalpha"] == -3.5
        assert cells[-1]["highalpha"] == -2.0

    def test_thresholds_match_the_pre_registered_design(self):
        D = _module("campaign_stage_decision")
        assert D.STOP_AT_20_STEEPEST == 0.30
        assert D.STOP_AT_50_STEEPEST == 0.20
        assert D.ALL_CELLS_LOW == 0.10
        assert D.STAGES == (20, 50, 100)


class TestBaselineScreen:
    """P_max is chosen at the pool's shortest baseline so that no fit gets a
    period prior its own data cannot support. That was enforced by hand when
    the pool was a pre-filtered list; simple random sampling over the whole
    master file quietly reintroduced short-baseline objects."""

    @pytest.fixture
    def master(self, tmp_path):
        df = pd.DataFrame({
            "object_id": [f"OBJ{i}" for i in range(5)],
            "rmag": 19.0,
            "n_epochs": 500,
            # 3.81 yr (the real ZTF minimum) through 7.59 yr (its maximum)
            "baseline_days": [1392.0, 2000.0, 2192.0, 2500.0, 2772.0],
            "dec": 30.0,
            "matched": True,
        })
        path = tmp_path / "m.csv"
        df.to_csv(path, index=False)
        return str(path)

    def test_objects_shorter_than_the_prior_are_dropped(self, master):
        R = _module("make_real_cadence_csv")
        pop = R.eligible_objects(master, min_baseline_years=6.0)
        # 6.0 yr = 2191.5 d, so only the 2192/2500/2772 d objects survive
        assert len(pop) == 3
        assert pop["baseline_days"].min() >= 6.0 * R.DAYS_PER_YEAR

    def test_screen_is_off_when_not_requested(self, master):
        R = _module("make_real_cadence_csv")
        assert len(R.eligible_objects(master)) == 5

    def test_pair_name_is_case_insensitive(self):
        """aggregate_results.py keys pairs "DRW"; a lowercase --pair must not
        silently report an empty campaign."""
        D = _module("campaign_stage_decision")
        summ = _summary({-3.5: (12, 20), -2.0: (0, 20)})
        assert len(D.cells_from_summary(summ, "drw")) == 2
        assert len(D.cells_from_summary(summ, "DRW")) == 2

    def test_unknown_pair_fails_loudly(self):
        D = _module("campaign_stage_decision")
        with pytest.raises(SystemExit, match="OBPL|no model pair"):
            D.cells_from_summary(_summary({-3.5: (1, 20)}), "OBPL")
