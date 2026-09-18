"""Smoke tests for the simulated-light-curve figure scripts.

These run real GP predictions on a deliberately tiny light curve -- they
check the plumbing (npz loading, band handling, file output), not the
science, which is covered by tests/test_predict.py.
"""

from __future__ import annotations

import json
import math
import os
import sys

import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import plot_sim_examples  # noqa: E402
from plot_sim_examples import (  # noqa: E402
    MODELS,
    load_lightcurve,
    make_corner_figure,
    make_panels_figure,
)
from pioran_periodicity.means import sine_mean  # noqa: E402


def _write_npz(lc_dir, lc_id, n=40, bands=None):
    rng = np.random.default_rng(3)
    t = np.sort(rng.uniform(0, 5, n)) + np.arange(n) * 1e-9
    payload = dict(
        t=t,
        y=rng.normal(0, 0.2, n),
        yerr=np.full(n, 0.05),
        highalpha=-2.0,
        n_points=n,
    )
    if bands is not None:
        payload["band"] = np.array([bands[i % len(bands)] for i in range(n)])
    np.savez(os.path.join(lc_dir, f"{lc_id}.npz"), **payload)


def _write_fits(results_dir, lc_id):
    rng = np.random.default_rng(4)
    for i, model in enumerate(MODELS):
        samples = {
            "log10_variance": rng.normal(-1, 0.1, 200).tolist(),
            "log10_fbend": rng.normal(-1, 0.1, 200).tolist(),
            "mu0": rng.normal(0, 0.05, 200).tolist(),
        }
        if "sine" in model:
            samples |= {
                "A_cos": rng.normal(0, 0.1, 200).tolist(),
                "A_sin": rng.normal(0, 0.1, 200).tolist(),
                "period": rng.uniform(1, 3, 200).tolist(),
            }
        payload = {
            "model": model,
            "logz": 100.0 + i,
            "logzerr": 0.2,
            "samples": samples,
            "ess": 2000.0,
            "converged": True,
            "meta": {
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


def _write_fits_with_logz(results_dir, lc_id, logz, logzerr):
    """Like _write_fits, but with caller-chosen (distinct, sign-varying)
    logz/logzerr per model, so a test can verify the evidence text box's
    Bayes-factor arithmetic against values it -- not the function under
    test -- computed.
    """
    rng = np.random.default_rng(6)
    for model in MODELS:
        samples = {
            "log10_variance": rng.normal(-1, 0.1, 50).tolist(),
            "log10_fbend": rng.normal(-1, 0.1, 50).tolist(),
            "mu0": rng.normal(0, 0.05, 50).tolist(),
        }
        if "sine" in model:
            samples |= {
                "A_cos": rng.normal(0, 0.1, 50).tolist(),
                "A_sin": rng.normal(0, 0.1, 50).tolist(),
                "period": rng.uniform(1, 3, 50).tolist(),
            }
        payload = {
            "model": model,
            "logz": logz[model],
            "logzerr": logzerr[model],
            "samples": samples,
            "ess": 2000.0,
            "converged": True,
            "meta": {
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


def _write_fits_for_panels(results_dir, lc_id):
    """Fixture for the make_panels_figure plumbing test: constant-valued
    samples (so the posterior median is exact and known without computing
    it), one sine model using the legacy A1/A2 keys and the other the
    current A_cos/A_sin keys, so both branches of the rename fallback in
    make_panels_figure fire in a single run.
    """
    n = 50
    base = {
        "log10_variance": np.full(n, -1.0).tolist(),
        "log10_fbend": np.full(n, -1.0).tolist(),
        "mu0": np.full(n, 0.05).tolist(),
    }
    sine_extra = {
        "drw_sine": {  # legacy keys
            "A1": np.full(n, 0.2).tolist(),
            "A2": np.full(n, -0.1).tolist(),
            "period": np.full(n, 2.5).tolist(),
        },
        "obpl_sine": {  # current keys
            "A_cos": np.full(n, -0.15).tolist(),
            "A_sin": np.full(n, 0.3).tolist(),
            "period": np.full(n, 1.8).tolist(),
        },
    }
    for model in MODELS:
        samples = dict(base)
        if model in sine_extra:
            samples |= sine_extra[model]
        payload = {
            "model": model,
            "logz": 10.0,
            "logzerr": 0.2,
            "samples": samples,
            "ess": 2000.0,
            "converged": True,
            "meta": {
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


def test_load_lightcurve_without_band(tmp_path):
    _write_npz(tmp_path, 1)
    lc = load_lightcurve(str(tmp_path), 1)
    assert lc["band"] is None
    assert lc["t"].shape == lc["y"].shape == lc["yerr"].shape == (40,)


def test_load_lightcurve_with_band(tmp_path):
    _write_npz(tmp_path, 2, bands=["g", "r"])
    lc = load_lightcurve(str(tmp_path), 2)
    assert set(lc["band"]) == {"g", "r"}


def test_corner_figure_is_written(tmp_path):
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    _write_fits(str(results_dir), 7)
    out = tmp_path / "7_corner.png"
    make_corner_figure(str(results_dir), 7, str(out), caption="test cell")
    assert out.exists() and out.stat().st_size > 0


def test_corner_figure_evidence_textbox_content(tmp_path, monkeypatch):
    """The evidence text box must carry the right logZ lines and the right
    Bayes factors, including sign and pairing (drw/drw_sine -> DRW,
    obpl/obpl_sine -> OBPL) -- not just exist. Expected strings are built
    here from the fixture's own logz/logzerr, independently of
    make_corner_figure's arithmetic, and the four logz values are chosen so
    that a DRW/OBPL pairing swap or a sign flip would change the expected
    text (unlike a fixture where every pair differs by the same amount).
    """
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    logz = {"drw": 50.0, "drw_sine": 55.0, "obpl": 80.0, "obpl_sine": 77.0}
    logzerr = {"drw": 0.30, "drw_sine": 0.25, "obpl": 0.40, "obpl_sine": 0.35}
    _write_fits_with_logz(str(results_dir), 9, logz, logzerr)
    out = tmp_path / "9_corner.png"

    # Capture the string handed to SubFigure.text (the evidence box lives on
    # its own subfigure -- see make_corner_figure) while still calling the
    # real method so the figure renders and saves normally.
    captured = {}
    original_text = matplotlib.figure.SubFigure.text

    def _capture_text(self, x, y, s, *args, **kwargs):
        captured["s"] = s
        return original_text(self, x, y, s, *args, **kwargs)

    monkeypatch.setattr(matplotlib.figure.SubFigure, "text", _capture_text)

    make_corner_figure(str(results_dir), 9, str(out), caption="content check")

    assert "s" in captured, "make_corner_figure never called SubFigure.text"
    text = captured["s"]

    # Independently computed from the fixture, per
    # aggregate_results.build_table's convention: log10 B = (logz_base -
    # logz_sine) / ln(10). DRW and OBPL are chosen to differ in both value
    # and sign, so a swapped pairing changes both numbers, not just their
    # order.
    expected_bf_drw = (logz["drw"] - logz["drw_sine"]) / math.log(10)
    expected_bf_obpl = (logz["obpl"] - logz["obpl_sine"]) / math.log(10)
    assert expected_bf_drw == pytest.approx(-2.1715, abs=1e-3)
    assert expected_bf_obpl == pytest.approx(1.3029, abs=1e-3)

    # Build each line the way the brief's target format specifies (not by
    # calling any of make_corner_figure's own formatting code), then check
    # each *whole line* is present verbatim. A whole-line check -- not a
    # per-number "somewhere in text" check -- is required here: with this
    # fixture a DRW/OBPL pairing swap still leaves both numbers -2.17 and
    # 1.30 present in the text, just attached to the wrong label, so
    # checking the numbers in isolation would not catch a swap. Checking
    # the full "label = value" line does.
    expected_logz_lines = [
        f"logZ  {model:<11s}= {logz[model]:.2f} +- {logzerr[model]:.2f}"
        for model in MODELS
    ]
    expected_bf_line = (
        f"log10 B (DRW)  = {expected_bf_drw:<10.2f}"
        f"log10 B (OBPL) = {expected_bf_obpl:.2f}"
    )
    for line in expected_logz_lines:
        assert line in text
    assert expected_bf_line in text
    # And nothing else claims to be that line with a different pairing.
    assert text == "\n".join(expected_logz_lines + [expected_bf_line])


def test_panels_figure_plumbing(tmp_path, monkeypatch):
    """Exercise make_panels_figure's own logic -- the A1/A2 legacy-key
    fallback, the sine-only-for-+sine-models branch, axes.flat/MODELS
    alignment, and output-directory creation -- without a real GP/Julia
    call: posterior_predictive is monkeypatched to a cheap canned function.
    """
    lc_dir = tmp_path / "lc"
    lc_dir.mkdir()
    _write_npz(str(lc_dir), 8, n=20)

    results_dir = tmp_path / "results"
    results_dir.mkdir()
    _write_fits_for_panels(str(results_dir), 8)

    def _fake_posterior_predictive(
        meta, params, t, y, yerr, t_grid, band_labels=None, need_std=False
    ):
        mu = np.full_like(np.asarray(t_grid, dtype=float), params.get("mu0", 0.0))
        sd = np.full_like(mu, 0.1) if need_std else None
        return mu, sd

    monkeypatch.setattr(
        plot_sim_examples, "posterior_predictive", _fake_posterior_predictive
    )

    captured = {}
    original_subplots = plt.subplots

    def _capture_subplots(*args, **kwargs):
        fig, axes = original_subplots(*args, **kwargs)
        captured["fig"] = fig
        captured["axes"] = axes
        return fig, axes

    monkeypatch.setattr(plt, "subplots", _capture_subplots)

    # nested, not-yet-existing output directory: exercises os.makedirs
    out = tmp_path / "nested" / "dir" / "8_panels.png"
    rng = np.random.default_rng(1)
    make_panels_figure(str(results_dir), str(lc_dir), 8, str(out), "panels smoke", rng)

    assert out.exists() and out.stat().st_size > 0

    axes = captured["axes"]
    assert axes.shape == (2, 2)
    # axes.flat/MODELS alignment: panel i's title must name model i's family.
    for ax, model in zip(axes.flat, MODELS):
        assert plot_sim_examples.PANEL_TITLES[model] in ax.get_title()

        _, labels = ax.get_legend_handles_labels()
        has_sine_line = "periodic mean (median)" in labels
        assert has_sine_line == ("sine" in model)

        if model in ("drw_sine", "obpl_sine"):
            # Every curve in a panel shares the same t_grid x-values, so any
            # plotted line's xdata recovers it.
            t_grid = ax.get_lines()[0].get_xdata()
            # legacy A1/A2 keys for drw_sine, current A_cos/A_sin for
            # obpl_sine (constant samples, see _write_fits_for_panels), so
            # the median sine curve is exact and independently
            # reproducible here -- this is what actually exercises the
            # rename fallback for both key spellings.
            a_cos, a_sin, period = (
                (0.2, -0.1, 2.5) if model == "drw_sine" else (-0.15, 0.3, 1.8)
            )
            expected = sine_mean(t_grid, a_cos, a_sin, period)
            (line,) = [
                ln
                for ln in ax.get_lines()
                if ln.get_label() == "periodic mean (median)"
            ]
            np.testing.assert_allclose(line.get_ydata(), expected)
