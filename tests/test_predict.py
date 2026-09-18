"""Tests for predict.py's reconstruction of a stored fit.

The multi-band conventions under test are the ones pinned in
tests/test_multiband.py and models.loglike: mu0 is the REFERENCE BAND's
offset in observed units (band_mu[0]), while the sine mean lives in latent
units and is therefore scaled by a_b. Getting that order wrong is the MB1
defect, which a test with unit amplitudes cannot see -- so the amplitudes
here are deliberately non-unit.
"""

from __future__ import annotations

import numpy as np
import pytest

from pioran_periodicity.means import sine_mean
from pioran_periodicity.predict import (
    build_kernel_and_mean,
    is_multiband,
    posterior_predictive,
)

SINGLE_META = {
    "noise": "drw",
    "variant": "sine",
    "band": None,
    "photometric_bands": [],
    "reference_band": "r",
    "n_components": None,
    "basis_function": None,
}
MULTI_META = dict(SINGLE_META, photometric_bands=["g"])


def test_is_multiband_distinguishes_on_photometric_bands():
    assert not is_multiband(SINGLE_META)
    assert is_multiband(MULTI_META)
    # 22 of the 400 ztf_multi objects have no non-reference band and are
    # therefore single-band fits despite the campaign being multi-band
    assert not is_multiband(dict(MULTI_META, photometric_bands=[]))


def test_single_band_mean_includes_mu0():
    params = {
        "log10_variance": -1.0,
        "log10_fbend": -1.0,
        "mu0": 0.7,
        "A_cos": 0.0,
        "A_sin": 0.0,
        "period": 2.0,
    }
    _, mean_func = build_kernel_and_mean(SINGLE_META, params)
    t = np.linspace(0.0, 1.0, 5)
    # A_cos = A_sin = 0 kills the sine, leaving the constant mu0 alone
    assert mean_func(t) == pytest.approx(np.full(5, 0.7))


def test_multiband_mean_excludes_mu0():
    """mu0 belongs in band_mu[0], not in the latent-unit mean function."""
    params = {
        "log10_variance": -1.0,
        "log10_fbend": -1.0,
        "mu0": 0.7,
        "a_g": 1.5,
        "mu_g": 0.2,
        "A_cos": 0.0,
        "A_sin": 0.0,
        "period": 2.0,
    }
    _, mean_func = build_kernel_and_mean(MULTI_META, params)
    t = np.linspace(0.0, 1.0, 5)
    assert mean_func(t) == pytest.approx(np.zeros(5))


def test_missing_err_scale_defaults_to_one():
    """No v2 fit stores err_scale; reconstruction must not require it."""
    params = {"log10_variance": -1.0, "log10_fbend": -1.0, "mu0": 0.0}
    meta = dict(SINGLE_META, variant="plain")
    rng = np.random.default_rng(0)
    t = np.sort(rng.uniform(0, 5, 40)) + np.arange(40) * 1e-9
    y = rng.normal(0, 0.3, 40)
    yerr = np.full(40, 0.05)
    mu, sd = posterior_predictive(meta, params, t, y, yerr, t_grid=t, need_std=True)
    assert np.all(np.isfinite(mu)) and np.all(np.isfinite(sd))


def test_multiband_prediction_is_in_reference_band_units():
    """With a_g != 1 and mu_g != 0, predicting on reference-band epochs must
    not inherit the g-band offset."""
    rng = np.random.default_rng(1)
    n = 60
    t = np.sort(rng.uniform(0, 5, n)) + np.arange(n) * 1e-9
    band_labels = np.where(np.arange(n) % 2 == 0, "r", "g")
    params = {
        "log10_variance": -1.0,
        "log10_fbend": -1.0,
        "mu0": 0.4,
        "a_g": 1.6,
        "mu_g": 0.9,
    }
    meta = dict(MULTI_META, variant="plain")
    # band_mu = [mu0, mu_g] = [0.4, 0.9]: mu_g IS the g-band offset, not an
    # increment on mu0 (models.py:564). With the latent process near zero the
    # g points sit at 0.9 and the r points at 0.4; a correct reconstruction
    # divides out a_g and removes mu_g, leaving the reference-band prediction
    # at mu0. Getting this wrong pulls the prediction towards ~0.52.
    y = np.where(band_labels == "g", 0.9, 0.4) + rng.normal(0, 0.01, n)
    yerr = np.full(n, 0.05)
    mu, _ = posterior_predictive(
        meta, params, t, y, yerr, t_grid=t, band_labels=band_labels
    )
    assert np.median(mu) == pytest.approx(0.4, abs=0.1)


def test_single_band_constant_mean_applies_nonzero_mu0():
    """shape is None (variant="plain") but mu0 != 0 must still be applied.

    Guards the behaviour change vs. the old paper/plot_fits.py, which
    returned mean_func=None for plain variants so mu0 was never subtracted
    -- models.py:545-554 always includes it. The existing mu0 test
    (test_single_band_mean_includes_mu0) goes through the sine branch with
    A_cos=A_sin=0, and test_missing_err_scale_defaults_to_one uses mu0=0.0,
    so neither exercises "shape is None, mu0 != 0" specifically.
    """
    params = {"log10_variance": -1.0, "log10_fbend": -1.0, "mu0": 0.6}
    meta = dict(SINGLE_META, variant="plain")
    _, mean_func = build_kernel_and_mean(meta, params)
    t = np.linspace(0.0, 1.0, 5)
    assert mean_func(t) == pytest.approx(np.full(5, 0.6))


def test_multiband_sine_reconstruction_matches_hand_computed_residuals():
    """Discriminates the MB1 mean-before-division defect AND a missing /a.

    test_multiband_prediction_is_in_reference_band_units uses variant=
    "plain", so mean_func is None there and the "subtract the mean after
    dividing" step (predict.py:160-162) never runs -- kernels.py:377-382
    states outright that a mean_func=None test cannot see the MB1 defect.
    That test also picks y already equal to band_mu, so both bands'
    residuals are ~0 before any division and deleting the "/ a" at
    predict.py:160 would still pass. This test fixes both gaps: variant=
    "sine" with non-zero A_cos/A_sin turns mean_func on, a_g is far from 1,
    and the injected values are not pre-cancelled.

    Data is built as y_i = band_mu[b_i] + a_[b_i] * (u(t_i) + m(t_i)) for
    an arbitrary smooth "shared process" u(t) (well away from zero; its
    value is a free choice -- see below) and the model's own sine mean
    m(t). With near-noiseless conditioning (tiny yerr) and t_grid == t, the
    GP posterior mean nearly interpolates the conditioning residuals AT the
    conditioning points, independent of the kernel's actual shape. The
    correct reconstruction removes m(t) in latent units then adds the same
    m(t) back unchanged, so u(t_i) and m(t_i) cancel completely out of the
    predicted value, which must equal, at every point regardless of band,

        mu_pred(t_i) ~= mu0 + (y_i - band_mu[b_i]) / a_[b_i]

    Under the MB1 defect (mean subtracted before dividing by a_b) that
    cancellation breaks specifically where a_b != 1 and m(t) != 0, i.e. the
    g-band points, biasing the prediction there by m(t_i)*(1 - 1/a_g).
    Under a missing "/ a", every g-band residual is wrong by a factor of
    a_g. Both mutations are exercised and shown to fail this test in the
    task report (fix round 1); this file only carries the correct-code
    assertion.
    """
    rng = np.random.default_rng(2)
    n = 80
    t = np.sort(rng.uniform(0, 6, n)) + np.arange(n) * 1e-9
    band_labels = np.where(np.arange(n) % 2 == 0, "r", "g")

    mu0, a_g, mu_g = 0.4, 1.6, 0.9
    A_cos, A_sin, period = 0.35, -0.25, 2.2
    params = {
        "log10_variance": -1.0,
        "log10_fbend": -1.0,
        "mu0": mu0,
        "a_g": a_g,
        "mu_g": mu_g,
        "A_cos": A_cos,
        "A_sin": A_sin,
        "period": period,
    }
    meta = dict(MULTI_META, variant="sine")

    band_mu = np.where(band_labels == "g", mu_g, mu0)
    band_amp = np.where(band_labels == "g", a_g, 1.0)
    # an arbitrary smooth "shared process", well away from zero -- its
    # value cancels out of the expected result (see the docstring above)
    u = 0.5 * np.sin(2.0 * np.pi * t / 1.7) + 0.2 * t
    m = sine_mean(t, A_cos, A_sin, period)
    y = band_mu + band_amp * (u + m)
    yerr = np.full(n, 1e-3)  # near-noiseless: forces near-exact interpolation

    mu, _ = posterior_predictive(
        meta, params, t, y, yerr, t_grid=t, band_labels=band_labels
    )
    expected = mu0 + (y - band_mu) / band_amp
    assert mu == pytest.approx(expected, abs=0.03)


def test_beta_sine_is_rejected():
    params = {"log10_variance": -1.0, "log10_fbend": -1.0, "mu0": 0.0, "beta_sine": 0.3}
    with pytest.raises(NotImplementedError, match="beta_sine"):
        build_kernel_and_mean(MULTI_META, params)
