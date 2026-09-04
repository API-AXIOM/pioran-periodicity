"""Multi-band (shared-latent-process) model tests.

Model under test: ``y_b(t) = mu_b + a_b * x(t) + noise_b(t)``, one shared
latent GP ``x(t) ~ GP(0, k_theta)`` read out through a per-band amplitude
``a_b`` and mean ``mu_b``. Since ZTF/LSST observe one band per epoch (never
simultaneously), this reduces to a per-point scalar rescale rather than a
genuine k-dimensional multivariate GP, so ``gp_log_likelihood`` can be reused
unmodified on rescaled data plus a Jacobian term.

The load-bearing regression here is
``test_rescale_trick_matches_brute_force``: it checks that likelihood
identity against a brute-force dense N x N reference covariance
``Sigma_ij = a_bi a_bj k(t_i,t_j) + diag(sigma^2)``, computed without any
pioranpy/Julia call. That is the one part of the scheme that cannot be
verified by reading the Python wrapper alone -- it depends on
``pa.ScalableGP`` behaving as a plain zero-mean GP evaluation with no hidden
per-call renormalisation. Ported from the validated prototype
(``pioran_periodicity_ai/workspace/tests/test_multiband_rescale_prototype.py``,
2026-08-14), rewired to call the production
``gp_log_likelihood_multiband``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pioran_periodicity.cadence import CadenceLibrary
from pioran_periodicity.inference import SamplerSettings, run_nested
from pioran_periodicity.kernels import (
    drw_kernel,
    gp_log_likelihood,
    gp_log_likelihood_multiband,
)
from pioran_periodicity.models import PriorConfig, build_family
from pioran_periodicity.multiband import (
    EFFECTIVE_WAVELENGTHS,
    BandEncoding,
    cadence_to_multiband_series,
    power_law_band_amplitudes,
)
from pioran_periodicity.priors import LogNormal, Parameter, PriorTransform

RNG = np.random.default_rng(20260814)


# ===========================================================================
# rescale-trick likelihood identity  (ported prototype regression)
# ===========================================================================


def drw_covariance(t, log10_variance, log10_fbend):
    """Closed-form DRW covariance (k(tau) = var * exp(-2*pi*f_bend*|tau|)).

    Reference side only -- never calls into pioranpy/Julia. ``t``: (n,).
    Returns (n, n).
    """
    variance = 10.0**log10_variance
    rate = 2.0 * np.pi * 10.0**log10_fbend
    tau = np.abs(t[:, None] - t[None, :])  # (n, n)
    return variance * np.exp(-rate * tau)


def brute_force_logpdf(t, r, sigma2, band_amp, log10_variance, log10_fbend):
    """log p(r) under Sigma = outer(a,a) * K + diag(sigma2), dense Cholesky.

    All inputs 1-D of length n except the (n, n) intermediates.
    """
    n = len(t)
    K = drw_covariance(t, log10_variance, log10_fbend)  # (n, n)
    Sigma = np.outer(band_amp, band_amp) * K + np.diag(sigma2)  # (n, n)
    L = np.linalg.cholesky(Sigma)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, r))
    logdet = 2.0 * np.sum(np.log(np.diag(L)))
    return -0.5 * (r @ alpha + logdet + n * np.log(2.0 * np.pi))


def _make_toy_case(n_per_band, band_amp_true, seed):
    """Interleaved single-band-per-epoch toy data. Returns (t, r, sigma2, band)."""
    rng = np.random.default_rng(seed)
    n_bands = len(band_amp_true)
    t = np.concatenate(
        [np.sort(rng.uniform(0, 50, n_per_band)) for _ in range(n_bands)]
    )
    band = np.repeat(np.arange(n_bands), n_per_band)
    order = np.argsort(t)
    t, band = t[order], band[order]
    # jitter ties apart (drw_kernel/pioranpy assume distinct times)
    t = t + np.arange(len(t)) * 1e-9

    sigma2 = rng.uniform(0.02, 0.08, len(t)) ** 2
    # arbitrary (not GP-drawn) residuals -- this tests a likelihood
    # *identity*, not sampling statistics
    r = rng.normal(0, 1, len(t)) * np.asarray(band_amp_true)[band]
    return t, r, sigma2, band


@pytest.mark.parametrize("n_bands,n_per_band", [(2, 8), (3, 6), (4, 5)])
def test_rescale_trick_matches_brute_force(n_bands, n_per_band):
    """Covers the covariance and the Jacobian ONLY -- mean_func is None here.

    Do not read this as validating the whole multi-band likelihood: with no
    mean function the "mean in latent units" and "mean in observed units"
    conventions coincide, and this test passed throughout the MB1 defect
    (2026-08 to 2026-09-03). The mean-function conventions are pinned in
    tests/test_model_conventions.py.
    """
    band_amp_true = RNG.uniform(0.4, 2.5, n_bands)
    band_amp_true[0] = 1.0  # pinned reference band
    band_mu_true = np.zeros(n_bands)
    log10_variance, log10_fbend = -0.5, -1.0

    t, r, sigma2, band = _make_toy_case(n_per_band, band_amp_true, seed=1)

    ref = brute_force_logpdf(
        t, r, sigma2, band_amp_true[band], log10_variance, log10_fbend
    )
    got = gp_log_likelihood_multiband(
        drw_kernel(log10_variance, log10_fbend),
        t,
        r,
        np.sqrt(sigma2),
        band,
        band_amp_true,
        band_mu_true,
    )
    assert got == pytest.approx(ref, rel=1e-6, abs=1e-6)


def test_band_mu_is_subtracted_before_the_gp():
    """Non-zero mu_b must shift the residuals, matching the brute-force
    reference evaluated on the already-offset data."""
    band_amp_true = np.array([1.0, 1.7, 0.6])
    band_mu_true = np.array([0.0, 0.35, -0.2])
    log10_variance, log10_fbend = -0.5, -1.0

    t, r, sigma2, band = _make_toy_case(6, band_amp_true, seed=11)
    y = r + band_mu_true[band]  # (n,), observed data carrying band offsets

    ref = brute_force_logpdf(
        t, r, sigma2, band_amp_true[band], log10_variance, log10_fbend
    )
    got = gp_log_likelihood_multiband(
        drw_kernel(log10_variance, log10_fbend),
        t,
        y,
        np.sqrt(sigma2),
        band,
        band_amp_true,
        band_mu_true,
    )
    assert got == pytest.approx(ref, rel=1e-6, abs=1e-6)


def test_unit_amplitudes_reduce_to_plain_single_band():
    """All amplitudes 1 and means 0 must reduce exactly to the existing
    (already-validated) single-series gp_log_likelihood, zero Jacobian."""
    n = 20
    rng = np.random.default_rng(2)
    t = np.sort(rng.uniform(0, 50, n)) + np.arange(n) * 1e-9
    y = rng.normal(0, 1, n)
    yerr = rng.uniform(0.02, 0.08, n)
    band = rng.integers(0, 3, n)
    log10_variance, log10_fbend = -0.3, -0.8

    kernel = drw_kernel(log10_variance, log10_fbend)
    direct = gp_log_likelihood(kernel, t, y, yerr, mean_func=None)
    via_trick = gp_log_likelihood_multiband(
        kernel, t, y, yerr, band, np.ones(3), np.zeros(3)
    )
    assert via_trick == pytest.approx(direct, rel=1e-10, abs=1e-10)


# ===========================================================================
# LogNormal prior
# ===========================================================================


class TestLogNormal:
    def test_median_is_ten_to_the_mu(self):
        assert LogNormal(0.0, 0.3).transform(0.5, {}) == pytest.approx(1.0)
        assert LogNormal(1.0, 0.3).transform(0.5, {}) == pytest.approx(10.0)

    def test_sigma_is_in_log10_units(self):
        # u = normal 1-sigma quantile -> exactly one sigma_log10 in log10
        u = 0.8413447460685429
        got = LogNormal(0.0, 0.3).transform(u, {})
        assert np.log10(got) == pytest.approx(0.3, rel=1e-6)

    def test_monotone_and_positive(self):
        u = np.linspace(0.01, 0.99, 25)
        vals = np.array([LogNormal(0.0, 0.3).transform(ui, {}) for ui in u])
        assert np.all(vals > 0)
        assert np.all(np.diff(vals) > 0)

    def test_rejects_nonpositive_sigma(self):
        with pytest.raises(ValueError):
            LogNormal(0.0, 0.0)

    def test_works_inside_prior_transform(self):
        pt = PriorTransform([Parameter("a_g", LogNormal(0.0, 0.3))])
        assert pt.as_dict(pt(np.array([0.5])))["a_g"] == pytest.approx(1.0)


# ===========================================================================
# BandEncoding
# ===========================================================================


class TestBandEncoding:
    def test_reference_is_most_observed_band(self):
        enc = BandEncoding.from_counts(["g"] * 3 + ["r"] * 7 + ["i"] * 5)
        assert enc.reference == "r"
        assert enc.others == ("g", "i")  # sorted, reference excluded
        assert enc.names == ("r", "g", "i")

    def test_ties_broken_alphabetically(self):
        enc = BandEncoding.from_counts(["r"] * 4 + ["g"] * 4)
        assert enc.reference == "g"

    def test_explicit_reference_override(self):
        enc = BandEncoding.from_counts(["g"] * 3 + ["r"] * 7, reference="g")
        assert enc.reference == "g"
        assert enc.others == ("r",)

    def test_explicit_reference_must_be_observed(self):
        with pytest.raises(ValueError, match="not observed"):
            BandEncoding.from_counts(["g", "r"], reference="u")

    def test_empty_labels_rejected(self):
        with pytest.raises(ValueError, match="no band labels"):
            BandEncoding.from_counts([])

    def test_encode_puts_reference_at_code_zero(self):
        labels = ["g", "r", "r", "i", "r"]
        enc = BandEncoding.from_counts(labels)
        codes = enc.encode(labels)
        assert enc.reference == "r"
        assert codes.tolist() == [1, 0, 0, 2, 0]
        assert codes.dtype == np.int64

    def test_encode_rejects_unknown_band(self):
        enc = BandEncoding(reference="r", others=("g",))
        with pytest.raises(ValueError, match="not in encoding"):
            enc.encode(["r", "z"])

    def test_single_band_object_has_no_free_bands(self):
        enc = BandEncoding.from_counts(["r"] * 5)
        assert enc.reference == "r" and enc.others == ()
        assert enc.encode(["r"] * 5).tolist() == [0] * 5


# ===========================================================================
# build_family(photometric_bands=...)
# ===========================================================================


class TestMultibandFamily:
    def test_band_parameters_added_for_non_reference_bands_only(self):
        spec = build_family(
            "drw", PriorConfig(), variants=("plain",), photometric_bands=("g", "i")
        )["drw"]
        names = spec.param_names
        assert {"a_g", "mu_g", "a_i", "mu_i"} <= set(names)
        assert not any(n.endswith("_r") for n in names)  # reference not fit
        assert "log10_variance" in names  # shared latent variance unchanged

    def test_default_is_unchanged_single_band(self):
        plain = build_family("drw", PriorConfig(), variants=("plain",))["drw"]
        assert not any(n.startswith(("a_", "mu_")) for n in plain.param_names)
        assert plain.meta["photometric_bands"] is None

    def test_variants_share_the_same_band_parameter_objects(self):
        fam = build_family(
            "drw",
            PriorConfig(),
            variants=("plain", "sine"),
            photometric_bands=("g",),
        )
        p_plain = {p.name: p for p in fam["drw"].prior.parameters}
        p_sine = {p.name: p for p in fam["drw+sine"].prior.parameters}
        assert p_plain["a_g"] is p_sine["a_g"]  # M1/B4 sharing
        assert p_plain["mu_g"] is p_sine["mu_g"]

    def test_prior_config_scales_are_used(self):
        cfg = PriorConfig(band_log_amp_sigma=0.7, band_mu_scale=1.5)
        spec = build_family(
            "drw", cfg, variants=("plain",), photometric_bands=("g",)
        )["drw"]
        by_name = {p.name: p for p in spec.prior.parameters}
        assert by_name["a_g"].prior.sigma_log10 == 0.7
        # mu_g median is 0 regardless of scale; check the scale actually bites
        lo = by_name["mu_g"].prior.transform(0.1, {})
        assert lo == pytest.approx(-1.5 * 1.2815515655446004, rel=1e-6)

    def test_meta_records_photometric_bands(self):
        spec = build_family(
            "drw", PriorConfig(), variants=("plain",), photometric_bands=("g", "i")
        )["drw"]
        assert spec.meta["photometric_bands"] == ["g", "i"]

    def test_loglike_matches_direct_multiband_call(self):
        spec = build_family(
            "drw", PriorConfig(), variants=("plain",), photometric_bands=("g",)
        )["drw"]
        rng = np.random.default_rng(7)
        n = 24
        t = np.sort(rng.uniform(0, 50, n)) + np.arange(n) * 1e-9
        y = rng.normal(0, 1, n)
        yerr = np.full(n, 0.05)
        band = rng.integers(0, 2, n)
        pdict = {
            "log10_variance": -0.5,
            "log10_fbend": -1.0,
            "mu0": -0.3,  # reference band's own offset, band_mu[0]
            "a_g": 1.6,
            "mu_g": 0.2,
        }
        got = spec.loglike(pdict, t, y, yerr, band)
        ref = gp_log_likelihood_multiband(
            drw_kernel(-0.5, -1.0),
            t,
            y,
            yerr,
            band,
            np.array([1.0, 1.6]),
            np.array([-0.3, 0.2]),
        )
        assert got == pytest.approx(ref, rel=1e-10)

    def test_band_none_falls_through_to_single_band_path(self):
        """A family built WITHOUT photometric_bands ignores a passed band=None
        and reproduces gp_log_likelihood exactly (single-band regression)."""
        spec = build_family("drw", PriorConfig(), variants=("plain",))["drw"]
        rng = np.random.default_rng(8)
        n = 20
        t = np.sort(rng.uniform(0, 50, n)) + np.arange(n) * 1e-9
        y = rng.normal(0, 1, n)
        yerr = np.full(n, 0.05)
        pdict = {"log10_variance": -0.5, "log10_fbend": -1.0, "mu0": 0.0}
        got = spec.loglike(pdict, t, y, yerr, None)
        ref = gp_log_likelihood(drw_kernel(-0.5, -1.0), t, y, yerr, mean_func=None)
        assert got == pytest.approx(ref, rel=1e-12)


@pytest.mark.slow
def test_run_nested_multiband_tiny():
    """Genuine (small) ultranest run through the band= path: finite logz,
    samples keyed by the multi-band parameter names."""
    pytest.importorskip("ultranest")
    rng = np.random.default_rng(3)
    n = 30
    t = np.sort(rng.uniform(0.0, 10.0, n)) + np.arange(n) * 1e-9
    y = rng.normal(0.0, 0.3, n)
    yerr = np.full(n, 0.1)
    band = rng.integers(0, 2, n)
    spec = build_family(
        "drw", PriorConfig(), variants=("plain",), photometric_bands=("g",)
    )["drw"]
    settings = SamplerSettings(
        # 40000, not 20000: mu0 added a dimension to every model and the
        # old cap truncated this deliberately-tiny run
        min_num_live_points=50, max_ncalls=40000, frac_remain=0.1, seed=42
    )
    res = run_nested(
        spec,
        t,
        y,
        yerr,
        settings=settings,
        show_status=False,
        n_posterior_samples=500,
        band=band,
    )
    assert np.isfinite(res.logz)
    assert set(res.samples) == set(spec.param_names)
    assert res.ncall > 0


# ===========================================================================
# cadence_to_multiband_series  (Phase 2 adapter)
# ===========================================================================


def _cadence_df(mjd, band, mag, magerr, depth=None):
    return pd.DataFrame(
        {
            "mjd": mjd,
            "band": band,
            "mag": mag,
            "magerr": magerr,
            "depth": np.nan if depth is None else depth,
            "seeing": np.nan,
        }
    )


class TestCadenceToMultibandSeries:
    def _simple(self):
        # r is the most-observed band -> reference; median(r mag) = 19.0
        return _cadence_df(
            mjd=[60000.0, 60365.0, 60730.0, 61095.0, 61460.0],
            band=["g", "r", "r", "i", "r"],
            mag=[18.0, 18.5, 19.0, 20.0, 19.5],
            magerr=[0.01, 0.02, 0.03, 0.04, 0.05],
        )

    def test_shapes_dtypes_and_encoding(self):
        t, y, yerr, code, enc = cadence_to_multiband_series(self._simple())
        assert t.shape == y.shape == yerr.shape == code.shape == (5,)
        assert enc.reference == "r"
        assert enc.others == ("g", "i")
        assert code.tolist() == [1, 0, 0, 2, 0]

    def test_time_in_years_zeroed_and_sorted(self):
        t, *_ = cadence_to_multiband_series(self._simple())
        assert t[0] == 0.0
        assert np.all(np.diff(t) > 0)
        assert t[-1] == pytest.approx((61460.0 - 60000.0) / 365.0)

    def test_zero_time_can_be_disabled(self):
        t, *_ = cadence_to_multiband_series(self._simple(), zero_time=False)
        assert t[0] == pytest.approx(60000.0 / 365.0)

    def test_centred_on_reference_band_median_not_global(self):
        t, y, yerr, code, enc = cadence_to_multiband_series(self._simple())
        assert np.median(y[code == 0]) == pytest.approx(0.0)  # mu_ref = 0 valid
        # global median (19.0 here) coincides, so check a case where it can't:
        df = _cadence_df(
            mjd=[60000.0, 60100.0, 60200.0, 60300.0],
            band=["g", "g", "r", "r"],
            mag=[10.0, 10.0, 20.0, 22.0],
            magerr=[0.01] * 4,
        )
        _, y2, _, code2, enc2 = cadence_to_multiband_series(df, reference="r")
        assert enc2.reference == "r"
        assert np.median(y2[code2 == 0]) == pytest.approx(0.0)
        assert y2.tolist() == [-11.0, -11.0, -1.0, 1.0]  # ref median 21 removed

    def test_magnitudes_are_not_converted_to_flux(self):
        _, y, yerr, code, _ = cadence_to_multiband_series(self._simple())
        # y is centred magnitude: differences preserve the magnitude scale
        assert y[code == 0].tolist() == [-0.5, 0.0, 0.5]
        assert yerr.tolist() == [0.01, 0.02, 0.03, 0.04, 0.05]  # magerr verbatim

    def test_drops_depth_only_and_bad_rows(self):
        df = _cadence_df(
            mjd=[60000.0, 60100.0, 60200.0, 60300.0, np.nan],
            band=["r", "r", "g", "g", "r"],
            mag=[19.0, 19.5, np.nan, 18.0, 19.0],   # row 2 is depth-only
            magerr=[0.01, 0.02, 0.03, -0.01, 0.02],  # row 3 has bad magerr
            depth=[np.nan, np.nan, 23.5, np.nan, np.nan],
        )
        t, y, yerr, code, enc = cadence_to_multiband_series(df)
        assert len(t) == 2  # only the two clean r rows survive
        assert enc.reference == "r" and enc.others == ()

    def test_rejects_all_bad(self):
        df = _cadence_df(
            mjd=[60000.0], band=["r"], mag=[np.nan], magerr=[np.nan], depth=[23.0]
        )
        with pytest.raises(ValueError, match="no rows with finite"):
            cadence_to_multiband_series(df)

    def test_rejects_missing_columns(self):
        with pytest.raises(ValueError, match="missing required columns"):
            cadence_to_multiband_series(pd.DataFrame({"mjd": [1.0], "band": ["r"]}))

    def test_explicit_reference_override(self):
        _, _, _, code, enc = cadence_to_multiband_series(self._simple(), reference="g")
        assert enc.reference == "g"
        assert code.tolist() == [0, 2, 2, 1, 2]

    def test_output_feeds_straight_into_loglike(self):
        t, y, yerr, code, enc = cadence_to_multiband_series(self._simple())
        spec = build_family(
            "drw", PriorConfig(), variants=("plain",), photometric_bands=enc.others
        )["drw"]
        pdict = {
            "log10_variance": -0.5,
            "log10_fbend": -1.0,
            "mu0": 0.05,
            "a_g": 1.2,
            "mu_g": 0.1,
            "a_i": 0.8,
            "mu_i": -0.1,
        }
        assert set(pdict) == set(spec.param_names)
        assert np.isfinite(spec.loglike(pdict, t, y, yerr, code))

    def test_real_cadence_library_object(self, tmp_path):
        """End-to-end on a CadenceLibrary-loaded ZTF object (g/r bands)."""
        from tests.test_cadence import _make_ztf_dir

        outdir, _ = _make_ztf_dir(tmp_path)
        lib = CadenceLibrary.from_survey_dirs({"ztf": outdir})
        cad = lib.get("ztf", "obj_matched")
        t, y, yerr, code, enc = cadence_to_multiband_series(cad)

        assert len(t) == len(y) == len(yerr) == len(code)
        assert set(enc.names) == {"g", "r"}
        assert np.all(np.diff(t) >= 0)
        assert np.all(yerr > 0)
        assert np.median(y[code == 0]) == pytest.approx(0.0)


# ===========================================================================
# Phase 5: colour-dependent injection into the simulator
# ===========================================================================


def _powerlaw_psd(freq, index):
    return freq ** (-index)


@pytest.fixture(scope="module")
def sim_lc():
    """Small red-noise realisation to sample cadences from (mirrors the
    fixture in test_package.py; stingray-dependent, skipped without it)."""
    pytest.importorskip("stingray", reason="stingray not installed")
    from pioran_periodicity.simulate import simulate_lightcurve

    import warnings as _warnings

    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore")
        return simulate_lightcurve(
            _powerlaw_psd,
            (2.0,),
            n_samples=40000,
            dt_minutes=60.0,
            mean_mag=0.0,
            sigma_mag=0.15,
            seed=1,
        )


class TestPowerLawBandAmplitudes:
    def test_beta_zero_is_all_unit_amplitude(self):
        amps = power_law_band_amplitudes("ugrizy", 0.0, "r")
        assert set(amps) == set("ugrizy")
        assert all(v == pytest.approx(1.0) for v in amps.values())

    def test_reference_band_is_exactly_one(self):
        for beta in (0.3, 1.0, -0.5):
            assert power_law_band_amplitudes("ugrizy", beta, "i")["i"] == 1.0

    def test_positive_beta_makes_bluer_bands_more_variable(self):
        amps = power_law_band_amplitudes("ugrizy", 0.5, "r")
        ordered = [amps[b] for b in "ugrizy"]  # u bluest -> y reddest
        assert all(a > b for a, b in zip(ordered, ordered[1:]))
        assert amps["u"] > 1.0 > amps["y"]

    def test_matches_the_closed_form(self):
        beta = 0.4
        amps = power_law_band_amplitudes(["u", "g"], beta, "r")
        lam = EFFECTIVE_WAVELENGTHS["lsst"]
        assert amps["u"] == pytest.approx((lam["u"] / lam["r"]) ** (-beta))

    def test_ztf_wavelengths_differ_from_lsst(self):
        assert power_law_band_amplitudes("gri", 0.5, "r", survey="ztf")["g"] != (
            power_law_band_amplitudes("gri", 0.5, "r", survey="lsst")["g"]
        )

    def test_rejects_unknown_band_and_survey(self):
        with pytest.raises(ValueError, match="no wavelength"):
            power_law_band_amplitudes(["g", "Q"], 0.3, "g")
        with pytest.raises(ValueError, match="no filter wavelengths"):
            power_law_band_amplitudes("gri", 0.3, "r", survey="des")


def _multiband_cadence(n_per_band=40, bands=("g", "r", "i"), seed=5):
    rng = np.random.default_rng(seed)
    n = n_per_band * len(bands)
    mjd = np.sort(rng.uniform(60000.0, 60060.0, n))
    band = np.array(list(bands) * n_per_band, dtype=object)[:n]
    return pd.DataFrame(
        {
            "mjd": mjd,
            "band": band,
            "mag": np.nan,
            "magerr": np.nan,
            "depth": np.full(n, 23.0),
            "seeing": np.nan,
        }
    )


class TestColourDependentInjection:
    """sample_real_cadence(band_amp=...) -- the Phase 5 simulator hook."""

    def test_no_band_amp_is_bit_for_bit_unchanged(self, sim_lc):
        """Omitting band_amp/band_mu must be identical to passing None, so the
        colour hook costs nothing when it is not used."""
        from pioran_periodicity.simulate import sample_real_cadence

        cad = _multiband_cadence()
        a = sample_real_cadence(sim_lc, cad, noise_model=None, ref_mag=19.0, seed=11)
        b = sample_real_cadence(
            sim_lc, cad, noise_model=None, ref_mag=19.0, seed=11,
            band_amp=None, band_mu=None,
        )
        for xa, xb in zip(a, b):
            assert np.array_equal(xa, xb)

    def test_unit_band_amp_leaves_mag_essentially_unchanged(self, sim_lc):
        from pioran_periodicity.simulate import sample_real_cadence

        cad = _multiband_cadence()
        _, f0, _ = sample_real_cadence(
            sim_lc, cad, noise_model=None, ref_mag=19.0, seed=11
        )
        _, f1, _ = sample_real_cadence(
            sim_lc, cad, noise_model=None, ref_mag=19.0, seed=11,
            band_amp={b: 1.0 for b in "gri"},
        )
        np.testing.assert_allclose(f1, f0, rtol=1e-12, atol=1e-12)

    def test_return_band_gives_labels_in_light_curve_order(self, sim_lc):
        from pioran_periodicity.simulate import sample_real_cadence

        cad = _multiband_cadence()
        t, f, e, band = sample_real_cadence(
            sim_lc, cad, noise_model=None, ref_mag=19.0, seed=11, return_band=True
        )
        assert len(band) == len(t) == len(f) == len(e)
        # sample_real_cadence sorts by mjd; labels must follow that order
        assert list(band) == list(cad.sort_values("mjd")["band"])

    def test_amplitude_scales_variability_not_the_mean(self, sim_lc):
        """a_b multiplies the scatter about the mean level; the mean level
        itself (and hence the magnitude zero point) must not move."""
        from pioran_periodicity.simulate import sample_real_cadence

        cad = _multiband_cadence(n_per_band=120, bands=("g", "r"), seed=6)
        amp = {"g": 3.0, "r": 1.0}
        _, f0, _, band = sample_real_cadence(
            sim_lc, cad, noise_model=None, ref_mag=19.0, seed=21, return_band=True
        )
        _, f1, _, _ = sample_real_cadence(
            sim_lc, cad, noise_model=None, ref_mag=19.0, seed=21,
            band_amp=amp, return_band=True,
        )
        g = np.asarray(band) == "g"
        # r band untouched; g band's spread about the mean grows ~3x
        np.testing.assert_allclose(f1[~g], f0[~g], rtol=1e-12, atol=1e-12)
        base = np.mean(f0)
        ratio = np.std(f1[g] - base) / np.std(f0[g] - base)
        assert 2.5 < ratio < 3.5

    def test_band_mu_offsets_only_its_own_band(self, sim_lc):
        from pioran_periodicity.simulate import sample_real_cadence

        cad = _multiband_cadence(n_per_band=100, bands=("g", "r"), seed=7)
        _, f0, _, band = sample_real_cadence(
            sim_lc, cad, noise_model=None, ref_mag=19.0, seed=31, return_band=True
        )
        _, f1, _, _ = sample_real_cadence(
            sim_lc, cad, noise_model=None, ref_mag=19.0, seed=31,
            band_mu={"g": 0.25, "r": 0.0}, return_band=True,
        )
        g = np.asarray(band) == "g"
        np.testing.assert_allclose(f1[g] - f0[g], 0.25, rtol=1e-10)
        np.testing.assert_allclose(f1[~g], f0[~g], rtol=1e-12, atol=1e-12)

    def test_missing_band_entry_is_rejected(self, sim_lc):
        from pioran_periodicity.simulate import sample_real_cadence

        cad = _multiband_cadence()
        with pytest.raises(ValueError, match="band_amp has no entry"):
            sample_real_cadence(
                sim_lc, cad, noise_model=None, ref_mag=19.0, seed=11,
                band_amp={"g": 1.0},  # missing r, i
            )
        with pytest.raises(ValueError, match="band_mu has no entry"):
            sample_real_cadence(
                sim_lc, cad, noise_model=None, ref_mag=19.0, seed=11,
                band_mu={"g": 0.0},
            )

    def test_injected_amplitude_ratio_matches_beta(self, sim_lc):
        """End-to-end: inject with power_law_band_amplitudes and recover the
        per-band scatter ratio the colour index implies."""
        from pioran_periodicity.simulate import sample_real_cadence

        beta = 0.8
        cad = _multiband_cadence(n_per_band=200, bands=("g", "r"), seed=8)
        amps = power_law_band_amplitudes(["g", "r"], beta, "r")
        _, f, _, band = sample_real_cadence(
            sim_lc, cad, noise_model=None, ref_mag=19.0, seed=41,
            band_amp=amps, return_band=True,
        )
        g = np.asarray(band) == "g"
        base = np.mean(f)
        ratio = np.std(f[g] - base) / np.std(f[~g] - base)
        assert ratio == pytest.approx(amps["g"], rel=0.25)
        assert amps["g"] > 1.0  # g is bluer than r, so more variable


class TestFitBandMeans:
    """fit_band_means=False drops mu_b (cost lever for large campaigns)."""

    def test_drops_mu_parameters_only(self):
        spec = build_family(
            "drw", PriorConfig(), variants=("plain",),
            photometric_bands=("g", "i"), fit_band_means=False,
        )["drw"]
        names = set(spec.param_names)
        assert {"a_g", "a_i"} <= names
        assert not any(n.startswith("mu_") for n in names)
        assert spec.prior.ndim == 5  # 2 noise + mu0 + 2 amplitudes

    def test_default_still_fits_means(self):
        spec = build_family(
            "drw", PriorConfig(), variants=("plain",), photometric_bands=("g",)
        )["drw"]
        assert "mu_g" in spec.param_names
        assert spec.meta["fit_band_means"] is True

    def test_meta_records_the_choice(self):
        spec = build_family(
            "drw", PriorConfig(), variants=("plain",),
            photometric_bands=("g",), fit_band_means=False,
        )["drw"]
        assert spec.meta["fit_band_means"] is False

    def test_single_band_meta_unaffected(self):
        spec = build_family("drw", PriorConfig(), variants=("plain",))["drw"]
        assert spec.meta["fit_band_means"] is None

    def test_loglike_equals_fitting_mu_at_zero(self):
        """Dropping mu_b must be exactly equivalent to fitting it at 0."""
        rng = np.random.default_rng(12)
        n = 30
        t = np.sort(rng.uniform(0, 50, n)) + np.arange(n) * 1e-9
        y = rng.normal(0, 1, n)
        yerr = np.full(n, 0.05)
        band = rng.integers(0, 2, n)
        common = dict(variants=("plain",), photometric_bands=("g",))
        free = build_family("drw", PriorConfig(), **common)["drw"]
        fixed = build_family(
            "drw", PriorConfig(), fit_band_means=False, **common
        )["drw"]
        base = {"mu0": 0.0, "log10_variance": -0.5, "log10_fbend": -1.0, "a_g": 1.4}
        got = fixed.loglike(base, t, y, yerr, band)
        ref = free.loglike({**base, "mu_g": 0.0}, t, y, yerr, band)
        assert got == pytest.approx(ref, rel=1e-12)
