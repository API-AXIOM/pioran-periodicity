"""Unit-test suite for the ``pioran_periodicity`` package.

Organised by module. Beyond ordinary correctness, the suite locks in the
bug-fixes documented in ``comparison_reports/simulation_code_bug_review.md``
(M1-M7, S1-S4) and ``comparison_reports/original_code_bug_review.md``
(B1-B5) as regression invariants:

* B1 / no dead parameters  -> DRW likelihood responds to variance AND fbend.
* M2 / B2 free variance     -> OBPL likelihood responds to log10_variance.
* M1 / B4 shared priors     -> every family shares identical Parameter objects.
* M5 / S1-real positive lo  -> period/err_scale singular limits rejected.
* M3 / B5 robust band       -> median f_max, density check, approximation error.
* M7 base-10 logs           -> one convention everywhere.
* S2 distinct nights        -> minimum epoch spacing floor.
* S1-sim leakage margin     -> simulated baseline >> observed baseline.
* S3 seeded resampling / S4 recorded settings.

Run with ``pytest`` from anywhere in the package (paths are resolved via the
``data_dir`` fixture, not the current working directory).
"""

from __future__ import annotations

import os
import warnings

import numpy as np
import pytest
from scipy.stats import norm as scipy_norm

import pioran_periodicity as pp
from pioran_periodicity.inference import (
    FitResult,
    SamplerSettings,
    load_result,
    log10_bayes_factors,
    run_nested,
    save_result,
)
from pioran_periodicity.kernels import (
    MIN_COMPONENTS_PER_DECADE,
    FrequencyBand,
    carma_kernel,
    drw_kernel,
    gp_log_likelihood,
    obpl_kernel,
    psd_approximation_error,
)
from pioran_periodicity.means import (
    combine_means,
    linear_mean,
    sine_amplitude_phase,
    sine_mean,
)
from pioran_periodicity.models import ModelFamily, PriorConfig, build_family
from pioran_periodicity.priors import (
    ConditionalUniform,
    LogUniform,
    Normal,
    Parameter,
    PriorTransform,
    Uniform,
)

# ===========================================================================
# priors
# ===========================================================================


class TestPriors:
    def test_uniform_transform_endpoints(self):
        u = Uniform(2.0, 6.0)
        assert u.transform(0.0, {}) == pytest.approx(2.0)
        assert u.transform(0.5, {}) == pytest.approx(4.0)
        assert u.transform(1.0, {}) == pytest.approx(6.0)

    def test_uniform_validates_lo_lt_hi(self):
        with pytest.raises(ValueError):
            Uniform(5.0, 5.0)
        with pytest.raises(ValueError):
            Uniform(6.0, 2.0)

    def test_normal_median_at_mu(self):
        n = Normal(1.5, 2.0)
        assert n.transform(0.5, {}) == pytest.approx(1.5)

    def test_normal_quantile_matches_scipy(self):
        mu, sigma = 1.0, 2.0
        n = Normal(mu, sigma)
        for u in (0.1, 0.25, 0.975):
            expected = mu + sigma * scipy_norm.ppf(u)
            assert n.transform(u, {}) == pytest.approx(expected, rel=1e-12)

    def test_normal_requires_positive_sigma(self):
        with pytest.raises(ValueError):
            Normal(0.0, 0.0)

    def test_loguniform_transform(self):
        lu = LogUniform(1e-2, 1e2)
        assert lu.transform(0.0, {}) == pytest.approx(1e-2)
        assert lu.transform(0.5, {}) == pytest.approx(1.0)  # log10 midpoint = 0
        assert lu.transform(1.0, {}) == pytest.approx(1e2)

    def test_loguniform_validates_bounds(self):
        with pytest.raises(ValueError):
            LogUniform(-1.0, 1.0)
        with pytest.raises(ValueError):
            LogUniform(1.0, 0.5)

    def test_conditional_uniform_respects_earlier_bound(self):
        # alpha_high ~ U(alpha_low, 4): must always land in [alpha_low, 4] (M1/B4)
        prior = PriorTransform(
            [
                Parameter("alpha_low", Uniform(0.0, 2.0)),
                Parameter("alpha_high", ConditionalUniform("alpha_low", 4.0)),
            ]
        )
        rng = np.random.default_rng(0)
        for _ in range(50):
            cube = rng.uniform(size=2)
            vals = prior(cube)
            alpha_low, alpha_high = vals
            assert alpha_low <= alpha_high <= 4.0

    def test_conditional_uniform_keyerror_when_reference_is_later(self):
        # referencing a parameter declared *after* this one must raise
        prior = PriorTransform(
            [
                Parameter("alpha_high", ConditionalUniform("alpha_low", 4.0)),
                Parameter("alpha_low", Uniform(0.0, 2.0)),
            ]
        )
        with pytest.raises(KeyError):
            prior(np.array([0.5, 0.5]))

    def test_prior_transform_ordering_asdict_ndim(self):
        prior = PriorTransform(
            [
                Parameter("a", Uniform(0.0, 1.0)),
                Parameter("b", Uniform(10.0, 20.0)),
            ]
        )
        assert prior.ndim == 2
        assert prior.names == ["a", "b"]
        vals = prior(np.array([0.0, 0.5]))
        assert vals[0] == pytest.approx(0.0)
        assert vals[1] == pytest.approx(15.0)
        d = prior.as_dict(vals)
        assert d == pytest.approx({"a": 0.0, "b": 15.0})
        assert list(d) == ["a", "b"]  # order preserved


# ===========================================================================
# means
# ===========================================================================


class TestMeans:
    def test_sine_mean_equals_amplitude_phase_form(self):
        rng = np.random.default_rng(1)
        t = np.linspace(0.0, 7.0, 40)
        period = 2.3
        for _ in range(10):
            A1, A2 = rng.normal(size=2)
            A, phi = sine_amplitude_phase(A1, A2)
            direct = sine_mean(t, A1, A2, period)
            equiv = A * np.sin(2.0 * np.pi * t / period + phi)
            assert np.allclose(direct, equiv, atol=1e-12)

    def test_sine_mean_rejects_nonpositive_period(self):
        with pytest.raises(ValueError):
            sine_mean(np.array([0.0, 1.0]), 1.0, 1.0, 0.0)
        with pytest.raises(ValueError):
            sine_mean(np.array([0.0, 1.0]), 1.0, 1.0, -3.0)

    def test_linear_mean(self):
        t = np.array([0.0, 1.0, 2.0])
        assert np.allclose(linear_mean(t, 2.0, -1.0), np.array([-1.0, 1.0, 3.0]))

    def test_combine_means_sums(self):
        t = np.linspace(0.0, 5.0, 11)
        f = lambda tt: sine_mean(tt, 0.5, 0.3, 2.0)  # noqa: E731
        g = lambda tt: linear_mean(tt, 0.1, -0.2)  # noqa: E731
        total = combine_means(f, g)
        assert np.allclose(total(t), f(t) + g(t))


# ===========================================================================
# kernels  (Julia-backed; keep parameter sets small)
# ===========================================================================


class TestFrequencyBand:
    def test_median_band_much_narrower_than_min(self):
        # one artificially close pair (0.0, 0.001) inflates min f_max hugely;
        # the median-based band must be several decades narrower (fix M3/B5).
        t = np.array([0.0, 0.001, 1.0, 2.0, 3.0, 4.0, 5.0])
        med = FrequencyBand.from_times(t, f_max_method="median")
        mn = FrequencyBand.from_times(t, f_max_method="min")
        assert mn.f_max > med.f_max
        assert (mn.grid_decades - med.grid_decades) >= 2.0  # several decades

    def test_invalid_construction_raises(self):
        with pytest.raises(ValueError):
            FrequencyBand(f_min=1.0, f_max=1.0)  # need f_min < f_max
        with pytest.raises(ValueError):
            FrequencyBand(f_min=2.0, f_max=1.0)
        with pytest.raises(ValueError):
            FrequencyBand(f_min=1.0, f_max=10.0, S_low=0.5)  # S_low >= 1

    def test_from_times_invalid_inputs_raise(self):
        with pytest.raises(ValueError):
            FrequencyBand.from_times([1.0])  # < 2 distinct stamps
        with pytest.raises(ValueError):
            FrequencyBand.from_times([1.0, 1.0, 1.0])  # no positive diffs
        with pytest.raises(ValueError):
            FrequencyBand.from_times([0.0, 1.0, 2.0], f_max_method="bogus")

    def test_grid_decades_hand_check(self):
        # f0 = 1/10 = 0.1, fM = 10*10 = 100 -> log10(100/0.1) = 3 decades
        band = FrequencyBand(f_min=1.0, f_max=10.0, S_low=10.0, S_high=10.0)
        assert band.grid_decades == pytest.approx(3.0)

    def test_check_density_warns_below_threshold(self):
        band = FrequencyBand(f_min=1.0, f_max=10.0, S_low=10.0, S_high=10.0)  # 3 dec
        n = 3  # 3/3 = 1.0 < 1.5
        with pytest.warns(UserWarning, match="issue M3"):
            density = band.check_density(n)
        assert density == pytest.approx(1.0)
        assert density < MIN_COMPONENTS_PER_DECADE

    def test_check_density_strict_raises(self):
        band = FrequencyBand(f_min=1.0, f_max=10.0, S_low=10.0, S_high=10.0)
        with pytest.raises(ValueError, match="issue M3"):
            band.check_density(3, strict=True)

    def test_check_density_silent_above_threshold(self):
        band = FrequencyBand(f_min=1.0, f_max=10.0, S_low=10.0, S_high=10.0)  # 3 dec
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # any warning -> failure
            density = band.check_density(6)  # 6/3 = 2.0 >= 1.5
        assert density == pytest.approx(2.0)


class TestDRWKernel:
    def test_variance_is_live_parameter(self, synthetic):
        # B1 lock-in: the amplitude/variance must affect the likelihood.
        t, y, yerr = synthetic
        base = gp_log_likelihood(drw_kernel(0.0, -0.3), t, y, yerr)
        bumped = gp_log_likelihood(drw_kernel(1.0, -0.3), t, y, yerr)
        assert abs(base - bumped) > 1e-3

    def test_fbend_is_live_parameter(self, synthetic):
        # B1 lock-in: the bend frequency must affect the likelihood.
        t, y, yerr = synthetic
        base = gp_log_likelihood(drw_kernel(0.0, -0.3), t, y, yerr)
        bumped = gp_log_likelihood(drw_kernel(0.0, 0.5), t, y, yerr)
        assert abs(base - bumped) > 1e-3

    def test_k0_equals_variance_convention(self, synthetic):
        # documented convention: k(0) = variance = 10**log10_variance, built as
        # Pioran Exp(2*var, 2*pi*fbend). Compare against the same Exp directly.
        import pioranpy as pa

        t, y, yerr = synthetic
        V, F = -0.5, -0.3
        variance = 10.0**V
        rate = 2.0 * np.pi * 10.0**F
        via_helper = gp_log_likelihood(drw_kernel(V, F), t, y, yerr)
        via_direct = gp_log_likelihood(pa.Exp(2.0 * variance, rate), t, y, yerr)
        assert via_helper == pytest.approx(via_direct, rel=1e-10)


class TestOBPLKernel:
    def test_variance_is_live_parameter(self, synthetic):
        # THE M2/B2 regression test: with free variance the OBPL likelihood
        # MUST respond to log10_variance (the legacy var-pinned code did not).
        t, y, yerr = synthetic
        band = FrequencyBand.from_times(t)
        low = gp_log_likelihood(
            obpl_kernel(
                -1.0, 1.0, -0.3, 3.0, band, n_components=20, check_density=False
            ),
            t,
            y,
            yerr,
        )
        high = gp_log_likelihood(
            obpl_kernel(
                1.0, 1.0, -0.3, 3.0, band, n_components=20, check_density=False
            ),
            t,
            y,
            yerr,
        )
        assert abs(low - high) > 1e-2

    def test_inverted_bend_raises(self, synthetic):
        # M1: alpha_high < alpha_low is not a red-noise bend.
        t, _, _ = synthetic
        band = FrequencyBand.from_times(t)
        with pytest.raises(ValueError, match="fix M1"):
            obpl_kernel(0.0, 3.0, -0.3, 1.0, band, n_components=20, check_density=False)

    def test_density_check_respected(self):
        # With check_density=True and a sparse grid the M3 warning must fire.
        band = FrequencyBand(f_min=1.0, f_max=10.0, S_low=10.0, S_high=10.0)  # 3 dec
        with pytest.warns(UserWarning, match="issue M3"):
            obpl_kernel(0.0, 1.0, -0.3, 3.0, band, n_components=3, check_density=True)


class TestCARMAKernel:
    def test_complex_root_p2_finite(self, synthetic):
        # a1^2 - 4 a0 < 0 -> complex conjugate AR roots
        t, y, yerr = synthetic
        k = carma_kernel(2, 1, [np.log10(0.25), np.log10(0.25)], [0.0], 0.0)
        assert np.isfinite(gp_log_likelihood(k, t, y, yerr))

    def test_real_root_p2_finite(self, synthetic):
        # r^2 + 3r + 2 -> roots -1, -2 (real): exercises the manual celerite branch
        t, y, yerr = synthetic
        k = carma_kernel(2, 1, [np.log10(2.0), np.log10(3.0)], [0.0], 0.0)
        assert np.isfinite(gp_log_likelihood(k, t, y, yerr))


class TestPSDApproximationError:
    def test_error_decreases_with_more_components(self, synthetic):
        t, _, _ = synthetic
        band = FrequencyBand.from_times(t)
        coarse = psd_approximation_error(1.0, -0.3, 3.0, band, n_components=5)
        fine = psd_approximation_error(1.0, -0.3, 3.0, band, n_components=20)
        assert fine["max_rel_error"] < coarse["max_rel_error"]

    def test_returns_documented_keys(self, synthetic):
        t, _, _ = synthetic
        band = FrequencyBand.from_times(t)
        out = psd_approximation_error(1.0, -0.3, 3.0, band, n_components=20)
        assert set(out) == {
            "max_rel_error",
            "median_rel_error",
            "n_components",
            "grid_decades",
            "components_per_decade",
        }
        assert out["n_components"] == 20


class TestGPLogLikelihood:
    def test_err_scale_nonpositive_raises(self, synthetic):
        # fix S2-real: nu = 0 collapses the diagonal.
        t, y, yerr = synthetic
        k = drw_kernel(0.0, -0.3)
        with pytest.raises(ValueError):
            gp_log_likelihood(k, t, y, yerr, err_scale=0.0)
        with pytest.raises(ValueError):
            gp_log_likelihood(k, t, y, yerr, err_scale=-1.0)

    def test_err_scale_default_equals_explicit_one(self, synthetic):
        t, y, yerr = synthetic
        k = drw_kernel(0.0, -0.3)
        assert gp_log_likelihood(k, t, y, yerr) == pytest.approx(
            gp_log_likelihood(k, t, y, yerr, err_scale=1.0)
        )

    def test_err_scale_changes_likelihood(self, synthetic):
        t, y, yerr = synthetic
        k = drw_kernel(0.0, -0.3)
        a = gp_log_likelihood(k, t, y, yerr, err_scale=1.0)
        b = gp_log_likelihood(k, t, y, yerr, err_scale=2.0)
        assert abs(a - b) > 1e-3

    def test_mean_subtraction_equivalence(self, synthetic):
        # fitting (y + m) with mean m must equal fitting y with no mean.
        t, y, yerr = synthetic
        k = drw_kernel(0.0, -0.3)
        m = lambda tt: sine_mean(tt, 0.7, -0.4, 3.0)  # noqa: E731
        with_mean = gp_log_likelihood(k, t, y + m(t), yerr, mean_func=m)
        no_mean = gp_log_likelihood(k, t, y, yerr)
        assert with_mean == pytest.approx(no_mean, rel=1e-10)


# ===========================================================================
# models
# ===========================================================================


def _prior_of(spec, name):
    for p in spec.prior.parameters:
        if p.name == name:
            return p.prior
    raise KeyError(name)


class TestModels:
    ALL_VARIANTS = ("plain", "sine", "linear", "sine+linear")

    def _band(self):
        return FrequencyBand(f_min=0.1, f_max=2.0, S_low=20.0, S_high=20.0)

    def _family(self, noise, cfg=None, variants=ALL_VARIANTS):
        cfg = cfg or PriorConfig()
        band = self._band() if noise == "obpl" else None
        return build_family(noise, cfg, variants=variants, band=band, n_components=20)

    @pytest.mark.parametrize("noise", ["drw", "obpl", "carma"])
    def test_build_family_all_variants(self, noise):
        fam = self._family(noise)
        assert isinstance(fam, ModelFamily)
        expected = {noise, f"{noise}+sine", f"{noise}+linear", f"{noise}+sine+linear"}
        assert set(fam.names()) == expected

    @pytest.mark.parametrize("noise", ["drw", "obpl", "carma"])
    def test_shared_priors_are_identical_objects(self, noise):
        # M1/B4 invariant: every shared parameter name must map to the SAME
        # prior object in every member of the family.
        fam = self._family(noise)
        members = list(fam.members.values())
        # collect noise-parameter names (present in the "plain" member)
        shared_names = fam[noise].param_names
        for name in shared_names:
            priors = [_prior_of(m, name) for m in members if name in m.param_names]
            first = priors[0]
            for other in priors[1:]:
                assert (
                    other is first
                ), f"prior for shared '{name}' differs across {noise} family"

    def test_obpl_without_band_raises(self):
        with pytest.raises(ValueError):
            build_family("obpl", PriorConfig(), variants=("plain", "sine"), band=None)

    def test_priorconfig_period_lower_bound(self):
        with pytest.raises(ValueError):
            PriorConfig(period=(0.0, 5.0))
        with pytest.raises(ValueError):
            PriorConfig(period=(-1.0, 5.0))

    def test_priorconfig_err_scale_lower_bound(self):
        with pytest.raises(ValueError):
            PriorConfig(err_scale=(0.0, 1.5))

    @pytest.mark.parametrize("noise", ["drw", "obpl", "carma"])
    def test_loglike_finite_on_prior_draws(self, noise, synthetic):
        t, y, yerr = synthetic
        fam = self._family(noise)
        rng = np.random.default_rng(7)
        for spec in fam.members.values():
            for _ in range(5):
                cube = rng.uniform(size=spec.prior.ndim)
                pdict = spec.prior.as_dict(spec.prior(cube))
                val = spec.loglike(pdict, t, y, yerr)
                assert np.isfinite(val), f"{spec.name} non-finite at {pdict}"

    def test_err_scale_parameter_appears_and_matters(self, synthetic):
        t, y, yerr = synthetic
        cfg = PriorConfig(err_scale=(0.5, 2.0))
        fam = build_family("drw", cfg, variants=("plain",))
        spec = fam["drw"]
        assert "err_scale" in spec.param_names
        base = dict(log10_variance=0.0, log10_fbend=-0.3, err_scale=1.0)
        scaled = dict(base, err_scale=2.0)
        a = spec.loglike(base, t, y, yerr)
        b = spec.loglike(scaled, t, y, yerr)
        assert abs(a - b) > 1e-3


# ===========================================================================
# inference
# ===========================================================================


class TestInference:
    def test_sampler_settings_serialization(self):
        s = SamplerSettings(min_num_live_points=50, seed=123)
        d = s.as_dict()
        assert d["min_num_live_points"] == 50
        assert d["seed"] == 123
        assert set(d) == {
            "min_num_live_points",
            "frac_remain",
            "max_ncalls",
            "max_num_improvement_loops",
            "seed",
            "step_sampler_min_ndim",
            "step_nsteps_per_dim",
        }

    def test_log10_bayes_factors_arithmetic(self):
        ln10 = np.log(10.0)
        results = {
            "A": FitResult("A", logz=10.0, logzerr=0.1, samples={}, settings={}),
            "B": FitResult("B", logz=3.0, logzerr=0.1, samples={}, settings={}),
        }
        bf = log10_bayes_factors(results)
        assert bf["A/B"] == pytest.approx((10.0 - 3.0) / ln10)

    def test_save_load_roundtrip(self, tmp_path):
        r = FitResult(
            model="drw",
            logz=-12.3,
            logzerr=0.2,
            samples={"log10_variance": [0.1, 0.2], "log10_fbend": [-1.0, -0.5]},
            settings=SamplerSettings(seed=9).as_dict(),
            meta={"noise": "drw"},
            ncall=1234,
            runtime_s=5.6,
        )
        path = tmp_path / "res.json"
        save_result(r, str(path))
        loaded = load_result(str(path))
        assert loaded.model == r.model
        assert loaded.logz == pytest.approx(r.logz)
        assert loaded.logzerr == pytest.approx(r.logzerr)
        assert loaded.samples == r.samples
        assert loaded.settings == r.settings
        assert loaded.meta == r.meta
        assert loaded.ncall == r.ncall
        assert loaded.runtime_s == pytest.approx(r.runtime_s)

    def test_seeded_resampling_is_reproducible(self, monkeypatch):
        # S3 lock-in: run_nested's equal-weight resampling is seeded, so with a
        # fixed set of weighted samples two runs with the same seed yield the
        # IDENTICAL resampled sample lists (and a different seed differs).
        # We stub ultranest so the test is deterministic (no sampler stochasticity).
        import pioran_periodicity.inference as inf

        rng = np.random.default_rng(0)
        pts = rng.normal(size=(200, 2))
        wts = rng.uniform(size=200)
        fake_result = {
            "logz": -10.0,
            "logzerr": 0.1,
            "ncall": 500,
            "weighted_samples": {"points": pts, "weights": wts},
        }

        class FakeSampler:
            def __init__(self, names, loglike, transform, **kw):
                pass

            def run(self, **kw):
                return fake_result

        fake_ultranest = type("M", (), {"ReactiveNestedSampler": FakeSampler})
        monkeypatch.setitem(__import__("sys").modules, "ultranest", fake_ultranest)

        spec = build_family("drw", PriorConfig(), variants=("plain",))["drw"]
        t = np.linspace(0, 10, 30)
        y = np.zeros(30)
        yerr = np.full(30, 0.1)

        # disable the step sampler for this stub (the fake ultranest module
        # has no .stepsampler submodule); this test exercises the seeded
        # resampling, not the sampler choice.
        no_slice = dict(step_sampler_min_ndim=99)
        r1 = inf.run_nested(
            spec,
            t,
            y,
            yerr,
            settings=SamplerSettings(seed=42, **no_slice),
            n_posterior_samples=100,
        )
        r2 = inf.run_nested(
            spec,
            t,
            y,
            yerr,
            settings=SamplerSettings(seed=42, **no_slice),
            n_posterior_samples=100,
        )
        r3 = inf.run_nested(
            spec,
            t,
            y,
            yerr,
            settings=SamplerSettings(seed=7, **no_slice),
            n_posterior_samples=100,
        )
        assert r1.samples == r2.samples  # same seed -> identical (S3)
        assert r1.samples != r3.samples  # different seed -> different

    @pytest.mark.slow
    def test_run_nested_tiny(self):
        # Genuine (small) ultranest run: finite logz, keyed samples, recorded
        # settings (S4), ncall > 0.
        pytest.importorskip("ultranest")
        rng = np.random.default_rng(3)
        t = np.sort(rng.uniform(0.0, 10.0, 30))
        y = rng.normal(0.0, 0.3, 30)
        yerr = np.full(30, 0.1)
        spec = build_family("drw", PriorConfig(), variants=("plain",))["drw"]
        settings = SamplerSettings(
            min_num_live_points=50, max_ncalls=20000, frac_remain=0.1, seed=42
        )
        res = run_nested(
            spec,
            t,
            y,
            yerr,
            settings=settings,
            show_status=False,
            n_posterior_samples=500,
        )
        assert np.isfinite(res.logz)
        assert set(res.samples) == set(spec.param_names)  # keyed by param names
        assert all(len(v) > 0 for v in res.samples.values())
        assert res.settings == settings.as_dict()  # S4 recorded
        assert res.ncall > 0


# ===========================================================================
# simulate   (needs stingray; skip gracefully if unavailable)
# ===========================================================================

stingray = pytest.importorskip("stingray", reason="stingray not installed")


def _powerlaw_psd(freq, index):
    return freq ** (-index)


@pytest.fixture(scope="module")
def sim_lc():
    from pioran_periodicity.simulate import simulate_lightcurve

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return simulate_lightcurve(
            _powerlaw_psd,
            (2.0,),
            n_samples=40000,
            dt_minutes=60.0,
            mean=1.0,
            rms=0.15,
            seed=1,
        )


# common seasonal-pattern kwargs that keep t_obs small and the sim short
_PAT = dict(
    n_windows=3,
    nights_per_window=3,
    window_period_months=2.0,
    window_width_days=10.0,
    obs_per_night=1,
    night_window_hours=8.0,
)


class TestSimulate:
    def test_lightcurve_length_dt_and_rms(self, sim_lc):
        assert len(sim_lc.time) == 40000
        assert sim_lc.dt_days == pytest.approx(60.0 / (60.0 * 24.0))
        frac_rms = np.std(sim_lc.flux) / np.mean(sim_lc.flux)
        assert frac_rms == pytest.approx(0.15, rel=0.2)

    def test_distinct_night_spacing_floor(self, sim_lc):
        # S2: obs_per_night=1, night_window_hours=8 -> spacing >= 16 h.
        from pioran_periodicity.simulate import sample_seasonal_pattern

        t, _, _ = sample_seasonal_pattern(sim_lc, seed=11, **_PAT)
        min_hours = np.min(np.diff(np.sort(t))) * 365.0 * 24.0
        assert min_hours >= 16.0

    def test_obs_per_night_exceeds_window_raises(self, sim_lc):
        from pioran_periodicity.simulate import sample_seasonal_pattern

        bad = dict(_PAT, obs_per_night=50)  # >> 8-sample night window
        with pytest.raises(ValueError):
            sample_seasonal_pattern(sim_lc, seed=1, **bad)

    def test_leakage_margin_enforced(self, sim_lc):
        # S1: a large observed baseline (relative to the sim) must raise, and
        # the raised error must mention S1.
        from pioran_periodicity.simulate import sample_seasonal_pattern

        greedy = dict(_PAT, n_windows=9, window_period_months=8.0)
        with pytest.raises(ValueError, match="S1"):
            sample_seasonal_pattern(sim_lc, seed=1, **greedy)

    def test_leakage_margin_override_warns(self, sim_lc):
        # margin < 10 (so the S1 warning fires) but the pattern still fits in
        # the simulated span: t_obs = 2*120 + 10 = 250 d, margin ~ 6.7.
        from pioran_periodicity.simulate import sample_seasonal_pattern

        greedy = dict(_PAT, window_period_months=4.0)
        with pytest.warns(UserWarning, match="S1"):
            sample_seasonal_pattern(
                sim_lc, seed=1, enforce_leakage_margin=False, **greedy
            )

    def test_data_loss_reduces_count(self, sim_lc):
        from pioran_periodicity.simulate import sample_seasonal_pattern

        t_full, _, _ = sample_seasonal_pattern(sim_lc, seed=2, **_PAT)
        t_lost, _, _ = sample_seasonal_pattern(
            sim_lc, seed=2, data_loss_frac=0.5, **_PAT
        )
        assert len(t_lost) < len(t_full)
        assert len(t_lost) == len(t_full) - int(len(t_full) * 0.5)

    def test_mean_signal_shifts_flux(self, sim_lc):
        from pioran_periodicity.simulate import sample_seasonal_pattern

        _, flux0, _ = sample_seasonal_pattern(sim_lc, seed=4, **_PAT)
        _, flux1, _ = sample_seasonal_pattern(
            sim_lc, seed=4, mean_signal=lambda tt: 5.0, **_PAT
        )
        assert np.mean(flux1) - np.mean(flux0) == pytest.approx(5.0, abs=1e-9)

    def test_same_seed_identical_output(self, sim_lc):
        from pioran_periodicity.simulate import sample_seasonal_pattern

        a = sample_seasonal_pattern(sim_lc, seed=99, **_PAT)
        b = sample_seasonal_pattern(sim_lc, seed=99, **_PAT)
        for xa, xb in zip(a, b):
            assert np.array_equal(xa, xb)


# ===========================================================================
# data
# ===========================================================================


class TestData:
    def test_load_pg1302(self, data_dir):
        path = os.path.join(data_dir, "AGNobsdata", "graham2015data.csv")
        t, y, yerr = pp.load_pg1302(path=path)
        assert len(t) == 290
        assert abs(np.median(y)) < 1e-12  # median-subtracted
        assert t[0] == pytest.approx(0.0)  # zeroed at first epoch
        assert np.all(np.diff(t) >= 0)  # sorted
        assert len(y) == len(t) == len(yerr)

    def test_load_pg1553(self, data_dir):
        path = os.path.join(data_dir, "AGNobsdata", "PG1553_113_logbase.txt")
        t, y, yerr = pp.load_pg1553(path=path)
        assert len(t) == 154
        assert abs(np.median(y)) < 1e-12  # median-subtracted
        assert np.all(np.diff(t) >= 0)  # sorted
        assert len(y) == len(t) == len(yerr)


# ===========================================================================
# NOTE: informational tests reproducing the legacy (pre-translation)
# utils/NSmodels.py likelihoods under documented convention transforms lived
# here historically. They depend on the quarantined translation layer and
# are kept only in the thesis-replication archive (workspace/tests/), not in
# this package's own test suite.
# ===========================================================================
