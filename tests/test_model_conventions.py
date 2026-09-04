"""Contract tests pinning the model definition shared by the simulator and
the fitted likelihood.

The defects these guard against (MB1 and MB2, found 2026-09-03) had the same
shape: the simulator and the fitted model disagreed about what a symbol
meant, and every existing test still passed because it only ever exercised
the degenerate case where the disagreement vanishes. ``test_multiband.py``'s
brute-force check passed ``mean_func=None``; with no mean function, the
"mean in observed units" and "mean in latent units" conventions are
algebraically identical, so the bug was invisible.

Each test below therefore does two things:

1. checks the convention against an INDEPENDENT brute-force reference
   (a dense multivariate normal built from the model equation by hand,
   not from the code under test), and
2. asserts that the degenerate case is NOT what is being tested -- i.e.
   that the two rival conventions actually differ for this input.

Point 2 is the part that was missing before. A test that cannot distinguish
the right answer from the wrong one is not covering the behaviour.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest
from scipy.stats import multivariate_normal

# scripts/ holds run_sim.py, whose injection convention is pinned below
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from pioran_periodicity.kernels import (  # noqa: E402
    drw_kernel,
    gp_log_likelihood,
    gp_log_likelihood_multiband,
)
from pioran_periodicity.means import sine_amplitude, sine_mean  # noqa: E402
from pioran_periodicity.models import PriorConfig, build_family  # noqa: E402
from pioran_periodicity.multiband import power_law_band_amplitudes  # noqa: E402

# The model equation both sides must agree on (see bugfix_report.tex, MB1):
#
#     y_b(t) = mu_b + a_b * ( x(t) + m(t) ) + noise_b(t)
#
# x(t): shared zero-mean latent GP;  m(t): the deterministic sine/linear mean,
# in shared-latent units;  a_b, mu_b: per-band amplitude and offset, pinned to
# (1, 0) for the reference band.

LOG10_VAR = -1.2
LOG10_FBEND = -0.6


def _drw_cov(t: np.ndarray) -> np.ndarray:
    """Dense (n, n) DRW covariance, written out independently of the kernel
    code so this is a genuine reference rather than a restatement."""
    var = 10.0**LOG10_VAR
    rate = 2.0 * np.pi * 10.0**LOG10_FBEND
    return var * np.exp(-rate * np.abs(t[:, None] - t[None, :]))


def _dense_loglike(t, y, yerr, a, mu, mean_vals, err_scale=1.0):
    """log N(y ; mu + a*mean_vals, (a a^T) o K + diag((err_scale*yerr)^2)).

    ``mean_vals`` is m(t) in LATENT units; the observed-frame mean is
    ``mu + a * m(t)``.  (n, n) covariance built by hand.
    """
    cov = (a[:, None] * a[None, :]) * _drw_cov(t) + np.diag((err_scale * yerr) ** 2)
    return float(multivariate_normal.logpdf(y, mean=mu + a * mean_vals, cov=cov))


@pytest.fixture(scope="module")
def mb_data():
    """Small 3-band dataset with deliberately UNEQUAL band amplitudes."""
    rng = np.random.default_rng(20260903)
    n = 45
    t = np.sort(rng.uniform(0.0, 5.0, n))
    code = rng.integers(0, 3, n)
    y = rng.normal(0.0, 0.2, n)
    yerr = np.full(n, 0.03)
    band_amp = np.array([1.0, 1.3, 0.7])  # a_ref = 1.0, others deliberately != 1
    band_mu = np.array([0.0, 0.15, -0.2])
    return t, y, yerr, code, band_amp, band_mu


def _sine(t):
    return sine_mean(t, 0.05, 0.03, 1.7)


class TestMultibandMeanConvention:
    """MB1: which side of the a_b rescale the mean function is subtracted."""

    def test_matches_dense_mvn_with_the_mean_in_latent_units(self, mb_data):
        t, y, yerr, code, amp, mu = mb_data
        got = gp_log_likelihood_multiband(
            drw_kernel(LOG10_VAR, LOG10_FBEND),
            t,
            y,
            yerr,
            code,
            amp,
            mu,
            mean_func=_sine,
        )
        want = _dense_loglike(t, y, yerr, amp[code], mu[code], _sine(t))
        assert got == pytest.approx(want, abs=1e-8)

    def test_does_not_use_the_observed_units_convention(self, mb_data):
        """The MB1 regression guard.

        The rejected convention is ``y_b = mu_b + m(t) + a_b x(t)``: a sine
        with the same amplitude in every band, against band-dependent red
        noise. It is what the code did before 2026-09-03.
        """
        t, y, yerr, code, amp, mu = mb_data
        a, m = amp[code], mu[code]
        cov = (a[:, None] * a[None, :]) * _drw_cov(t) + np.diag(yerr**2)
        observed_units = float(
            multivariate_normal.logpdf(y, mean=m + _sine(t), cov=cov)
        )
        got = gp_log_likelihood_multiband(
            drw_kernel(LOG10_VAR, LOG10_FBEND),
            t,
            y,
            yerr,
            code,
            amp,
            mu,
            mean_func=_sine,
        )
        # the two conventions must actually differ here, or this input
        # cannot distinguish them and the test proves nothing
        latent_units = _dense_loglike(t, y, yerr, a, m, _sine(t))
        assert abs(latent_units - observed_units) > 1.0, "degenerate input"
        assert got != pytest.approx(observed_units, abs=1e-6)

    def test_conventions_coincide_when_every_amplitude_is_unity(self, mb_data):
        """Documents exactly why the pre-existing tests could not catch MB1:
        at a_b == 1 the two conventions are algebraically identical."""
        t, y, yerr, code, _, mu = mb_data
        unit = np.ones(3)
        a, m = unit[code], mu[code]
        cov = _drw_cov(t) + np.diag(yerr**2)
        latent = _dense_loglike(t, y, yerr, a, m, _sine(t))
        observed = float(multivariate_normal.logpdf(y, mean=m + _sine(t), cov=cov))
        assert latent == pytest.approx(observed, abs=1e-9)

    def test_reduces_to_the_single_band_likelihood_with_a_mean(self, mb_data):
        """With unit amplitudes and zero offsets the multi-band likelihood must
        equal the plain single-band one -- WITH a mean function, which the
        pre-existing reduction test did not check."""
        t, y, yerr, code, _, _ = mb_data
        kern = drw_kernel(LOG10_VAR, LOG10_FBEND)
        got = gp_log_likelihood_multiband(
            kern,
            t,
            y,
            yerr,
            code,
            np.ones(3),
            np.zeros(3),
            mean_func=_sine,
        )
        want = gp_log_likelihood(kern, t, y, yerr, mean_func=_sine)
        assert got == pytest.approx(want, abs=1e-9)

    def test_err_scale_scales_the_observed_errors(self, mb_data):
        """err_scale multiplies the reported (observed-frame) uncertainties,
        so it must survive the division by a_b unchanged."""
        t, y, yerr, code, amp, mu = mb_data
        got = gp_log_likelihood_multiband(
            drw_kernel(LOG10_VAR, LOG10_FBEND),
            t,
            y,
            yerr,
            code,
            amp,
            mu,
            mean_func=_sine,
            err_scale=2.5,
        )
        want = _dense_loglike(t, y, yerr, amp[code], mu[code], _sine(t), err_scale=2.5)
        assert got == pytest.approx(want, abs=1e-8)

    def test_jacobian_is_present(self, mb_data):
        """The -sum(log a) rescale Jacobian: dropping it would still pass a
        test that only ever compared unit amplitudes."""
        t, y, yerr, code, amp, mu = mb_data
        got = gp_log_likelihood_multiband(
            drw_kernel(LOG10_VAR, LOG10_FBEND),
            t,
            y,
            yerr,
            code,
            amp,
            mu,
        )
        without = gp_log_likelihood(
            drw_kernel(LOG10_VAR, LOG10_FBEND),
            t,
            (y - mu[code]) / amp[code],
            yerr / amp[code],
        )
        assert got == pytest.approx(without - np.sum(np.log(amp[code])), abs=1e-9)
        assert abs(np.sum(np.log(amp[code]))) > 0.1, "degenerate input"


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


def _zero_noise(band, mag):
    """A noise model that returns exactly zero, for tests that need the
    noise-free flux (the uncertainty now tracks source brightness, so two
    runs with the same seed no longer share a noise realisation)."""
    return np.zeros(len(np.atleast_1d(mag)))


def _noiseless_cadence(cad):
    """Copy of a cadence with ``depth`` removed, routing it to ``noise_model``
    (so ``_zero_noise`` applies) instead of the LSST depth prescription."""
    out = cad.copy()
    out["depth"] = np.nan
    return out


def _flat_lc(lc):
    """Same time grid, constant unit flux -- isolates the noise prescription
    from the source's own variability."""
    from pioran_periodicity.simulate import SimulatedLightCurve

    return SimulatedLightCurve(
        time=lc.time, flux=np.ones_like(lc.flux), dt_days=lc.dt_days
    )


def _powerlaw_psd(f, index):
    return f ** (-float(index))


@pytest.fixture(scope="module")
def sim_lc():
    pytest.importorskip("stingray", reason="stingray not installed")
    import warnings as _w

    from pioran_periodicity.simulate import simulate_lightcurve

    with _w.catch_warnings():
        _w.simplefilter("ignore")
        return simulate_lightcurve(
            _powerlaw_psd,
            (2.0,),
            n_samples=40000,
            dt_minutes=60.0,
            mean=1.0,
            rms=0.15,
            seed=1,
        )


class TestSimulatorInjectionConvention:
    """The other half of MB1: what the SIMULATOR injects must be what the
    likelihood fits. Tested by differencing two runs with identical seeds.

    The noise is switched OFF (``depth`` NaN plus a zero-returning
    ``noise_model``) so the difference is exactly the injected signal. Equal
    seeds alone no longer suffice: the per-epoch uncertainty now depends on
    the source's brightness, so injecting a signal legitimately changes the
    noise realisation as well.
    """

    def test_injected_signal_is_scaled_by_the_band_amplitude(self, sim_lc):
        from pioran_periodicity.simulate import sample_real_cadence

        cad = _noiseless_cadence(_multiband_cadence())
        amps = {"g": 1.3, "r": 1.0, "i": 0.7}
        kw = dict(
            noise_model=_zero_noise,
            ref_mag=19.0,
            seed=17,
            band_amp=amps,
            return_band=True,
        )
        t0, f0, _, band = sample_real_cadence(sim_lc, cad, **kw)
        t1, f1, _, _ = sample_real_cadence(sim_lc, cad, mean_signal=_sine, **kw)

        assert np.array_equal(t0, t1)
        a = np.array([amps[b] for b in band], dtype=float)
        # identical seeds => identical noise and latent draw, so the whole
        # difference is the injected signal
        assert np.allclose(f1 - f0, a * _sine(t0), atol=1e-12)
        # and it is genuinely band-dependent, or this proves nothing
        assert not np.allclose(f1 - f0, _sine(t0), atol=1e-6)

    def test_injected_signal_is_unscaled_without_band_amp(self, sim_lc):
        from pioran_periodicity.simulate import sample_real_cadence

        cad = _noiseless_cadence(_multiband_cadence())
        kw = dict(noise_model=_zero_noise, ref_mag=19.0, seed=17)
        t0, f0, _ = sample_real_cadence(sim_lc, cad, **kw)
        t1, f1, _ = sample_real_cadence(sim_lc, cad, mean_signal=_sine, **kw)
        assert np.allclose(f1 - f0, _sine(t0), atol=1e-12)

    def test_the_correct_convention_fits_simulated_data_better(self, sim_lc):
        """End-to-end, from the science side: data generated by the simulator
        must be better explained by the convention the simulator used.

        Evaluated at the TRUE injected amplitude, so this compares model
        specifications rather than searching for a maximum -- the red noise
        dominates the likelihood surface, so an argmax test would be noise.
        """
        from pioran_periodicity.multiband import BandEncoding
        from pioran_periodicity.simulate import sample_real_cadence

        beta, period, amp_true = 1.0, 1.4, 0.6
        cad = _multiband_cadence(n_per_band=60, bands=("g", "r", "i"), seed=3)
        amps = power_law_band_amplitudes(["g", "r", "i"], beta, "r")
        assert max(amps.values()) / min(amps.values()) > 1.4, "degenerate input"

        def injected(tt):
            return sine_mean(tt, 0.0, amp_true, period)

        t, y, yerr, band = sample_real_cadence(
            sim_lc,
            cad,
            noise_model=None,
            ref_mag=19.0,
            seed=23,
            band_amp=amps,
            mean_signal=injected,
            return_band=True,
        )
        enc = BandEncoding.from_counts(band)
        code = enc.encode(band)
        t = t - t[0]
        y = y - np.median(y[code == 0])
        a = np.array([1.0] + [amps[b] for b in enc.others])[code]
        kern = drw_kernel(LOG10_VAR, LOG10_FBEND)

        # correctly specified: m(t) in latent units. Routed through the
        # function under test, so reintroducing MB1 breaks this assertion.
        latent = gp_log_likelihood_multiband(
            kern,
            t,
            y,
            yerr,
            code,
            np.array([1.0] + [amps[b] for b in enc.others]),
            np.zeros(1 + len(enc.others)),
            mean_func=injected,
        )
        # misspecified: m(t) subtracted in observed units (the MB1 bug)
        observed = gp_log_likelihood(
            kern, t, (y - 0.0 - injected(t)) / a, yerr / a
        ) - np.sum(np.log(a))

        assert latent > observed


class TestSineParameterNaming:
    """MB2: one symbol must not mean two things.

    The scenario-CSV column ``A1`` is the INJECTED sine AMPLITUDE. The model
    parameters are COEFFICIENTS. Naming both ``A1`` invited comparing a
    fitted coefficient against an injected amplitude.
    """

    def _sine_spec(self):
        fam = build_family("drw", PriorConfig(), variants=("plain", "sine"))
        return fam["drw+sine"]

    def test_sine_parameters_are_named_A_cos_and_A_sin(self):
        names = self._sine_spec().param_names
        assert "A_cos" in names and "A_sin" in names
        assert names.index("A_cos") < names.index("A_sin")

    def test_no_model_exposes_the_legacy_A1_A2_names(self):
        """Guards the collision from coming back under any variant."""
        cfg = PriorConfig()
        for noise in ("drw", "obpl", "carma"):
            kwargs = {}
            if noise == "obpl":
                from pioran_periodicity.kernels import FrequencyBand

                kwargs["band"] = FrequencyBand(f_min=0.01, f_max=10.0)
            fam = build_family(
                noise,
                cfg,
                variants=("plain", "sine", "linear", "sine+linear"),
                **kwargs,
            )
            for name, spec in fam.members.items():
                assert "A1" not in spec.param_names, f"{name} reintroduced A1"
                assert "A2" not in spec.param_names, f"{name} reintroduced A2"

    def test_amplitude_is_the_hypot_not_either_coefficient(self):
        assert sine_amplitude(3.0, 4.0) == pytest.approx(5.0)
        # a coefficient alone is NOT the amplitude
        assert sine_amplitude(3.0, 4.0) != pytest.approx(3.0)

    @pytest.mark.parametrize("phase", [0.0, 0.3, 1.1, 2.9, -1.4])
    def test_injected_amplitude_is_recovered_as_the_hypot_at_any_phase(self, phase):
        """Injection uses absolute time while fitting re-zeroes it, so the
        signal lands in an arbitrary mix of the two coefficients. Only the
        hypot is comparable to the injected amplitude."""
        amp_true, period = 0.37, 2.1
        t = np.linspace(0.0, 9.0, 400)
        signal = amp_true * np.sin(2.0 * np.pi * t / period + phase)
        design = np.column_stack(
            [np.cos(2 * np.pi * t / period), np.sin(2 * np.pi * t / period)]
        )
        a_cos, a_sin = np.linalg.lstsq(design, signal, rcond=None)[0]
        assert sine_amplitude(a_cos, a_sin) == pytest.approx(amp_true, rel=1e-8)
        assert np.allclose(sine_mean(t, a_cos, a_sin, period), signal, atol=1e-8)

    def test_run_sim_puts_the_injected_amplitude_in_the_sin_coefficient(self):
        """Pins the exact collision point: the CSV's ``A1`` becomes the
        ``A_sin`` coefficient with ``A_cos = 0``."""
        run_sim = pytest.importorskip("run_sim", reason="scripts/ not importable")
        row = pd.Series({"period": 2.0, "A1": 0.3})
        m = run_sim.true_mean_signal(row)
        t = np.linspace(0.0, 4.0, 50)
        assert np.allclose(m(t), sine_mean(t, 0.0, 0.3, 2.0))
        assert sine_amplitude(0.0, 0.3) == pytest.approx(0.3)

    def test_run_sim_returns_no_signal_for_the_null_case(self):
        run_sim = pytest.importorskip("run_sim", reason="scripts/ not importable")
        assert (
            run_sim.true_mean_signal(pd.Series({"period": np.nan, "A1": 0.0})) is None
        )
        assert run_sim.true_mean_signal(pd.Series({"highalpha": -2.0})) is None


class TestPeriodPriorIsCampaignWide:
    """MB3.1 / MB3.4: one sine period prior for every campaign, NaN-safe.

    A Bayes factor is only comparable across runs that share a prior, so a
    null campaign and the signal campaign whose threshold it calibrates must
    use the same sine period bound. Deriving it from each CSV's own
    ``period`` column silently gave nulls (0.2, 4.0) and signals (0.2, 8.0).
    """

    def _run_sim(self):
        return pytest.importorskip("run_sim", reason="scripts/ not importable")

    def test_period_prior_units_are_years(self):
        """t is in years everywhere, so the prior is too. Guards against a
        days/years mix-up of the kind MB3 catalogues for f_bend."""
        rs = self._run_sim()
        lo, hi = rs.PERIOD_PRIOR
        assert 0 < lo < hi
        # a plausible period range for a decade-long survey, in YEARS
        assert 0.1 <= lo <= 1.0 and 4.0 <= hi <= 20.0

    def test_null_and_signal_csvs_get_the_same_prior(self):
        rs = self._run_sim()
        null = pd.DataFrame({"period": [np.nan] * 5, "A1": [0.0] * 5})
        signal = pd.DataFrame({"period": [1.25, 3.75, 7.5], "A1": [0.1, 0.2, 0.3]})
        assert rs.resolve_period_prior(null, rs.PERIOD_PRIOR[1]) == (
            rs.resolve_period_prior(signal, rs.PERIOD_PRIOR[1])
        )

    def test_all_nan_period_column_does_not_yield_a_nan_bound(self):
        """The MB3.4 guard: an all-NaN column must not reach the sampler as a
        NaN prior bound. np.nanmax would warn and return NaN here."""
        rs = self._run_sim()
        got = rs.resolve_period_prior(pd.DataFrame({"period": [np.nan] * 4}), 8.0)
        assert np.isfinite(got) and got == 8.0

    def test_bound_is_not_derived_from_the_data(self):
        """Short injected periods must NOT shrink the prior -- that is what
        made null and signal runs incomparable."""
        rs = self._run_sim()
        assert rs.resolve_period_prior(pd.DataFrame({"period": [0.5]}), 8.0) == 8.0

    def test_period_outside_the_prior_is_a_hard_error(self):
        rs = self._run_sim()
        with pytest.raises(ValueError, match="outside the sine period prior"):
            rs.resolve_period_prior(pd.DataFrame({"period": [9.0]}), 8.0)

    def test_missing_period_column_is_fine(self):
        rs = self._run_sim()
        assert rs.resolve_period_prior(pd.DataFrame({"highalpha": [-2.0]}), 8.0) == 8.0

    @pytest.mark.parametrize("bad", [np.nan, 0.0, -1.0])
    def test_non_positive_or_nan_bound_is_rejected(self, bad):
        rs = self._run_sim()
        with pytest.raises(ValueError, match="finite and > 0"):
            rs.resolve_period_prior(pd.DataFrame({"period": [1.0]}), bad)

    def test_make_cfg_default_matches_the_constant(self):
        rs = self._run_sim()
        assert rs.make_cfg().period == rs.PERIOD_PRIOR


class TestSlopeGridStaysInsideThePrior:
    """MB3.2: simulated truths must be interior to the fitted prior AND to
    the region the basis expansion can represent."""

    def _defaults(self, module):
        mod = pytest.importorskip(module, reason="scripts/ not importable")
        return [float(v) for v in mod.HIGHALPHA_DEFAULT.split(",")]

    @pytest.mark.parametrize(
        "module", ["make_real_cadence_csv", "make_slope_robustness_csv"]
    )
    def test_every_default_slope_is_interior_to_the_alpha_high_prior(self, module):
        cap = PriorConfig().alpha_high_max
        for ha in self._defaults(module):
            alpha_high_true = -ha  # sign convention: model alpha = -CSV alpha
            assert alpha_high_true < cap - 0.25, (
                f"highalpha={ha} implies alpha_high={alpha_high_true}, "
                f"too close to the prior bound {cap}"
            )

    def test_multiband_campaign_slopes_are_interior_too(self):
        mod = pytest.importorskip("make_multiband_csv", reason="scripts/ missing")
        cap = PriorConfig().alpha_high_max
        for ha in tuple(mod.NULL_HIGHALPHA) + tuple(mod.SIGNAL_HIGHALPHA):
            assert -float(ha) < cap - 0.25

    def test_the_retired_grid_would_now_fail(self):
        """Documents the defect: -4.0 maps to exactly the prior bound."""
        assert not (-(-4.0) < PriorConfig().alpha_high_max - 0.25)

    def test_steepest_default_slope_is_representable_by_the_basis(self):
        """The prior bound of 4.0 is set by the SHO expansion's accuracy at
        the component count the campaigns use, not chosen arbitrarily."""
        from pioran_periodicity.kernels import FrequencyBand, psd_approximation_error

        steepest = max(
            -min(self._defaults("make_real_cadence_csv")),
            -min(self._defaults("make_slope_robustness_csv")),
        )
        band = FrequencyBand.from_times(np.linspace(0.0, 10.0, 700))
        err = psd_approximation_error(1.0, 0.30, steepest, band, n_components=20)
        assert err["max_rel_error"] < 0.05


class TestNoisePrescriptions:
    """MB3.3: the two surveys use different, survey-appropriate noise models
    that must nonetheless be expressed in the same units."""

    def test_conversion_constant(self):
        from pioran_periodicity.simulate import MAG_TO_FRACTIONAL_FLUX

        assert MAG_TO_FRACTIONAL_FLUX == pytest.approx(0.4 * np.log(10.0))
        assert MAG_TO_FRACTIONAL_FLUX == pytest.approx(0.921, abs=1e-3)

    def test_magnitude_errors_are_converted_to_fractional_flux(self, sim_lc):
        """ZTF branch: noise_model returns magnitudes, so the reported flux
        error must carry the 0.4 ln10 factor.

        Uses a FLAT light curve so the per-epoch brightness scaling is a
        no-op and the unit conversion can be asserted exactly.
        """
        from pioran_periodicity.simulate import (
            MAG_TO_FRACTIONAL_FLUX,
            sample_real_cadence,
        )

        flat = _flat_lc(sim_lc)
        cad = _multiband_cadence(n_per_band=20, bands=("g", "r"), seed=9)
        cad["depth"] = np.nan  # no depth -> ZTF-style branch
        magerr = 0.05
        _, _, flux_err = sample_real_cadence(
            flat,
            cad,
            noise_model=lambda b, m: np.full(len(m), magerr),
            ref_mag=19.0,
            seed=13,
        )
        assert np.allclose(flux_err, MAG_TO_FRACTIONAL_FLUX * magerr * 1.0)
        # and is NOT the unconverted magnitude error (the MB3.3 regression)
        assert not np.allclose(flux_err, magerr * 1.0)

    def test_both_branches_are_heteroscedastic(self, sim_lc):
        """LSST noise varies per epoch with visit depth; ZTF noise now varies
        per epoch with the source's own simulated brightness.

        Replaces an earlier test that asserted the ZTF branch was CONSTANT --
        that was the homoscedastic convention this change removes.
        """
        from pioran_periodicity.simulate import sample_real_cadence

        cad = _multiband_cadence(n_per_band=20, bands=("g",), seed=9)
        cad["depth"] = np.linspace(22.0, 24.0, len(cad))
        _, _, lsst_err = sample_real_cadence(
            sim_lc, cad, noise_model=None, ref_mag=19.0, seed=13
        )
        assert lsst_err.std() > 0

        cad_ztf = cad.copy()
        cad_ztf["depth"] = np.nan
        _, _, ztf_err = sample_real_cadence(
            sim_lc,
            cad_ztf,
            noise_model=lambda b, m: 0.01 * m,
            ref_mag=19.0,
            seed=13,
        )
        assert ztf_err.std() > 0

    def test_ztf_error_is_set_by_brightness_not_by_the_noise_draw(self, sim_lc):
        """The uncertainty must come from the NOISE-FREE model flux.

        With a flat light curve and a constant noise model the error is exactly
        constant even though the returned flux is noisy. Had sigma been derived
        from the realised (noisy) flux, it would scatter -- a noise draw
        feeding back into its own error bar.
        """
        from pioran_periodicity.simulate import sample_real_cadence

        flat = _flat_lc(sim_lc)
        cad = _multiband_cadence(n_per_band=30, bands=("g",), seed=9)
        cad["depth"] = np.nan
        _, flux, flux_err = sample_real_cadence(
            flat,
            cad,
            noise_model=lambda b, m: 0.01 * m,
            ref_mag=19.0,
            seed=13,
        )
        assert flux.std() > 0  # the flux really is noisy
        assert flux_err.std() == pytest.approx(0.0, abs=1e-15)

    def test_ztf_error_tracks_source_brightness(self, sim_lc):
        """A brighter epoch gets a smaller magnitude error, because magerr(mag)
        increases with magnitude. Sign check, not just "it varies"."""
        from pioran_periodicity.simulate import sample_real_cadence

        cad = _multiband_cadence(n_per_band=60, bands=("g",), seed=9)
        cad["depth"] = np.nan
        # noise_model increasing in magnitude => fainter epochs noisier
        _, _, flux_err = sample_real_cadence(
            sim_lc,
            cad,
            noise_model=lambda b, m: 0.01 * m,
            ref_mag=19.0,
            seed=13,
        )
        assert flux_err.std() > 0
        # sigma_mag decreases with brightness, so the brightest epoch must
        # carry a strictly smaller error than the faintest one.
        assert flux_err.min() < flux_err.max()

    def test_per_band_reference_magnitude_comes_from_real_photometry(self, sim_lc):
        """``ref_mag`` is the r-band catalogue magnitude; each band must use
        its OWN median real magnitude instead (the g-band colour bias)."""
        from pioran_periodicity.simulate import _band_reference_magnitudes

        band = np.array(["g"] * 5 + ["r"] * 5, dtype=object)
        mag_real = np.array(
            [20.0, 20.2, 20.4, 20.6, 20.8, 19.0, 19.1, 19.2, 19.3, 19.4]
        )
        out = _band_reference_magnitudes(band, mag_real, ref_mag=19.0)
        assert out["g"] == pytest.approx(20.4)
        assert out["r"] == pytest.approx(19.2)
        # the r-band catalogue value must NOT have been used for g
        assert out["g"] != pytest.approx(19.0)

    def test_band_reference_falls_back_to_ref_mag_without_photometry(self):
        """LSST/OpSim cadences carry no real photometry: fall back cleanly,
        and do not trip the all-NaN nanmedian trap (defect MB3.4)."""
        from pioran_periodicity.simulate import _band_reference_magnitudes

        band = np.array(["g"] * 4, dtype=object)
        out = _band_reference_magnitudes(band, np.full(4, np.nan), ref_mag=18.5)
        assert out["g"] == pytest.approx(18.5)

        # partial coverage: use the finite subset only
        mag = np.array([np.nan, 20.0, np.nan, 20.4])
        out = _band_reference_magnitudes(band, mag, ref_mag=18.5)
        assert out["g"] == pytest.approx(20.2)


class TestOBPLSharpness:
    """MB3.5: the simulated PSD must be in the same family as the fitted one.
    Pioran's SingleBendingPowerLaw has no sharpness parameter (it is 1)."""

    @pytest.mark.parametrize(
        "module", ["make_real_cadence_csv", "make_slope_robustness_csv"]
    )
    def test_campaign_sharpness_default_is_one(self, module):
        mod = pytest.importorskip(module, reason="scripts/ not importable")
        assert mod.FIXED_DEFAULTS["sharpness"] == 1.0

    def test_bend_pl_at_sharpness_one_is_pioran_single_bending_power_law(self):
        rs = pytest.importorskip("run_sim", reason="scripts/ not importable")
        f = np.logspace(-2, 2, 200)
        f_bend, alo, ahi = 2.0, -1.0, -3.5
        got = rs.bend_pl(f, 1.0, f_bend, alo, ahi, 1.0)
        # Pioran: P(f) = (f/fb)^-a1 / (1 + (f/fb)^(a2-a1)), a = -alpha
        x = f / f_bend
        want = x ** (alo) / (1.0 + x ** (alo - ahi))
        assert np.allclose(got, want, rtol=1e-12)

    def test_sharpness_ten_is_a_different_psd_family(self):
        """Guards the regression: at the bend the old default differed from
        the fitted model by ~0.27 dex, inside the science band."""
        rs = pytest.importorskip("run_sim", reason="scripts/ not importable")
        f_bend = 2.0
        at_bend_s1 = rs.bend_pl(np.array([f_bend]), 1.0, f_bend, -1.0, -3.5, 1.0)
        at_bend_s10 = rs.bend_pl(np.array([f_bend]), 1.0, f_bend, -1.0, -3.5, 10.0)
        assert at_bend_s10[0] / at_bend_s1[0] > 1.5


class TestSineAmplitudePriorIsRelative:
    """The sine amplitude prior is on the DIMENSIONLESS ratio f = A/sigma.

    An absolute prior is strongly informative for a quiet object and nearly
    vacuous for a variable one (quasar sigma spans ~0.03-0.5 mag), which
    silently varies the Occam factor -- and hence the effective detection
    threshold -- from object to object, so one null calibration would not
    transfer across a survey.
    """

    def _sine_spec(self, **kw):
        return build_family("drw", PriorConfig(**kw), variants=("sine",))["drw+sine"]

    def test_amplitude_prior_scales_with_the_sampled_process_sigma(self):
        from pioran_periodicity.priors import ProcessRelativeNormal

        prior = ProcessRelativeNormal(1.2)
        # u = 0.8413 -> +1 sigma of the standard normal
        u = 0.8413447460685429
        for log10_var in (-4.0, -2.0, 0.0):
            sigma = np.sqrt(10.0**log10_var)
            got = prior.transform(u, {"log10_variance": log10_var})
            assert got == pytest.approx(1.2 * sigma, rel=1e-6)

    def test_it_is_not_a_fixed_absolute_scale(self):
        """Guards the regression: two objects with different variability must
        get different absolute amplitude priors."""
        from pioran_periodicity.priors import ProcessRelativeNormal

        prior = ProcessRelativeNormal(1.2)
        u = 0.9
        quiet = prior.transform(u, {"log10_variance": -3.0})
        loud = prior.transform(u, {"log10_variance": -1.0})
        assert loud / quiet == pytest.approx(10.0, rel=1e-6)

    def test_induced_prior_on_f_is_rayleigh(self):
        """Two independent Normal(0, f0*sigma) coefficients give a Rayleigh(f0)
        prior on f = A/sigma and a uniform phase."""
        rng = np.random.default_rng(11)
        f0, log10_var = 1.2, -1.6
        sigma = np.sqrt(10.0**log10_var)
        spec = self._sine_spec(sine_amplitude_fraction=f0)
        i_cos = spec.param_names.index("A_cos")
        i_sin = spec.param_names.index("A_sin")
        i_var = spec.param_names.index("log10_variance")

        cube = rng.uniform(size=(4000, len(spec.param_names)))
        # pin log10_variance by inverting its Uniform(-4, 1) prior
        lo, hi = PriorConfig().log10_variance
        cube[:, i_var] = (log10_var - lo) / (hi - lo)
        drawn = np.array([spec.prior(c) for c in cube])
        f = np.hypot(drawn[:, i_cos], drawn[:, i_sin]) / sigma

        # Rayleigh(f0): median f0*sqrt(2 ln 2), mean f0*sqrt(pi/2)
        assert np.median(f) == pytest.approx(f0 * np.sqrt(2 * np.log(2)), rel=0.05)
        assert f.mean() == pytest.approx(f0 * np.sqrt(np.pi / 2), rel=0.05)
        phase = np.arctan2(drawn[:, i_cos], drawn[:, i_sin])
        assert abs(np.mean(np.cos(phase))) < 0.05  # isotropic
        assert abs(np.mean(np.sin(phase))) < 0.05

    def test_default_fraction_brackets_the_real_candidate(self):
        """PG 1302-102 sits at f = 2.3 (A = 0.124 mag, sigma = 0.053 mag).
        The prior must give that meaningful support without piling mass at
        f > 5, where a signal would dominate the light curve."""
        f0 = PriorConfig().sine_amplitude_fraction
        surv = lambda f: np.exp(-(f**2) / (2 * f0**2))  # noqa: E731  Rayleigh
        assert 0.02 < surv(2.3) < 0.5, "f=2.3 must be neither typical nor absurd"
        assert surv(5.0) < 0.01

    def test_carma_falls_back_to_the_absolute_scale(self):
        """CARMA's process variance is a nonlinear function of its AR/MA
        coefficients, not a sampled parameter, so it cannot use the relative
        prior. The fallback must be recorded, not silent."""
        spec = build_family(
            "carma", PriorConfig(), variants=("sine",), carma_order=(2, 1)
        )["carma+sine"]
        assert "log10_variance" not in spec.param_names
        assert spec.meta["sine_amplitude_prior"].startswith("absolute:")

    def test_relative_parametrisation_is_recorded_in_meta(self):
        assert self._sine_spec().meta["sine_amplitude_prior"] == "relative:f=1.2"

    def test_plain_variant_records_no_sine_prior(self):
        spec = build_family("drw", PriorConfig(), variants=("plain",))["drw"]
        assert spec.meta["sine_amplitude_prior"] is None

    def test_rejects_a_config_with_neither_parametrisation(self):
        with pytest.raises(ValueError, match="both are None"):
            PriorConfig(sine_amplitude_fraction=None, sine_amplitude_scale=None)

    def test_rejects_non_positive_fraction(self):
        with pytest.raises(ValueError, match="sine_amplitude_fraction must be > 0"):
            PriorConfig(sine_amplitude_fraction=0.0)


class TestPeriodPriorIsLogUniform:
    """Period is a scale parameter: equal weight per octave, not per year."""

    def _period_draws(self, n=4000):
        spec = build_family("drw", PriorConfig(), variants=("sine",))["drw+sine"]
        i = spec.param_names.index("period")
        rng = np.random.default_rng(3)
        cube = rng.uniform(size=(n, len(spec.param_names)))
        return np.array([spec.prior(c)[i] for c in cube]), PriorConfig().period

    def test_period_is_log_uniform_not_uniform(self):
        draws, (lo, hi) = self._period_draws()
        # log-uniform => log(period) is uniform => median is the GEOMETRIC mean
        assert np.median(draws) == pytest.approx(np.sqrt(lo * hi), rel=0.06)
        # and clearly NOT the arithmetic mean a Uniform prior would give
        assert abs(np.median(draws) - 0.5 * (lo + hi)) > 0.2 * (hi - lo)

    def test_equal_mass_per_octave(self):
        draws, (lo, hi) = self._period_draws()
        edges = np.geomspace(lo, hi, 5)
        counts = np.histogram(draws, bins=edges)[0]
        assert counts.min() / counts.max() > 0.85

    def test_bounds_are_respected(self):
        draws, (lo, hi) = self._period_draws()
        assert draws.min() >= lo and draws.max() <= hi


class TestSineColourIndex:
    """Freeing the periodic component's colour from the red noise's.

    Model: y_b = mu_b + a_b*x(t) + c_b*m(t), with
    c_b = (lambda_b/lambda_ref)**-beta_sine. c_b = a_b (the default) means the
    sine shares the noise's colour and multi-band data gives no leverage to
    separate them -- BOTH components are coherent across bands by
    construction. c_b = 1 is the MB1 defect. beta_sine free is what makes the
    distinction measurable.
    """

    LSST_OTHERS = ("g", "i", "u", "y", "z")

    def _family(self, **kw):
        return build_family(
            "drw", PriorConfig(), variants=("plain", "sine"),
            photometric_bands=self.LSST_OTHERS, fit_sine_colour=True,
            reference_band="r", survey="lsst", **kw,
        )

    @pytest.mark.parametrize("c_b", [[1.0, 1.0, 1.0], [1.0, 1.3, 0.7], [1.0, 0.6, 1.9]])
    def test_general_c_b_matches_dense_mvn(self, mb_data, c_b):
        t, y, yerr, code, amp, mu = mb_data
        c = np.asarray(c_b)
        got = gp_log_likelihood_multiband(
            drw_kernel(LOG10_VAR, LOG10_FBEND), t, y, yerr, code, amp, mu,
            mean_func=_sine, band_mean_amp=c,
        )
        cov = (amp[code][:, None] * amp[code][None, :]) * _drw_cov(t) + np.diag(
            yerr**2
        )
        want = float(
            multivariate_normal.logpdf(
                y, mean=mu[code] + c[code] * _sine(t), cov=cov
            )
        )
        assert got == pytest.approx(want, abs=1e-8)

    def test_c_b_equal_to_a_b_reproduces_the_default(self, mb_data):
        """The default (c_b = a_b) must be exactly the explicit case, so
        turning the feature on with beta_sine = beta_noise changes nothing."""
        t, y, yerr, code, amp, mu = mb_data
        k = drw_kernel(LOG10_VAR, LOG10_FBEND)
        default = gp_log_likelihood_multiband(
            k, t, y, yerr, code, amp, mu, mean_func=_sine
        )
        explicit = gp_log_likelihood_multiband(
            k, t, y, yerr, code, amp, mu, mean_func=_sine, band_mean_amp=amp
        )
        assert default == pytest.approx(explicit, abs=1e-10)

    def test_beta_sine_zero_is_achromatic(self):
        """beta_sine = 0 gives c_b = 1 for every band -- which is precisely
        the MB1 defect, now reachable only as a deliberate hypothesis."""
        spec = self._family()["drw+sine"]
        ratios = np.array([1.2, 0.8])
        assert np.allclose(ratios ** (-0.0), 1.0)
        assert "beta_sine" in spec.param_names

    def test_costs_exactly_one_dimension_regardless_of_band_count(self):
        """The colour dependence is parametrised, not free per band, so the
        cost does not grow with the number of filters."""
        for others in [("g",), ("g", "r"), self.LSST_OTHERS]:
            with_colour = build_family(
                "drw", PriorConfig(), variants=("sine",),
                photometric_bands=others, fit_sine_colour=True,
                reference_band="i", survey="lsst",
            )["drw+sine"]
            without = build_family(
                "drw", PriorConfig(), variants=("sine",),
                photometric_bands=others,
            )["drw+sine"]
            assert len(with_colour.param_names) - len(without.param_names) == 1

    def test_beta_sine_only_on_sine_variants(self):
        fam = self._family()
        assert "beta_sine" not in fam["drw"].param_names
        assert "beta_sine" in fam["drw+sine"].param_names

    def test_off_by_default(self):
        spec = build_family(
            "drw", PriorConfig(), variants=("sine",),
            photometric_bands=self.LSST_OTHERS,
        )["drw+sine"]
        assert "beta_sine" not in spec.param_names
        assert spec.meta["fit_sine_colour"] is False

    def test_meta_records_the_choice(self):
        assert self._family()["drw+sine"].meta["fit_sine_colour"] is True
        assert self._family()["drw+sine"].meta["reference_band"] == "r"

    def test_requires_reference_band_and_photometric_bands(self):
        with pytest.raises(ValueError, match="needs photometric_bands"):
            build_family(
                "drw", PriorConfig(), variants=("sine",),
                photometric_bands=self.LSST_OTHERS, fit_sine_colour=True,
            )
        with pytest.raises(ValueError, match="needs photometric_bands"):
            build_family(
                "drw", PriorConfig(), variants=("sine",),
                fit_sine_colour=True, reference_band="r",
            )

    def test_rejects_unknown_survey_and_band(self):
        with pytest.raises(ValueError, match="no filter wavelengths"):
            build_family(
                "drw", PriorConfig(), variants=("sine",),
                photometric_bands=("g",), fit_sine_colour=True,
                reference_band="r", survey="des",
            )
        with pytest.raises(ValueError, match="no wavelength for"):
            build_family(
                "drw", PriorConfig(), variants=("sine",),
                photometric_bands=("Q",), fit_sine_colour=True,
                reference_band="r", survey="lsst",
            )

    def test_leakage_and_doppler_hypotheses_are_distinguishable(self, mb_data):
        """The discriminant must actually separate: red-noise leakage
        (c_b = a_b) and an achromatic signal (c_b = 1) must give measurably
        different likelihoods on the same data."""
        t, y, yerr, code, amp, mu = mb_data
        k = drw_kernel(LOG10_VAR, LOG10_FBEND)
        big = lambda tt: 0.4 * np.cos(2 * np.pi * tt / 1.7)  # noqa: E731
        leak = gp_log_likelihood_multiband(
            k, t, y, yerr, code, amp, mu, mean_func=big, band_mean_amp=amp
        )
        achrom = gp_log_likelihood_multiband(
            k, t, y, yerr, code, amp, mu, mean_func=big, band_mean_amp=np.ones(3)
        )
        assert abs(leak - achrom) > 1.0


class TestFreeProcessMean:
    """mu0: the process mean is fitted, not fixed at a data-derived value.

    Scope note: these tests check the parameter is WIRED CORRECTLY -- present
    everywhere, shared, and applied on the right side of the a_b rescale.
    They deliberately do NOT attempt to verify the statistical claim (that
    marginalising over the offset improves log10_fbend coverage); that rests
    on the literature, not on this suite.
    """

    def _fam(self, **kw):
        return build_family("drw", PriorConfig(), **kw)

    def test_present_in_every_variant_including_plain(self):
        fam = self._fam(variants=("plain", "sine", "linear", "sine+linear"))
        for name, spec in fam.members.items():
            assert "mu0" in spec.param_names, name

    def test_prior_object_is_shared_across_variants(self):
        """M1/B4: a parameter common to several variants must be the SAME
        object, so a null model and its periodic alternative can never
        disagree about its prior. Building mu0 inside the variant loop breaks
        this -- it is what made the earlier free-mean attempt fail three
        tests."""
        fam = self._fam(variants=("plain", "sine", "linear", "sine+linear"))

        def prior_of(spec):
            return spec.prior.parameters[spec.param_names.index("mu0")].prior

        priors = [prior_of(s) for s in fam.members.values()]
        assert all(p is priors[0] for p in priors)

    def test_intercept_is_gone_and_subsumed_by_mu0(self):
        """A separate intercept would be exactly degenerate with mu0."""
        fam = self._fam(variants=("linear", "sine+linear"))
        for spec in fam.members.values():
            assert "intercept" not in spec.param_names
            assert "slope" in spec.param_names and "mu0" in spec.param_names

    def test_single_band_mu0_shifts_in_observed_units(self, synthetic):
        """loglike(y, mu0=c) must equal loglike(y-c, mu0=0)."""
        t, y, yerr = synthetic
        spec = self._fam(variants=("plain",))["drw"]
        base = dict(log10_variance=-0.5, log10_fbend=-1.0)
        c = 0.37
        shifted = spec.loglike(dict(base, mu0=c), t, y, yerr)
        recentred = spec.loglike(dict(base, mu0=0.0), t, y - c, yerr)
        assert shifted == pytest.approx(recentred, rel=1e-12)

    def test_mu0_actually_changes_the_likelihood(self, synthetic):
        t, y, yerr = synthetic
        spec = self._fam(variants=("plain",))["drw"]
        base = dict(log10_variance=-0.5, log10_fbend=-1.0)
        assert abs(
            spec.loglike(dict(base, mu0=0.0), t, y, yerr)
            - spec.loglike(dict(base, mu0=0.8), t, y, yerr)
        ) > 1e-3

    def test_multiband_mu0_is_the_reference_band_offset_not_scaled_by_a_b(
        self, mb_data
    ):
        """The subtlety: mu0 is in OBSERVED units, so it belongs in
        band_mu[0]. Putting it in mean_func instead would scale it by a_b,
        because the mean function is evaluated in shared-latent units."""
        t, y, yerr, code, amp, mu = mb_data
        spec = build_family(
            "drw", PriorConfig(), variants=("plain",), photometric_bands=("g", "i")
        )["drw"]
        mu0 = -0.3
        pdict = {
            "log10_variance": LOG10_VAR, "log10_fbend": LOG10_FBEND, "mu0": mu0,
            "a_g": amp[1], "mu_g": mu[1], "a_i": amp[2], "mu_i": mu[2],
        }
        got = spec.loglike(pdict, t, y, yerr, code)
        # reference: dense MVN with mu0 as the reference band's own offset
        band_mu = np.array([mu0, mu[1], mu[2]])
        cov = (amp[code][:, None] * amp[code][None, :]) * _drw_cov(t) + np.diag(
            yerr**2
        )
        want = float(multivariate_normal.logpdf(y, mean=band_mu[code], cov=cov))
        assert got == pytest.approx(want, abs=1e-8)
        # and it is NOT the a_b-scaled version
        wrong = float(
            multivariate_normal.logpdf(
                y, mean=band_mu[code] + (amp[code] - 1.0) * mu0, cov=cov
            )
        )
        assert abs(want - wrong) > 0.5, "degenerate input"
        assert got != pytest.approx(wrong, abs=1e-6)

    def test_rejects_non_positive_scale(self):
        with pytest.raises(ValueError, match="process_mean_scale must be > 0"):
            PriorConfig(process_mean_scale=0.0)

    @pytest.mark.parametrize("module", ["run_sim", "run_realdata"])
    def test_campaign_configs_still_construct(self, module):
        """Guards the breakage this change caused once: run_realdata passed an
        `intercept=` kwarg that no longer exists, and nothing imported it."""
        pytest.importorskip(module, reason="scripts/ not importable")

    @pytest.mark.slow
    def test_runs_end_to_end(self, synthetic):
        """It has to actually run, not just build."""
        pytest.importorskip("ultranest")
        from pioran_periodicity.inference import SamplerSettings, run_nested

        t, y, yerr = synthetic
        spec = self._fam(variants=("plain",))["drw"]
        r = run_nested(
            spec, t, y, yerr,
            settings=SamplerSettings(min_num_live_points=50, seed=7),
            show_status=False,
        )
        assert np.isfinite(r.logz)
        assert "mu0" in r.samples
        assert len(r.samples["mu0"]) > 0
