"""Tests for pioran_periodicity.cadence against small synthetic directories
shaped like ztf_crossmatch.py / lsst_crossmatch.py output (from the sibling
pioran_periodicity_ai repo).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pioran_periodicity.cadence import CadenceLibrary, NoiseModel, fit_magerr_relation


def _make_ztf_dir(tmp_path):
    """One matched object with a real bad row of each kind (NaN hmjd,
    non-positive magerr) plus clean epochs, one unmatched object."""
    outdir = tmp_path / "ztf_data"
    (outdir / "lightcurves").mkdir(parents=True)

    rng = np.random.default_rng(0)
    n = 40
    mag = rng.uniform(18.0, 21.0, n)
    magerr = 0.01 + 0.05 * (mag - 18.0)  # monotonically increasing with mag
    hmjd = np.sort(rng.uniform(58200.0, 60800.0, n))
    band = rng.choice(["g", "r"], n)

    lc = pd.DataFrame({"ztf_objectid": 1, "band": band, "hmjd": hmjd,
                        "mag": mag, "magerr": magerr, "catflags": 0})
    lc.loc[0, "hmjd"] = np.nan          # bad: NaN timestamp
    lc.loc[1, "magerr"] = -0.001        # bad: non-positive magerr
    lc.to_csv(outdir / "lightcurves" / "obj_matched.csv", index=False)

    master = pd.DataFrame([
        {"object_id": "obj_matched", "ra": 10.0, "dec": 20.0, "z": 1.0, "rmag": 19.0,
         "file": "lightcurves/obj_matched.csv", "n_epochs": n, "matched": True},
        {"object_id": "obj_unmatched", "ra": 11.0, "dec": 21.0, "z": 1.5, "rmag": 20.0,
         "file": "", "n_epochs": 0, "matched": False},
    ])
    master.to_csv(outdir / "master.csv", index=False)
    return outdir, n


def _make_lsst_dir(tmp_path):
    outdir = tmp_path / "lsst_data"
    (outdir / "lightcurves").mkdir(parents=True)

    rng = np.random.default_rng(1)
    n = 25
    mjd = np.sort(rng.uniform(61200.0, 64800.0, n))
    band = rng.choice(list("ugrizy"), n)
    depth = rng.uniform(22.0, 25.0, n)
    seeing = rng.uniform(0.7, 1.5, n)

    lc = pd.DataFrame({"mjd": mjd, "band": band, "depth": depth,
                        "seeing": seeing, "is_too": None})
    lc.to_csv(outdir / "lightcurves" / "obj_a.csv", index=False)

    master = pd.DataFrame([
        {"object_id": "obj_a", "ra": 30.0, "dec": -40.0, "z": 2.0, "rmag": 20.5,
         "file": "lightcurves/obj_a.csv", "n_epochs": n, "matched": True},
    ])
    master.to_csv(outdir / "master.csv", index=False)
    return outdir, n


@pytest.fixture
def library(tmp_path):
    ztf_dir, n_ztf = _make_ztf_dir(tmp_path)
    lsst_dir, n_lsst = _make_lsst_dir(tmp_path)
    lib = CadenceLibrary.from_survey_dirs({"ztf": ztf_dir, "lsst": lsst_dir})
    return lib, n_ztf, n_lsst


def test_fit_magerr_relation_shape_mismatch():
    with pytest.raises(ValueError):
        fit_magerr_relation(np.array([1.0, 2.0]), np.array([0.1, 0.2, 0.3]))


def test_fit_magerr_relation_too_few_points():
    with pytest.raises(ValueError):
        fit_magerr_relation(np.array([1.0]), np.array([0.1]), degree=2)


def test_noise_model_floors_negative_predictions():
    # A legacy poly fit extrapolated far outside its training range can dip
    # negative; __call__ must floor it, never return <= 0.
    nm = NoiseModel({"g": np.array([1.0, 0.0])}, kind="poly_magerr")
    out = nm("g", np.array([-5.0, 0.0, 5.0]))
    assert np.all(out > 0)
    assert out[-1] == pytest.approx(5.0)


def test_noise_model_unknown_band_raises():
    nm = NoiseModel({"g": np.array([0.0, 0.05])})
    with pytest.raises(KeyError):
        nm("r", np.array([19.0]))


def test_noise_model_default_is_log_space():
    """Default coefficients are fitted to log10(magerr), so evaluation must
    exponentiate: magerr = 10**poly(mag)."""
    nm = NoiseModel({"g": np.array([0.25, -6.0])})
    assert nm.kind == "log10_linear"
    assert nm("g", np.array([20.0]))[0] == pytest.approx(10 ** (0.25 * 20.0 - 6.0))


def test_noise_model_log_linear_is_monotonic_and_positive():
    """The reason for the log-space fit: the old degree-2 fit to magerr turned
    over near 16.6 mag, so a brighter epoch got a LARGER error. That inverted
    the brightness-error correlation inside a light curve once the relation
    started being evaluated per epoch."""
    nm = NoiseModel({"g": np.array([0.256, -6.034])})
    mags = np.linspace(13.0, 24.0, 200)
    out = nm("g", mags)
    assert np.all(out > 0)
    assert np.all(np.diff(out) > 0)

    # the legacy quadratic really is non-monotonic over the same range
    legacy = NoiseModel(
        {"g": np.array([0.00654, -0.2189, 1.878])}, kind="poly_magerr"
    )
    assert not np.all(np.diff(legacy("g", mags)) > 0)


def test_noise_model_roundtrip_carries_kind():
    nm = NoiseModel({"g": np.array([0.25, -6.0])})
    back = NoiseModel.from_dict(nm.to_dict())
    assert back.kind == "log10_linear"
    assert back("g", np.array([19.5])) == pytest.approx(nm("g", np.array([19.5])))


def test_noise_model_legacy_cache_without_kind_is_read_as_poly():
    """A cache written before the log-space fit must still evaluate the way it
    was written, not be silently reinterpreted as log-space."""
    legacy = {"g": [0.0, 0.05]}
    nm = NoiseModel.from_dict(legacy)
    assert nm.kind == "poly_magerr"
    assert nm("g", np.array([19.0]))[0] == pytest.approx(0.05)


def test_noise_model_rejects_unknown_kind():
    with pytest.raises(ValueError, match="unknown NoiseModel kind"):
        NoiseModel({"g": np.array([0.25, -6.0])}, kind="nonsense")


def test_fit_magerr_relation_fits_in_log_space():
    mag = np.linspace(16.0, 21.0, 200)
    truth = 10 ** (0.27 * mag - 6.3)
    coeffs = fit_magerr_relation(mag, truth)
    assert coeffs[0] == pytest.approx(0.27, rel=1e-6)
    nm = NoiseModel({"g": coeffs})
    assert np.allclose(nm("g", mag), truth, rtol=1e-6)


def test_fit_magerr_relation_rejects_nonpositive_magerr():
    with pytest.raises(ValueError, match="strictly positive"):
        fit_magerr_relation(np.array([18.0, 19.0]), np.array([0.01, 0.0]))


def test_from_survey_dirs_skips_unmatched(library):
    lib, n_ztf, n_lsst = library
    assert lib.surveys() == ["lsst", "ztf"]
    assert lib.object_ids("ztf") == ["obj_matched"]
    assert lib.object_ids("lsst") == ["obj_a"]


def test_from_survey_dirs_drops_bad_rows(library):
    lib, n_ztf, n_lsst = library
    df = lib.get("ztf", "obj_matched")
    # 2 bad rows (NaN hmjd, negative magerr) dropped from n_ztf total.
    assert len(df) == n_ztf - 2
    assert df["mjd"].notna().all()
    assert (df["magerr"] > 0).all()
    assert df["mjd"].is_monotonic_increasing


def test_lsst_object_has_no_photometry_ztf_has_no_depth(library):
    lib, _, _ = library
    ztf_df = lib.get("ztf", "obj_matched")
    lsst_df = lib.get("lsst", "obj_a")
    assert ztf_df["depth"].isna().all()
    assert lsst_df["mag"].isna().all()
    assert lsst_df["mjd"].notna().all()


def test_get_missing_object_raises(library):
    lib, _, _ = library
    with pytest.raises(KeyError):
        lib.get("ztf", "does_not_exist")


def test_noise_model_fit_for_ztf_bands(library):
    lib, _, _ = library
    nm = lib.noise_models["ztf"]
    pred = nm("g", np.array([19.0]))
    assert pred[0] > 0
    assert "lsst" not in lib.noise_models  # OpSim gives depth directly, no fit needed


def test_random_is_deterministic_given_seed(library):
    lib, _, _ = library
    oid1, df1 = lib.random("lsst", np.random.default_rng(42))
    oid2, df2 = lib.random("lsst", np.random.default_rng(42))
    assert oid1 == oid2
    pd.testing.assert_frame_equal(df1, df2)


def test_random_returns_absolute_unmodified_mjds(library):
    lib, _, _ = library
    oid, df = lib.random("ztf", np.random.default_rng(0))
    expected = lib.get("ztf", oid)
    pd.testing.assert_frame_equal(df, expected)


def test_cache_roundtrip(library, tmp_path):
    lib, _, _ = library
    cache_dir = tmp_path / "cache"
    lib.to_cache(cache_dir)
    lib2 = CadenceLibrary.from_cache(cache_dir)

    assert lib2.surveys() == lib.surveys()
    for survey in lib.surveys():
        assert lib2.object_ids(survey) == lib.object_ids(survey)
        for object_id in lib.object_ids(survey):
            pd.testing.assert_frame_equal(
                lib2.get(survey, object_id), lib.get(survey, object_id),
                check_dtype=False,
            )
    for survey, nm in lib.noise_models.items():
        nm2 = lib2.noise_models[survey]
        for band, coeffs in nm.coeffs.items():
            np.testing.assert_allclose(nm2.coeffs[band], coeffs)
    assert lib2.metadata == lib.metadata
