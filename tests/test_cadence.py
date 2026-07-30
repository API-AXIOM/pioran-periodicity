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
    # A steep fit extrapolated far outside the training range can dip
    # negative; __call__ must floor it, never return <= 0.
    nm = NoiseModel({"g": np.array([1.0, 0.0])})  # magerr = mag
    out = nm("g", np.array([-5.0, 0.0, 5.0]))
    assert np.all(out > 0)
    assert out[-1] == pytest.approx(5.0)


def test_noise_model_unknown_band_raises():
    nm = NoiseModel({"g": np.array([0.0, 0.05])})
    with pytest.raises(KeyError):
        nm("r", np.array([19.0]))


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
