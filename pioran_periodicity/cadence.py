"""Real survey cadence libraries (ZTF, LSST) for sampling simulated quasar
light curves onto genuine observing patterns, instead of the synthetic
seasonal-window pattern in ``simulate.sample_seasonal_pattern``.

Ingests the ``master.csv`` + per-object light curve CSVs produced by
``lsst_crossmatch.py`` / ``ztf_crossmatch.py`` in the sibling
``pioran_periodicity_ai`` repo (``workspace/claude_cadences_discussion/``) --
that data-acquisition step is project-specific and stays there; this module
only consumes its validated output.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# ZTF's hmjd (heliocentric MJD) is treated as MJD directly when combined with
# OpSim's plain MJD: the correction is at most ~8 minutes, negligible against
# DRW bend timescales (~182 days). Recorded here, not silently assumed.
ZTF_TIME_CONVENTION = "ztf_hmjd_as_mjd"

CADENCE_COLUMNS = ["mjd", "band", "mag", "magerr", "depth", "seeing"]
_MAGERR_FLOOR = 1e-4  # a fitted magerr(mag) polynomial must not predict <= 0


def fit_magerr_relation(
    mag: np.ndarray, magerr: np.ndarray, degree: int = 1
) -> np.ndarray:
    """Fit an empirical magerr(mag) relation from real photometry, in LOG space.

    Fits ``log10(magerr) = poly(mag)``, so the recovered relation is positive
    and (for the default ``degree=1``) MONOTONIC by construction.

    This replaces a degree-2 fit to ``magerr`` itself. That parabola turned
    over at 16.5-16.7 mag, so below the vertex a BRIGHTER source was assigned
    a LARGER error -- harmless while the relation was evaluated once per
    object at a fixed magnitude, but sign-inverting once it is evaluated per
    epoch, where it sets the direction of the brightness-error correlation
    inside a single light curve. It affected 6 of 243 per-band zero points in
    the ZTF pool (2 of 100 objects, faintest-case zero point 14.76 mag).

    The log-linear form is not a compromise: fitted to the same real ZTF
    photometry it matches the quadratic's accuracy (median |log10| residual
    0.0415 vs 0.0417 in g, 0.0396 vs 0.0388 in r, 0.0429 vs 0.0401 in i) while
    being monotonic everywhere. It is also the standard empirical form in the
    literature, e.g. ``log10(sigma) = 0.3416 m - 7.7095``; our ZTF slopes come
    out at 0.256 (g), 0.275 (r), 0.272 (i). Clamping the quadratic at its
    vertex -- the other published approach, equivalent to repeating the
    brightest magnitude bin -- would leave a kink where this has none.

    ``mag``, ``magerr`` must already be cleaned by the caller (finite,
    magerr > 0) -- this function does not filter. Returns ``np.polyfit``
    coefficients IN LOG SPACE, highest power first; pair them with
    ``NoiseModel(kind="log10_linear")``.
    """
    mag = np.asarray(mag, dtype=float)
    magerr = np.asarray(magerr, dtype=float)
    if mag.shape != magerr.shape:
        raise ValueError(
            f"mag and magerr must have the same shape, got {mag.shape} vs {magerr.shape}"
        )
    if len(mag) < degree + 1:
        raise ValueError(
            f"need at least {degree + 1} points to fit a degree-{degree} "
            f"polynomial, got {len(mag)}"
        )
    if np.any(magerr <= 0):
        raise ValueError("magerr must be strictly positive to fit in log space")
    return np.polyfit(mag, np.log10(magerr), degree)


# LSST single-visit photometric error model, Ivezic et al. (2019):
#   sigma^2 = sigma_sys^2 + sigma_rand^2
#   sigma_rand^2 = (0.04 - gamma) x + gamma x^2,   x = 10**(0.4 (m - m5))
# The calibration system is specified to hold the systematic floor below
# 0.005 mag; simulations conventionally adopt that value.
LSST_SIGMA_SYS = 0.005
LSST_GAMMA_DEFAULT = 0.039
LSST_GAMMA = {"u": 0.038}


def lsst_magnitude_error(
    mag: np.ndarray,
    depth: np.ndarray,
    band: str | None = None,
    sigma_sys: float = LSST_SIGMA_SYS,
    gamma: float | None = None,
) -> np.ndarray:
    """LSST single-visit photometric error in MAGNITUDES (Ivezic et al. 2019).

    ``depth`` is the visit's ``fiveSigmaDepth`` (m5). ``band`` selects gamma
    (0.038 for u, 0.039 otherwise); pass ``gamma`` to override directly.

    Replaces an earlier ``0.2 * 10**(0.4*(mag - depth))`` fractional-flux
    approximation, which used the 5-sigma definition alone: linear in x, with
    no gamma x^2 term and no systematic floor. That was accurate near the
    limiting magnitude but badly wrong for bright sources -- 1.6x low at 4 mag
    above the depth, 2.9x at 5, 6.3x at 6 -- and it let 25.6% of real campaign
    visits be assigned an uncertainty below LSST's own 5 mmag systematic
    floor, a precision Rubin will never deliver.

    Validated against 78,714 real LSSTCam alert-stream detections (60 AGN,
    2025-12-15 to 2026-06-14, i.e. LSSTCam not ComCam): with m5 fitted per
    band this form reproduces the real magnitude errors to 0.047-0.074 dex
    (12-18%) in all six bands, against 0.052-0.163 dex for the old model.
    Fitted effective depths ran 0.0-0.46 mag shallower than nominal, worst in
    u and g -- consistent with DP1's documented finding that u and y errors
    are underestimated and their depths overestimated. Treat u and y results
    as the least trustworthy.

    ``photerr`` (github.com/jfcrenshaw/photerr) generalises this to the
    low-SNR regime and to extended sources. Not adopted here: it is an extra
    runtime dependency for long campaigns, and the plain high-SNR form already
    matches real LSSTCam point-source photometry to 12-18%.
    """
    mag = np.asarray(mag, dtype=float)
    depth = np.asarray(depth, dtype=float)
    if gamma is None:
        gamma = LSST_GAMMA.get(band, LSST_GAMMA_DEFAULT)
    x = 10.0 ** (0.4 * (mag - depth))
    return np.sqrt(sigma_sys**2 + (0.04 - gamma) * x + gamma * x * x)


@dataclass
class NoiseModel:
    """Per-band magerr(mag) fit, one per survey.

    ``kind`` selects how ``coeffs`` are interpreted:

    * ``"log10_linear"`` (default, current): ``magerr = 10**poly(mag)`` --
      positive and monotonic by construction. See ``fit_magerr_relation``.
    * ``"poly_magerr"`` (legacy): ``magerr = poly(mag)`` directly, the old
      degree-2 fit that turned over at ~16.6 mag. Retained ONLY so previously
      cached noise models still evaluate as they did when they were written;
      caches without a ``kind`` field are assumed to be this.
    """

    coeffs: dict[str, np.ndarray]
    kind: str = "log10_linear"

    def __post_init__(self):
        if self.kind not in ("log10_linear", "poly_magerr"):
            raise ValueError(
                f"unknown NoiseModel kind {self.kind!r}; expected "
                "'log10_linear' or 'poly_magerr'"
            )

    def __call__(self, band: str, mag: np.ndarray) -> np.ndarray:
        if band not in self.coeffs:
            raise KeyError(f"no noise-model fit for band {band!r}; have {sorted(self.coeffs)}")
        pred = np.polyval(self.coeffs[band], mag)
        if self.kind == "log10_linear":
            pred = 10.0**pred
        return np.clip(pred, _MAGERR_FLOOR, None)

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "coeffs": {band: c.tolist() for band, c in self.coeffs.items()},
        }

    @classmethod
    def from_dict(cls, d: dict) -> "NoiseModel":
        if "coeffs" in d and "kind" in d:
            return cls(
                {band: np.asarray(c) for band, c in d["coeffs"].items()},
                kind=str(d["kind"]),
            )
        # legacy cache: a bare {band: coeffs} mapping, written before the
        # log-space fit existed. Interpret it the way it was written.
        return cls({band: np.asarray(c) for band, c in d.items()}, kind="poly_magerr")


@dataclass
class CadenceLibrary:
    """Real per-object survey cadences plus each survey's noise model.

    ``cadences[survey][object_id]`` is a DataFrame with columns
    ``mjd, band, mag, magerr, depth, seeing``, NaN wherever a column doesn't
    apply to that survey -- ZTF has real photometry (mag/magerr) but no
    depth/seeing; LSST/OpSim gives cadence and depth, not real photometry.
    """

    cadences: dict[str, dict[str, pd.DataFrame]]
    noise_models: dict[str, NoiseModel]
    metadata: dict

    def surveys(self) -> list[str]:
        return sorted(self.cadences)

    def object_ids(self, survey: str) -> list[str]:
        return sorted(self.cadences[survey])

    def get(self, survey: str, object_id: str) -> pd.DataFrame:
        try:
            return self.cadences[survey][object_id]
        except KeyError:
            raise KeyError(f"{object_id!r} not found for survey {survey!r}") from None

    def random(self, survey: str, rng: np.random.Generator) -> tuple[str, pd.DataFrame]:
        """Draw one object's cadence with replacement.

        Absolute MJDs are returned unmodified, so seasonal gaps and survey
        epochs stay aligned across simulated objects -- unlike
        ``sample_seasonal_pattern``, which regenerates synthetic gap timing
        fresh on every draw, a real cadence's gaps are fixed by the survey's
        actual history.
        """
        ids = self.object_ids(survey)
        object_id = ids[rng.integers(len(ids))]
        return object_id, self.cadences[survey][object_id]

    # -- construction --------------------------------------------------

    @classmethod
    def from_survey_dirs(
        cls, survey_dirs: dict[str, str | Path], magerr_degree: int = 2
    ) -> "CadenceLibrary":
        """Build a library directly from crossmatch-script output directories,
        e.g. ``{"ztf": ztf_data_dir, "lsst": lsst_data_dir}``.

        Each directory must contain ``master.csv`` (with ``matched`` and
        ``file`` columns) and the per-object CSVs ``file`` points to.
        Unmatched targets are skipped. Rows with a NaN timestamp or a
        non-positive ``magerr`` are dropped -- not marginally-faint mag
        outliers near a survey's depth limit, which are real photometry
        (see the ZTF validation report, 2026-07-29: ~0.007% of ZTF epochs
        flagged this way, confirmed benign).
        """
        cadences: dict[str, dict[str, pd.DataFrame]] = {}
        noise_models: dict[str, NoiseModel] = {}
        counts: dict[str, dict[str, int]] = {}

        for survey, data_dir in survey_dirs.items():
            objects, mag_pool, err_pool, n_dropped_rows = cls._load_survey_dir(Path(data_dir))
            cadences[survey] = objects
            counts[survey] = {
                "n_objects_loaded": len(objects),
                "n_rows_dropped": n_dropped_rows,
            }
            if mag_pool:
                coeffs = {
                    band: fit_magerr_relation(
                        np.concatenate(mag_pool[band]), np.concatenate(err_pool[band]),
                        degree=magerr_degree,
                    )
                    for band in mag_pool
                }
                noise_models[survey] = NoiseModel(coeffs)

        metadata = {
            "surveys": {s: str(d) for s, d in survey_dirs.items()},
            "counts": counts,
            "time_convention": {"ztf": ZTF_TIME_CONVENTION},
        }
        return cls(cadences=cadences, noise_models=noise_models, metadata=metadata)

    @staticmethod
    def _load_survey_dir(data_dir: Path):
        master = pd.read_csv(data_dir / "master.csv")
        matched = master[master["matched"]]

        objects: dict[str, pd.DataFrame] = {}
        mag_pool: dict[str, list[np.ndarray]] = {}
        err_pool: dict[str, list[np.ndarray]] = {}
        n_dropped_rows = 0

        for row in matched.itertuples():
            lc = pd.read_csv(data_dir / row.file)
            df = pd.DataFrame(
                {c: np.nan for c in CADENCE_COLUMNS}, index=lc.index
            )
            df["band"] = lc["band"]
            if "hmjd" in lc.columns:
                df["mjd"] = lc["hmjd"]
                df["mag"] = lc["mag"]
                df["magerr"] = lc["magerr"]
            elif "mjd" in lc.columns:
                df["mjd"] = lc["mjd"]
                if "depth" in lc.columns:
                    df["depth"] = lc["depth"]
                if "seeing" in lc.columns:
                    df["seeing"] = lc["seeing"]
            else:
                raise KeyError(f"{data_dir / row.file}: no mjd/hmjd time column")

            bad = df["mjd"].isna()
            if "magerr" in df.columns:
                bad = bad | (df["magerr"].notna() & (df["magerr"] <= 0))
            n_dropped_rows += int(bad.sum())
            df = df.loc[~bad].sort_values("mjd").reset_index(drop=True)
            if len(df) == 0:
                continue

            objects[row.object_id] = df

            if df["mag"].notna().any():
                for band, g in df.groupby("band"):
                    good = g["mag"].notna() & g["magerr"].notna()
                    if not good.any():
                        continue
                    mag_pool.setdefault(band, []).append(g.loc[good, "mag"].to_numpy())
                    err_pool.setdefault(band, []).append(g.loc[good, "magerr"].to_numpy())

        return objects, mag_pool, err_pool, n_dropped_rows

    # -- caching ----------------------------------------------------------

    def to_cache(self, path: str | Path) -> None:
        """Serialize to a directory: one gzip CSV per survey (long format,
        one ``object_id`` column plus the cadence columns) plus
        ``noise_models.json`` and ``metadata.json``."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        for survey, objects in self.cadences.items():
            frames = []
            for object_id, df in objects.items():
                d = df.copy()
                d.insert(0, "object_id", object_id)
                frames.append(d)
            long = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
                columns=["object_id", *CADENCE_COLUMNS]
            )
            long.to_csv(path / f"{survey}_cadences.csv.gz", index=False)

        (path / "noise_models.json").write_text(
            json.dumps({s: nm.to_dict() for s, nm in self.noise_models.items()}, indent=2)
        )
        (path / "metadata.json").write_text(json.dumps(self.metadata, indent=2))

    @classmethod
    def from_cache(cls, path: str | Path) -> "CadenceLibrary":
        path = Path(path)
        metadata = json.loads((path / "metadata.json").read_text())
        noise_json = json.loads((path / "noise_models.json").read_text())
        noise_models = {s: NoiseModel.from_dict(d) for s, d in noise_json.items()}

        cadences: dict[str, dict[str, pd.DataFrame]] = {}
        for f in sorted(path.glob("*_cadences.csv.gz")):
            survey = f.name[: -len("_cadences.csv.gz")]
            long = pd.read_csv(f)
            cadences[survey] = {
                object_id: g.drop(columns="object_id").reset_index(drop=True)
                for object_id, g in long.groupby("object_id")
            }
        return cls(cadences=cadences, noise_models=noise_models, metadata=metadata)
