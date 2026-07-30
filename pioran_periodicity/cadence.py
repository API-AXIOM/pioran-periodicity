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


def fit_magerr_relation(mag: np.ndarray, magerr: np.ndarray, degree: int = 2) -> np.ndarray:
    """Fit an empirical magerr(mag) polynomial from real photometry.

    ``mag``, ``magerr`` must already be cleaned by the caller (finite,
    magerr > 0) -- this function does not filter. Returns ``np.polyfit``
    coefficients, highest power first.
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
    return np.polyfit(mag, magerr, degree)


def depth_to_fractional_error(mag: np.ndarray, depth: np.ndarray) -> np.ndarray:
    """Fractional flux error from an LSST/OpSim ``fiveSigmaDepth`` visit depth.

    Uses the definition of ``fiveSigmaDepth`` directly: SNR = 5 at
    ``mag == depth``, and SNR scales with flux (``propto 10**(-0.4*mag)``)
    in the background-noise-dominated regime, i.e.
    ``sigma_flux / flux = 0.2 * 10**(0.4*(mag - depth))``. A simplified
    version of the full LSST photometric error model (which also splits out
    source vs. sky Poisson noise at bright magnitudes) -- accurate enough
    for the cadence-realism check this pipeline exists to do.
    """
    mag = np.asarray(mag, dtype=float)
    depth = np.asarray(depth, dtype=float)
    return 0.2 * 10.0 ** (0.4 * (mag - depth))


@dataclass
class NoiseModel:
    """Per-band magerr(mag) polynomial fit, one per survey."""

    coeffs: dict[str, np.ndarray]

    def __call__(self, band: str, mag: np.ndarray) -> np.ndarray:
        if band not in self.coeffs:
            raise KeyError(f"no noise-model fit for band {band!r}; have {sorted(self.coeffs)}")
        pred = np.polyval(self.coeffs[band], mag)
        return np.clip(pred, _MAGERR_FLOOR, None)

    def to_dict(self) -> dict:
        return {band: c.tolist() for band, c in self.coeffs.items()}

    @classmethod
    def from_dict(cls, d: dict) -> "NoiseModel":
        return cls({band: np.asarray(c) for band, c in d.items()})


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
