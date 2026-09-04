"""Band-identity bookkeeping for multi-band (ZTF/LSST) light curves.

Shared-latent-process model: each photometric band ``b`` observes an affine
readout of one shared latent process, ``y_b(t) = mu_b + a_b * x(t) + noise``.
Since ZTF/LSST observe a single band per epoch (never simultaneously), the
covariance of the observed points is a diagonal rescale of the shared-latent
covariance, so the existing single-band GP machinery can be reused via
:func:`pioran_periodicity.kernels.gp_log_likelihood_multiband` -- this
module only carries the (reference band, other bands) bookkeeping needed to
build the per-band parameter arrays that function expects.

One band is designated the *reference*: its amplitude and mean offset are
pinned (``a_ref=1``, ``mu_ref=0``) rather than fit, which is what makes the
per-band amplitudes identifiable against the noise model's own
``log10_variance`` (an unpinned amplitude would be perfectly degenerate with
an inverse rescale of the whole shared process). See
``pioran_periodicity/models.py`` for how the pinned/free split is applied.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = [
    "BandEncoding",
    "cadence_to_multiband_series",
    "EFFECTIVE_WAVELENGTHS",
    "power_law_band_amplitudes",
]

DAYS_PER_YEAR = 365.0  # matches data.DAYS_PER_YEAR / simulate's t_years contract

# Filter effective wavelengths in Angstrom, used only to turn a single
# colour-dependence index into per-band amplitudes (see
# power_law_band_amplitudes). Approximate central/effective values -- the
# amplitude ratios depend on wavelength RATIOS, so few-percent differences
# between definitions (effective vs pivot wavelength) are immaterial here.
EFFECTIVE_WAVELENGTHS = {
    "lsst": {"u": 3671.0, "g": 4827.0, "r": 6223.0, "i": 7546.0,
             "z": 8691.0, "y": 9712.0},
    "ztf": {"g": 4746.0, "r": 6366.0, "i": 7829.0},
}


def power_law_band_amplitudes(
    bands, beta: float, reference: str, survey: str = "lsst"
) -> dict[str, float]:
    """Per-band variability amplitudes from a single colour-dependence index.

    ``a_b = (lambda_b / lambda_ref) ** (-beta)``, normalised so the reference
    band has ``a_ref = 1`` exactly (the convention pinned by
    :class:`BandEncoding` and ``models.build_family``). ``beta = 0`` gives
    ``a_b = 1`` for every band, i.e. the identical-variability-in-every-band
    behaviour the simulator had before colour dependence existed; ``beta > 0``
    makes bluer (shorter-wavelength) bands more variable, the direction real
    quasar structure functions show.

    ``bands``: iterable of band names. Returns {band: amplitude}.
    """
    table = EFFECTIVE_WAVELENGTHS.get(survey)
    if table is None:
        raise ValueError(
            f"no filter wavelengths for survey {survey!r}; "
            f"have {sorted(EFFECTIVE_WAVELENGTHS)}"
        )
    bands = list(bands)
    unknown = sorted(set(bands + [reference]) - set(table))
    if unknown:
        raise ValueError(f"no wavelength for {survey} band(s) {unknown}")
    lam_ref = table[reference]
    return {b: float((table[b] / lam_ref) ** (-beta)) for b in bands}


@dataclass(frozen=True)
class BandEncoding:
    """Fixes a band ordering: index 0 is always the pinned reference band."""

    reference: str
    others: tuple[str, ...]

    @property
    def names(self) -> tuple[str, ...]:
        """Band names in code order: (reference, *others)."""
        return (self.reference, *self.others)

    def encode(self, band_labels) -> np.ndarray:
        """Map per-epoch band strings to integer codes indexing ``names``.

        ``band_labels``: 1-D array-like of str, length n_points (one label
        per data point, aligned with t/y/yerr). Returns an int64 array of
        the same length; code 0 always means the reference band.
        """
        labels = np.asarray(band_labels, dtype=object)
        # index lookup rather than pd.Categorical(categories=...): the latter
        # is deprecated for values outside the categories, which is exactly
        # the unknown-band case we want to detect and report.
        codes = pd.Index(self.names).get_indexer(labels)  # (n_points,), -1 = unknown
        if (codes < 0).any():
            unknown = sorted(set(labels[codes < 0]))
            raise ValueError(
                f"band labels {unknown} not in encoding {self.names}"
            )
        return codes.astype(np.int64)

    @classmethod
    def from_counts(cls, band_labels, reference: str | None = None) -> "BandEncoding":
        """Build an encoding from observed band labels.

        Default reference band is the most-observed band (ties broken
        alphabetically) -- always exists for the object being fit, and
        minimizes how many free per-band parameters have to be estimated
        from the sparsest bands. Pass ``reference`` explicitly to override,
        e.g. to pin a fixed canonical band across a population study.
        """
        labels = np.asarray(band_labels, dtype=object)
        counts = pd.Series(labels).value_counts()
        if len(counts) == 0:
            raise ValueError("no band labels given")
        if reference is None:
            max_count = counts.max()
            reference = sorted(counts.index[counts == max_count])[0]
        elif reference not in counts.index:
            raise ValueError(f"reference band {reference!r} not observed in data")
        others = tuple(sorted(b for b in counts.index if b != reference))
        return cls(reference=reference, others=others)


def cadence_to_multiband_series(
    cadence: pd.DataFrame,
    reference: str | None = None,
    zero_time: bool = True,
):
    """Convert a real per-object cadence DataFrame to fittable multi-band arrays.

    ``cadence`` is one object's DataFrame as returned by
    ``cadence.CadenceLibrary.get`` -- columns ``mjd, band, mag, magerr,
    depth, seeing``, NaN wherever a column doesn't apply to that survey.
    Only rows carrying real photometry (finite ``mag`` and ``magerr > 0``)
    are usable here; depth-only rows (an LSST visit schedule with no
    measured magnitude) are out of scope and dropped.

    Conventions, all inherited from the existing single-band real-data path
    (``data.load_photometry_csv``):

    * ``y = mag``, ``yerr = magerr`` -- magnitude is fit directly as the GP
      variable, with no mag-to-flux conversion anywhere in this codebase.
    * time in **years** (``mjd / 365``), sorted, re-zeroed to the first
      epoch when ``zero_time``.
    * median-subtracted -- but by the **reference band's own median**,
      applied to every band, not each band's own median and not a global
      median. That is what makes the pinned ``mu_ref = 0`` of
      :func:`pioran_periodicity.kernels.gp_log_likelihood_multiband` valid
      by construction, while leaving each non-reference ``mu_b`` free to
      absorb that band's genuine colour offset relative to the reference
      rather than a centring artefact.

    Returns ``(t_years, y, yerr, band_code, encoding)``: four 1-D arrays of
    equal length n_points plus the :class:`BandEncoding` whose
    ``.others`` should be passed to ``models.build_family(
    photometric_bands=...)`` and whose ``band_code`` goes to
    ``inference.run_nested(..., band=...)``.
    """
    missing = {"mjd", "band", "mag", "magerr"} - set(cadence.columns)
    if missing:
        raise ValueError(f"cadence is missing required columns {sorted(missing)}")

    mjd = cadence["mjd"].to_numpy(dtype=float)
    mag = cadence["mag"].to_numpy(dtype=float)
    magerr = cadence["magerr"].to_numpy(dtype=float)
    band_labels = cadence["band"].to_numpy(dtype=object)

    # (n_rows,) boolean: keep only rows with genuine photometry
    keep = np.isfinite(mjd) & np.isfinite(mag) & np.isfinite(magerr) & (magerr > 0)
    if not keep.any():
        raise ValueError(
            "cadence has no rows with finite mjd/mag/magerr > 0 "
            "(depth-only visit schedules are out of scope for this adapter)"
        )
    mjd, mag = mjd[keep], mag[keep]
    magerr, band_labels = magerr[keep], band_labels[keep]

    order = np.argsort(mjd)
    mjd, mag, magerr, band_labels = (
        mjd[order],
        mag[order],
        magerr[order],
        band_labels[order],
    )

    encoding = BandEncoding.from_counts(band_labels, reference=reference)
    band_code = encoding.encode(band_labels)  # (n_points,), 0 = reference band

    y = mag - np.median(mag[band_code == 0])  # reference-band centring
    t_years = mjd / DAYS_PER_YEAR
    if zero_time:
        t_years = t_years - t_years[0]
    return t_years, y, magerr, band_code, encoding
