"""Build a HEALPix visit-count map from an OpSim visit history.

The map is the basis of both the LSST footprint definition and the sampling
density used to stratify objects. It reproduces the footprint used when the
target lists were built (nside=64, 1.75 deg circular FOV, >= 100 visits)
but counts per *pixel* with a KD-tree rather than per *visit* with
``query_disc``, which is orders of magnitude faster and gives the same map.

Validation: every object in an archived target list should land inside the
resulting footprint. ``--validate-targets`` checks exactly that.

    conda run -n <env> python scripts/lsst/build_visit_map.py \
        --opsim baseline_v5.3.0_10yrs.db --out visit_counts_nside64.npy \
        --validate-targets targets_lsst.csv

Requires ``healpy`` and ``scipy`` in addition to the package's runtime deps.
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd


def visit_counts(opsim: Path, nside: int, fov_radius_deg: float) -> np.ndarray:
    """-> (npix,) number of visits whose pointing centre lies within
    ``fov_radius_deg`` of each HEALPix pixel centre."""
    import healpy as hp
    from scipy.spatial import cKDTree

    con = sqlite3.connect(str(opsim))
    try:
        ptg = pd.read_sql_query(
            "SELECT fieldRA AS ra, fieldDec AS dec FROM observations", con
        )
    finally:
        con.close()
    print(f"{len(ptg)} visits")

    # (n_visits, 3) and (npix, 3) unit vectors
    vis = np.asarray(hp.ang2vec(ptg["ra"].to_numpy(), ptg["dec"].to_numpy(), lonlat=True))
    npix = hp.nside2npix(nside)
    pix = np.asarray(hp.pix2vec(nside, np.arange(npix))).T

    chord = 2.0 * np.sin(np.radians(fov_radius_deg) / 2.0)
    counts = cKDTree(vis).query_ball_point(pix, chord, return_length=True)
    return np.asarray(counts, dtype=np.int32)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--opsim", type=Path, required=True, help="OpSim sqlite database")
    ap.add_argument("--out", type=Path, required=True, help="output .npy visit map")
    ap.add_argument("--nside", type=int, default=64)
    ap.add_argument("--fov-radius-deg", type=float, default=1.75)
    ap.add_argument("--min-visits", type=int, default=100,
                    help="footprint threshold, for the summary only")
    ap.add_argument("--validate-targets", type=Path, default=None,
                    help="CSV with ra/dec columns that must fall inside the footprint")
    args = ap.parse_args()

    counts = visit_counts(args.opsim, args.nside, args.fov_radius_deg)
    np.save(args.out, counts)

    good = counts >= args.min_visits
    print(f"footprint: {good.sum()} pixels ({100 * good.mean():.1f}% of sky), "
          f">= {args.min_visits} visits; max count {counts.max()}")
    print("written", args.out)

    if args.validate_targets is not None:
        import healpy as hp

        t = pd.read_csv(args.validate_targets)
        pix = hp.ang2pix(args.nside, t["ra"].to_numpy(), t["dec"].to_numpy(), lonlat=True)
        inside = int(good[pix].sum())
        print(f"validation: {inside}/{len(t)} targets inside the footprint")
        if inside != len(t):
            raise SystemExit("FAILED: the map does not reproduce the archived selection")


if __name__ == "__main__":
    main()
