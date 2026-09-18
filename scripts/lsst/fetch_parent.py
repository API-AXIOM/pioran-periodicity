"""Rebuild the UNSTRATIFIED parent population the LSST targets were drawn from.

The target lists were built by drawing from Milliquas with a declination-
stratified sample, and the catalogue itself was never cached. Any attempt to
express a campaign result as a population rate therefore needs the parent
reconstructed: same catalogue, same cuts, no stratification.

Cuts, in the order the original target builder applied them: type-Q only,
``|b| > --gal-lat-min``, ``--mag-bright <= Rmag <= --mag-faint``, then inside
the OpSim footprint (``>= --min-visits`` in the map from
``build_visit_map.py``).

    conda run -n <env> python scripts/lsst/fetch_parent.py \
        --visit-map visit_counts_nside64.npy --out parent_lsst.csv

Requires ``healpy``, ``astroquery`` and ``astropy``. The VizieR query pulls
~1e6 rows and takes a few minutes.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def fetch_milliquas(catalog: str, gal_lat_min: float) -> pd.DataFrame:
    """-> type-Q quasars with valid coordinates and Rmag, outside the plane."""
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    from astroquery.vizier import Vizier

    v = Vizier(columns=["RAJ2000", "DEJ2000", "z", "Rmag", "Type"], row_limit=-1)
    print(f"querying VizieR {catalog} ...", flush=True)
    cats = v.get_catalogs(catalog)
    if len(cats) == 0:
        raise RuntimeError(f"VizieR returned no catalogs for {catalog}")
    df = cats[0].to_pandas()
    print(f"  {len(df)} rows returned")

    missing = {"RAJ2000", "DEJ2000", "Rmag", "Type"} - set(df.columns)
    if missing:
        raise KeyError(f"VizieR columns changed. Missing {missing}")

    df = df[df["Type"].astype(str).str.startswith("Q")]
    df = df.dropna(subset=["RAJ2000", "DEJ2000", "Rmag"])
    c = SkyCoord(ra=df["RAJ2000"].values * u.deg, dec=df["DEJ2000"].values * u.deg)
    df = df.assign(glat=c.galactic.b.deg)
    df = df[df["glat"].abs() > gal_lat_min]
    print(f"  {len(df)} type-Q with |b| > {gal_lat_min}")
    return df


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--visit-map", type=Path, required=True,
                    help=".npy visit map from build_visit_map.py")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--catalog", default="VII/294", help="VizieR catalogue id")
    ap.add_argument("--nside", type=int, default=64)
    ap.add_argument("--gal-lat-min", type=float, default=15.0)
    ap.add_argument("--mag-bright", type=float, default=16.0)
    ap.add_argument("--mag-faint", type=float, default=23.0)
    ap.add_argument("--min-visits", type=int, default=100)
    args = ap.parse_args()

    import healpy as hp

    df = fetch_milliquas(args.catalog, args.gal_lat_min)

    df = df[(df["Rmag"] >= args.mag_bright) & (df["Rmag"] <= args.mag_faint)]
    print(f"  {len(df)} after {args.mag_bright} <= R <= {args.mag_faint}")

    counts = np.load(args.visit_map)
    pix = hp.ang2pix(args.nside, df["RAJ2000"].to_numpy(),
                     df["DEJ2000"].to_numpy(), lonlat=True)
    df = df.assign(n_visits=counts[pix])
    df = df[df["n_visits"] >= args.min_visits]
    print(f"  {len(df)} inside the footprint  <-- PARENT")

    (df.rename(columns={"RAJ2000": "ra", "DEJ2000": "dec"})
       [["ra", "dec", "z", "Rmag", "glat", "n_visits"]]
       .to_csv(args.out, index=False))
    print("written", args.out)


if __name__ == "__main__":
    main()
