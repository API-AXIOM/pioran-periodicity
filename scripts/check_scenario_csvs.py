"""Pre-launch consistency check for scenario CSVs that must be comparable.

Nulls and their signal/multiband counterparts must share every simulation
constant except the axes that are swept on purpose. A stale default once put
the multiband nulls at bendfreq 2/yr while every single-band CSV was at
0.35/yr and nothing noticed (2026-09-25). Run this on every CSV of a
campaign family BEFORE launching anything:

    conda run -n <env> python scripts/check_scenario_csvs.py \
        v2_ztf_n400_final.csv v2_ztf_signal_n50.csv v2_ztf_multiband_null_fb035.csv

Exit status 1 if any checked column is not single-valued within a file, or
differs between files. ``--expect-bendfreq-per-yr`` (default 0.35) also pins
the value itself, so a family that is consistently WRONG is caught too.
"""
from __future__ import annotations

import argparse
import sys
from typing import List

import numpy as np
import pandas as pd

# Constants that must agree across a comparable family. Cadence-dependent
# columns (period_max differs per survey) are only compared within a family
# the caller passes together, which is the intended use: one survey per call.
CHECKED = ["units", "bendfreq", "lowalpha", "sharpness", "period_max"]


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("csvs", nargs="+")
    ap.add_argument("--expect-bendfreq-per-yr", type=float, default=0.35)
    args = ap.parse_args(argv)

    seen = {}  # column -> (first file, value)
    bad = 0
    for path in args.csvs:
        df = pd.read_csv(path)
        for col in CHECKED:
            if col not in df.columns:
                continue
            vals = df[col].dropna().unique()
            if len(vals) != 1:
                print(f"FAIL {path}: {col} not single-valued: {vals[:5]}")
                bad += 1
                continue
            v = vals[0]
            if col in seen:
                ref_path, ref = seen[col]
                same = (
                    v == ref if isinstance(v, str) else np.isclose(v, ref, rtol=1e-6)
                )
                if not same:
                    print(f"FAIL {col}: {path}={v!r} but {ref_path}={ref!r}")
                    bad += 1
            else:
                seen[col] = (path, v)
        if "bendfreq" in df.columns:
            per_yr = float(df["bendfreq"].iloc[0]) * 365.25
            if not np.isclose(per_yr, args.expect_bendfreq_per_yr, rtol=1e-3):
                print(
                    f"FAIL {path}: bendfreq {per_yr:.3f}/yr, expected "
                    f"{args.expect_bendfreq_per_yr}/yr"
                )
                bad += 1
    print("OK" if not bad else f"{bad} problem(s)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
