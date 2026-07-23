"""Loaders for real light curves used in the thesis Section 3.7 analysis.

Generic loaders plus convenience wrappers for the two thesis sources.
Preprocessing follows the original notebooks exactly (verified against
original/real_data_test_analysis.ipynb, 2026-07-17).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["load_photometry_csv", "load_three_column", "load_pg1302", "load_pg1553"]

DAYS_PER_YEAR = 365.0


def load_photometry_csv(
    path,
    time_col="MJD",
    value_col="Mag",
    err_col="Magerr",
    time_to_years=True,
    subtract_median=True,
    zero_time=True,
):
    """Load survey photometry (CRTS-style CSV) as (t, y, yerr).

    Time converted MJD -> years (divide by 365, matching the original
    analysis); value median-subtracted; sorted by time; optionally
    re-zeroed to the first observation.
    """
    df = pd.read_csv(path)
    t = df[time_col].to_numpy(dtype=float)
    y = df[value_col].to_numpy(dtype=float)
    yerr = df[err_col].to_numpy(dtype=float)

    if time_to_years:
        t = t / DAYS_PER_YEAR
    order = np.argsort(t)
    t, y, yerr = t[order], y[order], yerr[order]
    if subtract_median:
        y = y - np.median(y)
    if zero_time:
        t = t - t[0]
    return t, y, yerr


def load_three_column(path, subtract_median=True):
    """Load a whitespace-separated (time, value, error) text file."""
    t, y, yerr = np.loadtxt(path).T
    order = np.argsort(t)
    t, y, yerr = t[order], y[order], yerr[order]
    if subtract_median:
        y = y - np.median(y)
    return t, y, yerr


def load_pg1302(path="AGNobsdata/graham2015data.csv"):
    """PG 1302-102 CRTS V-band photometry (Graham et al. 2015).

    Returns (t_years, mag - median, magerr), t zeroed at first epoch.
    """
    return load_photometry_csv(path)


def load_pg1553(path="AGNobsdata/PG1553_113_logbase.txt"):
    """PG 1553+113 Fermi-LAT monthly ln-flux light curve.

    Returns (t_years, lnflux - median, err). The file is the original
    analysis's preprocessed product (3-day LCR flux, monthly rebin,
    natural log).
    """
    return load_three_column(path)
