#!/usr/bin/env python
"""
Validate ZTF light curve output from ztf_crossmatch.py.

Runs three tiers of checks:
  1. Structural  -- files well-formed, columns present, master consistent.
  2. Photometric -- values physically plausible for ZTF quasars.
  3. Cadence     -- sampling usable as a cadence (the actual downstream purpose).

Prints a report and flags anything suspect. Optionally writes a per-object
diagnostics table and a few sanity plots.

Usage:
    python validate_ztf.py ztf_out/
    python validate_ztf.py ztf_out/ --plots
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ZTF plausibility bounds (single-exposure PSF photometry)
MAG_MIN, MAG_MAX = 12.0, 22.0        # saturation to faint limit, generous
MAGERR_MAX = 1.0                     # anything above is junk
BANDS = {"g", "r", "i"}
# ZTF public survey started 2018-03; MJD ~58200. Nothing should predate it.
MJD_FLOOR = 58100.0
MJD_CEIL = 61000.0                   # ~2025; adjust if using a later release


def load(outdir):
    master_path = outdir / "master.csv"
    if not master_path.exists():
        sys.exit(f"No master.csv in {outdir}")
    master = pd.read_csv(master_path)
    return master


def check_structure(master, outdir):
    print("=" * 60)
    print("1. STRUCTURAL")
    print("=" * 60)

    issues = []
    n = len(master)
    matched = master[master["matched"]] if "matched" in master else master[master["n_epochs"] > 0]
    print(f"targets in master:     {n}")
    print(f"matched:               {len(matched)}  ({len(matched)/n*100:.0f}%)")
    print(f"unmatched:             {n - len(matched)}")

    # Every matched target should have a file that exists and reads.
    missing_files, bad_reads, count_mismatch = [], [], []
    expected_cols = {"ztf_objectid", "band", "hmjd", "mag", "magerr", "catflags"}

    for r in matched.itertuples():
        fp = outdir / r.file
        if not fp.exists():
            missing_files.append(r.object_id)
            continue
        try:
            lc = pd.read_csv(fp)
        except Exception as e:
            bad_reads.append((r.object_id, str(e)))
            continue
        miss = expected_cols - set(lc.columns)
        if miss:
            bad_reads.append((r.object_id, f"missing cols {miss}"))
            continue
        # master's n_epochs should match the file's row count.
        if len(lc) != r.n_epochs:
            count_mismatch.append((r.object_id, r.n_epochs, len(lc)))

    if missing_files:
        issues.append(f"{len(missing_files)} matched targets have no file on disk")
        print(f"  MISSING FILES: {missing_files[:5]}")
    if bad_reads:
        issues.append(f"{len(bad_reads)} files unreadable or malformed")
        print(f"  BAD READS: {bad_reads[:5]}")
    if count_mismatch:
        issues.append(f"{len(count_mismatch)} files disagree with master n_epochs")
        print(f"  COUNT MISMATCH (id, master, file): {count_mismatch[:5]}")

    # Unmatched targets should NOT have files.
    stray = []
    for r in master[~master.index.isin(matched.index)].itertuples():
        f = getattr(r, "file", "")
        if isinstance(f, str) and f and (outdir / f).exists():
            stray.append(r.object_id)
    if stray:
        issues.append(f"{len(stray)} unmatched targets unexpectedly have files")

    if not issues:
        print("  OK: all matched targets have well-formed files.")
    return issues


def check_photometry(master, outdir):
    print("\n" + "=" * 60)
    print("2. PHOTOMETRIC")
    print("=" * 60)

    issues = []
    all_mags, all_errs, all_bands = [], [], []
    bad_band, bad_mag, bad_err, has_nan = [], [], [], []

    matched = master[master["matched"]] if "matched" in master else master[master["n_epochs"] > 0]
    for r in matched.itertuples():
        fp = outdir / r.file
        if not fp.exists():
            continue
        lc = pd.read_csv(fp)

        if not set(lc["band"].unique()).issubset(BANDS):
            bad_band.append(r.object_id)
        if lc[["mag", "magerr", "hmjd"]].isna().any().any():
            has_nan.append(r.object_id)

        m = lc["mag"].to_numpy()
        e = lc["magerr"].to_numpy()
        if np.any(m < MAG_MIN) or np.any(m > MAG_MAX):
            bad_mag.append(r.object_id)
        if np.any(e > MAGERR_MAX) or np.any(e <= 0):
            bad_err.append(r.object_id)

        all_mags.append(m)
        all_errs.append(e)
        all_bands.append(lc["band"].to_numpy())

    if not all_mags:
        return ["no photometry to check"]

    mags = np.concatenate(all_mags)
    errs = np.concatenate(all_errs)
    bands = np.concatenate(all_bands)

    print(f"total epochs:          {len(mags)}")
    print(f"mag    range:          {mags.min():.2f} - {mags.max():.2f}")
    print(f"magerr range:          {errs.min():.3f} - {errs.max():.3f}")
    print(f"median magerr:         {np.median(errs):.3f}")
    print("\nepochs per band:")
    for b in sorted(set(bands)):
        sel = bands == b
        print(f"  {b}: {sel.sum():>7d}   median mag {np.median(mags[sel]):.2f}")

    if bad_band:
        issues.append(f"{len(bad_band)} targets have unexpected band codes")
    if bad_mag:
        issues.append(f"{len(bad_mag)} targets have mags outside [{MAG_MIN},{MAG_MAX}]")
        print(f"  SUSPECT MAGS: {bad_mag[:5]}")
    if bad_err:
        issues.append(f"{len(bad_err)} targets have magerr <=0 or >{MAGERR_MAX}")
        print(f"  SUSPECT ERRORS: {bad_err[:5]}")
    if has_nan:
        issues.append(f"{len(has_nan)} targets contain NaN in mag/magerr/hmjd")
        print(f"  NANs: {has_nan[:5]}")

    # A ZTF quasar should vary but not wildly. Flag suspiciously flat or
    # explosive scatter as something to eyeball, not necessarily an error.
    flat, wild = [], []
    for r in matched.itertuples():
        fp = outdir / r.file
        if not fp.exists():
            continue
        lc = pd.read_csv(fp)
        for b, g in lc.groupby("band"):
            if len(g) < 10:
                continue
            s = g["mag"].std()
            if s < 0.01:
                flat.append((r.object_id, b, round(s, 4)))
            elif s > 2.0:
                wild.append((r.object_id, b, round(s, 2)))
    if flat:
        print(f"\n  NOTE: {len(flat)} band-curves nearly constant (std<0.01) "
              f"-- possible non-variable/reference artifact: {flat[:3]}")
    if wild:
        print(f"  NOTE: {len(wild)} band-curves with std>2 mag "
              f"-- possible blends/bad points: {wild[:3]}")

    if not issues:
        print("\n  OK: photometry within physical bounds.")
    return issues


def check_cadence(master, outdir):
    print("\n" + "=" * 60)
    print("3. CADENCE")
    print("=" * 60)

    issues = []
    rows = []
    matched = master[master["matched"]] if "matched" in master else master[master["n_epochs"] > 0]

    for r in matched.itertuples():
        fp = outdir / r.file
        if not fp.exists():
            continue
        lc = pd.read_csv(fp)
        t = np.sort(lc["hmjd"].to_numpy())

        if np.any(t < MJD_FLOOR) or np.any(t > MJD_CEIL):
            issues.append(f"{r.object_id}: hmjd outside [{MJD_FLOOR},{MJD_CEIL}]")

        # Exact-duplicate timestamps across the whole target: should be gone
        # (dedup was on objectid+hmjd, but different objectids can share a time).
        dt = np.diff(t)
        n_zero = int(np.sum(dt == 0))

        rows.append({
            "object_id": r.object_id,
            "n_epochs": len(t),
            "baseline_days": t[-1] - t[0],
            "median_gap": np.median(dt) if len(dt) else np.nan,
            "max_gap": dt.max() if len(dt) else np.nan,
            "n_simultaneous": n_zero,
        })

    diag = pd.DataFrame(rows)
    if len(diag) == 0:
        return ["no cadence to check"]

    print(f"baseline (days):   median {diag['baseline_days'].median():.0f}, "
          f"range {diag['baseline_days'].min():.0f}-{diag['baseline_days'].max():.0f}")
    print(f"n_epochs:          median {diag['n_epochs'].median():.0f}, "
          f"range {diag['n_epochs'].min():.0f}-{diag['n_epochs'].max():.0f}")
    print(f"median gap (days): median {diag['median_gap'].median():.1f}")
    print(f"max gap (days):    median {diag['max_gap'].median():.0f}, "
          f"worst {diag['max_gap'].max():.0f}")

    # ZTF has a ~4 month seasonal gap; a max gap under ~90d for a multi-year
    # baseline would be surprising. A baseline near 0 means a broken curve.
    short = diag[diag["baseline_days"] < 30]
    if len(short):
        issues.append(f"{len(short)} targets span <30 days -- likely broken curves")
        print(f"  SHORT BASELINE: {short['object_id'].tolist()[:5]}")

    simul = diag[diag["n_simultaneous"] > 0]
    if len(simul):
        print(f"\n  NOTE: {len(simul)} targets have same-timestamp epochs across "
              "objectids (expected -- multiple ZTF objects observed in one visit).")

    diag.to_csv(outdir / "cadence_diagnostics.csv", index=False)
    print(f"\n  Wrote {outdir/'cadence_diagnostics.csv'}")

    if not [i for i in issues if "broken" in i]:
        print("  OK: cadences span realistic baselines.")
    return issues


def make_plots(master, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matched = master[master["matched"]] if "matched" in master else master[master["n_epochs"] > 0]
    sample = matched.sample(min(6, len(matched)), random_state=0)

    fig, axes = plt.subplots(2, 3, figsize=(15, 7))
    colors = {"g": "green", "r": "red", "i": "black"}
    for ax, r in zip(axes.flat, sample.itertuples()):
        lc = pd.read_csv(outdir / r.file)
        for b, g in lc.groupby("band"):
            ax.errorbar(g["hmjd"], g["mag"], yerr=g["magerr"], fmt=".",
                        ms=3, alpha=0.6, color=colors.get(b, "gray"), label=b)
        ax.invert_yaxis()
        ax.set_title(f"{r.object_id}  (N={r.n_epochs})", fontsize=8)
        ax.set_xlabel("HMJD"); ax.set_ylabel("mag"); ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(outdir / "sample_lightcurves.png", dpi=110)
    print(f"\nWrote {outdir/'sample_lightcurves.png'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir", type=Path)
    ap.add_argument("--plots", action="store_true")
    args = ap.parse_args()

    master = load(args.outdir)
    all_issues = []
    all_issues += check_structure(master, args.outdir)
    all_issues += check_photometry(master, args.outdir)
    all_issues += check_cadence(master, args.outdir)

    if args.plots:
        make_plots(master, args.outdir)

    print("\n" + "=" * 60)
    if all_issues:
        print(f"FLAGGED {len(all_issues)} ISSUE(S):")
        for i in all_issues:
            print(f"  - {i}")
        sys.exit(1)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
