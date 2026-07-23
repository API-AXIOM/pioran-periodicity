"""Minimal end-to-end demo: simulate -> fit -> Bayes factor.

Simulates a short DRW-like light curve, fits it with a DRW noise model
against a DRW+sine alternative via nested sampling, and prints the log10
Bayes factor between the two.

    conda run -n <env> python tutorials/01_simulate_and_fit.py
"""

from __future__ import annotations

import numpy as np

from pioran_periodicity import PriorConfig, SamplerSettings, build_family
from pioran_periodicity.inference import fit_family, log10_bayes_factors
from pioran_periodicity.simulate import sample_seasonal_pattern, simulate_lightcurve


def bend_pl(f, norm, f_bend, alpha_lo, alpha_hi):
    """A simple one-bend power-law PSD (day^-1 frequency convention)."""
    return (norm * (f / f_bend) ** alpha_lo) / (
        1.0 + (f / f_bend) ** (alpha_lo - alpha_hi)
    )


def main():
    # Simulate a ~4 year DRW-like light curve (bend at 1 yr^-1) sampled in
    # 4 observing seasons of 3 nights each -- short enough to fit quickly,
    # long enough relative to the simulated baseline to avoid low-frequency
    # leakage (the package enforces a >=10x margin by default).
    lc = simulate_lightcurve(
        bend_pl,
        psd_params=[20.0, 0.00274, 0.0, -2.0],
        n_samples=2**21,
        dt_minutes=10.0,
        mean=1.0,
        rms=0.15,
        seed=42,
    )
    t, y, yerr = sample_seasonal_pattern(
        lc,
        n_windows=4,
        nights_per_window=3,
        window_period_months=12.0,
        window_width_days=10.0,
        data_loss_frac=0.1,
        noise_sigma=0.05,
        seed=43,
    )
    t = t - t[0]
    y = y - np.median(y)
    print(f"simulated {len(t)} points spanning {t[-1]:.2f} years")

    # A DRW noise model plus a DRW+sine periodic alternative, sharing every
    # noise-model parameter and prior.
    cfg = PriorConfig(
        log10_variance=(-4.0, 1.0),
        log10_fbend=(-3.0, 2.0),
        sine_amplitude_scale=0.15,
        period=(0.2, 4.0),
        err_scale=None,
    )
    family = build_family("drw", cfg, variants=("plain", "sine"))

    results = fit_family(
        family,
        t,
        y,
        yerr,
        settings=SamplerSettings(seed=1, min_num_live_points=200, frac_remain=0.05),
        show_status=False,
    )
    for name, r in results.items():
        print(f"{name:10s} logZ = {r.logz:8.2f} +/- {r.logzerr:.2f}")

    bf = log10_bayes_factors(results)
    print("\nlog10 Bayes factors:")
    for pair, val in bf.items():
        print(f"  {pair}: {val:+.2f}")


if __name__ == "__main__":
    main()
