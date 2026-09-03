"""Deterministic mean functions for the GP models.

One sine parametrization for the whole package (fix M4):

    m(t) = A_cos * cos(2*pi*t / period) + A_sin * sin(2*pi*t / period)

which is equivalent to A*sin(2*pi*t/period + phi) with the AMPLITUDE
A = sqrt(A_cos^2 + A_sin^2) and phi = atan2(A_cos, A_sin). The
(A_cos, A_sin) form avoids the phase-wrap boundary at +-pi and, with
independent Normal(0, s) priors on each, gives an isotropic (uniform-phase)
prior with a Rayleigh(s) marginal on the amplitude.

NAMING (fix MB2, 2026-09-03): these coefficients were called ``A1``/``A2``
until 2026-09-03, which collided with the scenario-CSV column ``A1`` -- an
entirely different quantity (the INJECTED sine amplitude, which
``run_sim.true_mean_signal`` passes into the ``A_sin`` slot with
``A_cos = 0``). Comparing a fitted ``A1`` against the injected ``A1`` was
therefore wrong twice over: wrong coefficient, and a coefficient rather
than an amplitude. The injected amplitude must be compared against
``sine_amplitude(A_cos, A_sin)``, never against a single coefficient --
injection uses absolute time while fitting re-zeroes it, so the phase is
deliberately randomised and the signal lands in an arbitrary mix of the two.
The original code mixed two parametrizations and, in one place, gave the
phase a Normal(-pi, pi) prior by mistake (bug M4).

Periods are validated to be strictly positive (fix M5: the singular
period -> 0 limit is not evaluable).
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "sine_mean",
    "linear_mean",
    "sine_amplitude",
    "sine_amplitude_phase",
    "combine_means",
]


def sine_mean(t: np.ndarray, A_cos: float, A_sin: float, period: float) -> np.ndarray:
    """Sinusoid A_cos*cos(2 pi t / period) + A_sin*sin(2 pi t / period).

    ``A_cos``/``A_sin`` are COEFFICIENTS, not the amplitude; the amplitude is
    :func:`sine_amplitude` of the two. See the module docstring on naming.
    """
    if period <= 0:
        raise ValueError(f"period must be > 0, got {period}")
    phase = 2.0 * np.pi * np.asarray(t, dtype=float) / period
    return A_cos * np.cos(phase) + A_sin * np.sin(phase)


def linear_mean(t: np.ndarray, slope: float, intercept: float) -> np.ndarray:
    """Linear trend slope*t + intercept."""
    return slope * np.asarray(t, dtype=float) + intercept


def sine_amplitude_phase(A_cos: float, A_sin: float) -> tuple[float, float]:
    """Convert (A_cos, A_sin) to (amplitude, phase) of A*sin(2 pi t/P + phi)."""
    return float(np.hypot(A_cos, A_sin)), float(np.arctan2(A_cos, A_sin))


def sine_amplitude(A_cos: float, A_sin: float) -> float:
    """Amplitude sqrt(A_cos^2 + A_sin^2) of the fitted sine.

    This -- NOT either coefficient alone -- is the quantity comparable to a
    scenario CSV's injected ``A1`` column. In the multi-band model it is in
    shared-latent (reference-band) units; the observed amplitude in band b
    is ``a_b`` times it.
    """
    return float(np.hypot(A_cos, A_sin))


def combine_means(*terms):
    """Combine mean callables f(t) -> array into one summed callable."""

    def total(t):
        out = np.zeros_like(np.asarray(t, dtype=float))
        for f in terms:
            out = out + f(t)
        return out

    return total
