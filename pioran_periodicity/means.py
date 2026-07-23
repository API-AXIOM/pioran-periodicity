"""Deterministic mean functions for the GP models.

One sine parametrization for the whole package (fix M4):

    m(t) = A1 * cos(2*pi*t / period) + A2 * sin(2*pi*t / period)

which is equivalent to A*sin(2*pi*t/period + phi) with A = sqrt(A1^2 + A2^2)
and phi = atan2(A1, A2). The (A1, A2) form avoids the phase-wrap boundary at
+-pi and, with independent Normal(0, s) priors on A1 and A2, gives an
isotropic (uniform-phase) prior with a Rayleigh(s) marginal on the amplitude.
The original code mixed two parametrizations and, in one place, gave the
phase a Normal(-pi, pi) prior by mistake (bug M4).

Periods are validated to be strictly positive (fix M5: the singular
period -> 0 limit is not evaluable).
"""

from __future__ import annotations

import numpy as np

__all__ = ["sine_mean", "linear_mean", "sine_amplitude_phase", "combine_means"]


def sine_mean(t: np.ndarray, A1: float, A2: float, period: float) -> np.ndarray:
    """Sinusoid A1*cos(2 pi t / period) + A2*sin(2 pi t / period)."""
    if period <= 0:
        raise ValueError(f"period must be > 0, got {period}")
    phase = 2.0 * np.pi * np.asarray(t, dtype=float) / period
    return A1 * np.cos(phase) + A2 * np.sin(phase)


def linear_mean(t: np.ndarray, slope: float, intercept: float) -> np.ndarray:
    """Linear trend slope*t + intercept."""
    return slope * np.asarray(t, dtype=float) + intercept


def sine_amplitude_phase(A1: float, A2: float) -> tuple[float, float]:
    """Convert (A1, A2) to (amplitude, phase) of A*sin(2 pi t/period + phi)."""
    return float(np.hypot(A1, A2)), float(np.arctan2(A1, A2))


def combine_means(*terms):
    """Combine mean callables f(t) -> array into one summed callable."""

    def total(t):
        out = np.zeros_like(np.asarray(t, dtype=float))
        for f in terms:
            out = out + f(t)
        return out

    return total
