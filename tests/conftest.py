"""Shared fixtures for the pioran_periodicity package tests.

Suppresses the known stingray/numba warnings and provides a small seeded
synthetic dataset reused across the kernel/model tests to keep the (slow)
Julia-backed likelihood evaluations to a minimum.
"""

from __future__ import annotations

import os
import warnings

import numpy as np
import pytest

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(TESTS_DIR, "data")


def pytest_configure(config):
    warnings.filterwarnings("ignore", message=".*numba.*")
    warnings.filterwarnings("ignore", message=".*Some functionality might be slower.*")
    config.addinivalue_line("markers", "slow: genuine (small) ultranest run")


@pytest.fixture
def data_dir():
    """Absolute path to tests/data (shared fixtures + small real-data samples)."""
    return DATA_DIR


@pytest.fixture(scope="session")
def synthetic():
    """A tiny seeded (t, y, yerr) dataset in the thesis' (years, mag) units."""
    rng = np.random.default_rng(20260717)
    t = np.sort(rng.uniform(0.0, 10.0, 30))
    y = rng.normal(0.0, 1.0, 30)
    yerr = np.full(30, 0.1)
    return t, y, yerr
