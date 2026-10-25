"""Study constants, numerical tolerances and deterministic seeding.

Lengths are in micrometres and the spectral variable is the vacuum wavenumber
k0 = 2 pi / lambda (1/um). Only dimensionless ratios such as Delta/gamma enter
the results, so the choice of spectral variable does not affect them.
"""
from __future__ import annotations

import hashlib

import numpy as np

# Master seed. Every random stream in the study is derived from it by tag.
MASTER_SEED = 20260623

# Cavity platform: quarter-wave Bragg mirrors around a 30-half-wave cavity.
LAMBDA0 = 1.55          # design wavelength (um)
N_LOW = 1.45            # low-index mirror layer
N_CAVITY = 1.45         # cavity layer
M_HALFWAVE = 30         # cavity length in half-waves; sets the mode comb

# Target separations Delta/gamma (log-spaced, fixed before the production runs).
SEPARATION_RANGE = (1.0, 50.0)
SEPARATION_COUNT = 24

# Excitation-phase grid used where the error is sampled directly.
PHASE_COUNT = 64

# Measurement window around the target: half-width = min(a * Delta, b * gamma).
WINDOW_A = 0.4
WINDOW_B = 12.0
WINDOW_POINTS = 1600

# Acceptance tolerances for the solver benchmarks.
TOL_AIRY = 1e-10
TOL_UNITARITY = 1e-9
TOL_ISOLATED = 1e-3
TOL_ENERGY = 1e-10
TOL_CONVERGENCE = 5e-3

# Resampling.
BOOTSTRAP_RESAMPLES = 1000
CONFIDENCE = 0.95


def rng(tag: str) -> np.random.Generator:
    """Independent, reproducible random stream identified by a text tag."""
    digest = hashlib.sha256(tag.encode("utf-8")).digest()
    return np.random.default_rng(
        np.random.SeedSequence([MASTER_SEED, int.from_bytes(digest[:8], "little")]))


def separation_targets() -> np.ndarray:
    lo, hi = SEPARATION_RANGE
    return np.geomspace(lo, hi, SEPARATION_COUNT)


def phase_grid(n: int = PHASE_COUNT) -> np.ndarray:
    return np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
