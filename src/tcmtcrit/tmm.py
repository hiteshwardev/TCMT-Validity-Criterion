"""Transfer-matrix solution of one-dimensional stratified media at normal incidence.

Conventions: time dependence exp(-i omega t); a forward wave varies as exp(+i k z).
For a stack between identical input and output media the power coefficients are
|r|^2 and |t|^2. Layer thicknesses are in micrometres and k0 = 2 pi / lambda in 1/um.

The matrices are evaluated for a whole array of wavenumbers at once, so a full
spectrum costs one batched sequence of 2 x 2 products per layer.
"""
from __future__ import annotations

import numpy as np


def _interface(n_a: complex, n_b: complex) -> np.ndarray:
    """Interface matrix relating amplitudes on side a to those on side b."""
    r = (n_a - n_b) / (n_a + n_b)
    t = 2.0 * n_a / (n_a + n_b)
    return np.array([[1.0, r], [r, 1.0]], dtype=complex) / t


def _stack(indices, thicknesses, k0, n_in, n_out) -> np.ndarray:
    indices = np.asarray(indices, dtype=complex)
    thicknesses = np.asarray(thicknesses, dtype=float)
    k0 = np.atleast_1d(np.asarray(k0, dtype=float))
    first = indices[0] if len(indices) else n_out
    M = np.broadcast_to(_interface(n_in, first), (len(k0), 2, 2)).copy()
    P = np.zeros((len(k0), 2, 2), dtype=complex)
    for i, (n, d) in enumerate(zip(indices, thicknesses)):
        phase = k0 * n * d
        P[:, 0, 0] = np.exp(-1j * phase)
        P[:, 1, 1] = np.exp(1j * phase)
        M = M @ P
        nxt = indices[i + 1] if i + 1 < len(indices) else n_out
        M = M @ _interface(n, nxt)
    return M


def reflection_transmission(indices, thicknesses, k0, n_in=1.0, n_out=1.0):
    """Amplitude reflection and transmission for incidence from the input side."""
    M = _stack(indices, thicknesses, k0, n_in, n_out)
    return M[:, 1, 0] / M[:, 0, 0], 1.0 / M[:, 0, 0]


def smatrix(indices, thicknesses, k0, n_in=1.0, n_out=1.0) -> np.ndarray:
    """Two-port scattering matrix [[r_L, t], [t, r_R]] with shape (len(k0), 2, 2)."""
    r_left, t = reflection_transmission(indices, thicknesses, k0, n_in, n_out)
    r_right, _ = reflection_transmission(list(indices)[::-1], list(thicknesses)[::-1],
                                         k0, n_out, n_in)
    S = np.empty((len(t), 2, 2), dtype=complex)
    S[:, 0, 0] = r_left
    S[:, 0, 1] = t
    S[:, 1, 0] = t
    S[:, 1, 1] = r_right
    return S


def airy_slab(n0: float, n1: float, d: float, k0):
    """Closed-form reflection and transmission of a single slab n0 | n1 | n0."""
    k0 = np.asarray(k0, dtype=float)
    r01 = (n0 - n1) / (n0 + n1)
    beta = k0 * n1 * d
    e = np.exp(2j * beta)
    denom = 1.0 - r01 ** 2 * e
    r = r01 * (1.0 - e) / denom
    t = (2.0 * n0 / (n0 + n1)) * (2.0 * n1 / (n0 + n1)) * np.exp(1j * beta) / denom
    return r, t
