"""Coherent two-port excitation and the error functional of a reduced model.

A normalised input s(phi) = (1, exp(i phi)) / sqrt(2) produces the output S s(phi).
The relative error of a model S_model against the exact S over a window of N
frequency samples is

    E(phi)^2 = sum_w |dS s|^2 / sum_w |S s|^2,      dS = S_model - S.

Both sums are Hermitian quadratic forms in s, with matrices
M = sum_w dS^H dS and P = sum_w S^H S. Writing m0 = (M11 + M22)/2 and
p0 = (P11 + P22)/2,

    E(phi)^2 = (m0 + Re[M12 exp(i phi)]) / (p0 + Re[P12 exp(i phi)]).

For a lossless reference S is unitary at every frequency, P = N I, and the phase
dependence reduces exactly to a single harmonic,

    E(phi)^2 = A [1 + v cos(phi - psi)],  A = m0 / N,  v = |M12| / m0,
    psi = -arg M12,

so that the worst case over excitation phase is E_max = sqrt(A (1 + v)), reached
at phi = psi. Mirror symmetry of both the structure and the model makes M12 real,
which pins psi to 0 or pi.
"""
from __future__ import annotations

import numpy as np


def coherent_input(phi: float) -> np.ndarray:
    return np.array([1.0, np.exp(1j * phi)], dtype=complex) / np.sqrt(2.0)


def sampled_error(S_model, S_exact, phi) -> np.ndarray:
    """E(phi) evaluated directly from its definition on an array of phases."""
    out = []
    for p in np.atleast_1d(phi):
        s = coherent_input(p)
        num = np.linalg.norm(np.einsum("nij,j->ni", S_model - S_exact, s))
        den = np.linalg.norm(np.einsum("nij,j->ni", S_exact, s))
        out.append(num / den)
    return np.array(out)


def quadratic_forms(S_model, S_exact):
    """Matrices M and P of the numerator and denominator quadratic forms."""
    D = S_model - S_exact
    M = np.einsum("nij,nik->jk", D.conj(), D)
    P = np.einsum("nij,nik->jk", S_exact.conj(), S_exact)
    return M, P


def error_curve(M, P, phi) -> np.ndarray:
    """E(phi) from the quadratic forms; valid for lossless and lossy references."""
    phi = np.asarray(phi, dtype=float)
    num = 0.5 * (M[0, 0] + M[1, 1]).real + np.real(M[0, 1] * np.exp(1j * phi))
    den = 0.5 * (P[0, 0] + P[1, 1]).real + np.real(P[0, 1] * np.exp(1j * phi))
    return np.sqrt(np.maximum(num, 0.0) / den)


def phase_law(M, P):
    """Coefficients (A, v, psi) of E^2 = A [1 + v cos(phi - psi)] for a lossless reference."""
    n = 0.5 * (P[0, 0] + P[1, 1]).real
    m0 = 0.5 * (M[0, 0] + M[1, 1]).real
    return m0 / n, abs(M[0, 1]) / m0, float(-np.angle(M[0, 1]))


def worst_case(M, P) -> float:
    """Largest E(phi) over the excitation phase, from the quadratic forms.

    For a lossless reference the maximum is sqrt(A (1 + v)). Otherwise it is located
    on a fine grid and refined by a bounded scalar search.
    """
    if losslessness(P) < 1e-8:
        A, v, _ = phase_law(M, P)
        return float(np.sqrt(A * (1.0 + v)))
    from scipy.optimize import minimize_scalar
    grid = np.linspace(0.0, 2.0 * np.pi, 2048, endpoint=False)
    i = int(np.argmax(error_curve(M, P, grid)))
    step = grid[1] - grid[0]
    res = minimize_scalar(lambda p: -error_curve(M, P, p),
                          bounds=(grid[i] - step, grid[i] + step), method="bounded")
    return float(-res.fun)


def losslessness(P) -> float:
    """Departure of P / N from the identity; zero when the reference is unitary."""
    n = 0.5 * (P[0, 0] + P[1, 1]).real
    return float(np.linalg.norm(P / n - np.eye(2)))
