"""Bragg-mirror Fabry-Perot cavities and the location of their resonances.

A cavity of m half-waves between two quarter-wave mirrors supports a comb of
longitudinal modes inside the mirror stop band. The mirror reflectivity, set by
the number of periods and the high index, fixes the finesse and therefore the
adjacent-mode separation in linewidths, Delta/gamma; the cavity length fixes the
quality factor independently.

Resonances are located as maxima of the spectral activity ||dS/dk0||_F. For a
Lorentzian term d d^T / (gamma + i(w0 - w)) the activity is proportional to
1 / (gamma^2 + (w - w0)^2), so its half-width at half-maximum equals gamma.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks

from . import config, tmm


def quarter_wave_mirror(n_high, n_low, periods, wavelength, high_first=True):
    """Layer indices and thicknesses (um) of a quarter-wave Bragg mirror."""
    pair = [(n_high, wavelength / (4 * n_high)), (n_low, wavelength / (4 * n_low))]
    if not high_first:
        pair = pair[::-1]
    indices, thicknesses = [], []
    for _ in range(periods):
        for n, d in pair:
            indices.append(n)
            thicknesses.append(d)
    return indices, thicknesses


def fabry_perot(n_high, periods, m_halfwave=config.M_HALFWAVE, n_low=config.N_LOW,
                n_cavity=config.N_CAVITY, wavelength=config.LAMBDA0):
    """Mirror-symmetric cavity: mirror | m half-wave layer | mirrored mirror."""
    li, ld = quarter_wave_mirror(n_high, n_low, periods, wavelength, high_first=True)
    ri, rd = quarter_wave_mirror(n_high, n_low, periods, wavelength, high_first=False)
    return (li + [n_cavity] + ri,
            ld + [m_halfwave * wavelength / (2 * n_cavity)] + rd)


def design_wavenumber(wavelength=config.LAMBDA0) -> float:
    return 2.0 * np.pi / wavelength


def activity(S: np.ndarray, k0: np.ndarray) -> np.ndarray:
    """Spectral activity ||dS/dk0||_F evaluated on the sampling grid."""
    return np.linalg.norm(np.gradient(S, k0, axis=0).reshape(len(k0), -1), axis=1)


def half_width(a: np.ndarray, k0: np.ndarray, i_peak: int):
    """Half-width at half-maximum of an activity peak.

    Returns (gamma, resolved). When the half-maximum is not reached inside the
    sampled interval the width is inferred from the Lorentzian profile using the
    lower of the two edge values, and resolved is False.
    """
    half = a[i_peak] / 2.0
    lo = i_peak
    while lo > 0 and a[lo] > half:
        lo -= 1
    hi = i_peak
    while hi < len(a) - 1 and a[hi] > half:
        hi += 1
    if a[lo] <= half and a[hi] <= half:
        return 0.5 * (k0[hi] - k0[lo]), True
    reach = 0.5 * (k0[-1] - k0[0])
    edge = min(a[0], a[-1])
    if a[i_peak] > edge > 0:
        return reach / np.sqrt(max(a[i_peak] / edge - 1.0, 1e-9)), False
    return reach, False


def linewidth(indices, thicknesses, k_res, spacing, n_points=3000):
    """Decay rate gamma of the mode at k_res from a window scaled to the mode spacing."""
    reach = max(0.45 * spacing, 1e-4 * k_res)
    k0 = np.linspace(k_res - reach, k_res + reach, n_points)
    a = activity(tmm.smatrix(indices, thicknesses, k0), k0)
    return half_width(a, k0, int(np.argmin(np.abs(k0 - k_res))))[0]


def comb(indices, thicknesses, k_center, span=0.30, n_scan=20000, with_widths=True):
    """Positions (sorted) and, optionally, linewidths of the modes near k_center."""
    k0 = np.linspace(k_center * (1 - span), k_center * (1 + span), n_scan)
    a = activity(tmm.smatrix(indices, thicknesses, k0), k0)
    peaks, _ = find_peaks(a, prominence=a.max() * 1e-3, distance=max(1, n_scan // 300))
    positions = k0[peaks]
    if not with_widths:
        return positions, None
    widths = np.empty(len(positions))
    for j, p in enumerate(positions):
        others = np.delete(positions, j)
        spacing = np.min(np.abs(others - p)) if len(others) else 0.1 * k_center
        widths[j] = linewidth(indices, thicknesses, p, spacing)
    return positions, widths


def target_pair(indices, thicknesses, k_center):
    """Target mode nearest k_center and its nearest neighbour.

    Returns (k_target, k_neighbour, gamma_target, gamma_neighbour), or None when
    fewer than two modes are resolved.
    """
    positions, _ = comb(indices, thicknesses, k_center, with_widths=False)
    if len(positions) < 2:
        return None
    i = int(np.argmin(np.abs(positions - k_center)))
    right = positions[i + 1] - positions[i] if i + 1 < len(positions) else np.inf
    left = positions[i] - positions[i - 1] if i > 0 else np.inf
    j = i + 1 if right <= left else i - 1
    spacing = abs(positions[j] - positions[i])
    return (positions[i], positions[j],
            linewidth(indices, thicknesses, positions[i], spacing),
            linewidth(indices, thicknesses, positions[j], spacing))


def separation(n_high, periods, k_center) -> float:
    """Measured Delta/gamma of the target mode of a cavity."""
    idx, thk = fabry_perot(n_high, periods)
    pair = target_pair(idx, thk, k_center)
    if pair is None:
        return np.nan
    kt, kn, gt, _ = pair
    return abs(kn - kt) / gt


def separation_table(k_center, periods=(1, 2, 3, 4), n_high=np.linspace(1.75, 2.6, 14)):
    """Measured Delta/gamma on a grid of mirror high indices for each period count."""
    table = {}
    for p in periods:
        x = np.array([separation(nh, p, k_center) for nh in n_high])
        ok = np.isfinite(x)
        table[p] = (np.asarray(n_high)[ok], x[ok])
    return table


def design_for_separation(target, table):
    """(periods, n_high) whose interpolated separation equals target, or None."""
    for p in sorted(table):
        nh, x = table[p]
        if len(x) < 2:
            continue
        order = np.argsort(x)
        if x[order][0] <= target <= x[order][-1]:
            return p, float(np.interp(target, x[order], nh[order]))
    return None


def measurement_window(k_target, spacing, gamma, a=config.WINDOW_A, b=config.WINDOW_B,
                       n_points=config.WINDOW_POINTS):
    """Window centred on the target with half-width min(a * Delta, b * gamma)."""
    half = min(a * spacing, b * gamma)
    return np.linspace(k_target - half, k_target + half, n_points)
