"""Fourier modal method for a one-dimensional binary grating at normal incidence.

TE polarisation (electric field along the grating lines). With the period set to
one and the frequency f expressed in units of c / period, the field in the grating
layer is expanded in Floquet orders exp(i 2 pi m x), and the coupled-wave equations
reduce to

    d^2 S / dz^2 = (K^2 - k0^2 [[eps]]) S,    K = diag(2 pi m),  k0 = 2 pi f,

where [[eps]] is the Toeplitz matrix of the Fourier coefficients of the permittivity.
For TE polarisation this direct (Laurent) factorisation converges without the
inverse rule required in TM. Matching the field and its z-derivative at both
interfaces gives the reflected and transmitted amplitudes. With identical media on
both sides and f < 1, only the zeroth order propagates and the two-port matrix
[[r, t], [t, r]] is unitary and symmetric.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks


def permittivity_harmonics(orders, eps_high, eps_low, fill):
    """Fourier coefficients of a binary profile with eps_high on 0 < x < fill."""
    p = np.asarray(orders, dtype=float)
    out = np.full(p.shape, fill * eps_high + (1.0 - fill) * eps_low, dtype=complex)
    nz = p != 0
    out[nz] = ((eps_high - eps_low) * fill * np.sinc(p[nz] * fill)
               * np.exp(-1j * np.pi * p[nz] * fill))
    return out


def smatrix(freqs, eps_high, fill, thickness, eps_low=1.0, n_orders=10,
            eps_cover=1.0, eps_substrate=1.0, chunk=128):
    """Two-port scattering matrix of the grating on the frequency array freqs.

    n_orders is the truncation N; orders -N..N are retained. Returns an array of
    shape (len(freqs), 2, 2).
    """
    f_all = np.atleast_1d(np.asarray(freqs, dtype=float))
    M = 2 * n_orders + 1
    m = np.arange(-n_orders, n_orders + 1)
    K = 2.0 * np.pi * m
    coeffs = permittivity_harmonics(np.arange(-2 * n_orders, 2 * n_orders + 1),
                                    eps_high, eps_low, fill)
    EPS = coeffs[(m[:, None] - m[None, :]) + 2 * n_orders]
    eye = np.eye(M)
    c = n_orders
    out = np.empty((len(f_all), 2, 2), dtype=complex)

    for s0 in range(0, len(f_all), chunk):
        f = f_all[s0:s0 + chunk]
        nb = len(f)
        k0 = 2.0 * np.pi * f
        kz_c = np.sqrt((k0[:, None] ** 2 * eps_cover - K[None, :] ** 2).astype(complex))
        kz_s = np.sqrt((k0[:, None] ** 2 * eps_substrate - K[None, :] ** 2).astype(complex))
        kz_c = np.where(kz_c.imag < 0, -kz_c, kz_c)
        kz_s = np.where(kz_s.imag < 0, -kz_s, kz_s)

        A = np.diag(K ** 2)[None] - (k0 ** 2)[:, None, None] * EPS[None]
        lam, W = np.linalg.eig(A)
        q = np.sqrt(lam.astype(complex))
        flip = (q.real < -1e-12) | ((np.abs(q.real) <= 1e-12) & (q.imag > 0))
        q = np.where(flip, -q, q)
        X = np.exp(-q * thickness)

        WX = W * X[:, None, :]
        WQ = W * (-q)[:, None, :]
        WQX = WQ * X[:, None, :]
        I = np.broadcast_to(eye, (nb, M, M))
        Z = np.zeros((nb, M, M), dtype=complex)
        Dc = np.zeros((nb, M, M), dtype=complex)
        Ds = np.zeros((nb, M, M), dtype=complex)
        Dc[:, np.arange(M), np.arange(M)] = 1j * kz_c
        Ds[:, np.arange(M), np.arange(M)] = 1j * kz_s

        # unknowns: reflected orders, forward and backward layer modes, transmitted orders
        system = np.concatenate([
            np.concatenate([I, -W, -WX, Z], axis=2),
            np.concatenate([-Dc, -WQ, WQX, Z], axis=2),
            np.concatenate([Z, WX, W, -I], axis=2),
            np.concatenate([Z, WQX, -WQ, -Ds], axis=2)], axis=1)
        rhs = np.zeros((nb, 4 * M), dtype=complex)
        rhs[:, c] = -1.0
        rhs[:, M + c] = -1j * kz_c[:, c]
        sol = np.linalg.solve(system, rhs[..., None])[..., 0]

        r = sol[:, c]
        t = sol[:, 3 * M + c] * np.sqrt(kz_s[:, c].real / kz_c[:, c].real)
        out[s0:s0 + nb, 0, 0] = r
        out[s0:s0 + nb, 1, 1] = r
        out[s0:s0 + nb, 0, 1] = t
        out[s0:s0 + nb, 1, 0] = t
    return out


def activity(S, f):
    return np.linalg.norm(np.gradient(S, f, axis=0).reshape(len(f), -1), axis=1)


def resonances(geometry, f_min=0.30, f_max=0.999, n_scan=24000, n_local=1500,
               n_orders=10, rel_prominence=1e-3):
    """Catalogue of guided-mode resonances of a grating.

    Activity peaks are located on a coarse scan, each is refined on a local window
    of +-0.45 times the distance to its nearest neighbour, and its decay rate is the
    half-width at half-maximum of the local activity. Returns a list of dicts with
    keys f, gamma and resolved (False when the half-maximum is not reached locally).
    """
    from .cavity import half_width
    f = np.linspace(f_min, f_max, n_scan)
    a = activity(smatrix(f, n_orders=n_orders, **geometry), f)
    peaks, _ = find_peaks(a, prominence=a.max() * rel_prominence)
    centres = f[peaks]
    out = []
    for j, f0 in enumerate(centres):
        others = np.delete(centres, j)
        spacing = np.min(np.abs(others - f0)) if len(others) else 0.1
        loc = np.linspace(f0 - 0.45 * spacing, f0 + 0.45 * spacing, n_local)
        al = activity(smatrix(loc, n_orders=n_orders, **geometry), loc)
        i = int(np.argmax(al))
        g, ok = half_width(al, loc, i)
        out.append(dict(f=float(loc[i]), gamma=float(g), resolved=bool(ok)))
    return out
