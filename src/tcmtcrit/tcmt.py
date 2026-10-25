"""Temporal coupled-mode models of a two-port and their least-squares fitting.

A model with modes m = 1..M sharing one direct (background) matrix reads

    S(w) = exp(i[a + b (w - w_ref)]) { C(w) + sum_m d_m d_m^T / (gamma_m + i (w_m - w)) },

where d_m are complex coupling vectors and the common factor in front describes the
position of the port reference planes. In standard TCMT the background is constant,
C(w) = C_0, complex symmetric by reciprocity; the single-mode model of Fan, Suh and
Joannopoulos is the case M = 1. As a phenomenological extension the background may be
given a polynomial dependence C(w) = sum_k C_k u^k on the reduced frequency
u = (w - w_ref) / h, where h is the half-width of the window.

The constraints that energy conservation and time reversal impose on a lossless
model (|d|^2 = 2 gamma and C d* = -d) are not enforced. The fitted model is
therefore the best approximation available to the M-pole family on the window,
and its error is a lower bound on that of any constrained model with the same
number of modes. How closely the fitted single-mode parameters satisfy
|d|^2 = 2 gamma is reported as a diagnostic.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares
from scipy.signal import find_peaks


def evaluate(w, poles, C, couplings, phase=(0.0, 0.0), w_ref=None, half=None):
    """Model scattering matrix on the grid w, shape (len(w), 2, 2).

    C is either one 2 x 2 matrix (constant background) or a sequence of matrices
    holding the polynomial background coefficients C_0, C_1, ...
    """
    w = np.asarray(w, dtype=float)
    if w_ref is None:
        w_ref = 0.5 * (w[0] + w[-1])
    if half is None:
        half = 0.5 * (w[-1] - w[0])
    terms = [C] if np.ndim(C) == 2 else list(C)
    u = (w - w_ref) / half
    S = sum(np.asarray(Ck, dtype=complex)[None, :, :] * (u ** k)[:, None, None]
            for k, Ck in enumerate(terms))
    for (w0, g), d in zip(poles, couplings):
        S += np.outer(d, d)[None, :, :] / (g + 1j * (w0 - w))[:, None, None]
    return S * np.exp(1j * (phase[0] + phase[1] * (w - w_ref)))[:, None, None]


def relative_l2(S_model, S_exact) -> float:
    """Relative Frobenius-norm error of a scattering matrix over a window."""
    return float(np.linalg.norm((S_model - S_exact).ravel())
                 / np.linalg.norm(S_exact.ravel()))


def rank_one_factor(R: np.ndarray) -> np.ndarray:
    """Vector d with d d^T = R for a complex-symmetric rank-one 2 x 2 matrix R."""
    if abs(R[0, 0]) >= abs(R[1, 1]):
        d0 = np.sqrt(R[0, 0])
        d1 = R[0, 1] / d0 if abs(d0) > 0 else np.sqrt(R[1, 1])
    else:
        d1 = np.sqrt(R[1, 1])
        d0 = R[0, 1] / d1 if abs(d1) > 0 else np.sqrt(R[0, 0])
    return np.array([d0, d1], dtype=complex)


def strongest_resonance(w, S):
    """Position and decay rate of the dominant activity peak inside the window."""
    w = np.asarray(w, dtype=float)
    a = np.linalg.norm(np.gradient(S, w, axis=0).reshape(len(w), -1), axis=1)
    peaks, props = find_peaks(a, prominence=1e-12, distance=max(1, len(a) // 4))
    i = int(peaks[np.argmax(props["prominences"])]) if len(peaks) else int(np.argmax(a))
    half = a[i] / 2.0
    lo = i
    while lo > 0 and a[lo] > half:
        lo -= 1
    hi = i
    while hi < len(a) - 1 and a[hi] > half:
        hi += 1
    return float(w[i]), float(max(0.5 * (w[hi] - w[lo]), w[1] - w[0]))


def fit(w, S, poles, free, S_at=None, background_order=0, start=None):
    """Least-squares fit of an M-mode model to exact data on the window w.

    poles  : list of (w_m, gamma_m) initial values, one per mode
    free   : list of booleans; True lets the pole position and width vary,
             False keeps them at the supplied values (used for neighbours whose
             resonance lies outside the window and is known from the spectrum)
    S_at   : callable returning the exact 2 x 2 scattering matrix at a pole
             position; used only to initialise the coupling vectors. If None,
             the window sample closest to each pole is used.
    background_order
           polynomial order of the background; 0 is standard TCMT.
    start  : parameters of a nested model (fewer modes or a lower background
             order) used as the initial point. Its modes are matched to the
             leading entries of poles, missing couplings and background
             coefficients start at zero, and surplus background coefficients
             are dropped.

    The coupling vectors, the background and the reference-plane phase are always
    fitted. Returns (parameters, S_model); parameters["cost"] is the final value
    of the least-squares objective.
    """
    w = np.asarray(w, dtype=float)
    w_ref = 0.5 * (w[0] + w[-1])
    half = 0.5 * (w[-1] - w[0])
    n_bg = background_order + 1
    if start is None:
        C_init = [np.mean(S[[0, 1, -2, -1]], axis=0)] + [np.zeros((2, 2))] * background_order
        d_init = []
        for w0, g in poles:
            S0 = S_at(w0) if S_at is not None else S[int(np.argmin(np.abs(w - w0)))]
            d_init.append(rank_one_factor(g * (S0 - C_init[0])))
        phase_init = (0.0, 0.0)
        pole_init = list(poles)
    else:
        k = len(start["poles"])
        C_init = [np.asarray(c, complex) for c in start["C"][:n_bg]]
        C_init += [np.zeros((2, 2))] * (n_bg - len(C_init))
        d_init = list(start["couplings"]) + [np.zeros(2)] * (len(poles) - k)
        phase_init = start["phase"]
        pole_init = [start["poles"][m] if (m < k and f) else poles[m]
                     for m, f in enumerate(free)]

    def unpack(p):
        i, out = 0, []
        for (w0, g), f in zip(poles, free):
            if f:
                out.append((p[i], abs(p[i + 1])))
                i += 2
            else:
                out.append((w0, g))
        C = []
        for _ in range(n_bg):
            C.append(np.array([[p[i] + 1j * p[i + 1], p[i + 2] + 1j * p[i + 3]],
                               [p[i + 2] + 1j * p[i + 3], p[i + 4] + 1j * p[i + 5]]]))
            i += 6
        ds = []
        for _ in poles:
            ds.append(np.array([p[i] + 1j * p[i + 1], p[i + 2] + 1j * p[i + 3]]))
            i += 4
        return out, C, ds, (p[i], p[i + 1])

    def residual(p):
        pl, C, ds, ph = unpack(p)
        r = (evaluate(w, pl, C, ds, ph, w_ref, half) - S).ravel()
        return np.concatenate([r.real, r.imag])

    p0 = []
    for (w0, g), f in zip(pole_init, free):
        if f:
            p0 += [w0, g]
    for Ck in C_init:
        p0 += [Ck[0, 0].real, Ck[0, 0].imag, Ck[0, 1].real, Ck[0, 1].imag,
               Ck[1, 1].real, Ck[1, 1].imag]
    for d in d_init:
        p0 += [d[0].real, d[0].imag, d[1].real, d[1].imag]
    p0 += [phase_init[0], phase_init[1]]
    sol = least_squares(residual, np.asarray(p0, dtype=float), method="lm",
                        max_nfev=20000)
    pl, C, ds, ph = unpack(sol.x)
    params = dict(poles=pl, C=C, couplings=ds, phase=ph, w_ref=w_ref,
                  converged=bool(sol.success), cost=float(sol.cost))
    return params, evaluate(w, pl, C, ds, ph, w_ref, half)


def best_fit(w, S, poles, free, S_at=None, background_order=0, starts=()):
    """Fit from the default initial point and from each nested start; keep the best.

    Starting from the optimum of a nested model guarantees that the extended model
    is never worse than the nested one, which a single local search from a generic
    initial point does not.
    """
    runs = [fit(w, S, poles, free, S_at, background_order)]
    runs += [fit(w, S, poles, free, S_at, background_order, start=s) for s in starts]
    return min(runs, key=lambda r: r[0]["cost"])


def single_mode_family(w, S, max_order=0):
    """Single-mode fits with background orders 0..max_order.

    The models are nested, so each order is also started from the optimum of the
    order below, and the constant-background model is refitted from the
    linear-background optimum with its linear term removed. This protects the
    constant-background fit against local minima when the background varies
    strongly across the window. Returns a list of (parameters, S_model).
    """
    w0, g = strongest_resonance(w, S)
    pole, free = [(w0, g)], [True]

    def better(*runs):
        return min(runs, key=lambda r: r[0]["cost"])

    const = fit(w, S, pole, free)
    lin = better(fit(w, S, pole, free, background_order=1),
                 fit(w, S, pole, free, background_order=1, start=const[0]))
    refit = fit(w, S, pole, free, start=lin[0])
    if refit[0]["cost"] < const[0]["cost"]:
        const = refit
        lin = better(lin, fit(w, S, pole, free, background_order=1, start=const[0]))
    fits = [const, lin][:max_order + 1]
    for k in range(2, max_order + 1):
        fits.append(best_fit(w, S, pole, free, background_order=k, starts=[fits[-1][0]]))
    return fits


def single_mode(w, S, background_order=0):
    """Single-mode model initialised from the dominant activity peak in the window."""
    return single_mode_family(w, S, background_order)[background_order]


def energy_mismatch(params, mode=0) -> float:
    """|d|^2 / (2 gamma) - 1 for one fitted mode; zero for a lossless TCMT mode."""
    d = params["couplings"][mode]
    g = params["poles"][mode][1]
    return float(np.vdot(d, d).real / (2.0 * g) - 1.0)
