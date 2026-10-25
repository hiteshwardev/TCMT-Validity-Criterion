"""Fitting, selection and validation of the worst-case validity criterion.

The worst-case error is modelled as ln E_max = sum_k c_k (ln x)^k with x = Delta/gamma.
The polynomial degree is chosen by information criteria and by leave-one-structure-out
cross-validation. Uncertainties come from resampling structures, the unit of
replication, with replacement.
"""
from __future__ import annotations

import numpy as np


def fit(x, e, degree=2):
    """Coefficients of ln E = polyval(c, ln x), highest power first."""
    return np.polyfit(np.log(np.asarray(x, float)), np.log(np.asarray(e, float)), degree)


def predict(coef, x):
    return np.exp(np.polyval(np.asarray(coef, float), np.log(np.asarray(x, float))))


def crossings(coef, level, x_range=(1e-2, 1e4)):
    """Values of x at which the fitted criterion equals level, in increasing order."""
    coef = np.asarray(coef, float)
    shifted = coef.copy()
    shifted[-1] -= np.log(level)
    roots = np.roots(shifted)
    real = roots[np.abs(roots.imag) < 1e-9].real
    x = np.exp(np.sort(real))
    return [float(v) for v in x if x_range[0] <= v <= x_range[1]]


def peak(coef):
    """Location and height of the maximum of a quadratic-in-log criterion."""
    c2, c1, _ = np.asarray(coef, float)
    if c2 >= 0:
        return None, None
    lx = -c1 / (2.0 * c2)
    return float(np.exp(lx)), float(np.exp(np.polyval(coef, lx)))


def log_slope(coef, x):
    """Local exponent d ln E / d ln x."""
    return float(np.polyval(np.polyder(np.asarray(coef, float)), np.log(x)))


def discrepancy(pred, truth):
    """Median multiplicative discrepancy max(p/t, t/p); one means exact."""
    r = np.asarray(pred, float) / np.asarray(truth, float)
    return float(np.median(np.maximum(r, 1.0 / r)))


def leave_one_out(x, e, degree=2):
    """Predict each structure from a fit to all the others.

    Returns the per-structure predictions in the input order.
    """
    x = np.asarray(x, float)
    e = np.asarray(e, float)
    pred = np.empty_like(e)
    for i in range(len(x)):
        keep = np.arange(len(x)) != i
        pred[i] = predict(fit(x[keep], e[keep], degree), x[i])
    return pred


def information_criteria(x, e, degree):
    """Gaussian AIC and BIC of the log-space fit, counting the noise variance."""
    r = np.log(e) - np.polyval(fit(x, e, degree), np.log(x))
    n = len(r)
    k = degree + 2
    rss = float(np.sum(r ** 2))
    loglik = -0.5 * n * (np.log(2.0 * np.pi * rss / n) + 1.0)
    return 2 * k - 2 * loglik, k * np.log(n) - 2 * loglik, rss


def model_selection(x, e, degrees=(1, 2, 3)):
    rows = []
    for d in degrees:
        aic, bic, rss = information_criteria(x, e, d)
        rows.append(dict(degree=int(d), rss=rss, aic=float(aic), bic=float(bic),
                         loo=discrepancy(leave_one_out(x, e, d), e)))
    return rows


def bootstrap(x, e, degree, n, rng, levels=(0.05, 0.01), slope_at=50.0):
    """Resample structures with replacement and refit.

    Returns arrays of the peak location and height, the local slope at slope_at,
    and, for each level, the lower and upper crossings (NaN when absent).
    """
    x = np.asarray(x, float)
    e = np.asarray(e, float)
    keys = ["peak_x", "peak_e", "slope"] + [f"{s}_{l}" for l in levels for s in ("lo", "hi")]
    out = {k: [] for k in keys}
    for _ in range(n):
        idx = rng.integers(0, len(x), len(x))
        if len(np.unique(x[idx])) <= degree + 1:
            continue
        c = fit(x[idx], e[idx], degree)
        px, pe = peak(c)
        out["peak_x"].append(np.nan if px is None else px)
        out["peak_e"].append(np.nan if pe is None else pe)
        out["slope"].append(log_slope(c, slope_at))
        for l in levels:
            cr = [v for v in crossings(c, l) if v > 1.0]
            lo = [v for v in cr if px is not None and v < px]
            hi = [v for v in cr if px is None or v >= px]
            out[f"lo_{l}"].append(lo[-1] if lo else np.nan)
            out[f"hi_{l}"].append(hi[0] if hi else np.nan)
    return {k: np.asarray(v) for k, v in out.items()}


def interval(samples, confidence=0.95):
    """Percentile interval of the finite samples; NaNs when there are none."""
    s = np.asarray(samples, float)
    s = s[np.isfinite(s)]
    if s.size == 0:
        return [float("nan"), float("nan")]
    a = 100.0 * (1.0 - confidence) / 2.0
    return [float(np.percentile(s, a)), float(np.percentile(s, 100.0 - a))]

