"""
Pure pricing math for the worst-of put on two assets. No Streamlit, no I/O —
everything here is deterministic and unit-testable.

Conventions: performance space, i.e. X_i(0) = 1 and the strike K is a fraction
of spot (0.70 = 70%). Payoff of the worst-of put: max(K - min(X1(T), X2(T)), 0).
"""

from math import exp, log, sqrt

import numpy as np
from scipy.stats import multivariate_normal, norm


# ============================================================================
# ANALYTIC PRICER — Stulz (1982) option on the minimum of two assets
# ============================================================================

def _bvn_cdf(a, b, rho):
    """P(Z1 <= a, Z2 <= b) for standard bivariate normal with correlation rho."""
    rho = float(np.clip(rho, -0.99999, 0.99999))
    a = float(np.clip(a, -12.0, 12.0))
    b = float(np.clip(b, -12.0, 12.0))
    return float(multivariate_normal(mean=[0.0, 0.0],
                                     cov=[[1.0, rho], [rho, 1.0]]).cdf([a, b]))


def stulz_put_on_min(K, T, r, q1, q2, s1, s2, rho):
    """
    European put on min(X1, X2) in performance space (X(0)=1), strike K.
    Put = Call_min(K) − PV[min] + K·e^{−rT}   (parity on the minimum).
    Exact under correlated lognormals.
    """
    T = max(float(T), 1e-8)
    s1 = max(float(s1), 1e-6)
    s2 = max(float(s2), 1e-6)
    b1, b2 = r - q1, r - q2
    sig = sqrt(max(s1 * s1 + s2 * s2 - 2 * rho * s1 * s2, 1e-12))
    srt = sqrt(T)
    d = ((b1 - b2 + 0.5 * sig * sig) * T) / (sig * srt)          # ln(X1/X2)=0
    y1 = (log(1.0 / K) + (b1 + 0.5 * s1 * s1) * T) / (s1 * srt)
    y2 = (log(1.0 / K) + (b2 + 0.5 * s2 * s2) * T) / (s2 * srt)
    r1 = (rho * s2 - s1) / sig
    r2 = (rho * s1 - s2) / sig
    call_min = (exp((b1 - r) * T) * _bvn_cdf(y1, -d, r1)
                + exp((b2 - r) * T) * _bvn_cdf(y2, d - sig * srt, r2)
                - K * exp(-r * T) * _bvn_cdf(y1 - s1 * srt, y2 - s2 * srt, rho))
    pv_min = (exp((b1 - r) * T) * norm.cdf(-d)
              + exp((b2 - r) * T) * norm.cdf(d - sig * srt))
    return call_min - pv_min + K * exp(-r * T)


def bs_put_pct(K, T, r, q, s):
    """Vanilla European put in performance space (analytic)."""
    T = max(float(T), 1e-8)
    s = max(float(s), 1e-6)
    d1 = (log(1.0 / K) + (r - q + 0.5 * s * s) * T) / (s * sqrt(T))
    d2 = d1 - s * sqrt(T)
    return K * exp(-r * T) * norm.cdf(-d2) - exp(-q * T) * norm.cdf(-d1)


# ============================================================================
# MONTE CARLO — independent confirmation + distribution for charts
# ============================================================================

def price_wo_implied(sig1, sig2, corr, K_pct, T, r, q1, q2,
                     n_paths=500000, seed=42, antithetic=True):
    """Exact terminal correlated-lognormal draw. payoff = max(K − min(X1,X2), 0)."""
    rng = np.random.default_rng(seed)
    n_base = n_paths // 2 if antithetic else n_paths
    Z = rng.standard_normal((n_base, 2))
    if antithetic:
        Z = np.vstack([Z, -Z])
    e1 = Z[:, 0]
    e2 = corr * Z[:, 0] + np.sqrt(max(1.0 - corr ** 2, 0.0)) * Z[:, 1]
    X1 = np.exp((r - q1 - 0.5 * sig1 ** 2) * T + sig1 * np.sqrt(T) * e1)
    X2 = np.exp((r - q2 - 0.5 * sig2 ** 2) * T + sig2 * np.sqrt(T) * e2)
    wo = np.minimum(X1, X2)                      # WORST-of: the minimum, never an average
    disc = np.exp(-r * T)

    def stats(pay):
        if antithetic:
            paired = 0.5 * (pay[:n_base] + pay[n_base:])
            return disc * paired.mean(), disc * paired.std(ddof=1) / np.sqrt(n_base)
        return disc * pay.mean(), disc * pay.std(ddof=1) / np.sqrt(n_base)

    price, se = stats(np.maximum(K_pct - wo, 0.0))
    return {"price": price, "se": se, "ci": (price - 1.96 * se, price + 1.96 * se),
            "wo_T": wo, "prob_exercise": float((wo < K_pct).mean())}


def price_wo_heston(v0s, thetas, kappas, xis, rho_svs, corr_assets, K_pct, T, r, qs,
                    n_paths=50000, n_steps=252, antithetic=True, seed=42):
    """Per-asset Heston, full-truncation Euler (stochastic-vol comparison)."""
    rng = np.random.default_rng(seed)
    v0s, thetas = np.asarray(v0s, float), np.asarray(thetas, float)
    kappas, xis, rho_svs = np.asarray(kappas, float), np.asarray(xis, float), np.asarray(rho_svs, float)
    qs = np.asarray(qs, float)
    dt_ = T / n_steps
    sqrt_dt = np.sqrt(dt_)
    L = np.linalg.cholesky(np.array([[1.0, corr_assets], [corr_assets, 1.0]]))
    n_base = n_paths // 2 if antithetic else n_paths
    total = n_base * 2 if antithetic else n_base
    X = np.ones((total, 2))
    v = np.tile(v0s, (total, 1))
    orth = np.sqrt(1.0 - rho_svs ** 2)
    for _ in range(n_steps):
        Z1 = rng.standard_normal((n_base, 2))
        Z2 = rng.standard_normal((n_base, 2))
        if antithetic:
            Z1, Z2 = np.vstack([Z1, -Z1]), np.vstack([Z2, -Z2])
        eps_S = Z1 @ L.T
        eps_v = rho_svs * eps_S + orth * Z2
        v_pos = np.maximum(v, 0.0)
        X *= np.exp((r - qs - 0.5 * v_pos) * dt_ + np.sqrt(v_pos) * sqrt_dt * eps_S)
        v += kappas * (thetas - v_pos) * dt_ + xis * np.sqrt(v_pos) * sqrt_dt * eps_v
    wo = X.min(axis=1)
    disc = np.exp(-r * T)
    pay = np.maximum(K_pct - wo, 0.0)
    if antithetic:
        paired = 0.5 * (pay[:n_base] + pay[n_base:])
        return {"price": disc * paired.mean(),
                "se": disc * paired.std(ddof=1) / np.sqrt(n_base)}
    return {"price": disc * pay.mean(), "se": disc * pay.std(ddof=1) / np.sqrt(n_base)}


# ============================================================================
# BLACK–SCHOLES UTILITIES — vectorized pricing and implied-vol inversion
# ============================================================================

def bs_price_vec(S, K, T, r, q, sig, is_put):
    """Vectorized Black–Scholes price in dollar space."""
    sig = np.maximum(np.asarray(sig, dtype=float), 1e-9)
    K = np.asarray(K, dtype=float)
    d1 = (np.log(S / K) + (r - q + 0.5 * sig ** 2) * T) / (sig * np.sqrt(T))
    d2 = d1 - sig * np.sqrt(T)
    if is_put:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
    return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def implied_vol_vec(mids, S, Ks, T, r, q, is_put):
    """Bisection inversion of BS from MID prices (Yahoo's IV column is stale)."""
    mids = np.asarray(mids, dtype=float)
    Ks = np.asarray(Ks, dtype=float)
    lo = np.full_like(mids, 0.005)
    hi = np.full_like(mids, 4.0)
    if is_put:
        lower = np.maximum(Ks * np.exp(-r * T) - S * np.exp(-q * T), 0.0)
        upper = Ks * np.exp(-r * T)
    else:
        lower = np.maximum(S * np.exp(-q * T) - Ks * np.exp(-r * T), 0.0)
        upper = S * np.exp(-q * T)
    ok = (mids > lower + 1e-6) & (mids < upper - 1e-6)
    for _ in range(80):
        mid_sig = 0.5 * (lo + hi)
        px = bs_price_vec(S, Ks, T, r, q, mid_sig, is_put)
        too_low = px < mids
        lo = np.where(too_low, mid_sig, lo)
        hi = np.where(too_low, hi, mid_sig)
    iv = 0.5 * (lo + hi)
    return iv, ok & (iv > 0.02) & (iv < 3.5)
