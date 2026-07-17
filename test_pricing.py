"""Math validation: run with  `python tests/test_pricing.py`  from the repo root.

Proves the pricing core is right without any market data:
  1. Stulz (1982) closed form == independent Monte Carlo across a parameter grid.
  2. rho -> 1 with identical assets collapses to the vanilla Black-Scholes put.
  3. No-arbitrage bounds:  max(P1, P2) <= P_wo <= min(P1 + P2, K e^{-rT}).
  4. Price is monotone decreasing in correlation.
  5. Implied-vol inversion round-trips Black-Scholes prices to the 4th decimal.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pricing import (bs_price_vec, bs_put_pct, implied_vol_vec,  # noqa: E402
                     price_wo_implied, stulz_put_on_min)

failures = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}  {detail}")
    if not cond:
        failures.append(name)


# 1) Closed form vs independent MC over a parameter grid
grid = [
    # K,    T,    r,     q1,    q2,    s1,   s2,   rho
    (0.70, 1.00, 0.042, 0.000, 0.007, 0.55, 0.28, 0.50),
    (1.00, 0.50, 0.050, 0.010, 0.000, 0.20, 0.30, 0.30),
    (0.90, 2.00, 0.030, 0.000, 0.020, 0.40, 0.40, 0.80),
    (0.50, 1.50, 0.040, 0.000, 0.000, 0.60, 0.60, -0.20),
    (1.20, 0.25, 0.045, 0.005, 0.005, 0.25, 0.45, 0.65),
    (0.80, 3.00, 0.035, 0.000, 0.010, 0.35, 0.50, 0.10),
]
for K, T, r, q1, q2, s1, s2, rho in grid:
    a = stulz_put_on_min(K, T, r, q1, q2, s1, s2, rho)
    mc = price_wo_implied(s1, s2, rho, K, T, r, q1, q2, n_paths=1_000_000, seed=123)
    z = abs(mc["price"] - a) / mc["se"]
    check(f"Stulz vs MC (K={K}, T={T}, rho={rho})", z < 4.0,
          f"analytic={a:.5f} mc={mc['price']:.5f}±{mc['se']:.5f} |z|={z:.2f}")

# 2) Degenerate limit: rho -> 1 with identical assets => vanilla BS put
a = stulz_put_on_min(0.9, 1.0, 0.04, 0.01, 0.01, 0.3, 0.3, 0.99999)
v = bs_put_pct(0.9, 1.0, 0.04, 0.01, 0.3)
check("rho->1 identical assets == vanilla put", abs(a - v) < 2e-4,
      f"stulz={a:.6f} vanilla={v:.6f}")

# 3) No-arbitrage bounds
K, T, r, q1, q2, s1, s2, rho = 0.85, 1.0, 0.04, 0.0, 0.01, 0.5, 0.25, 0.4
p1, p2 = bs_put_pct(K, T, r, q1, s1), bs_put_pct(K, T, r, q2, s2)
pw = stulz_put_on_min(K, T, r, q1, q2, s1, s2, rho)
check("no-arbitrage bounds",
      max(p1, p2) - 1e-10 <= pw <= min(p1 + p2, K * np.exp(-r * T)) + 1e-10,
      f"p1={p1:.5f} p2={p2:.5f} p_wo={pw:.5f}")

# 4) Monotone decreasing in correlation
prices = [stulz_put_on_min(0.8, 1.0, 0.04, 0.0, 0.0, 0.45, 0.3, rho)
          for rho in np.linspace(-0.9, 0.95, 12)]
check("monotone decreasing in rho", all(b <= a + 1e-9 for a, b in zip(prices, prices[1:])))

# 5) Implied-vol inversion round-trip
S, r, q, T = 250.0, 0.042, 0.005, 0.75
Ks = np.array([150.0, 200.0, 250.0, 300.0])
true_iv = np.array([0.62, 0.55, 0.48, 0.44])
for is_put in (True, False):
    px = bs_price_vec(S, Ks, T, r, q, true_iv, is_put)
    iv, ok = implied_vol_vec(px, S, Ks, T, r, q, is_put)
    err = np.abs(iv[ok] - true_iv[ok]).max() if ok.any() else np.inf
    check(f"IV inversion round-trip ({'puts' if is_put else 'calls'})",
          ok.any() and err < 1e-4, f"max |err|={err:.2e} over {int(ok.sum())} strikes")

print()
if failures:
    print(f"{len(failures)} FAILURES: {failures}")
    sys.exit(1)
print("ALL PASS")
