"""Fake yfinance with a realistic, internally consistent TSLA/MSFT market."""
import datetime as dt
import numpy as np
import pandas as pd
from math import erf, exp, log, sqrt

_SPOTS = {"TSLA": 250.0, "MSFT": 300.0, "SPCX": 95.0}
_ATM1Y = {"TSLA": 0.52, "MSFT": 0.26, "SPCX": 0.62}
_SKEW = {"TSLA": 0.22, "MSFT": 0.12, "SPCX": 0.28}   # IV pickup per unit (1 - m)
_RVOL = {"TSLA": 0.50, "MSFT": 0.24, "SPCX": 0.60}
_R, _CORR = 0.042, 0.48

def _cdf(x): return 0.5 * (1.0 + erf(x / sqrt(2.0)))

def _surface_iv(tkr, m, T):
    base = _ATM1Y[tkr] * (0.95 + 0.05 * sqrt(max(T, 0.02)))
    return base + _SKEW[tkr] * max(1.0 - m, -0.3)

def _bs(S, K, T, r, q, sig, is_put):
    d1 = (log(S / K) + (r - q + 0.5 * sig * sig) * T) / (sig * sqrt(T))
    d2 = d1 - sig * sqrt(T)
    if is_put:
        return K * exp(-r * T) * _cdf(-d2) - S * exp(-q * T) * _cdf(-d1)
    return S * exp(-q * T) * _cdf(d1) - K * exp(-r * T) * _cdf(d2)

def download(tickers, period="1y", auto_adjust=True, progress=False, **kw):
    if isinstance(tickers, str):
        tickers = [tickers]
    idx = pd.bdate_range(end=dt.date.today(), periods=252)
    rng = np.random.default_rng(7)
    z = rng.standard_normal((252, 2))
    z2 = _CORR * z[:, 0] + sqrt(1 - _CORR**2) * z[:, 1]
    data = {}
    for i, tkr in enumerate(tickers):
        shocks = z[:, 0] if i == 0 else z2
        vol = _RVOL.get(tkr, 0.5) / sqrt(252)
        path = _SPOTS.get(tkr, 100.0) * np.exp(np.cumsum(vol * shocks - 0.5 * vol**2))
        path *= _SPOTS.get(tkr, 100.0) / path[-1]
        data[("Close", tkr)] = path
    df = pd.DataFrame(data, index=idx)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df

class _Chain:
    def __init__(self, puts, calls):
        self.puts, self.calls = puts, calls

class Ticker:
    def __init__(self, symbol):
        self.symbol = symbol
        self.info = {"dividendYield": 0.007 if symbol == "MSFT" else 0.0}

    @property
    def options(self):
        if self.symbol.startswith("^"):
            return []
        today = dt.date.today()
        return [str(today + dt.timedelta(days=d)) for d in (30, 91, 182, 350, 380, 540)]

    def history(self, period="5d"):
        if self.symbol == "^IRX":
            idx = pd.bdate_range(end=dt.date.today(), periods=5)
            return pd.DataFrame({"Close": [4.2] * 5}, index=idx)
        raise ValueError("no history")

    def option_chain(self, expiry):
        S = _SPOTS[self.symbol]
        T = max((dt.date.fromisoformat(expiry) - dt.date.today()).days, 1) / 365.0
        q = 0.007 if self.symbol == "MSFT" else 0.0
        rows_p, rows_c = [], []
        for m in np.arange(0.40, 1.65, 0.05):
            K = round(S * m, 1)
            iv = _surface_iv(self.symbol, m, T)
            if m <= 1.0:
                mid = _bs(S, K, T, _R, q, iv, True)
                if mid >= 0.03:
                    rows_p.append({"strike": K, "bid": round(mid * 0.98, 2),
                                   "ask": round(mid * 1.02, 2),
                                   "impliedVolatility": iv * 0.72})  # deliberately stale
            else:
                mid = _bs(S, K, T, _R, q, iv, False)
                if mid >= 0.03:
                    rows_c.append({"strike": K, "bid": round(mid * 0.98, 2),
                                   "ask": round(mid * 1.02, 2),
                                   "impliedVolatility": iv * 0.72})
        return _Chain(pd.DataFrame(rows_p), pd.DataFrame(rows_c))
