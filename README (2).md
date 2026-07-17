# Worst-of Put Pricer

A Streamlit app that prices a **European worst-of put on two stocks** — the client sells
a put whose payoff at maturity is

```
max( K% − min(S₁(T)/S₁(0), S₂(T)/S₂(0)), 0 )
```

i.e. a put on the **minimum** performance of the two names (never an average). This is
the standard structured-products building block, and the pickup over a single-name put
is the dispersion premium the desk pays for.

## How it stays accurate

Every number on screen is checked by an independent layer, live, in the app:

1. **Implied vols** are computed from live bid/ask **mid prices** by our own
   Black–Scholes bisection inversion — Yahoo's stale `impliedVolatility` column is never
   used. Vols are read at the actual strike (skew included) and interpolated to the
   trade maturity linearly in total variance σ²T.
2. **Headline price** comes from the **Stulz (1982) closed form** for an option on the
   minimum of two correlated lognormals — exact, zero Monte Carlo noise.
3. **An independent Monte Carlo** (exact terminal lognormal draw, antithetic,
   100k–1M paths) runs beside the closed form; the app flags loudly if the two disagree
   beyond the MC confidence interval.
4. **Market round-trip**: a verification panel reprices actual listed puts with the same
   vol pipeline — model prices must land inside the market bid/ask.

Extras: realized-correlation seeding (editable — desks mark implied correlation above
realized), dividend yields, risk-free rate seeded from the 13-week T-bill (^IRX), a
worst-of terminal distribution chart, a client P&L payoff chart, and an optional
Heston stochastic-vol comparison.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Push this repo to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io), pick the repo, set the main
   file to `app.py`, deploy.

If Yahoo Finance rate-limits the cloud host, the app degrades gracefully: it falls back
to realized vol (with a visible warning) instead of failing.

## Tests

```bash
# Pure math: Stulz closed form vs independent MC, no-arbitrage bounds, IV round-trip
python tests/test_pricing.py

# End-to-end: drives the real app through streamlit.testing with a deterministic
# fake yfinance (realistic TSLA/MSFT surface, deliberately stale Yahoo IV column)
python tests/test_app.py
```

## Repo layout

```
app.py              Streamlit UI + market-data pipeline (yfinance, cached)
pricing.py          Pure pricing math: Stulz 1982, MC, Heston, BS + IV inversion
tests/test_pricing.py   Math validation suite
tests/test_app.py       End-to-end AppTest run
tests/yfinance.py       Deterministic fake yfinance used by the e2e test
```

## Model notes

- Listed US single-name options are American; using their mids for European IVs slightly
  overstates OTM put vol (small at typical moneyness, and conservative).
- Strike-vol lognormal pricing is the standard structurer shortcut vs a fully
  smile-calibrated model; the optional Heston panel shows the stochastic-vol sensitivity.
- Prices are fair value — no dealer margin. Client-facing term sheets typically print
  0.5–3% of notional below fair value.
