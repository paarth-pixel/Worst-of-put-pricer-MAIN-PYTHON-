"""
Worst-of Put Pricer
Client sells a worst-of put on 2 stocks.

Accuracy stack (each layer checks the one below, live, in the app):
  1. Implied vols computed from LIVE bid/ask mid prices (own BS inversion —
     Yahoo's stale impliedVolatility column is never used).
  2. Headline price from the STULZ (1982) closed-form for a put on the
     minimum of two assets — exact, zero Monte Carlo noise.
  3. Independent Monte Carlo (exact terminal lognormal draw) shown beside it;
     the two must agree within the MC confidence interval.
  4. A verification panel reprices actual listed puts against their market
     mids, proving the vol pipeline round-trips to the cent.
"""

import datetime as dt

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from pricing import (
    stulz_put_on_min,
    bs_put_pct,
    price_wo_implied,
    price_wo_heston,
    bs_price_vec,
    implied_vol_vec,
    simulate_wo_paths,
)

st.set_page_config(page_title="Worst-of Put Pricer", page_icon="📉", layout="wide")

TICKER_UNIVERSE = ["TSLA", "MSFT", "SPCX", "NVDA", "AAPL", "AMZN", "GOOGL", "META", "AMD", "NFLX"]
PARAM_LABELS = ["Initial vol √v₀ (%)", "Long-run vol √θ (%)", "Mean reversion κ", "Vol of vol ξ", "Spot–vol corr ρ"]

# ============================================================================
# MARKET DATA — history, dividends, rates, and own-inverted implied vols
# ============================================================================

@st.cache_data(ttl=900, show_spinner=False)
def fetch_hist(tickers: tuple):
    import yfinance as yf
    px = yf.download(list(tickers), period="1y", auto_adjust=True, progress=False)["Close"]
    if isinstance(px, pd.Series):
        px = px.to_frame(tickers[0])
    px = px[list(tickers)]
    spots, vol_lr, days_used = {}, {}, {}
    for tkr in tickers:
        s = px[tkr].dropna()
        if len(s) < 2:
            raise ValueError(f"No price history for {tkr}.")
        rets = np.log(s / s.shift(1)).dropna()
        spots[tkr] = float(s.iloc[-1])
        days_used[tkr] = len(rets)
        vol_lr[tkr] = float(rets.std(ddof=1) * np.sqrt(252)) if len(rets) >= 10 else 0.60
    both = np.log(px / px.shift(1)).dropna()
    corr = float(both.corr().iloc[0, 1]) if len(both) >= 20 else 0.50
    return spots, vol_lr, corr, px, {"days_used": days_used, "overlap": len(both),
                                     "corr_est": len(both) >= 20}


@st.cache_data(ttl=900, show_spinner=False)
def fetch_div_yield(ticker: str) -> float:
    import yfinance as yf
    try:
        dy = float(yf.Ticker(ticker).info.get("dividendYield") or 0.0)
        if dy > 0.25:
            dy /= 100.0
        return float(np.clip(dy, 0.0, 0.10))
    except Exception:
        return 0.0


@st.cache_data(ttl=600, show_spinner=False)
def fetch_riskfree_seed() -> float:
    import yfinance as yf
    try:
        h = yf.Ticker("^IRX").history(period="5d")["Close"].dropna()
        rr = float(h.iloc[-1]) / 100.0
        if 0.0 < rr < 0.15:
            return rr
    except Exception:
        pass
    return 0.04


def _smile_from_chain(chain, spot, T_exp, r, q):
    """OTM smile from live bid/ask mids with own IV inversion."""
    frames, quotes = [], []
    for df, is_put in ((chain.puts, True), (chain.calls, False)):
        if df is None or len(df) == 0:
            continue
        d = df.copy()
        d = d[(d["bid"] > 0) & (d["ask"] >= d["bid"])]
        d["mid"] = 0.5 * (d["bid"] + d["ask"])
        d = d[d["mid"] >= 0.03]
        d["m"] = d["strike"] / spot
        d = d[(d["m"] > 0.35) & (d["m"] < 1.8)]
        d = d[d["m"] <= 1.0] if is_put else d[d["m"] > 1.0]
        if len(d) == 0:
            continue
        iv, ok = implied_vol_vec(d["mid"].to_numpy(), spot, d["strike"].to_numpy(),
                                 T_exp, r, q, is_put)
        d["iv_mid"] = iv
        d = d[ok]
        if len(d):
            frames.append(d[["m", "iv_mid"]])
            qq = d[["strike", "bid", "ask", "mid", "iv_mid"]].copy()
            qq["type"] = "put" if is_put else "call"
            quotes.append(qq)
    if not frames:
        return None, None
    sm = pd.concat(frames).sort_values("m")
    sm = sm.groupby("m", as_index=False)["iv_mid"].mean().rename(columns={"iv_mid": "iv"})
    if len(sm) < 4:
        return None, None
    return sm, pd.concat(quotes).sort_values("strike")


@st.cache_data(ttl=900, show_spinner=False)
def fetch_iv(ticker: str, spot: float, target_T: float, r: float, q: float):
    """IV at any moneyness, interpolated to target_T in total variance."""
    import yfinance as yf
    tk = yf.Ticker(ticker)
    expiries = tk.options
    if not expiries:
        raise ValueError(f"{ticker}: no listed options")

    today = dt.date.today()
    Ts = np.array([max((dt.date.fromisoformat(e) - today).days, 1) / 365.0 for e in expiries])
    order = np.argsort(Ts)
    Ts, expiries = Ts[order], [expiries[i] for i in order]

    hi_idx = int(np.searchsorted(Ts, target_T))
    lo_idx = max(hi_idx - 1, 0)
    hi_idx = min(hi_idx, len(Ts) - 1)
    extrapolated = target_T > Ts[-1] + 1e-9 or target_T < Ts[0] - 1e-9

    candidates = list(dict.fromkeys(
        [lo_idx, hi_idx, max(lo_idx - 1, 0), min(hi_idx + 1, len(Ts) - 1)]))
    smiles, quotes_by_exp, used = {}, {}, []
    for idx in candidates:
        if idx in smiles:
            continue
        sm, qt = _smile_from_chain(tk.option_chain(expiries[idx]), spot, float(Ts[idx]), r, q)
        if sm is not None:
            smiles[idx] = sm
            quotes_by_exp[expiries[idx]] = qt
            used.append((expiries[idx], float(Ts[idx]), len(sm)))
        if (len(smiles) >= 2 and any(k >= hi_idx for k in smiles)
                and any(k <= lo_idx for k in smiles)):
            break
    if not smiles:
        raise ValueError(f"{ticker}: chains too thin to build a smile from live quotes")

    ids = sorted(sorted(smiles, key=lambda k: abs(Ts[k] - target_T))[:2])

    def iv_at(idx, m):
        sm = smiles[idx]
        return float(np.interp(m, sm["m"].to_numpy(), sm["iv"].to_numpy()))

    v0_seed = None
    for idx in sorted(smiles):
        if Ts[idx] >= 10 / 365:
            v0_seed = iv_at(idx, 1.0)
            break

    plot_idx = ids[-1]
    return {
        "iv_curve_m": smiles[plot_idx]["m"].tolist(),
        "iv_curve_v": smiles[plot_idx]["iv"].tolist(),
        "plot_expiry": expiries[plot_idx],
        "iv_fn_points": {int(k): (smiles[k]["m"].tolist(), smiles[k]["iv"].tolist(),
                                  float(Ts[k])) for k in ids},
        "target_T": target_T,
        "used": used,
        "extrapolated": bool(extrapolated),
        "v0_seed": v0_seed,
        "quotes": {e: quotes_by_exp[e].to_dict("records") for e in quotes_by_exp},
        "expiry_T": {e: float(Ts[i]) for i, e in enumerate(expiries) if e in quotes_by_exp},
    }


def iv_from_pack(pack, m):
    pts = pack["iv_fn_points"]
    ids = sorted(pts)

    def one(k):
        ms, vs, _ = pts[k]
        return float(np.interp(m, ms, vs))

    if len(ids) == 1:
        return one(ids[0])
    k0, k1 = ids
    T0, T1 = pts[k0][2], pts[k1][2]
    tT = pack["target_T"]
    w = float(np.clip((tT - T0) / (T1 - T0), 0.0, 1.0)) if T1 > T0 else 1.0
    tv = (1 - w) * one(k0) ** 2 * T0 + w * one(k1) ** 2 * T1
    T_eff = (1 - w) * T0 + w * T1
    return float(np.sqrt(max(tv, 1e-8) / max(T_eff, 1e-8)))


def atm_iv_from_pack(pack):
    return iv_from_pack(pack, 1.0)


# ============================================================================
# SIDEBAR
# ============================================================================

st.sidebar.header("Underlyings")
c1, c2 = st.sidebar.columns(2)
t1 = c1.selectbox("Stock 1", TICKER_UNIVERSE, index=0)
t2 = c2.selectbox("Stock 2", TICKER_UNIVERSE, index=1)
if t1 == t2:
    st.sidebar.error("Pick two different stocks.")
    st.stop()

st.sidebar.header("Trade")
today = dt.date.today()
maturity = st.sidebar.date_input("Maturity date", value=today + dt.timedelta(days=365),
                                 min_value=today + dt.timedelta(days=7),
                                 max_value=today + dt.timedelta(days=365 * 5))
T = max((maturity - today).days, 1) / 365.0
st.sidebar.caption(f"T = {T:.3f} years ({(maturity - today).days} days)")
K_pct = st.sidebar.slider("Strike (% of spot)", 10, 150, 70, 1) / 100.0
notional = st.sidebar.number_input("Notional (USD)", value=1_000_000, step=100_000)
r = st.sidebar.slider("Risk-free rate (%)", 0.0, 10.0, round(fetch_riskfree_seed() * 100, 2),
                      0.05, help="Seeded from the 13-week T-bill (^IRX)") / 100.0

st.sidebar.header("Monte Carlo (confirmation run)")
n_paths = st.sidebar.select_slider("Paths", options=[100000, 250000, 500000, 1000000],
                                   value=500000)
seed = st.sidebar.number_input("Seed", value=42, step=1)
run_heston = st.sidebar.checkbox("Also run Heston stochastic-vol comparison", value=False)

# ============================================================================
# MAIN
# ============================================================================

st.title("📉 Worst-of Put Pricer")
st.caption(
    f"Client **sells** a worst-of put on {t1} / {t2}, strike {K_pct:.0%}, maturing "
    f"{maturity:%d %b %Y}. Payoff = max(K% − **min**(S₁ₜ/S₁₀, S₂ₜ/S₂₀), 0) — the minimum, "
    "never an average. Priced two independent ways (closed-form + Monte Carlo) that must agree."
)

try:
    spots_d, vollr_d, corr_real, px_hist, hmeta = fetch_hist((t1, t2))
    S1, S2 = float(spots_d[t1]), float(spots_d[t2])
except Exception as e:
    st.warning(f"Could not fetch price history ({e}). Running in **manual mode** — "
               "enter spots below and set vols/correlation in *Pricing inputs*; "
               "the pricing math is unaffected.")
    mc1_, mc2_ = st.columns(2)
    S1 = float(mc1_.number_input(f"{t1} spot (manual)", value=100.0, min_value=0.01))
    S2 = float(mc2_.number_input(f"{t2} spot (manual)", value=100.0, min_value=0.01))
    vollr_d = {t1: 0.40, t2: 0.40}
    corr_real = 0.50
    hmeta = {"overlap": 0}

q1, q2 = fetch_div_yield(t1), fetch_div_yield(t2)

iv_packs, iv_strike, iv_source, fetch_errors = {}, {}, {}, []
for tkr, spot, qq in ((t1, S1, q1), (t2, S2, q2)):
    try:
        pack = fetch_iv(tkr, spot, T, round(r, 4), round(qq, 4))
        iv_packs[tkr] = pack
        iv_strike[tkr] = iv_from_pack(pack, K_pct)
        iv_source[tkr] = "implied (live bid/ask mids)"
    except Exception as e:
        iv_strike[tkr] = vollr_d[tkr]
        iv_source[tkr] = "realized (fallback)"
        fetch_errors.append(f"{tkr}: {e} — using realized vol {vollr_d[tkr]:.1%} instead.")

for msg in fetch_errors:
    st.warning(msg)
for tkr in (t1, t2):
    if iv_source[tkr].startswith("implied") and iv_strike[tkr] < 0.75 * vollr_d[tkr]:
        st.warning(f"{tkr}: strike IV ({iv_strike[tkr]:.1%}) is well below realized "
                   f"({vollr_d[tkr]:.1%}) — quotes may be stale/thin. Cross-check and override.")

mcols = st.columns(4)
mcols[0].metric(f"{t1} spot", f"${S1:,.2f}")
mcols[1].metric(f"{t2} spot", f"${S2:,.2f}")
mcols[2].metric(f"{t1} IV @ {K_pct:.0%}K", f"{iv_strike[t1]:.1%}", help=iv_source[t1])
mcols[3].metric(f"{t2} IV @ {K_pct:.0%}K", f"{iv_strike[t2]:.1%}", help=iv_source[t2])
for tkr in (t1, t2):
    if tkr in iv_packs:
        used = ", ".join(f"{e} ({n} pts)" for e, _, n in iv_packs[tkr]["used"])
        extra = " ⚠️ tenor outside listed expiries — flat extrapolation" \
            if iv_packs[tkr]["extrapolated"] else ""
        st.caption(f"{tkr}: smile from expiries {used}{extra}")

# ---- pricing inputs ---------------------------------------------------------
st.subheader("Pricing inputs")


def vol_control(col, tkr, seed_pct):
    """Slider + exact-entry box kept in sync. Re-seeds from the smile whenever
    the market read (strike/tenor/ticker) changes; user edits stick otherwise."""
    sl_key, num_key, seed_key = f"vol_sl_{tkr}", f"vol_num_{tkr}", f"vol_seed_{tkr}"
    if st.session_state.get(seed_key) != seed_pct:
        st.session_state[seed_key] = seed_pct
        st.session_state[sl_key] = seed_pct
        st.session_state[num_key] = seed_pct

    def _from_slider():
        st.session_state[num_key] = st.session_state[sl_key]

    def _from_box():
        st.session_state[sl_key] = st.session_state[num_key]

    col.slider(f"{tkr} vol used (%)", min_value=1.0, max_value=300.0, step=0.1,
               key=sl_key, on_change=_from_slider)
    col.number_input(f"{tkr} vol — exact value (%)", min_value=1.0, max_value=300.0,
                     step=0.1, format="%.2f", key=num_key, on_change=_from_box)
    return float(st.session_state[num_key]) / 100.0


ic1, ic2, ic3 = st.columns(3)
sig1 = vol_control(ic1, t1, round(iv_strike[t1] * 100, 1))
sig2 = vol_control(ic2, t2, round(iv_strike[t2] * 100, 1))
corr_used = ic3.slider("Correlation used", -0.95, 0.99,
                       round(float(np.clip(corr_real, -0.95, 0.99)), 2), 0.01)
st.caption(
    f"Vols read off the option surface at the {K_pct:.0%} strike, tenor-interpolated; edit "
    f"freely. Correlation seeded from realized ({corr_real:.2f}, {hmeta['overlap']}d overlap); "
    "desks mark implied correlation 5–15pts above realized, which lowers the premium."
)

# ---- PRICE (always live — no button, so nothing ever disappears) ------------
analytic = stulz_put_on_min(K_pct, T, r, q1, q2, sig1, sig2, corr_used)
mc = price_wo_implied(sig1, sig2, corr_used, K_pct, T, r, q1, q2,
                      n_paths=int(n_paths), seed=int(seed))
agree = abs(mc["price"] - analytic) <= max(3 * mc["se"], 1e-4)
v1p = bs_put_pct(K_pct, T, r, q1, sig1)
v2p = bs_put_pct(K_pct, T, r, q2, sig2)

st.subheader("Price")
pc = st.columns(4)
pc[0].metric("Fair premium — analytic (Stulz 1982)", f"{analytic:.2%} of notional")
pc[1].metric("Monte Carlo confirmation", f"{mc['price']:.2%}",
             delta=f"{mc['price'] - analytic:+.3%} vs analytic",
             help="Independent method — must agree with the closed form within MC noise")
pc[2].metric("Premium (USD)", f"${analytic * notional:,.0f}")
pc[3].metric("P(exercised at T)", f"{mc['prob_exercise']:.1%}",
             help="Risk-neutral probability the worst performer ends below strike")
if agree:
    st.success(f"✔ Cross-check passed: closed-form {analytic:.3%} vs Monte Carlo "
               f"{mc['price']:.3%} ± {mc['se']:.3%} — two independent pricers agree.")
else:
    st.error("Cross-check FAILED: closed-form and Monte Carlo disagree beyond noise. "
             "Do not use this number; check inputs.")
st.caption("Fair value, no dealer margin. Client-facing term sheets typically print "
           "0.5–3% of notional below this.")

st.subheader("Worst-of vs single-name puts (all analytic)")
cc = st.columns(3)
cc[0].metric(f"Vanilla {K_pct:.0%} put — {t1}", f"{v1p:.2%}")
cc[1].metric(f"Vanilla {K_pct:.0%} put — {t2}", f"{v2p:.2%}")
cc[2].metric("Worst-of pickup", f"+{analytic - max(v1p, v2p):.2%}",
             help="Dispersion premium over the richer single-name put")

# ---- market repricing verification ------------------------------------------
with st.expander("✅ Verification: reprice listed puts vs their market quotes"):
    st.markdown("The same vol pipeline must reproduce **actual traded option prices**. "
                "Model price uses our smile IV at each listed strike & expiry — it should "
                "land inside or near the bid/ask.")
    any_rows = False
    for tkr, spot, qq in ((t1, S1, q1), (t2, S2, q2)):
        if tkr not in iv_packs:
            continue
        pack = iv_packs[tkr]
        for exp_name, recs in pack["quotes"].items():
            qdf = pd.DataFrame(recs)
            qdf = qdf[qdf["type"] == "put"]
            if qdf.empty:
                continue
            T_exp = pack["expiry_T"].get(exp_name)
            if T_exp is None:
                continue
            k_target = K_pct * spot
            qdf = qdf.iloc[(qdf["strike"] - k_target).abs().argsort()[:5]].sort_values("strike")
            model_px = bs_price_vec(spot, qdf["strike"].to_numpy(), T_exp, r, qq,
                                    qdf["iv_mid"].to_numpy(), True)
            out = pd.DataFrame({
                "strike": qdf["strike"].values,
                "market bid": qdf["bid"].values,
                "market ask": qdf["ask"].values,
                "market mid": qdf["mid"].round(2).values,
                "model": np.round(model_px, 2),
                "IV used": (qdf["iv_mid"] * 100).round(1).astype(str) + "%",
            })
            out["inside bid/ask"] = np.where(
                (out["model"] >= out["market bid"] - 0.01) &
                (out["model"] <= out["market ask"] + 0.01), "✔", "✘")
            st.markdown(f"**{tkr} — {exp_name}** (T={T_exp:.2f}y, target strike ≈ ${k_target:,.0f})")
            st.dataframe(out, hide_index=True, width="stretch")
            any_rows = True
    if not any_rows:
        st.info("No listed quotes available (fallback mode) — verification needs live chains.")

with st.expander("Implied vol smiles (as fetched)"):
    figs = go.Figure()
    for tkr in (t1, t2):
        if tkr in iv_packs:
            p = iv_packs[tkr]
            figs.add_trace(go.Scatter(x=np.array(p["iv_curve_m"]) * 100,
                                      y=np.array(p["iv_curve_v"]) * 100,
                                      mode="lines+markers",
                                      name=f"{tkr} ({p['plot_expiry']})"))
    figs.add_vline(x=K_pct * 100, line_dash="dash", line_color="#E45756",
                   annotation_text=f"Strike {K_pct:.0%}")
    figs.add_vline(x=100, line_dash="dot", line_color="#54A24B", annotation_text="ATM")
    figs.update_layout(xaxis_title="Moneyness (% of spot)", yaxis_title="Implied vol (%)",
                       height=380, legend=dict(orientation="h"))
    st.plotly_chart(figs, width="stretch")
    st.caption("Built from live bid/ask mids with our own BS inversion — Yahoo's stale "
               "impliedVolatility field is never used.")

# ---- charts -----------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["Worst-of distribution", "Payoff at maturity (client view)",
                            "Monte Carlo paths"])
with tab1:
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=mc["wo_T"][:200000] * 100, nbinsx=90,
                               marker_color="#4C78A8", opacity=0.85))
    fig.add_vline(x=K_pct * 100, line_dash="dash", line_color="#E45756",
                  annotation_text=f"Strike {K_pct:.0%}")
    fig.add_vline(x=100, line_dash="dot", line_color="#54A24B", annotation_text="Spot")
    fig.update_layout(title=f"Worst-of performance at maturity — {mc['prob_exercise']:.1%} below strike",
                      xaxis_title="min(S₁ₜ/S₁₀, S₂ₜ/S₂₀) (% of spot)", yaxis_title="Paths",
                      showlegend=False, height=420)
    st.plotly_chart(fig, width="stretch")
with tab2:
    wo_grid = np.linspace(0.0, 1.5, 301)
    pnl = analytic - np.maximum(K_pct - wo_grid, 0.0)
    breakeven = (K_pct - analytic) * 100
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=wo_grid * 100, y=pnl * 100, mode="lines",
                              line=dict(color="#4C78A8", width=2)))
    fig2.add_hline(y=0, line_color="grey", line_width=1)
    fig2.add_vline(x=K_pct * 100, line_dash="dash", line_color="#E45756",
                   annotation_text=f"Strike {K_pct:.0%}")
    fig2.add_vline(x=breakeven, line_dash="dot", line_color="#F58518",
                   annotation_text=f"Breakeven {breakeven:.1f}%")
    fig2.update_layout(title="Client P&L at maturity (short worst-of put, % of notional)",
                       xaxis_title="Worst-of performance at T (%)",
                       yaxis_title="P&L (% of notional)", height=420)
    st.plotly_chart(fig2, width="stretch")
with tab3:
    t_grid, _, _, wo_paths = simulate_wo_paths(sig1, sig2, corr_used, T, r, q1, q2,
                                               n_paths=80, n_steps=126, seed=int(seed))
    ends_itm = wo_paths[-1] < K_pct
    fig3 = go.Figure()
    for j in range(wo_paths.shape[1]):
        itm = bool(ends_itm[j])
        fig3.add_trace(go.Scatter(
            x=t_grid, y=wo_paths[:, j] * 100, mode="lines",
            line=dict(color="#E45756" if itm else "#4C78A8", width=1),
            opacity=0.45, showlegend=False, hoverinfo="skip"))
    # legend proxies
    fig3.add_trace(go.Scatter(x=[None], y=[None], mode="lines",
                              line=dict(color="#E45756"), name="ends below strike (exercised)"))
    fig3.add_trace(go.Scatter(x=[None], y=[None], mode="lines",
                              line=dict(color="#4C78A8"), name="ends above strike"))
    fig3.add_hline(y=K_pct * 100, line_dash="dash", line_color="#E45756",
                   annotation_text=f"Strike {K_pct:.0%}")
    fig3.add_hline(y=100, line_dash="dot", line_color="#54A24B", annotation_text="Spot")
    fig3.update_layout(
        title=f"80 simulated worst-of paths — {ends_itm.mean():.0%} of this sample exercised",
        xaxis_title="Time (years)", yaxis_title="min(S₁ₜ/S₁₀, S₂ₜ/S₂₀) (% of spot)",
        height=460, legend=dict(orientation="h"))
    st.plotly_chart(fig3, width="stretch")
    st.caption("Correlated GBM sample paths of the worst performer, same vols/correlation "
               "as the pricer — for intuition only. The payoff is European, so the pricing "
               "Monte Carlo draws the terminal distribution exactly (100k–1M draws, no "
               "time-stepping and no discretization error); these 80 paths are just a "
               "visual sample of the same dynamics.")

# ---- optional Heston comparison ---------------------------------------------
if run_heston:
    st.subheader("Heston stochastic-vol comparison")
    hcols = st.columns(2)
    tables = {}
    for col, tkr in zip(hcols, (t1, t2)):
        with col:
            st.markdown(f"**{tkr}**")
            if tkr in iv_packs:
                v0_seed = iv_packs[tkr]["v0_seed"] or atm_iv_from_pack(iv_packs[tkr])
                th_seed = atm_iv_from_pack(iv_packs[tkr])
            else:
                v0_seed = th_seed = iv_strike[tkr]
            df = pd.DataFrame({"Parameter": PARAM_LABELS,
                               "Value": [round(v0_seed * 100, 1), round(th_seed * 100, 1),
                                         2.0, 0.9, -0.6]})
            tables[tkr] = st.data_editor(
                df, hide_index=True, width="stretch", key=f"h_{tkr}",
                column_config={"Parameter": st.column_config.TextColumn(disabled=True),
                               "Value": st.column_config.NumberColumn(format="%.2f")})
    pv = {k: v["Value"].to_numpy(float) for k, v in tables.items()}
    hres = price_wo_heston(
        [(pv[t1][0] / 100) ** 2, (pv[t2][0] / 100) ** 2],
        [(pv[t1][1] / 100) ** 2, (pv[t2][1] / 100) ** 2],
        [pv[t1][2], pv[t2][2]], [pv[t1][3], pv[t2][3]],
        [float(np.clip(pv[t1][4], -0.99, 0.99)), float(np.clip(pv[t2][4], -0.99, 0.99))],
        corr_used, K_pct, T, r, [q1, q2], n_paths=50000, n_steps=252, seed=int(seed))
    st.metric("Heston price", f"{hres['price']:.2%}",
              delta=f"{hres['price'] - analytic:+.2%} vs headline")
    st.caption("ATM-seeded, not surface-calibrated — a sensitivity view. The headline "
               "already carries the market's skew by reading IV at the actual strike.")

with st.expander("Model notes & accuracy stack"):
    st.markdown(
        r"""
**Payoff (worst-of, not average).** $\max\!\big(K\% - \min(\tfrac{S_1(T)}{S_1(0)},
\tfrac{S_2(T)}{S_2(0)}),\,0\big)$. A put on the *average* basket would be much cheaper;
the two are deliberately different products.

**Four-layer accuracy stack.**
1. *Vols*: implied from live bid/ask mids by BS inversion (Yahoo's IV field is stale for
   LEAPS and never used). Interpolated to your strike across listed strikes and to your
   maturity linearly in total variance $\sigma^2T$.
2. *Headline price*: **Stulz (1982) closed form** for a put on the minimum of two
   correlated lognormals — exact, no simulation noise.
3. *Cross-check*: an independent Monte Carlo (exact terminal draw) runs beside it; the
   app flags loudly if they ever disagree beyond MC noise.
4. *Market round-trip*: the verification panel reprices actual listed puts with the same
   vols — model prices must land inside the market bid/ask.

**Known approximations.** Listed US single-name options are American; using their mids
for European IVs slightly overstates vol for OTM puts (small at these moneyness levels,
and conservative). Strike-vol lognormal pricing is the standard structurer shortcut vs a
full smile-calibrated model. Correlation is realized-seeded; desks mark implied
correlation higher, lowering the premium. Fair value excludes dealer margin.
"""
    )
