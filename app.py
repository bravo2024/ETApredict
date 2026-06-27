"""app.py – ETApredict: comprehensive demand forecasting & ETA dashboard."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy import stats

from src.data import fetch_nyc_taxi, make_synthetic
from src.model import fit_and_evaluate, forecast_next

st.set_page_config(page_title="ETApredict | Demand & ETA Forecasting",
                   layout="wide", page_icon="🚕")

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙ Configuration")
    data_source = st.radio("Dataset",
        ["NYC Taxi TLC — NAB (live)", "Synthetic (demo)"], index=0)
    horizon = st.slider("Forecast horizon (hours)", 6, 72, 24, step=6)
    season_period = st.slider("Seasonal period (hours)", 12, 48, 24, step=1)
    n_folds = st.slider("Walk-forward folds", 3, 8, 5, step=1)
    st.divider()
    st.caption("NYC Taxi TLC pickups (Jul 2014 – Jan 2015) via Numenta NAB. 30-min → hourly aggregation.")
    st.code("streamlit run app.py", language="bash")

# ── Load data ──────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Fetching NYC taxi data...")
def _load(source: str):
    if "NYC" in source:
        try:
            r = fetch_nyc_taxi()
            return r["series"], r["source"], "Hourly pickups", "hours"
        except Exception:
            pass
    s = make_synthetic()
    return s["series"], "Synthetic demand series", "Demand", "steps"

series, source_label, y_label, freq_label = _load(data_source)
series_key = tuple(float(x) for x in series)

# ── Train & helpers ────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Training ridge model...")
def _train(key: tuple):
    return fit_and_evaluate({"series": np.asarray(key, float)})

@st.cache_data(show_spinner="Computing test residuals...")
def _residuals(key: tuple, n: int = 80):
    s = np.asarray(key, float)
    if len(s) < 35:
        return np.zeros(1)
    m, _ = fit_and_evaluate({"series": s})
    start = max(29, len(s) - n)
    out = []
    for t in range(start, len(s)):
        try:
            out.append(float(s[t]) - forecast_next(m, list(s[:t]), step_index=t))
        except Exception:
            pass
    return np.asarray(out) if out else np.zeros(1)

@st.cache_data(show_spinner="Generating forecast...")
def _forecast(key: tuple, h: int):
    m, _ = fit_and_evaluate({"series": np.asarray(key, float)})
    tail = list(np.asarray(key, float))
    out = []
    for _ in range(h):
        v = max(0.0, forecast_next(m, tail, step_index=len(tail)))
        out.append(v); tail.append(v)
    return np.asarray(out)

model, metrics = _train(series_key)
walk_resid = _residuals(series_key)
preds = _forecast(series_key, horizon)

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("🚕 ETApredict")
st.caption(f"{source_label}  |  {len(series):,} observations  |  frequency: {freq_label}")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Observations", f"{len(series):,}")
c2.metric("Train rows", f"{metrics['n_train']:,}")
c3.metric("Test rows", f"{metrics['n_test']:,}")
c4.metric("RMSE", f"{metrics['rmse']:.2f}")
c5.metric("SMAPE", f"{metrics['smape_pct']:.2f}%")

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Data Explorer", "📐 Stationarity & ACF",
    "🌀 Decomposition", "🔄 Walk-Forward CV", "🔮 Forecast"])

# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 — Data Explorer
# ════════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Raw Series & Descriptive Statistics")

    win = min(24, max(4, len(series) // 8))
    rm = np.convolve(series, np.ones(win) / win, mode="valid")
    rs = np.array([series[i:i + win].std(ddof=1) for i in range(len(series) - win + 1)])
    xr = np.arange(win - 1, len(series))

    fig, axes = plt.subplots(2, 1, figsize=(13, 5), sharex=True)
    axes[0].plot(series, color="#1d4ed8", lw=0.6, alpha=0.8, label="Raw demand")
    axes[0].plot(xr, rm, color="#dc2626", lw=1.5, label=f"{win}-pt rolling mean")
    axes[0].set_ylabel(y_label); axes[0].legend(fontsize=8)
    axes[0].set_title("Hourly taxi pickups with rolling mean")
    axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    axes[1].plot(xr, rs, color="#7c3aed", lw=1.2, label=f"{win}-pt rolling std (ddof=1)")
    axes[1].set_xlabel(f"Time ({freq_label})"); axes[1].set_ylabel("Std dev"); axes[1].legend(fontsize=8)
    axes[1].set_title("Rolling standard deviation — peaks reveal demand volatility windows")
    fig.tight_layout(); st.pyplot(fig, use_container_width=True)

    col_s, col_h = st.columns(2)
    with col_s:
        st.markdown("**Descriptive Statistics**")
        sk = float(stats.skew(series)); ku = float(stats.kurtosis(series))
        cv = series.std(ddof=1) / abs(series.mean()) * 100 if series.mean() != 0 else float("nan")
        q25, q75 = np.percentile(series, [25, 75])
        st.dataframe(pd.DataFrame({
            "Statistic": ["N", "Mean", "Median", "Std (σ, ddof=1)", "Min", "Max",
                          "Q25", "Q75", "IQR", "Skewness", "Excess Kurtosis", "CV (%)"],
            "Value": [f"{len(series):,}", f"{series.mean():.2f}", f"{np.median(series):.2f}",
                      f"{series.std(ddof=1):.2f}", f"{series.min():.0f}", f"{series.max():.0f}",
                      f"{q25:.2f}", f"{q75:.2f}", f"{q75 - q25:.2f}",
                      f"{sk:.4f}", f"{ku:.4f}", f"{cv:.2f}%"]
        }), use_container_width=True, hide_index=True)
        st.markdown("""
**Skewness > 0** → right tail (occasional surge hours — NYE, rush events).
**Excess kurtosis > 0** → heavy tails; extreme pickups more common than Gaussian.
**IQR** = Q75 − Q25 — robust spread measure (resistant to outliers).
**CV = σ/μ × 100** — relative variability across hours/days.
""")
    with col_h:
        st.markdown("**Distribution Histogram**")
        fig_h, ax_h = plt.subplots(figsize=(6, 4))
        ax_h.hist(series, bins=60, color="#1d4ed8", alpha=0.7, edgecolor="white")
        ax_h.axvline(series.mean(), color="#dc2626", ls="--", lw=1.5, label=f"Mean={series.mean():.0f}")
        ax_h.axvline(np.median(series), color="#f59e0b", ls="--", lw=1.5, label=f"Median={np.median(series):.0f}")
        ax_h.set_xlabel(y_label); ax_h.set_ylabel("Count"); ax_h.legend(fontsize=8)
        ax_h.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
        fig_h.tight_layout(); st.pyplot(fig_h, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 — Stationarity & ACF/PACF
# ════════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Stationarity Test & Autocorrelation Structure")

    try:
        from statsmodels.tsa.stattools import adfuller, acf as sm_acf, pacf as sm_pacf
        adf = adfuller(series, autolag="AIC")
        adf_stat, adf_p, adf_lag, adf_n, adf_crit = adf[0], adf[1], adf[2], adf[3], adf[4]
        sm_ok = True
    except Exception as exc:
        st.error(f"statsmodels unavailable: {exc}"); sm_ok = False

    if sm_ok:
        st.markdown("#### Augmented Dickey-Fuller (ADF) Unit Root Test")
        st.latex(r"\Delta y_t = \alpha + \beta t + \gamma y_{t-1} + \sum_{i=1}^{k}\delta_i\,\Delta y_{t-i} + \varepsilon_t")
        st.markdown(r"""
**H₀**: γ = 0 — unit root, **non-stationary** (mean/variance change over time)
**H₁**: γ < 0 — no unit root, **stationary**
Reject H₀ when ADF statistic < MacKinnon (1994) critical value at chosen confidence.
Lag count *k* chosen by AIC to whiten regression residuals.
""")
        col1, col2 = st.columns(2)
        with col1:
            verdict = "✅ Stationary — reject H₀" if adf_p < 0.05 else "❌ Non-stationary — fail to reject H₀"
            st.dataframe(pd.DataFrame({
                "Metric": ["ADF Statistic", "p-value", "Lags used (AIC)", "Obs in test", "Verdict"],
                "Value": [f"{adf_stat:.4f}", f"{adf_p:.6f}", str(adf_lag), str(adf_n), verdict]
            }), use_container_width=True, hide_index=True)
        with col2:
            st.dataframe(pd.DataFrame({
                "Confidence": ["1% (MacKinnon)", "5% (MacKinnon)", "10% (MacKinnon)"],
                "Critical Value": [f"{adf_crit['1%']:.4f}", f"{adf_crit['5%']:.4f}", f"{adf_crit['10%']:.4f}"],
                "ADF < CV?": ["✅ Reject H₀" if adf_stat < adf_crit[k] else "❌ No" for k in ["1%", "5%", "10%"]]
            }), use_container_width=True, hide_index=True)

        if adf_p >= 0.05:
            st.info("Series appears non-stationary. Consider first-differencing: Δyₜ = yₜ − yₜ₋₁ before fitting ARIMA.")

        st.markdown("---")
        st.markdown("#### ACF & PACF — Autocorrelation Structure")
        st.markdown(r"""
**ACF(k)** = Corr(yₜ, yₜ₋ₖ) — total autocorrelation at lag k (includes indirect paths through lags 1…k-1).
**PACF(k)** — direct autocorrelation at lag k, controlling for 1…k-1.
Confidence bands: **±1.96/√n** — bars exceeding the band are significant at α = 0.05.
For demand data: expect ACF spikes at lags 24 (daily), 168 (weekly).
""")
        nlags = min(72, len(series) // 4)
        acf_v = sm_acf(series, nlags=nlags, fft=True)
        pacf_v = sm_pacf(series, nlags=nlags, method="ywm")
        conf = 1.96 / np.sqrt(len(series))
        lx = np.arange(len(acf_v))

        fig_a, ax_a = plt.subplots(1, 2, figsize=(14, 4))
        for ax, vals, title, ylabel in [
            (ax_a[0], acf_v, "ACF — total autocorrelation (spikes at 24h = daily seasonality)", "ACF"),
            (ax_a[1], pacf_v, "PACF — direct lag effects (Box-Jenkins AR order selection)", "PACF"),
        ]:
            colors = ["#dc2626" if abs(v) > conf else "#93c5fd" for v in vals]
            ax.bar(lx, vals, color=colors, width=0.6)
            ax.axhline(conf, ls="--", color="gray", lw=0.8, label=f"±{conf:.3f}")
            ax.axhline(-conf, ls="--", color="gray", lw=0.8)
            ax.axhline(0, color="black", lw=0.5)
            ax.set_xlabel("Lag (hours)"); ax.set_ylabel(ylabel); ax.set_title(title); ax.legend(fontsize=7)
        fig_a.tight_layout(); st.pyplot(fig_a, use_container_width=True)

        st.markdown("""
**Demand-specific interpretation:**
- ACF spike at lag 24 → daily cycle (rush hour pattern repeats every 24h)
- ACF spike at lag 168 → weekly cycle (weekday vs weekend pattern)
- Slow ACF decay → non-stationarity; need differencing
- PACF cuts off at lag p → AR(p) component; our model uses lags [1,2,3,7,14,28]

**Box-Jenkins Model Identification:**

| ACF | PACF | Model |
|---|---|---|
| Cuts off at q | Tails off | MA(q) |
| Tails off | Cuts off at p | AR(p) |
| Both tail off | Both tail off | ARMA(p, q) |
| Slow decay | — | Difference → ARIMA |
| Spikes at 24, 48 | Spikes at 24 | Seasonal SARIMA(P,D,Q)₂₄ |
""")

# ════════════════════════════════════════════════════════════════════════════════
# TAB 3 — Decomposition
# ════════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Classical Decomposition & Frequency Analysis")
    st.markdown(r"""
**Additive model:** $y_t = T_t + S_t + R_t$

**T** = trend (slow long-run changes in overall demand level)
**S** = seasonal (recurring daily/weekly demand patterns)
**R** = residual (shocks — events, weather, strikes)

Additive appropriate when seasonal swings have constant amplitude.
""")

    if len(series) < 2 * season_period + 1:
        st.warning(f"Series too short for period={season_period}. Need ≥ {2 * season_period + 1} obs.")
    else:
        try:
            from statsmodels.tsa.seasonal import seasonal_decompose
            dec = seasonal_decompose(series, model="additive", period=season_period,
                                     extrapolate_trend="freq")
            T, S, R = dec.trend, dec.seasonal, dec.resid

            var_r = np.nanvar(R)
            fs = max(0.0, 1 - var_r / np.nanvar(S + R)) if np.nanvar(S + R) > 0 else 0.0
            ft = max(0.0, 1 - var_r / np.nanvar(T + R)) if np.nanvar(T + R) > 0 else 0.0

            fig_d, axd = plt.subplots(4, 1, figsize=(13, 9), sharex=True)
            pairs = [
                (axd[0], series, "Observed", "#1d4ed8", 0.6),
                (axd[1], T, "Trend (T)", "#dc2626", 1.5),
                (axd[2], S, f"Seasonal (S, period={season_period}h)", "#16a34a", 1.0),
                (axd[3], R, "Residual (R)", "#7c3aed", 0.7),
            ]
            for ax, vals, lbl, col, lw in pairs:
                ax.plot(vals, color=col, lw=lw); ax.set_ylabel(lbl)
                if lbl != "Observed":
                    ax.axhline(0, color="gray", lw=0.5, ls="--")
                ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
                ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
            axd[3].set_xlabel(f"Time ({freq_label})")
            fig_d.suptitle(f"Additive Decomposition — period={season_period}h (NYC Taxi demand)", y=1.01)
            fig_d.tight_layout(); st.pyplot(fig_d, use_container_width=True)

            mc1, mc2 = st.columns(2)
            mc1.metric("Seasonality Strength Fₛ", f"{fs:.4f}",
                       help="1 - Var(R)/Var(S+R). > 0.64 = strong (Wang et al. 2006)")
            mc2.metric("Trend Strength F_T", f"{ft:.4f}",
                       help="1 - Var(R)/Var(T+R). > 0.64 = strong trend")
        except Exception as exc:
            st.error(f"Decomposition failed: {exc}")

        st.markdown("---")
        st.markdown("#### Periodogram — Power Spectral Density via FFT")
        st.markdown(r"""
$$I(\omega_j) = \frac{1}{n}\left|\sum_{t=0}^{n-1} y_t\, e^{-2\pi i\, \omega_j t}\right|^2$$
Peaks reveal dominant cycle frequencies. Period = 1/frequency (in hours).
Expected dominant periods: **24h** (daily commute cycle), **168h** (weekly pattern).
""")
        fv = np.fft.rfft(series - series.mean())
        fr = np.fft.rfftfreq(len(series))[1:]
        pw = (np.abs(fv) ** 2)[1:]
        per = 1.0 / fr

        fig_fft, ax_fft = plt.subplots(figsize=(13, 3))
        ax_fft.plot(per, pw, color="#1d4ed8", lw=0.8)
        ax_fft.set_xscale("log")
        ax_fft.set_xlabel(f"Period ({freq_label}/cycle) — log scale")
        ax_fft.set_ylabel("Power"); ax_fft.set_title("Periodogram — peaks at 24h and 168h expected for taxi demand")
        for ref_per, lbl in [(24, "24h daily"), (168, "168h weekly")]:
            ax_fft.axvline(ref_per, color="#f59e0b", lw=1.0, ls=":", alpha=0.8)
            ax_fft.text(ref_per, pw.max() * 0.9, f" {lbl}", fontsize=7, color="#f59e0b")
        top_idx = np.argsort(pw)[-5:][::-1]
        for i in top_idx:
            ax_fft.axvline(per[i], color="#dc2626", lw=1.0, ls="--", alpha=0.5)
        fig_fft.tight_layout(); st.pyplot(fig_fft, use_container_width=True)

        st.dataframe(
            pd.DataFrame([(f"{per[i]:.1f}h", f"{pw[i]:.3e}") for i in top_idx[:5]],
                         columns=["Dominant Period", "Power"]),
            use_container_width=True, hide_index=True)

# ════════════════════════════════════════════════════════════════════════════════
# TAB 4 — Walk-Forward CV
# ════════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("Walk-Forward (Expanding-Window) Cross-Validation")
    st.markdown(r"""
**Walk-forward** is the only correct CV strategy for time series.
K-fold shuffling destroys temporal ordering → future data leaks into training → optimistically biased metrics.

Train on [1..t] → evaluate on [t+gap..t+gap+n_test].
Gap = max(lags) = 28 prevents lag-feature leakage across the boundary.

$$\text{SMAPE} = \frac{200}{n}\sum_t\frac{|\hat{y}_t - y_t|}{|\hat{y}_t|+|y_t|}$$
""")

    min_size = max(100, 29 * 3)
    if len(series) < min_size + 20:
        st.warning(f"Series too short for walk-forward CV (need ≥ {min_size + 20} obs).")
    else:
        with st.spinner("Running expanding-window cross-validation…"):
            incr = max(1, (len(series) - min_size) // n_folds)
            rows = []
            for fold in range(n_folds):
                fe = min(len(series), min_size + (fold + 1) * incr)
                try:
                    _, fm = fit_and_evaluate({"series": series[:fe]})
                    rows.append({
                        "Fold": fold + 1, "Train size": fe,
                        "RMSE": round(fm["rmse"], 2),
                        "MAE": round(fm.get("mae", float("nan")), 2),
                        "SMAPE (%)": round(fm["smape_pct"], 4),
                    })
                except Exception:
                    rows.append({"Fold": fold + 1, "Train size": fe,
                                 "RMSE": float("nan"), "MAE": float("nan"), "SMAPE (%)": float("nan")})

        fold_df = pd.DataFrame(rows)
        st.dataframe(fold_df, use_container_width=True, hide_index=True)

        fig_wf, axw = plt.subplots(1, 2, figsize=(12, 4))
        fols = fold_df["Fold"]
        axw[0].bar(fols, fold_df["RMSE"], color="#1d4ed8", alpha=0.8)
        axw[0].set_xlabel("Fold"); axw[0].set_ylabel("RMSE")
        axw[0].set_title("RMSE by fold (expanding training window)"); axw[0].set_xticks(fols)
        axw[1].bar(fols, fold_df["SMAPE (%)"], color="#dc2626", alpha=0.8)
        axw[1].set_xlabel("Fold"); axw[1].set_ylabel("SMAPE (%)")
        axw[1].set_title("SMAPE by fold"); axw[1].set_xticks(fols)
        fig_wf.tight_layout(); st.pyplot(fig_wf, use_container_width=True)

        st.markdown("---")
        st.markdown("#### Ljung-Box Q-Test — Residual Autocorrelation")
        st.markdown(r"""
**H₀**: residuals are iid white noise (no autocorrelation at any lag).
$$Q_m = n(n+2)\sum_{k=1}^{m}\frac{\hat{\rho}_k^2}{n-k} \;\overset{H_0}{\sim}\; \chi^2(m)$$
Reject H₀ at α = 0.05 → remaining structure in residuals → model under-specified.
""")
        if len(walk_resid) > 10:
            try:
                from statsmodels.stats.diagnostic import acorr_ljungbox
                lb = acorr_ljungbox(walk_resid, lags=[10, 20, 30], return_df=True)
                lb.index = [10, 20, 30]; lb.columns = ["Q-statistic", "p-value"]
                st.dataframe(lb.round(4), use_container_width=True)
                if (lb["p-value"] < 0.05).any():
                    st.warning("p < 0.05 → autocorrelated residuals; consider more lags or SARIMA.")
                else:
                    st.success("All p ≥ 0.05 → residuals consistent with white noise. ✓")
            except Exception as exc:
                st.error(f"Ljung-Box failed: {exc}")
        else:
            st.info("Too few residuals for Ljung-Box test.")

        st.markdown("---")
        st.markdown("#### Residual Diagnostics")
        if len(walk_resid) > 5:
            fig_r, axr = plt.subplots(1, 2, figsize=(12, 4))
            axr[0].hist(walk_resid, bins=min(30, max(5, len(walk_resid) // 3)),
                        color="#7c3aed", alpha=0.7, edgecolor="white")
            axr[0].axvline(0, color="black", lw=1.0, ls="--")
            axr[0].axvline(walk_resid.mean(), color="#dc2626", lw=1.2, ls="--",
                           label=f"Mean={walk_resid.mean():.2f}")
            axr[0].set_xlabel("Residual (pickups)"); axr[0].set_ylabel("Count"); axr[0].legend(fontsize=8)
            axr[0].set_title("Residual histogram — centred at 0 means unbiased predictions")
            axr[1].scatter(np.arange(len(walk_resid)), walk_resid, s=8, color="#7c3aed", alpha=0.6)
            axr[1].axhline(0, color="black", lw=1.0, ls="--")
            axr[1].set_xlabel("Test observation index"); axr[1].set_ylabel("Residual (pickups)")
            axr[1].set_title("Residuals vs time — random scatter = well-specified")
            fig_r.tight_layout(); st.pyplot(fig_r, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════════
# TAB 5 — Forecast
# ════════════════════════════════════════════════════════════════════════════════
with tab5:
    st.subheader(f"{horizon}-Hour Demand Forecast + Bootstrap Prediction Intervals (90%)")
    st.markdown(r"""
**Bootstrap PI construction (non-parametric):**
1. Collect signed walk-forward residuals: $e_t = y_t - \hat{y}_t$
2. Resample *B = 500* paths of length *h* with replacement
3. 90% PI at step h: $[Q_{0.05}(\hat{y}_h + e^{(b)}_h),\; Q_{0.95}(\hat{y}_h + e^{(b)}_h)]$

Captures asymmetric, heavy-tailed uncertainty without assuming Gaussian errors.
""")

    rng = np.random.default_rng(42)
    resid_pool = walk_resid if len(walk_resid) >= 5 else rng.normal(0, metrics["rmse"], 200)
    n_boot = 500
    boot = rng.choice(resid_pool, size=(n_boot, horizon), replace=True)
    boot_preds = np.maximum(0, preds[np.newaxis, :] + boot)
    lo90 = np.percentile(boot_preds, 5, axis=0)
    hi90 = np.percentile(boot_preds, 95, axis=0)

    fig_fc, axes_fc = plt.subplots(1, 2, figsize=(14, 4),
                                    gridspec_kw={"width_ratios": [3, 1]})
    hist_n = min(168, len(series))
    x_h = np.arange(len(series) - hist_n, len(series))
    x_f = np.arange(len(series), len(series) + horizon)

    axes_fc[0].plot(x_h, series[-hist_n:], color="#1d4ed8", lw=1.2, label=f"Last {hist_n}h observed")
    axes_fc[0].plot(x_f, preds, color="#dc2626", lw=2.0, ls="--", label="Point forecast")
    axes_fc[0].fill_between(x_f, lo90, hi90, color="#dc2626", alpha=0.2, label="90% Bootstrap PI")
    axes_fc[0].axvline(len(series) - 1, color="gray", lw=0.8, ls=":")
    axes_fc[0].set_xlabel(f"Time ({freq_label})")
    axes_fc[0].set_ylabel(y_label)
    axes_fc[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    axes_fc[0].legend(fontsize=8)
    axes_fc[0].set_title(f"Last {hist_n}h history + {horizon}h forecast")

    axes_fc[1].barh(np.arange(horizon), preds[::-1], color="#dc2626", alpha=0.7)
    axes_fc[1].set_yticks(np.arange(horizon))
    axes_fc[1].set_yticklabels([f"h+{i + 1}" for i in range(horizon)][::-1], fontsize=7)
    axes_fc[1].set_xlabel(y_label)
    axes_fc[1].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    axes_fc[1].set_title("Step-by-step")

    fig_fc.tight_layout(); st.pyplot(fig_fc, use_container_width=True)

    st.markdown("---")
    st.markdown("#### Horizon Decay — Forecast Uncertainty by Step Ahead")
    pi_width = hi90 - lo90
    fig_hd, ax_hd = plt.subplots(figsize=(10, 3))
    ax_hd.plot(np.arange(1, horizon + 1), pi_width, "o-", color="#7c3aed", ms=5)
    ax_hd.set_xlabel("Step ahead (h)"); ax_hd.set_ylabel("90% PI width (pickups)")
    ax_hd.set_title("Horizon decay — uncertainty compounds as recursive forecast substitutes predicted for actual values")
    ax_hd.grid(True, alpha=0.3)
    ax_hd.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    fig_hd.tight_layout(); st.pyplot(fig_hd, use_container_width=True)

    st.markdown("---")
    fc_df = pd.DataFrame({
        "Step (h)": np.arange(1, horizon + 1),
        "Forecast": np.round(preds, 0).astype(int),
        "Lower 90% PI": np.round(lo90, 0).astype(int),
        "Upper 90% PI": np.round(hi90, 0).astype(int),
        "PI Width": np.round(pi_width, 0).astype(int),
    })
    st.dataframe(fc_df, use_container_width=True, hide_index=True)
    st.download_button("⬇ Download forecast CSV",
                       fc_df.to_csv(index=False).encode(),
                       "etapredict_forecast.csv", "text/csv")

    st.markdown("---")
    st.markdown(f"""
**Model:** Ridge Regression on lag features
**Features:** lags [1, 2, 3, 7, 14, 28] + sin/cos cyclic encoding (daily period = {season_period}h)
**Split:** chronological 80/20 with gap = 28 steps (prevents lag-feature data leakage)
**Cyclic encoding:** sin(2πt/24), cos(2πt/24) — hour 23 and hour 0 are adjacent in feature space
**Ridge:** β = (X'X + αI)⁻¹ X'y — L2 penalty shrinks coefficients, excluding intercept
**Demand constraint:** all forecasts clipped to ≥ 0 (pickups cannot be negative)
""")
    st.json({k: round(v, 4) if isinstance(v, float) else v for k, v in metrics.items()})
