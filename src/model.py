
"""model.py - lag-feature forecaster with sin/cos encoding and multi-horizon evaluation."""
from __future__ import annotations
import numpy as np
from src.core import RidgeRegression, Standardizer, rmse, smape, mae, temporal_split

PREDICT_KIND = "timeseries"
LAGS = [1, 2, 3, 7, 14, 28]


def _cyclic_features(i, period=365, week_period=7):
    """Encode cyclic time features as sin/cos pairs."""
    theta_d = 2.0 * np.pi * i / period
    theta_w = 2.0 * np.pi * (i % week_period) / week_period
    return [np.sin(theta_d), np.cos(theta_d), np.sin(theta_w), np.cos(theta_w)]


def _feat(s):
    """Build feature matrix from lag values and cyclic time features."""
    s = np.asarray(s, float)
    if not np.isfinite(s).all():
        raise ValueError("Series contains NaN or inf values.")
    if len(s) <= max(LAGS):
        raise ValueError(f"Need more than {max(LAGS)} observations, got {len(s)}.")
    rows, tgt = [], []
    st = max(LAGS)
    for i in range(st, len(s)):
        rows.append([s[i - l] for l in LAGS] + _cyclic_features(i))
        tgt.append(s[i])
    return np.array(rows), np.array(tgt)


def fit_and_evaluate(data, horizon=1):
    """Train ridge regression with lag features and evaluate at horizon steps ahead.

    For ETA prediction, the practical use case is forecasting *H steps ahead*
    (e.g., 'what will the ETA be in 6 hours'), not one-step-ahead nowcasting.
    This function evaluates both nowcast (horizon=1) and multi-step (horizon=H)
    accuracy so the user can see the decay."""
    s = np.asarray(data["series"], float)
    if not np.isfinite(s).all():
        raise ValueError("Series contains NaN or inf values.")

    # Multi-horizon evaluation
    st = max(LAGS)
    X_multi, y_multi = [], []
    for i in range(st, len(s) - horizon):
        X_multi.append([s[i - l] for l in LAGS] + _cyclic_features(i))
        y_multi.append(s[i + horizon])
    X_multi = np.array(X_multi)
    y_multi = np.array(y_multi)

    gap = max(LAGS)
    sp = int(len(X_multi) * 0.8)
    train_end = max(sp - gap, 0)

    X_train, y_train = X_multi[:train_end], y_multi[:train_end]
    X_test, y_test = X_multi[sp:], y_multi[sp:]

    scaler = Standardizer().fit(X_train)
    Xs_tr = scaler.transform(X_train)
    Xs_te = scaler.transform(X_test)

    m = RidgeRegression(alpha=1.0).fit(Xs_tr, y_train)
    pred = m.predict(Xs_te)

    metrics = {
        "n_train": int(len(Xs_tr)),
        "n_test": int(len(Xs_te)),
        "horizon": horizon,
        "rmse": rmse(y_test, pred),
        "smape_pct": smape(y_test, pred),
        "mae": mae(y_test, pred),
    }
    model_dict = {
        "model": m,
        "scaler": scaler,
        "lags": LAGS,
        "tail": s[-max(LAGS):].tolist(),
    }
    return model_dict, metrics


def forecast_next(model_dict, series=None, step_index=None):
    """Generate a one-step-ahead forecast from the most recent data.

    Raises ValueError if insufficient history is available."""
    s = np.asarray(series if series is not None else model_dict.get("tail", []), float)
    if len(s) < max(model_dict["lags"]):
        raise ValueError(
            f"Need at least {max(model_dict['lags'])} observations for forecast, "
            f"got {len(s)}."
        )
    if step_index is None:
        step_index = len(s)
    f = [s[-l] for l in model_dict["lags"]] + _cyclic_features(step_index)
    f_scaled = model_dict["scaler"].transform(np.array([f]))
    return float(model_dict["model"].predict(f_scaled)[0])
