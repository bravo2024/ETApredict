"""data.py - real and synthetic demand series for ETApredict."""
from __future__ import annotations
import numpy as np
from pathlib import Path


def fetch_nyc_taxi():
    """Load NYC taxi pickup demand from Numenta Anomaly Benchmark (NAB).

    Real NYC Taxi and Limousine Commission (TLC) data covering
    Jul 2014 – Jan 2015 at 30-minute resolution.  Aggregated to hourly
    pickup counts here.  Demand spikes correlate with peak commute hours,
    events, and weather — the same signal used for real-time ETA prediction
    on routing platforms.

    Reference: Laptev et al. (2015) 'Anomaly Detection: A Survey';
    NAB GitHub: github.com/numenta/NAB.
    """
    import pandas as pd
    url = ("https://raw.githubusercontent.com/numenta/NAB/master/"
           "data/realKnownCause/nyc_taxi.csv")
    df = pd.read_csv(url, parse_dates=["timestamp"])
    df = df.set_index("timestamp").resample("h").sum().reset_index()
    arr = df["value"].astype(float).to_numpy()
    return {
        "series": arr,
        "timestamps": df["timestamp"].astype(str).tolist(),
        "source": "NYC Taxi TLC — NAB dataset (hourly pickups)",
        "frequency": "hourly",
    }


def make_synthetic(n=730, seed=42):
    """Synthetic demand with trend, weekly/yearly seasonality (fallback)."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    s = (20 + 0.02 * t
         + 3 * np.sin(2 * np.pi * t / 7)
         + 6 * np.sin(2 * np.pi * t / 365)
         + rng.normal(0, 1, n))
    return {"series": s, "source": "Synthetic"}


def load_real(csv_name, value_col, date_col=None):
    csv_path = Path("data/raw") / csv_name
    if not csv_path.exists():
        raise FileNotFoundError(f"Data file not found: {csv_path}")
    import pandas as pd
    df = pd.read_csv(csv_path)
    if date_col:
        df = df.sort_values(date_col)
    arr = df[value_col].astype(float).to_numpy()
    if np.isnan(arr).any():
        arr = np.nan_to_num(arr, nan=float(np.nanmean(arr)))
    return {"series": arr}
