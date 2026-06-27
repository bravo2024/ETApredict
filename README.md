# ETApredict

Shipment ETA prediction from lag features, with multi-horizon evaluation.

## Quickstart

```bash
pip install -r requirements.txt
python train.py
pytest -q
streamlit run app.py
```

Synthetic data is included so the pipeline runs immediately. To use real shipment or taxi data, place a CSV in `data/raw/` and call `load_real()`.

## Approach

- Ridge regression on lagged values with sin/cos calendar encoding
- Multi-horizon evaluation: the model reports accuracy at H=1, H=6, H=24 steps ahead separately
- Standardised features, chronological split with a lag gap, and multi-horizon target shifting

Most forecasting demos report one-step-ahead error, which is nowcasting rather than forecasting. For an ETA system, accuracy at longer horizons matters more. This project reports both so the decay is visible.

## Project structure

```
src/
  core.py          RidgeRegression, Standardizer, metrics
  data.py          synthetic generator and CSV loader
  model.py         feature engineering with multi-horizon evaluation
  evaluate.py      metric persistence
  persist.py       model save/load
train.py           training entry point
app.py             Streamlit dashboard
```

## Notes

The Ridge model is a baseline, not a production solution. A production ETA system would incorporate route features, weather data, real-time traffic, and a non-linear model. This project focuses on getting the evaluation protocol right — chronological split, multi-horizon reporting, no lookahead leakage.
