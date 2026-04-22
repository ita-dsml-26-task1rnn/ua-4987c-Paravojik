# Task 1 — One-step Time Series Forecasting with LSTM (Keras)

## Goal
Implement a correct time-series forecasting pipeline:
- convert series to sliding windows
- split by time (no shuffle!) to avoid leakage
- train an LSTM model
- evaluate on test set using MAE and RMSE

## Project structure
```
src/
  simple_rnn_forecast.py
  simple_rnn_forecast_solution.py   # mentor-only

tests/
  test_simple_forecast.py
```

## How to run
Install dependencies:
```bash
pip install -r requirements.txt
```

Run unit tests:
```bash
python -m unittest -q
```

Run optional slow quality test:
```bash
RUN_SLOW_TESTS=1 python -m unittest -q
```

## Notes
- **Do not shuffle** the time series when splitting.
- Model quality is not heavily tested in CI to avoid flaky tests.
- Use `python src/simple_rnn_forecast.py` to run the demo plot (after implementing functions).
