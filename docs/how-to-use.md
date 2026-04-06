# How To Use These Templates

Each folder is a PyP Quant project skeleton. Copy the files into a new PyP Quant project and run training from the dashboard.

## Required Files

- `strategy.py` contains `train()` and `predict()`.
- `quant.config.json` declares pair, timeframe, model family, artifact format, and requirements.
- `README.md` explains the project intent and tuning knobs.

## Recommended Workflow

1. Train the model.
2. Run PPE simulation.
3. Inspect trade count, drawdown, profit factor, win rate, and session behavior.
4. Adjust label thresholds or signal gates.
5. Train again.
6. Deploy only after out-of-sample validation.

## Common Tuning Knobs

- `threshold`: minimum forward move for UP/DOWN labels.
- `horizon`: bars ahead used for training labels.
- `min_confidence`: inference confidence gate.
- `lookback`: bars required before prediction.
- `sl_percent` and `tp_percent`: stop-loss and take-profit defaults for simulations.
