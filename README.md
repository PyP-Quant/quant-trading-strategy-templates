# PyP Quant Trading Strategy Templates

Open-source Python quant trading strategy templates for PyP Quant Mode.

This repository is a public, educational starter library for traders who want to build quantitative trading strategies with Python, validate them with PyP PPE simulation, and deploy live signals through the PyP platform.

These examples are intentionally safe starter projects. They are not financial advice, not live performance claims, and not recommendations to trade any instrument.

## What Is Inside

Each template follows the PyP Quant Mode contract:

```python
def train(data, config):
    return model, metrics

def predict(model, market_data, config):
    return {"signal": "UP|DOWN|HOLD", "confidence": 0.0, "metadata": {}}
```

Every project includes:

- `strategy.py`
- `quant.config.json`
- `README.md`

## Templates

| Template | Pair | Timeframe | Model family | Use case |
| --- | --- | --- | --- | --- |
| `eurusd-logistic-15m` | EURUSD | 15m | sklearn | Baseline directional classifier |
| `eurusd-xgboost-1h` | EURUSD | 1h | XGBoost | Feature-rich trend classifier |
| `gbpusd-breakout-rf` | GBPUSD | 30m | sklearn RandomForest | Range breakout classifier |
| `usdjpy-mean-reversion` | USDJPY | 1h | sklearn | Mean reversion baseline |
| `xauusd-regime-xgboost-v11` | XAUUSD | 1h | XGBoost | Less restrictive Au-79-style gold model |
| `xauusd-atr-breakout` | XAUUSD | 15m | custom Python | ATR breakout rules |
| `btcusdt-lightgbm-1h` | BTCUSDT | 1h | LightGBM | Crypto trend classifier |
| `ethusdt-volatility-classifier` | ETHUSDT | 30m | sklearn | Volatility regime classifier |
| `solusdt-scalp-baseline` | SOLUSDT | 1m | custom Python | High-volatility scalp baseline |
| `onnx-export-sklearn-starter` | EURUSD | 1h | sklearn to ONNX | ONNX export starter |
| `statsmodels-arima-direction` | EURUSD | 1h | statsmodels | Statistical direction baseline |
| `lightgbm-fx-multifeature` | EURUSD | 30m | LightGBM | Multi-feature FX classifier |

## Use With PyP

1. Open PyP Quant Mode:
   https://pyp.stanlink.online/projects/quant/new
2. Create a new quant project.
3. Copy a template's `strategy.py` and `quant.config.json`.
4. Run a training job.
5. Validate with PPE simulation.
6. Deploy only after the strategy produces acceptable out-of-sample behavior.

## PyP Links

- Quant landing page: https://pyp.stanl.ink/for-quant-traders
- Quant docs: https://pyp.stanl.ink/docs/quant/what-is-quant-mode
- Create a quant project: https://pyp.stanlink.online/projects/quant/new

## Risk Disclaimer

Trading foreign exchange, CFDs, crypto, and leveraged products involves substantial risk. These templates are educational examples only. Past performance and simulated results do not guarantee future performance.
