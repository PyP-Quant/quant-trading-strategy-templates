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

This repository currently contains 50 PyP Quant strategy templates.

| Template | Pair | Timeframe | Model family | Use case |
| --- | --- | --- | --- | --- |
| `adausdt-volatility-rf-15m` | ADAUSDT | 15m | sklearn RandomForest | sklearn RandomForest PyP Quant template for ADAUSDT 15m. |
| `audjpy-randomforest-1h` | AUDJPY | 1h | sklearn RandomForest | sklearn RandomForest PyP Quant template for AUDJPY 1h. |
| `audusd-logistic-30m` | AUDUSD | 30m | sklearn LogisticRegression | sklearn LogisticRegression PyP Quant template for AUDUSD 30m. |
| `avaxusdt-volatility-rf-30m` | AVAXUSDT | 30m | sklearn RandomForest | sklearn RandomForest PyP Quant template for AVAXUSDT 30m. |
| `bnbusdt-trend-rf-30m` | BNBUSDT | 30m | sklearn RandomForest | sklearn RandomForest PyP Quant template for BNBUSDT 30m. |
| `brentoil-trend-rf-1h` | BRENTUSD | 1h | sklearn RandomForest | sklearn RandomForest PyP Quant template for BRENTUSD 1h. |
| `btcusdt-breakout-rf-15m` | BTCUSDT | 15m | sklearn RandomForest | sklearn RandomForest PyP Quant template for BTCUSDT 15m. |
| `btcusdt-lightgbm-1h` | BTCUSDT | 1h | LightGBM | BTCUSDT LightGBM crypto trend classifier |
| `btcusdt-mean-reversion-5m` | BTCUSDT | 5m | custom Python | custom Python PyP Quant template for BTCUSDT 5m. |
| `cadjpy-breakout-1h` | CADJPY | 1h | sklearn RandomForest | sklearn RandomForest PyP Quant template for CADJPY 1h. |
| `chfjpy-mean-reversion-1h` | CHFJPY | 1h | custom Python | custom Python PyP Quant template for CHFJPY 1h. |
| `copperusd-mean-reversion-1h` | COPPERUSD | 1h | custom Python | custom Python PyP Quant template for COPPERUSD 1h. |
| `dogeusdt-scalp-logistic-1m` | DOGEUSDT | 1m | sklearn LogisticRegression | sklearn LogisticRegression PyP Quant template for DOGEUSDT 1m. |
| `dotusdt-mean-reversion-30m` | DOTUSDT | 30m | custom Python | custom Python PyP Quant template for DOTUSDT 30m. |
| `ethusdt-volatility-classifier` | ETHUSDT | 30m | sklearn RandomForest | ETHUSDT volatility-aware directional classifier |
| `eurgbp-range-classifier-30m` | EURGBP | 30m | sklearn LogisticRegression | sklearn LogisticRegression PyP Quant template for EURGBP 30m. |
| `eurjpy-trend-rf-1h` | EURJPY | 1h | sklearn RandomForest | sklearn RandomForest PyP Quant template for EURJPY 1h. |
| `eurnzd-volatility-rf-4h` | EURNZD | 4h | sklearn RandomForest | sklearn RandomForest PyP Quant template for EURNZD 4h. |
| `eurusd-logistic-15m` | EURUSD | 15m | sklearn LogisticRegression | EURUSD logistic regression baseline classifier |
| `eurusd-xgboost-1h` | EURUSD | 1h | XGBoost | EURUSD feature-rich XGBoost trend classifier |
| `gbpjpy-breakout-rf-1h` | GBPJPY | 1h | sklearn RandomForest | sklearn RandomForest PyP Quant template for GBPJPY 1h. |
| `gbpusd-breakout-rf` | GBPUSD | 30m | sklearn RandomForest | GBPUSD range breakout random forest classifier |
| `gbpusd-logistic-15m` | GBPUSD | 15m | sklearn LogisticRegression | sklearn LogisticRegression PyP Quant template for GBPUSD 15m. |
| `ger40-breakout-rf-30m` | GER40 | 30m | sklearn RandomForest | sklearn RandomForest PyP Quant template for GER40 30m. |
| `lightgbm-fx-multifeature` | EURUSD | 30m | LightGBM | Multi-feature FX LightGBM classifier |
| `linkusdt-trend-logistic-30m` | LINKUSDT | 30m | sklearn LogisticRegression | sklearn LogisticRegression PyP Quant template for LINKUSDT 30m. |
| `ltcusdt-trend-rf-1h` | LTCUSDT | 1h | sklearn RandomForest | sklearn RandomForest PyP Quant template for LTCUSDT 1h. |
| `maticusdt-scalp-baseline-5m` | MATICUSDT | 5m | custom Python | custom Python PyP Quant template for MATICUSDT 5m. |
| `nas100-breakout-rf-15m` | NAS100 | 15m | sklearn RandomForest | sklearn RandomForest PyP Quant template for NAS100 15m. |
| `naturalgas-volatility-rf-4h` | NATGAS | 4h | sklearn RandomForest | sklearn RandomForest PyP Quant template for NATGAS 4h. |
| `nzdusd-mean-reversion-1h` | NZDUSD | 1h | custom Python | custom Python PyP Quant template for NZDUSD 1h. |
| `oilwtico-atr-breakout-1h` | WTICOUSD | 1h | custom Python | custom Python PyP Quant template for WTICOUSD 1h. |
| `onnx-export-sklearn-starter` | EURUSD | 1h | sklearn to ONNX | Sklearn classifier starter for ONNX export |
| `solusdt-scalp-baseline` | SOLUSDT | 1m | custom Python | SOLUSDT high-volatility scalp baseline |
| `spx500-trend-logistic-1h` | SPX500 | 1h | sklearn LogisticRegression | sklearn LogisticRegression PyP Quant template for SPX500 1h. |
| `statsmodels-arima-direction` | EURUSD | 1h | statsmodels | ARIMA-style statistical direction baseline |
| `uk100-range-logistic-1h` | UK100 | 1h | sklearn LogisticRegression | sklearn LogisticRegression PyP Quant template for UK100 1h. |
| `us30-mean-reversion-30m` | US30 | 30m | custom Python | custom Python PyP Quant template for US30 30m. |
| `usdcad-trend-rf-1h` | USDCAD | 1h | sklearn RandomForest | sklearn RandomForest PyP Quant template for USDCAD 1h. |
| `usdchf-range-logistic-1h` | USDCHF | 1h | sklearn LogisticRegression | sklearn LogisticRegression PyP Quant template for USDCHF 1h. |
| `usdjpy-mean-reversion` | USDJPY | 1h | custom Python | USDJPY mean-reversion baseline |
| `usdmxn-volatility-rf-4h` | USDMXN | 4h | sklearn RandomForest | sklearn RandomForest PyP Quant template for USDMXN 4h. |
| `xagusd-atr-breakout-1h` | XAGUSD | 1h | custom Python | custom Python PyP Quant template for XAGUSD 1h. |
| `xagusd-mean-reversion-30m` | XAGUSD | 30m | custom Python | custom Python PyP Quant template for XAGUSD 30m. |
| `xauusd-atr-breakout` | XAUUSD | 15m | custom Python | XAUUSD ATR breakout rule baseline |
| `xauusd-london-breakout-15m` | XAUUSD | 15m | custom Python | custom Python PyP Quant template for XAUUSD 15m. |
| `xauusd-regime-xgboost-v11` | XAUUSD | 1h | XGBoost | Less restrictive Au-79-style XAUUSD regime classifier |
| `xauusd-scalp-logistic-5m` | XAUUSD | 5m | sklearn LogisticRegression | sklearn LogisticRegression PyP Quant template for XAUUSD 5m. |
| `xptusd-trend-rf-1h` | XPTUSD | 1h | sklearn RandomForest | sklearn RandomForest PyP Quant template for XPTUSD 1h. |
| `xrpusdt-breakout-rf-15m` | XRPUSDT | 15m | sklearn RandomForest | sklearn RandomForest PyP Quant template for XRPUSDT 15m. |

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
