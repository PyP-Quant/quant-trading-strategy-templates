import pandas as pd
from statsmodels.tsa.arima.model import ARIMA


def train(data, config):
    df = data.copy()
    df.columns = [str(c).lower() for c in df.columns]
    close = pd.to_numeric(df["close"], errors="coerce").dropna()
    ret = close.pct_change().dropna().tail(1500)
    fit = ARIMA(ret, order=(1, 0, 1)).fit()
    return {"fit": fit, "threshold": config.get("parameters", {}).get("entry_threshold", 0.0004)}, {"training_bars": int(len(ret)), "aic": float(fit.aic)}


def predict(model, market_data, config):
    threshold = float(config.get("parameters", {}).get("entry_threshold", model.get("threshold", 0.0004)))
    forecast = float(model["fit"].forecast(1).iloc[0])
    if forecast > threshold:
        return {"signal": "UP", "confidence": min(0.8, abs(forecast) / threshold / 3), "metadata": {"forecast_return": forecast}}
    if forecast < -threshold:
        return {"signal": "DOWN", "confidence": min(0.8, abs(forecast) / threshold / 3), "metadata": {"forecast_return": forecast}}
    return {"signal": "HOLD", "confidence": 0.25, "metadata": {"forecast_return": forecast}}
