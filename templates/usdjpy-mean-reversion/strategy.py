import pandas as pd


def train(data, config):
    return {"params": config.get("parameters", {}), "name": "usdjpy_mean_reversion"}, {"training_bars": int(len(data)), "model": "rule_baseline"}


def predict(model, market_data, config):
    p = {**model.get("params", {}), **config.get("parameters", {})}
    candles = market_data.get("candles", [])
    lookback = int(p.get("lookback", 100))
    if len(candles) < lookback:
        return {"signal": "HOLD", "confidence": 0.0, "metadata": {"reason": "not_enough_candles"}}
    close = pd.Series([float(c[3]) for c in candles[-lookback:]])
    n = int(p.get("z_window", 40))
    mean = close.rolling(n).mean().iloc[-1]
    std = close.rolling(n).std().iloc[-1] or 1e-9
    z = float((close.iloc[-1] - mean) / std)
    entry = float(p.get("entry_z", 1.4))
    if z <= -entry:
        return {"signal": "UP", "confidence": min(0.88, abs(z) / 3), "metadata": {"zscore": z}}
    if z >= entry:
        return {"signal": "DOWN", "confidence": min(0.88, abs(z) / 3), "metadata": {"zscore": z}}
    return {"signal": "HOLD", "confidence": 0.25, "metadata": {"zscore": z}}
