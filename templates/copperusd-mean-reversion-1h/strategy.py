import numpy as np
import pandas as pd


SYMBOL = "COPPERUSD"
MODEL_NAME = "copperusd-mean-reversion-1h"


def _prep(data):
    df = data.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    if "volume" not in df.columns:
        df["volume"] = 1.0
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)


def _zscore(close, window):
    mean = close.rolling(window).mean()
    std = close.rolling(window).std()
    return (close - mean) / (std + 1e-9)


def train(data, config):
    p = config.get("parameters", {})
    return {"window": int(p.get("window", 48)), "entry_z": float(p.get("entry_z", 1.4)), "exit_z": float(p.get("exit_z", 0.3)), "symbol": SYMBOL}, {"model_family": "rule_based_mean_reversion", "training_bars": int(len(data))}


def predict(model, market_data, config):
    p = config.get("parameters", {})
    candles = market_data.get("candles", [])
    lookback = int(p.get("lookback", 140))
    if len(candles) < lookback:
        return {"signal": "HOLD", "confidence": 0.0, "metadata": {"reason": "not_enough_candles", "model": MODEL_NAME}}
    df = _prep(pd.DataFrame(candles, columns=["open", "high", "low", "close", "volume"]))
    window = int(model.get("window", p.get("window", 48)))
    z = float(_zscore(df["close"], window).iloc[-1])
    entry_z = float(model.get("entry_z", p.get("entry_z", 1.4)))
    signal = "HOLD"
    if z <= -entry_z:
        signal = "UP"
    elif z >= entry_z:
        signal = "DOWN"
    confidence = min(abs(z) / max(entry_z, 1e-9), 1.0)
    return {"signal": signal, "confidence": round(float(confidence), 4), "metadata": {"zscore": round(z, 4), "window": window, "model": MODEL_NAME, "symbol": SYMBOL}}
