import numpy as np
import pandas as pd


SYMBOL = "MATICUSDT"
MODEL_NAME = "maticusdt-scalp-baseline-5m"


def _prep(data):
    df = data.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    if "volume" not in df.columns:
        df["volume"] = 1.0
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)


def _atr(df, n=14):
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([(h - l), (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(span=n, adjust=False).mean()


def train(data, config):
    p = config.get("parameters", {})
    return {"atr_window": int(p.get("atr_window", 14)), "breakout_window": int(p.get("breakout_window", 36)), "atr_mult": float(p.get("atr_mult", 0.25)), "symbol": SYMBOL}, {"model_family": "rule_based_atr_breakout", "training_bars": int(len(data))}


def predict(model, market_data, config):
    p = config.get("parameters", {})
    candles = market_data.get("candles", [])
    lookback = int(p.get("lookback", 100))
    if len(candles) < lookback:
        return {"signal": "HOLD", "confidence": 0.0, "metadata": {"reason": "not_enough_candles", "model": MODEL_NAME}}
    df = _prep(pd.DataFrame(candles, columns=["open", "high", "low", "close", "volume"]))
    atr_window = int(model.get("atr_window", p.get("atr_window", 14)))
    breakout_window = int(model.get("breakout_window", p.get("breakout_window", 36)))
    atr_mult = float(model.get("atr_mult", p.get("atr_mult", 0.25)))
    atr = _atr(df, atr_window)
    close = float(df["close"].iloc[-1])
    upper = float(df["high"].rolling(breakout_window).max().shift().iloc[-1] + atr.iloc[-1] * atr_mult)
    lower = float(df["low"].rolling(breakout_window).min().shift().iloc[-1] - atr.iloc[-1] * atr_mult)
    signal = "HOLD"
    distance = 0.0
    if close > upper:
        signal = "UP"
        distance = (close - upper) / max(atr.iloc[-1], 1e-9)
    elif close < lower:
        signal = "DOWN"
        distance = (lower - close) / max(atr.iloc[-1], 1e-9)
    confidence = min(max(distance, 0.0), 1.0)
    return {"signal": signal, "confidence": round(float(confidence), 4), "metadata": {"upper": round(upper, 6), "lower": round(lower, 6), "close": round(close, 6), "model": MODEL_NAME, "symbol": SYMBOL}}
