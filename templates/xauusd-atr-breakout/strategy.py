import numpy as np
import pandas as pd


def _df(candles):
    df = pd.DataFrame(candles, columns=["open", "high", "low", "close", "volume"])
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna().reset_index(drop=True)


def _atr(df, n):
    prev = df["close"].shift()
    tr = pd.concat([(df["high"] - df["low"]), (df["high"] - prev).abs(), (df["low"] - prev).abs()], axis=1).max(axis=1)
    return tr.ewm(span=n, adjust=False).mean()


def train(data, config):
    params = config.get("parameters", {})
    return {"params": params, "name": "xauusd_atr_breakout"}, {"training_bars": int(len(data)), "model": "rule_baseline"}


def predict(model, market_data, config):
    params = {**model.get("params", {}), **config.get("parameters", {})}
    lookback = int(params.get("lookback", 64))
    atr_window = int(params.get("atr_window", 14))
    breakout_window = int(params.get("breakout_window", 12))
    atr_mult = float(params.get("atr_mult", 0.05))
    candles = market_data.get("candles", [])
    if len(candles) < lookback:
        return {"signal": "HOLD", "confidence": 0.0, "metadata": {"reason": "not_enough_candles"}}
    df = _df(candles[-lookback:])
    atr = float(_atr(df, atr_window).iloc[-1])
    close = float(df["close"].iloc[-1])
    high = float(df["high"].iloc[-breakout_window:-1].max())
    low = float(df["low"].iloc[-breakout_window:-1].min())
    if close > high + atr * atr_mult:
        return {"signal": "UP", "confidence": 0.64, "metadata": {"breakout": "high", "atr": atr}}
    if close < low - atr * atr_mult:
        return {"signal": "DOWN", "confidence": 0.64, "metadata": {"breakout": "low", "atr": atr}}
    return {"signal": "HOLD", "confidence": 0.2, "metadata": {"high": high, "low": low, "atr": atr}}
