import pandas as pd


def train(data, config):
    return {"params": config.get("parameters", {}), "name": "solusdt_scalp_baseline"}, {"training_bars": int(len(data)), "model": "rule_baseline"}


def predict(model, market_data, config):
    p = {**model.get("params", {}), **config.get("parameters", {})}
    candles = market_data.get("candles", [])
    lookback = int(p.get("lookback", 90))
    if len(candles) < lookback:
        return {"signal": "HOLD", "confidence": 0.0, "metadata": {"reason": "not_enough_candles"}}
    df = pd.DataFrame(candles[-lookback:], columns=["open", "high", "low", "close", "volume"]).astype(float)
    close = df["close"]
    fast = close.ewm(span=int(p.get("fast", 8)), adjust=False).mean()
    slow = close.ewm(span=int(p.get("slow", 34)), adjust=False).mean()
    ret = close.pct_change()
    vol = ret.rolling(int(p.get("vol_window", 20))).std().iloc[-1]
    slope = (fast.iloc[-1] - slow.iloc[-1]) / close.iloc[-1]
    min_move = float(p.get("min_move", 0.0006))
    confidence = min(0.9, abs(slope) / max(float(vol or 1e-6), 1e-6))
    if slope > min_move:
        return {"signal": "UP", "confidence": round(float(confidence), 4), "metadata": {"slope": float(slope), "vol": float(vol)}}
    if slope < -min_move:
        return {"signal": "DOWN", "confidence": round(float(confidence), 4), "metadata": {"slope": float(slope), "vol": float(vol)}}
    return {"signal": "HOLD", "confidence": 0.2, "metadata": {"slope": float(slope), "vol": float(vol)}}
