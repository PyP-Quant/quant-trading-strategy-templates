def _clean_candles(candles):
    cleaned = []
    for candle in candles:
        try:
            cleaned.append({
                "open": float(candle[0]),
                "high": float(candle[1]),
                "low": float(candle[2]),
                "close": float(candle[3]),
                "volume": float(candle[4]) if len(candle) > 4 else 1.0,
            })
        except Exception:
            continue
    return cleaned


def _atr(candles, n):
    if len(candles) < 2:
        return 0.0
    alpha = 2.0 / (n + 1.0)
    value = None
    prev_close = candles[0]["close"]
    for candle in candles[1:]:
        tr = max(
            candle["high"] - candle["low"],
            abs(candle["high"] - prev_close),
            abs(candle["low"] - prev_close),
        )
        value = tr if value is None else value + alpha * (tr - value)
        prev_close = candle["close"]
    return float(value or 0.0)


def train(data, config):
    params = config.get("parameters", {})
    return {
        "params": {
            "lookback": int(params.get("lookback", 64)),
            "atr_window": int(params.get("atr_window", 14)),
            "breakout_window": int(params.get("breakout_window", 12)),
            "atr_mult": float(params.get("atr_mult", 0.05)),
        },
        "name": "xauusd_atr_breakout",
    }, {"training_bars": int(len(data)), "model": "edge_rule_baseline"}


def predict(model, market_data, config):
    params = {**model.get("params", {}), **config.get("parameters", {})}
    lookback = int(params.get("lookback", 64))
    atr_window = int(params.get("atr_window", 14))
    breakout_window = int(params.get("breakout_window", 12))
    atr_mult = float(params.get("atr_mult", 0.05))
    candles = _clean_candles(market_data.get("candles", []))

    if len(candles) < lookback:
        return {
            "signal": "HOLD",
            "confidence": 0.0,
            "metadata": {"reason": "not_enough_candles", "got": len(candles), "need": lookback},
        }

    window = candles[-lookback:]
    previous = window[-breakout_window - 1:-1]
    if len(previous) < breakout_window:
        return {"signal": "HOLD", "confidence": 0.0, "metadata": {"reason": "not_enough_breakout_window"}}

    atr = _atr(window, atr_window)
    close = window[-1]["close"]
    high = max(candle["high"] for candle in previous)
    low = min(candle["low"] for candle in previous)
    upper = high + atr * atr_mult
    lower = low - atr * atr_mult

    if close > upper:
        edge = (close - upper) / max(atr, 1e-9)
        return {
            "signal": "UP",
            "confidence": round(min(0.55 + edge, 0.92), 4),
            "metadata": {"breakout": "high", "atr": round(atr, 6), "upper": round(upper, 6), "close": round(close, 6)},
        }
    if close < lower:
        edge = (lower - close) / max(atr, 1e-9)
        return {
            "signal": "DOWN",
            "confidence": round(min(0.55 + edge, 0.92), 4),
            "metadata": {"breakout": "low", "atr": round(atr, 6), "lower": round(lower, 6), "close": round(close, 6)},
        }

    return {
        "signal": "HOLD",
        "confidence": 0.2,
        "metadata": {"high": round(high, 6), "low": round(low, 6), "atr": round(atr, 6), "upper": round(upper, 6), "lower": round(lower, 6)},
    }
