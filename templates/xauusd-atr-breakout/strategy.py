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


def _ema(series, n):
    return series.ewm(span=n, adjust=False).mean()


def _clip01(value):
    return float(np.clip(value, 0.0, 1.0))


def train(data, config):
    params = config.get("parameters", {})
    return {
        "params": {
            "lookback": int(params.get("lookback", 96)),
            "atr_window": int(params.get("atr_window", 14)),
            "breakout_window": int(params.get("breakout_window", 24)),
            "atr_mult": float(params.get("atr_mult", 0.03)),
            "near_breakout_atr": float(params.get("near_breakout_atr", 0.18)),
            "pullback_atr": float(params.get("pullback_atr", 0.35)),
            "min_momentum_atr": float(params.get("min_momentum_atr", 0.0)),
            "fast_ema": int(params.get("fast_ema", 8)),
            "slow_ema": int(params.get("slow_ema", 34)),
            "enable_near_breakout": bool(params.get("enable_near_breakout", False)),
            "enable_continuation": bool(params.get("enable_continuation", False)),
        },
        "name": "xauusd_atr_breakout_precision",
    }, {"training_bars": int(len(data)), "model": "modal_python_rule_precision_breakout"}


def predict(model, market_data, config):
    params = {**model.get("params", {}), **config.get("parameters", {})}
    lookback = int(params.get("lookback", 96))
    atr_window = int(params.get("atr_window", 14))
    breakout_window = int(params.get("breakout_window", 24))
    atr_mult = float(params.get("atr_mult", 0.03))
    near_breakout_atr = float(params.get("near_breakout_atr", 0.18))
    pullback_atr = float(params.get("pullback_atr", 0.35))
    min_momentum_atr = float(params.get("min_momentum_atr", 0.0))
    fast_ema = int(params.get("fast_ema", 8))
    slow_ema = int(params.get("slow_ema", 34))
    enable_near_breakout = bool(params.get("enable_near_breakout", False))
    enable_continuation = bool(params.get("enable_continuation", False))
    candles = market_data.get("candles", [])
    if len(candles) < lookback:
        return {"signal": "HOLD", "confidence": 0.0, "metadata": {"reason": "not_enough_candles"}}

    df = _df(candles[-lookback:])
    atr = float(_atr(df, atr_window).iloc[-1])
    if not np.isfinite(atr) or atr <= 0:
        return {"signal": "HOLD", "confidence": 0.0, "metadata": {"reason": "invalid_atr"}}

    close = float(df["close"].iloc[-1])
    prev_close = float(df["close"].iloc[-2])
    high = float(df["high"].iloc[-breakout_window:-1].max())
    low = float(df["low"].iloc[-breakout_window:-1].min())
    upper = high + atr * atr_mult
    lower = low - atr * atr_mult
    range_width = max(high - low, atr)

    ema_fast = float(_ema(df["close"], fast_ema).iloc[-1])
    ema_slow = float(_ema(df["close"], slow_ema).iloc[-1])
    ema_gap_atr = (ema_fast - ema_slow) / atr
    momentum_3 = (close - float(df["close"].iloc[-4])) / atr if len(df) >= 4 else 0.0
    momentum_6 = (close - float(df["close"].iloc[-7])) / atr if len(df) >= 7 else momentum_3
    body_atr = (close - float(df["open"].iloc[-1])) / atr
    close_position = (close - low) / range_width

    trend_up = ema_fast > ema_slow and close >= ema_slow and momentum_3 > -min_momentum_atr
    trend_down = ema_fast < ema_slow and close <= ema_slow and momentum_3 < min_momentum_atr

    meta = {
        "atr": round(atr, 5),
        "high": round(high, 5),
        "low": round(low, 5),
        "upper": round(upper, 5),
        "lower": round(lower, 5),
        "ema_gap_atr": round(float(ema_gap_atr), 4),
        "momentum_3_atr": round(float(momentum_3), 4),
        "momentum_6_atr": round(float(momentum_6), 4),
        "close_position": round(float(close_position), 4),
    }

    def confidence(edge_atr, quality, floor=0.56):
        return round(min(floor + edge_atr * 0.22 + quality * 0.16, 0.92), 4)

    if close > upper:
        edge = (close - upper) / max(atr, 1e-9)
        quality = _clip01(max(ema_gap_atr, 0.0) * 0.35 + max(momentum_3, 0.0) * 0.35 + max(body_atr, 0.0) * 0.20)
        return {"signal": "UP", "confidence": confidence(edge, quality), "metadata": {**meta, "setup": "breakout_high"}}
    if close < lower:
        edge = (lower - close) / max(atr, 1e-9)
        quality = _clip01(max(-ema_gap_atr, 0.0) * 0.35 + max(-momentum_3, 0.0) * 0.35 + max(-body_atr, 0.0) * 0.20)
        return {"signal": "DOWN", "confidence": confidence(edge, quality), "metadata": {**meta, "setup": "breakout_low"}}

    # If price is pressing a range boundary with trend/momentum confirmation,
    # enter before the hard breakout instead of waiting for an extreme close.
    upper_pressure = (high - close) / atr
    lower_pressure = (close - low) / atr
    if enable_near_breakout and trend_up and upper_pressure <= near_breakout_atr and momentum_3 >= min_momentum_atr:
        edge = max(near_breakout_atr - upper_pressure, 0.0)
        quality = _clip01(max(ema_gap_atr, 0.0) * 0.35 + max(momentum_3, 0.0) * 0.35 + close_position * 0.20)
        return {"signal": "UP", "confidence": confidence(edge, quality, floor=0.57), "metadata": {**meta, "setup": "near_breakout_high"}}
    if enable_near_breakout and trend_down and lower_pressure <= near_breakout_atr and momentum_3 <= -min_momentum_atr:
        edge = max(near_breakout_atr - lower_pressure, 0.0)
        quality = _clip01(max(-ema_gap_atr, 0.0) * 0.35 + max(-momentum_3, 0.0) * 0.35 + (1.0 - close_position) * 0.20)
        return {"signal": "DOWN", "confidence": confidence(edge, quality, floor=0.57), "metadata": {**meta, "setup": "near_breakout_low"}}

    # Continuation path: after a breakout, gold often retests the fast EMA
    # without closing outside the range again. This keeps the system alive
    # while still requiring trend and momentum context.
    if enable_continuation and trend_up and close > high - atr * pullback_atr and prev_close <= close and momentum_6 >= 0:
        edge = max((close - (high - atr * pullback_atr)) / atr, 0.0)
        quality = _clip01(max(ema_gap_atr, 0.0) * 0.35 + max(momentum_6, 0.0) * 0.25 + close_position * 0.25)
        return {"signal": "UP", "confidence": confidence(edge, quality, floor=0.55), "metadata": {**meta, "setup": "trend_continuation_high"}}
    if enable_continuation and trend_down and close < low + atr * pullback_atr and prev_close >= close and momentum_6 <= 0:
        edge = max(((low + atr * pullback_atr) - close) / atr, 0.0)
        quality = _clip01(max(-ema_gap_atr, 0.0) * 0.35 + max(-momentum_6, 0.0) * 0.25 + (1.0 - close_position) * 0.25)
        return {"signal": "DOWN", "confidence": confidence(edge, quality, floor=0.55), "metadata": {**meta, "setup": "trend_continuation_low"}}

    return {"signal": "HOLD", "confidence": 0.2, "metadata": {**meta, "setup": "inside_range"}}
