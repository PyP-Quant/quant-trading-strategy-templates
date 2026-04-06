import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


SYMBOL = "BTCUSDT"
MODEL_NAME = "btcusdt-breakout-rf-15m"


def _prep(data):
    df = data.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    if "volume" not in df.columns:
        df["volume"] = 1.0
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)


def _features(df):
    c = df["close"]
    h = df["high"]
    l = df["low"]
    f = pd.DataFrame(index=df.index)
    high_break = h.rolling(36).max().shift()
    low_break = l.rolling(36).min().shift()
    f["breakout_up"] = (c - high_break) / c
    f["breakout_down"] = (c - low_break) / c
    f["ret4"] = c.pct_change(4)
    f["ret12"] = c.pct_change(12)
    f["volatility"] = c.pct_change().rolling(24).std()
    f["range_pct"] = (h - l) / c
    f["ema_slope"] = c.ewm(span=12, adjust=False).mean().pct_change(6)
    return f.replace([np.inf, -np.inf], np.nan).dropna()


def train(data, config):
    p = config.get("parameters", {})
    df = _prep(data)
    x = _features(df)
    horizon = int(p.get("horizon", 4))
    threshold = float(p.get("threshold", 0.004))
    fwd = df["close"].pct_change(horizon).shift(-horizon)
    y = pd.Series(1, index=df.index)
    y[fwd > threshold] = 2
    y[fwd < -threshold] = 0
    y = y.reindex(x.index).fillna(1).astype(int)
    clf = RandomForestClassifier(n_estimators=300, max_depth=7, min_samples_leaf=6, class_weight="balanced_subsample", random_state=42, n_jobs=-1)
    clf.fit(x.values.astype(np.float32), y.values)
    pred = clf.predict(x.values.astype(np.float32))
    return {"model": clf, "features": list(x.columns), "symbol": SYMBOL}, {"training_bars": int(len(x)), "feature_count": int(x.shape[1]), "buy_signals": int((pred == 2).sum()), "sell_signals": int((pred == 0).sum()), "hold_signals": int((pred == 1).sum())}


def predict(model, market_data, config):
    p = config.get("parameters", {})
    candles = market_data.get("candles", [])
    if len(candles) < int(p.get("lookback", 100)):
        return {"signal": "HOLD", "confidence": 0.0, "metadata": {"reason": "not_enough_candles", "model": MODEL_NAME}}
    df = _prep(pd.DataFrame(candles, columns=["open", "high", "low", "close", "volume"]))
    row = _features(df).tail(1)
    if row.empty:
        return {"signal": "HOLD", "confidence": 0.0, "metadata": {"reason": "no_features", "model": MODEL_NAME}}
    prob = model["model"].predict_proba(row[model["features"]].values.astype(np.float32))[0]
    klass = int(np.argmax(prob))
    conf = float(np.max(prob))
    signal = {0: "DOWN", 1: "HOLD", 2: "UP"}[klass]
    if conf < float(p.get("min_confidence", 0.5)):
        signal = "HOLD"
    return {"signal": signal, "confidence": round(conf, 4), "metadata": {"p_sell": round(float(prob[0]), 4), "p_hold": round(float(prob[1]), 4), "p_buy": round(float(prob[2]), 4), "model": MODEL_NAME, "symbol": SYMBOL}}
