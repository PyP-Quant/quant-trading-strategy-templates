import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


SYMBOL = "AVAXUSDT"
MODEL_NAME = "avaxusdt-volatility-rf-30m"


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
    v = df["volume"]
    f = pd.DataFrame(index=df.index)
    for n in [1, 4, 8, 16, 32, 64]:
        f[f"ret{n}"] = c.pct_change(n)
    f["ema_12_48"] = (c.ewm(span=12, adjust=False).mean() - c.ewm(span=48, adjust=False).mean()) / c
    f["ema_24_96"] = (c.ewm(span=24, adjust=False).mean() - c.ewm(span=96, adjust=False).mean()) / c
    f["volatility_fast"] = c.pct_change().rolling(16).std()
    f["volatility_slow"] = c.pct_change().rolling(64).std()
    f["volatility_ratio"] = f["volatility_fast"] / (f["volatility_slow"] + 1e-9)
    f["volume_z"] = (v - v.rolling(48).mean()) / (v.rolling(48).std() + 1e-9)
    return f.replace([np.inf, -np.inf], np.nan).dropna()


def train(data, config):
    p = config.get("parameters", {})
    df = _prep(data)
    x = _features(df)
    horizon = int(p.get("horizon", 6))
    threshold = float(p.get("threshold", 0.004))
    fwd = df["close"].pct_change(horizon).shift(-horizon)
    y = pd.Series(1, index=df.index)
    y[fwd > threshold] = 2
    y[fwd < -threshold] = 0
    y = y.reindex(x.index).fillna(1).astype(int)
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=350, max_depth=8, min_samples_leaf=8, class_weight="balanced_subsample", random_state=42, n_jobs=-1)),
    ])
    model.fit(x.values.astype(np.float32), y.values)
    pred = model.predict(x.values.astype(np.float32))
    return {"model": model, "features": list(x.columns), "symbol": SYMBOL}, {"training_bars": int(len(x)), "feature_count": int(x.shape[1]), "buy_signals": int((pred == 2).sum()), "sell_signals": int((pred == 0).sum()), "hold_signals": int((pred == 1).sum())}


def predict(model, market_data, config):
    p = config.get("parameters", {})
    candles = market_data.get("candles", [])
    if len(candles) < int(p.get("lookback", 120)):
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
