import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def _prep(data):
    df = data.copy()
    df.columns = [str(c).lower() for c in df.columns]
    if "volume" not in df.columns:
        df["volume"] = 1.0
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna().reset_index(drop=True)


def _feat(df):
    c, v = df["close"], df["volume"]
    f = pd.DataFrame(index=df.index)
    for n in [1, 3, 6, 12, 24]:
        f[f"ret{n}"] = c.pct_change(n)
    f["volatility"] = c.pct_change().rolling(24).std()
    f["volume_z"] = (v - v.rolling(24).mean()) / (v.rolling(24).std() + 1e-9)
    f["ema_fast"] = (c.ewm(span=12, adjust=False).mean() - c.ewm(span=48, adjust=False).mean()) / c
    return f.replace([np.inf, -np.inf], np.nan).dropna()


def train(data, config):
    p = config.get("parameters", {})
    df = _prep(data)
    x = _feat(df)
    fwd = df["close"].pct_change(int(p.get("horizon", 3))).shift(-int(p.get("horizon", 3)))
    threshold = float(p.get("threshold", 0.006))
    y = pd.Series(1, index=df.index)
    y[fwd > threshold] = 2
    y[fwd < -threshold] = 0
    y = y.reindex(x.index).fillna(1).astype(int)
    model = Pipeline([("scaler", StandardScaler()), ("clf", LGBMClassifier(n_estimators=300, learning_rate=0.04, max_depth=5, random_state=42, verbose=-1))])
    model.fit(x.values, y.values)
    return {"model": model, "features": list(x.columns)}, {"training_bars": int(len(x)), "class_dist": {"SELL": int((y == 0).sum()), "HOLD": int((y == 1).sum()), "BUY": int((y == 2).sum())}}


def predict(model, market_data, config):
    candles = market_data.get("candles", [])
    if len(candles) < int(config.get("parameters", {}).get("lookback", 120)):
        return {"signal": "HOLD", "confidence": 0, "metadata": {"reason": "not_enough_candles"}}
    df = _prep(pd.DataFrame(candles, columns=["open", "high", "low", "close", "volume"]))
    row = _feat(df).tail(1)[model["features"]]
    prob = model["model"].predict_proba(row.values)[0]
    k, conf = int(np.argmax(prob)), float(np.max(prob))
    signal = {0: "DOWN", 1: "HOLD", 2: "UP"}[k] if conf >= float(config.get("parameters", {}).get("min_confidence", 0.46)) else "HOLD"
    return {"signal": signal, "confidence": round(conf, 4), "metadata": {"p_sell": round(float(prob[0]), 4), "p_hold": round(float(prob[1]), 4), "p_buy": round(float(prob[2]), 4)}}
