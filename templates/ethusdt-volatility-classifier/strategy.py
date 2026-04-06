import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


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


def _features(df):
    c = df["close"]
    v = df["volume"]
    ret = c.pct_change()
    f = pd.DataFrame(index=df.index)
    for n in [1, 2, 4, 8, 16, 32]:
        f[f"ret{n}"] = c.pct_change(n)
    f["atr_pct"] = _atr(df, 14) / c
    f["atr_expansion"] = (_atr(df, 8) / (_atr(df, 50) + 1e-9)).clip(0, 5)
    f["rv_12"] = ret.rolling(12).std()
    f["rv_48"] = ret.rolling(48).std()
    f["vol_regime"] = f["rv_12"] / (f["rv_48"] + 1e-9)
    f["volume_z"] = (v - v.rolling(48).mean()) / (v.rolling(48).std() + 1e-9)
    f["ema_9_34"] = (c.ewm(span=9, adjust=False).mean() - c.ewm(span=34, adjust=False).mean()) / c
    f["ema_21_89"] = (c.ewm(span=21, adjust=False).mean() - c.ewm(span=89, adjust=False).mean()) / c
    return f.replace([np.inf, -np.inf], np.nan).dropna()


def train(data, config):
    p = config.get("parameters", {})
    df = _prep(data)
    x = _features(df)
    horizon = int(p.get("horizon", 3))
    threshold = float(p.get("threshold", 0.004))
    fwd = df["close"].pct_change(horizon).shift(-horizon)
    y = pd.Series(1, index=df.index)
    y[fwd > threshold] = 2
    y[fwd < -threshold] = 0
    y = y.reindex(x.index).fillna(1).astype(int)
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=int(p.get("n_estimators", 300)),
            min_samples_leaf=int(p.get("min_samples_leaf", 8)),
            max_depth=int(p.get("max_depth", 8)),
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        )),
    ])
    model.fit(x.values, y.values)
    pred = model.predict(x.values)
    return {
        "model": model,
        "features": list(x.columns),
    }, {
        "training_bars": int(len(x)),
        "feature_count": int(x.shape[1]),
        "class_dist": {"SELL": int((y == 0).sum()), "HOLD": int((y == 1).sum()), "BUY": int((y == 2).sum())},
        "buy_signals": int((pred == 2).sum()),
        "sell_signals": int((pred == 0).sum()),
        "hold_signals": int((pred == 1).sum()),
    }


def predict(model, market_data, config):
    p = config.get("parameters", {})
    candles = market_data.get("candles", [])
    if len(candles) < int(p.get("lookback", 120)):
        return {"signal": "HOLD", "confidence": 0.0, "metadata": {"reason": "not_enough_candles"}}
    df = _prep(pd.DataFrame(candles, columns=["open", "high", "low", "close", "volume"]))
    row = _features(df).tail(1)
    if row.empty:
        return {"signal": "HOLD", "confidence": 0.0, "metadata": {"reason": "no_features"}}
    prob = model["model"].predict_proba(row[model["features"]].values)[0]
    klass = int(np.argmax(prob))
    conf = float(np.max(prob))
    signal = {0: "DOWN", 1: "HOLD", 2: "UP"}[klass]
    if conf < float(p.get("min_confidence", 0.47)):
        signal = "HOLD"
    return {"signal": signal, "confidence": round(conf, 4), "metadata": {"p_sell": round(float(prob[0]), 4), "p_hold": round(float(prob[1]), 4), "p_buy": round(float(prob[2]), 4), "model": "ethusdt-volatility-classifier"}}
