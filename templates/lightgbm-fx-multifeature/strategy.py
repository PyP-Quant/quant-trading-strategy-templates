import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.preprocessing import StandardScaler


def _prep(data):
    df = data.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    if "volume" not in df.columns:
        df["volume"] = 1.0
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)


def _rsi(close, n=14):
    d = close.diff()
    g = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    l = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + g / (l + 1e-9))


def _features(df):
    c = df["close"]
    v = df["volume"]
    f = pd.DataFrame(index=df.index)
    for n in [1, 2, 4, 8, 16, 32, 64]:
        f[f"ret{n}"] = c.pct_change(n)
    f["ema_8_21"] = (c.ewm(span=8, adjust=False).mean() - c.ewm(span=21, adjust=False).mean()) / c
    f["ema_21_55"] = (c.ewm(span=21, adjust=False).mean() - c.ewm(span=55, adjust=False).mean()) / c
    f["ema_55_144"] = (c.ewm(span=55, adjust=False).mean() - c.ewm(span=144, adjust=False).mean()) / c
    f["rsi14"] = (_rsi(c, 14) - 50) / 50
    f["rsi5"] = (_rsi(c, 5) - 50) / 50
    f["volatility_32"] = c.pct_change().rolling(32).std()
    f["volatility_96"] = c.pct_change().rolling(96).std()
    f["volume_z"] = (v - v.rolling(48).mean()) / (v.rolling(48).std() + 1e-9)
    return f.replace([np.inf, -np.inf], np.nan).dropna()


def train(data, config):
    p = config.get("parameters", {})
    df = _prep(data)
    x = _features(df)
    horizon = int(p.get("horizon", 6))
    threshold = float(p.get("threshold", 0.0015))
    fwd = df["close"].pct_change(horizon).shift(-horizon)
    y = pd.Series(1, index=df.index)
    y[fwd > threshold] = 2
    y[fwd < -threshold] = 0
    y = y.reindex(x.index).fillna(1).astype(int)
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x.values.astype(np.float32))
    clf = LGBMClassifier(
        n_estimators=int(p.get("n_estimators", 350)),
        learning_rate=float(p.get("learning_rate", 0.035)),
        num_leaves=int(p.get("num_leaves", 31)),
        max_depth=int(p.get("max_depth", -1)),
        subsample=float(p.get("subsample", 0.85)),
        colsample_bytree=float(p.get("colsample_bytree", 0.85)),
        random_state=42,
        verbose=-1,
    )
    clf.fit(x_scaled, y.values)
    pred = clf.predict(x_scaled)
    return {"model": clf, "scaler": scaler, "features": list(x.columns)}, {
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
    if len(candles) < int(p.get("lookback", 180)):
        return {"signal": "HOLD", "confidence": 0.0, "metadata": {"reason": "not_enough_candles"}}
    df = _prep(pd.DataFrame(candles, columns=["open", "high", "low", "close", "volume"]))
    row = _features(df).tail(1)
    if row.empty:
        return {"signal": "HOLD", "confidence": 0.0, "metadata": {"reason": "no_features"}}
    x = model["scaler"].transform(row[model["features"]].values.astype(np.float32))
    prob = model["model"].predict_proba(x)[0]
    klass = int(np.argmax(prob))
    conf = float(np.max(prob))
    signal = {0: "DOWN", 1: "HOLD", 2: "UP"}[klass]
    if conf < float(p.get("min_confidence", 0.48)):
        signal = "HOLD"
    return {"signal": signal, "confidence": round(conf, 4), "metadata": {"p_sell": round(float(prob[0]), 4), "p_hold": round(float(prob[1]), 4), "p_buy": round(float(prob[2]), 4), "model": "lightgbm-fx-multifeature"}}
