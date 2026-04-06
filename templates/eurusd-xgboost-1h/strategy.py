import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


def _normalise(data):
    df = data.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    aliases = {"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"}
    df.rename(columns=aliases, inplace=True)
    if "volume" not in df.columns:
        df["volume"] = 1.0
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)


def _rsi(close, n=14):
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + gain / (loss + 1e-9))


def _features(df):
    c = df["close"]
    h = df["high"]
    l = df["low"]
    o = df["open"]
    v = df["volume"]
    rng = (h - l).replace(0, np.nan)
    f = pd.DataFrame(index=df.index)
    for n in [1, 3, 6, 12, 24]:
        f[f"ret{n}"] = c.pct_change(n)
    f["range_pct"] = rng / c
    f["body_pct"] = (c - o) / rng
    f["close_pos"] = (c - l) / (rng + 1e-9)
    f["volatility_24"] = c.pct_change().rolling(24).std()
    f["volatility_72"] = c.pct_change().rolling(72).std()
    f["ema_12_48"] = (c.ewm(span=12, adjust=False).mean() - c.ewm(span=48, adjust=False).mean()) / c
    f["ema_24_96"] = (c.ewm(span=24, adjust=False).mean() - c.ewm(span=96, adjust=False).mean()) / c
    f["rsi14"] = (_rsi(c, 14) - 50) / 50
    f["volume_z"] = (v - v.rolling(48).mean()) / (v.rolling(48).std() + 1e-9)
    return f.replace([np.inf, -np.inf], np.nan).dropna()


def _labels(close, index, horizon, threshold):
    fwd = close.pct_change(horizon).shift(-horizon)
    y = pd.Series(1, index=close.index)
    y[fwd > threshold] = 2
    y[fwd < -threshold] = 0
    return y.reindex(index).fillna(1).astype(int)


def train(data, config):
    params = config.get("parameters", {})
    df = _normalise(data)
    feat = _features(df)
    horizon = int(params.get("horizon", 4))
    threshold = float(params.get("threshold", 0.0012))
    y = _labels(df["close"], feat.index, horizon, threshold)
    scaler = StandardScaler()
    x = scaler.fit_transform(feat.values.astype(np.float32))
    clf = XGBClassifier(
        n_estimators=int(params.get("n_estimators", 250)),
        max_depth=int(params.get("max_depth", 4)),
        learning_rate=float(params.get("learning_rate", 0.04)),
        subsample=float(params.get("subsample", 0.8)),
        colsample_bytree=float(params.get("colsample_bytree", 0.85)),
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=42,
    )
    clf.fit(x, y.values)
    preds = clf.predict(x)
    metrics = {
        "training_bars": int(len(feat)),
        "feature_count": int(feat.shape[1]),
        "class_dist": {
            "SELL": int((y == 0).sum()),
            "HOLD": int((y == 1).sum()),
            "BUY": int((y == 2).sum()),
        },
        "buy_signals": int((preds == 2).sum()),
        "sell_signals": int((preds == 0).sum()),
        "hold_signals": int((preds == 1).sum()),
    }
    return {"model": clf, "scaler": scaler, "features": list(feat.columns)}, metrics


def predict(model, market_data, config):
    params = config.get("parameters", {})
    candles = market_data.get("candles", [])
    if len(candles) < int(params.get("lookback", 140)):
        return {"signal": "HOLD", "confidence": 0.0, "metadata": {"reason": "not_enough_candles"}}
    df = _normalise(pd.DataFrame(candles, columns=["open", "high", "low", "close", "volume"]))
    row = _features(df).tail(1)
    if row.empty:
        return {"signal": "HOLD", "confidence": 0.0, "metadata": {"reason": "no_features"}}
    row = row[model["features"]]
    x = model["scaler"].transform(row.values.astype(np.float32))
    prob = model["model"].predict_proba(x)[0]
    klass = int(np.argmax(prob))
    conf = float(np.max(prob))
    signal = {0: "DOWN", 1: "HOLD", 2: "UP"}[klass]
    if conf < float(params.get("min_confidence", 0.48)):
        signal = "HOLD"
    return {
        "signal": signal,
        "confidence": round(conf, 4),
        "metadata": {
            "p_sell": round(float(prob[0]), 4),
            "p_hold": round(float(prob[1]), 4),
            "p_buy": round(float(prob[2]), 4),
            "model": "eurusd-xgboost-1h",
        },
    }
