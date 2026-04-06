import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
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


def _features(df):
    c = df["close"]
    rng = (df["high"] - df["low"]).replace(0, np.nan)
    f = pd.DataFrame(index=df.index)
    f["ret1"] = c.pct_change(1)
    f["ret5"] = c.pct_change(5)
    f["ret20"] = c.pct_change(20)
    f["range_pct"] = rng / c
    f["body_pct"] = (c - df["open"]) / (rng + 1e-9)
    f["ema_10_40"] = (c.ewm(span=10, adjust=False).mean() - c.ewm(span=40, adjust=False).mean()) / c
    f["volatility_20"] = c.pct_change().rolling(20).std()
    f["volume_ratio"] = df["volume"] / (df["volume"].rolling(20).mean() + 1e-9)
    return f.replace([np.inf, -np.inf], np.nan).dropna()


def train(data, config):
    p = config.get("parameters", {})
    df = _prep(data)
    x = _features(df)
    horizon = int(p.get("horizon", 4))
    threshold = float(p.get("threshold", 0.001))
    fwd = df["close"].pct_change(horizon).shift(-horizon)
    y = pd.Series(1, index=df.index)
    y[fwd > threshold] = 2
    y[fwd < -threshold] = 0
    y = y.reindex(x.index).fillna(1).astype(int)
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])
    model.fit(x.values.astype(np.float32), y.values)
    pred = model.predict(x.values.astype(np.float32))
    return {"model": model, "features": list(x.columns), "onnx_export_hint": "Convert this sklearn Pipeline with skl2onnx after training."}, {
        "training_bars": int(len(x)),
        "feature_count": int(x.shape[1]),
        "buy_signals": int((pred == 2).sum()),
        "sell_signals": int((pred == 0).sum()),
        "hold_signals": int((pred == 1).sum()),
    }


def predict(model, market_data, config):
    p = config.get("parameters", {})
    candles = market_data.get("candles", [])
    if len(candles) < int(p.get("lookback", 80)):
        return {"signal": "HOLD", "confidence": 0.0, "metadata": {"reason": "not_enough_candles"}}
    df = _prep(pd.DataFrame(candles, columns=["open", "high", "low", "close", "volume"]))
    row = _features(df).tail(1)
    if row.empty:
        return {"signal": "HOLD", "confidence": 0.0, "metadata": {"reason": "no_features"}}
    prob = model["model"].predict_proba(row[model["features"]].values.astype(np.float32))[0]
    klass = int(np.argmax(prob))
    conf = float(np.max(prob))
    signal = {0: "DOWN", 1: "HOLD", 2: "UP"}[klass]
    if conf < float(p.get("min_confidence", 0.48)):
        signal = "HOLD"
    return {"signal": signal, "confidence": round(conf, 4), "metadata": {"p_sell": round(float(prob[0]), 4), "p_hold": round(float(prob[1]), 4), "p_buy": round(float(prob[2]), 4), "model": "onnx-export-sklearn-starter"}}
