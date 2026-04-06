import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def _normalise(data):
    df = data.copy()
    df.columns = [str(c).lower() for c in df.columns]
    if "volume" not in df.columns:
        df["volume"] = 1.0
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)


def _features(df):
    c = df["close"]
    rng = (df["high"] - df["low"]).replace(0, np.nan)
    out = pd.DataFrame(index=df.index)
    out["ret1"] = c.pct_change()
    out["ret4"] = c.pct_change(4)
    out["ret12"] = c.pct_change(12)
    out["range_pct"] = rng / c
    out["body_pct"] = (c - df["open"]) / rng
    out["close_pos"] = (c - df["low"]) / rng
    out["volatility"] = out["ret1"].rolling(20).std()
    out["ma_fast"] = (c.rolling(8).mean() - c.rolling(21).mean()) / c
    out["ma_slow"] = (c.rolling(21).mean() - c.rolling(55).mean()) / c
    return out.replace([np.inf, -np.inf], np.nan).dropna()


def _labels(close, index, horizon, threshold):
    fwd = close.pct_change(horizon).shift(-horizon)
    y = pd.Series(1, index=close.index)
    y[fwd > threshold] = 2
    y[fwd < -threshold] = 0
    return y.reindex(index).fillna(1).astype(int)


def train(data, config):
    params = config.get("parameters", {})
    horizon = int(params.get("horizon", 4))
    threshold = float(params.get("threshold", 0.0008))
    df = _normalise(data)
    feat = _features(df)
    y = _labels(df["close"], feat.index, horizon, threshold)
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", multi_class="auto")),
    ])
    model.fit(feat.values, y.values)
    preds = model.predict(feat.values)
    metrics = {
        "training_bars": int(len(feat)),
        "feature_count": int(feat.shape[1]),
        "buy_signals": int((preds == 2).sum()),
        "sell_signals": int((preds == 0).sum()),
        "hold_signals": int((preds == 1).sum()),
    }
    return {"model": model, "features": list(feat.columns)}, metrics


def predict(model, market_data, config):
    candles = market_data.get("candles", [])
    lookback = int(config.get("parameters", {}).get("lookback", 80))
    min_conf = float(config.get("parameters", {}).get("min_confidence", 0.48))
    if len(candles) < lookback:
        return {"signal": "HOLD", "confidence": 0.0, "metadata": {"reason": "not_enough_candles"}}
    df = _normalise(pd.DataFrame(candles, columns=["open", "high", "low", "close", "volume"]))
    feat = _features(df).tail(1)
    if feat.empty:
        return {"signal": "HOLD", "confidence": 0.0, "metadata": {"reason": "no_features"}}
    proba = model["model"].predict_proba(feat.values)[0]
    klass = int(np.argmax(proba))
    conf = float(proba[klass])
    signal = {0: "DOWN", 1: "HOLD", 2: "UP"}[klass]
    if conf < min_conf:
        signal = "HOLD"
    return {"signal": signal, "confidence": round(conf, 4), "metadata": {"p_sell": round(float(proba[0]), 4), "p_hold": round(float(proba[1]), 4), "p_buy": round(float(proba[2]), 4)}}
