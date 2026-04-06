import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def _dataset(data, horizon=4, threshold=0.0012):
    df = data.copy()
    df.columns = [str(c).lower() for c in df.columns]
    if "volume" not in df.columns:
        df["volume"] = 1
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    c = df["close"]
    x = pd.DataFrame(index=df.index)
    x["range_break"] = (c - df["high"].rolling(24).max().shift()) / c
    x["range_floor"] = (c - df["low"].rolling(24).min().shift()) / c
    x["ret4"] = c.pct_change(4)
    x["ret12"] = c.pct_change(12)
    x["vol"] = c.pct_change().rolling(20).std()
    x = x.replace([np.inf, -np.inf], np.nan).dropna()
    fwd = c.pct_change(horizon).shift(-horizon)
    y = pd.Series(1, index=df.index)
    y[fwd > threshold] = 2
    y[fwd < -threshold] = 0
    return x, y.reindex(x.index).fillna(1).astype(int)


def train(data, config):
    p = config.get("parameters", {})
    x, y = _dataset(data, int(p.get("horizon", 4)), float(p.get("threshold", 0.0012)))
    clf = RandomForestClassifier(n_estimators=300, max_depth=7, class_weight="balanced_subsample", random_state=42)
    clf.fit(x.values, y.values)
    return {"model": clf, "features": list(x.columns)}, {"training_bars": int(len(x)), "feature_count": int(x.shape[1])}


def predict(model, market_data, config):
    candles = market_data.get("candles", [])
    if len(candles) < int(config.get("parameters", {}).get("lookback", 120)):
        return {"signal": "HOLD", "confidence": 0, "metadata": {"reason": "not_enough_candles"}}
    x, _ = _dataset(pd.DataFrame(candles, columns=["open", "high", "low", "close", "volume"]))
    row = x.tail(1)[model["features"]]
    prob = model["model"].predict_proba(row.values)[0]
    k = int(np.argmax(prob))
    conf = float(np.max(prob))
    return {"signal": {0: "DOWN", 1: "HOLD", 2: "UP"}[k] if conf >= float(config.get("parameters", {}).get("min_confidence", 0.5)) else "HOLD", "confidence": round(conf, 4), "metadata": {"proba": [round(float(v), 4) for v in prob]}}
