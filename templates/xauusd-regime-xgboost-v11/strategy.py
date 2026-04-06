import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def _normalise(data):
    df = data.copy()
    df.columns = [str(c).lower() for c in df.columns]
    if "volume" not in df.columns:
        df["volume"] = 1.0
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)


def _atr(df, n):
    prev = df["close"].shift()
    tr = pd.concat([(df["high"] - df["low"]), (df["high"] - prev).abs(), (df["low"] - prev).abs()], axis=1).max(axis=1)
    return tr.ewm(span=n, adjust=False).mean()


def _features(df):
    c = df["close"]
    atr8 = _atr(df, 8)
    atr50 = _atr(df, 50)
    f = pd.DataFrame(index=df.index)
    f["hurst_proxy"] = atr8 / (atr50 + 1e-10)
    f["atr_norm"] = _atr(df, 14) / (c + 1e-10)
    f["ret3"] = c.pct_change(3)
    f["ret6"] = c.pct_change(6)
    f["ret20"] = c.pct_change(20)
    f["ema8_21"] = (c.ewm(span=8, adjust=False).mean() - c.ewm(span=21, adjust=False).mean()) / c
    f["ema21_55"] = (c.ewm(span=21, adjust=False).mean() - c.ewm(span=55, adjust=False).mean()) / c
    f["price_vs_200"] = (c - c.ewm(span=200, adjust=False).mean()) / c
    rng = (df["high"] - df["low"]).replace(0, np.nan)
    f["body_pct"] = (c - df["open"]) / rng
    f["close_position"] = (c - df["low"]) / rng
    return f.iloc[200:].replace([np.inf, -np.inf], np.nan).dropna()


def _labels(df, index, horizon, threshold, atr_mult):
    close = df["close"]
    fwd = close.pct_change(horizon).shift(-horizon)
    dyn = ((_atr(df, 14) / close) * atr_mult).clip(lower=threshold)
    y = pd.Series(1, index=df.index)
    y[fwd > dyn] = 2
    y[fwd < -dyn] = 0
    return y.reindex(index).fillna(1).astype(int)


def train(data, config):
    params = config.get("parameters", {})
    df = _normalise(data)
    feat = _features(df)
    y = _labels(df, feat.index, int(params.get("horizon", 6)), float(params.get("threshold", 0.0015)), float(params.get("atr_mult", 1.0)))
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", XGBClassifier(n_estimators=350, max_depth=4, learning_rate=0.05, subsample=0.85, colsample_bytree=0.85, objective="multi:softprob", num_class=3, eval_metric="mlogloss", tree_method="hist", random_state=79)),
    ])
    model.fit(feat.values.astype(np.float32), y.values)
    preds = model.predict(feat.values.astype(np.float32))
    return {"model": model, "features": list(feat.columns)}, {"training_bars": int(len(feat)), "buy_signals": int((preds == 2).sum()), "sell_signals": int((preds == 0).sum()), "hold_signals": int((preds == 1).sum())}


def predict(model, market_data, config):
    p = config.get("parameters", {})
    candles = market_data.get("candles", [])
    if len(candles) < int(p.get("lookback", 220)):
        return {"signal": "HOLD", "confidence": 0.0, "metadata": {"reason": "not_enough_candles"}}
    df = _normalise(pd.DataFrame(candles, columns=["open", "high", "low", "close", "volume"]))
    row = _features(df).tail(1)
    if row.empty:
        return {"signal": "HOLD", "confidence": 0.0, "metadata": {"reason": "no_features"}}
    row = row[model["features"]]
    prob = model["model"].predict_proba(row.values.astype(np.float32))[0]
    p_sell, p_hold, p_buy = map(float, prob)
    edge = max(p_buy, p_sell) - p_hold
    min_prob = float(p.get("min_direction_prob", 0.38))
    min_edge = float(p.get("min_edge", 0.03))
    if p_buy >= min_prob and edge >= min_edge and p_buy > p_sell:
        signal, confidence = "UP", p_buy
    elif p_sell >= min_prob and edge >= min_edge and p_sell > p_buy:
        signal, confidence = "DOWN", p_sell
    else:
        signal, confidence = "HOLD", p_hold
    return {"signal": signal, "confidence": round(confidence, 4), "metadata": {"p_buy": round(p_buy, 4), "p_sell": round(p_sell, 4), "p_hold": round(p_hold, 4), "edge": round(edge, 4), "hurst_proxy": round(float(row["hurst_proxy"].iloc[0]), 4)}}
