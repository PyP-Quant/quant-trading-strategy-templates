from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_ROOT = ROOT / "templates"


SPECS = [
    ("audusd-logistic-30m", "AUDUSD", "30m", "sklearn LogisticRegression", "AUDUSD momentum classifier", "logistic"),
    ("audjpy-randomforest-1h", "AUDJPY", "1h", "sklearn RandomForest", "AUDJPY carry-risk classifier", "forest"),
    ("cadjpy-breakout-1h", "CADJPY", "1h", "sklearn RandomForest", "CADJPY breakout classifier", "breakout"),
    ("chfjpy-mean-reversion-1h", "CHFJPY", "1h", "custom Python", "CHFJPY mean-reversion baseline", "mean_reversion"),
    ("eurgbp-range-classifier-30m", "EURGBP", "30m", "sklearn LogisticRegression", "EURGBP range classifier", "logistic"),
    ("eurjpy-trend-rf-1h", "EURJPY", "1h", "sklearn RandomForest", "EURJPY trend classifier", "forest"),
    ("eurnzd-volatility-rf-4h", "EURNZD", "4h", "sklearn RandomForest", "EURNZD volatility classifier", "forest"),
    ("gbpjpy-breakout-rf-1h", "GBPJPY", "1h", "sklearn RandomForest", "GBPJPY breakout classifier", "breakout"),
    ("gbpusd-logistic-15m", "GBPUSD", "15m", "sklearn LogisticRegression", "GBPUSD directional baseline", "logistic"),
    ("nzdusd-mean-reversion-1h", "NZDUSD", "1h", "custom Python", "NZDUSD mean-reversion baseline", "mean_reversion"),
    ("usdcad-trend-rf-1h", "USDCAD", "1h", "sklearn RandomForest", "USDCAD trend classifier", "forest"),
    ("usdchf-range-logistic-1h", "USDCHF", "1h", "sklearn LogisticRegression", "USDCHF range classifier", "logistic"),
    ("usdmxn-volatility-rf-4h", "USDMXN", "4h", "sklearn RandomForest", "USDMXN volatility classifier", "forest"),
    ("xagusd-atr-breakout-1h", "XAGUSD", "1h", "custom Python", "Silver ATR breakout baseline", "atr_breakout"),
    ("xagusd-mean-reversion-30m", "XAGUSD", "30m", "custom Python", "Silver mean-reversion baseline", "mean_reversion"),
    ("xauusd-london-breakout-15m", "XAUUSD", "15m", "custom Python", "Gold London breakout baseline", "breakout"),
    ("xauusd-scalp-logistic-5m", "XAUUSD", "5m", "sklearn LogisticRegression", "Gold scalp classifier", "logistic"),
    ("xptusd-trend-rf-1h", "XPTUSD", "1h", "sklearn RandomForest", "Platinum trend classifier", "forest"),
    ("btcusdt-breakout-rf-15m", "BTCUSDT", "15m", "sklearn RandomForest", "BTC breakout classifier", "breakout"),
    ("btcusdt-mean-reversion-5m", "BTCUSDT", "5m", "custom Python", "BTC mean-reversion baseline", "mean_reversion"),
    ("bnbusdt-trend-rf-30m", "BNBUSDT", "30m", "sklearn RandomForest", "BNB trend classifier", "forest"),
    ("dogeusdt-scalp-logistic-1m", "DOGEUSDT", "1m", "sklearn LogisticRegression", "DOGE high-volatility scalp classifier", "logistic"),
    ("adausdt-volatility-rf-15m", "ADAUSDT", "15m", "sklearn RandomForest", "ADA volatility classifier", "forest"),
    ("xrpusdt-breakout-rf-15m", "XRPUSDT", "15m", "sklearn RandomForest", "XRP breakout classifier", "breakout"),
    ("linkusdt-trend-logistic-30m", "LINKUSDT", "30m", "sklearn LogisticRegression", "LINK trend classifier", "logistic"),
    ("avaxusdt-volatility-rf-30m", "AVAXUSDT", "30m", "sklearn RandomForest", "AVAX volatility classifier", "forest"),
    ("maticusdt-scalp-baseline-5m", "MATICUSDT", "5m", "custom Python", "MATIC scalp baseline", "atr_breakout"),
    ("dotusdt-mean-reversion-30m", "DOTUSDT", "30m", "custom Python", "DOT mean-reversion baseline", "mean_reversion"),
    ("ltcusdt-trend-rf-1h", "LTCUSDT", "1h", "sklearn RandomForest", "LTC trend classifier", "forest"),
    ("nas100-breakout-rf-15m", "NAS100", "15m", "sklearn RandomForest", "NASDAQ index breakout classifier", "breakout"),
    ("us30-mean-reversion-30m", "US30", "30m", "custom Python", "Dow index mean-reversion baseline", "mean_reversion"),
    ("spx500-trend-logistic-1h", "SPX500", "1h", "sklearn LogisticRegression", "S&P 500 trend classifier", "logistic"),
    ("ger40-breakout-rf-30m", "GER40", "30m", "sklearn RandomForest", "DAX breakout classifier", "breakout"),
    ("uk100-range-logistic-1h", "UK100", "1h", "sklearn LogisticRegression", "FTSE range classifier", "logistic"),
    ("oilwtico-atr-breakout-1h", "WTICOUSD", "1h", "custom Python", "WTI crude oil ATR breakout baseline", "atr_breakout"),
    ("brentoil-trend-rf-1h", "BRENTUSD", "1h", "sklearn RandomForest", "Brent crude trend classifier", "forest"),
    ("naturalgas-volatility-rf-4h", "NATGAS", "4h", "sklearn RandomForest", "Natural gas volatility classifier", "forest"),
    ("copperusd-mean-reversion-1h", "COPPERUSD", "1h", "custom Python", "Copper mean-reversion baseline", "mean_reversion"),
]


STRATEGIES = {
    "logistic": '''import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


SYMBOL = "{symbol}"
MODEL_NAME = "{slug}"


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
    for n in [1, 3, 6, 12, 24]:
        f[f"ret{{n}}"] = c.pct_change(n)
    f["range_pct"] = rng / c
    f["body_pct"] = (c - df["open"]) / (rng + 1e-9)
    f["close_pos"] = (c - df["low"]) / (rng + 1e-9)
    f["ema_8_21"] = (c.ewm(span=8, adjust=False).mean() - c.ewm(span=21, adjust=False).mean()) / c
    f["ema_21_55"] = (c.ewm(span=21, adjust=False).mean() - c.ewm(span=55, adjust=False).mean()) / c
    f["volatility"] = c.pct_change().rolling(24).std()
    f["volume_ratio"] = df["volume"] / (df["volume"].rolling(24).mean() + 1e-9)
    return f.replace([np.inf, -np.inf], np.nan).dropna()


def train(data, config):
    p = config.get("parameters", {{}})
    df = _prep(data)
    x = _features(df)
    horizon = int(p.get("horizon", 4))
    threshold = float(p.get("threshold", {threshold}))
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
    return {{"model": model, "features": list(x.columns), "symbol": SYMBOL}}, {{"training_bars": int(len(x)), "feature_count": int(x.shape[1]), "buy_signals": int((pred == 2).sum()), "sell_signals": int((pred == 0).sum()), "hold_signals": int((pred == 1).sum())}}


def predict(model, market_data, config):
    p = config.get("parameters", {{}})
    candles = market_data.get("candles", [])
    if len(candles) < int(p.get("lookback", {lookback})):
        return {{"signal": "HOLD", "confidence": 0.0, "metadata": {{"reason": "not_enough_candles", "model": MODEL_NAME}}}}
    df = _prep(pd.DataFrame(candles, columns=["open", "high", "low", "close", "volume"]))
    row = _features(df).tail(1)
    if row.empty:
        return {{"signal": "HOLD", "confidence": 0.0, "metadata": {{"reason": "no_features", "model": MODEL_NAME}}}}
    prob = model["model"].predict_proba(row[model["features"]].values.astype(np.float32))[0]
    klass = int(np.argmax(prob))
    conf = float(np.max(prob))
    signal = {{0: "DOWN", 1: "HOLD", 2: "UP"}}[klass]
    if conf < float(p.get("min_confidence", 0.48)):
        signal = "HOLD"
    return {{"signal": signal, "confidence": round(conf, 4), "metadata": {{"p_sell": round(float(prob[0]), 4), "p_hold": round(float(prob[1]), 4), "p_buy": round(float(prob[2]), 4), "model": MODEL_NAME, "symbol": SYMBOL}}}}
''',
    "forest": '''import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


SYMBOL = "{symbol}"
MODEL_NAME = "{slug}"


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
        f[f"ret{{n}}"] = c.pct_change(n)
    f["ema_12_48"] = (c.ewm(span=12, adjust=False).mean() - c.ewm(span=48, adjust=False).mean()) / c
    f["ema_24_96"] = (c.ewm(span=24, adjust=False).mean() - c.ewm(span=96, adjust=False).mean()) / c
    f["volatility_fast"] = c.pct_change().rolling(16).std()
    f["volatility_slow"] = c.pct_change().rolling(64).std()
    f["volatility_ratio"] = f["volatility_fast"] / (f["volatility_slow"] + 1e-9)
    f["volume_z"] = (v - v.rolling(48).mean()) / (v.rolling(48).std() + 1e-9)
    return f.replace([np.inf, -np.inf], np.nan).dropna()


def train(data, config):
    p = config.get("parameters", {{}})
    df = _prep(data)
    x = _features(df)
    horizon = int(p.get("horizon", 6))
    threshold = float(p.get("threshold", {threshold}))
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
    return {{"model": model, "features": list(x.columns), "symbol": SYMBOL}}, {{"training_bars": int(len(x)), "feature_count": int(x.shape[1]), "buy_signals": int((pred == 2).sum()), "sell_signals": int((pred == 0).sum()), "hold_signals": int((pred == 1).sum())}}


def predict(model, market_data, config):
    p = config.get("parameters", {{}})
    candles = market_data.get("candles", [])
    if len(candles) < int(p.get("lookback", {lookback})):
        return {{"signal": "HOLD", "confidence": 0.0, "metadata": {{"reason": "not_enough_candles", "model": MODEL_NAME}}}}
    df = _prep(pd.DataFrame(candles, columns=["open", "high", "low", "close", "volume"]))
    row = _features(df).tail(1)
    if row.empty:
        return {{"signal": "HOLD", "confidence": 0.0, "metadata": {{"reason": "no_features", "model": MODEL_NAME}}}}
    prob = model["model"].predict_proba(row[model["features"]].values.astype(np.float32))[0]
    klass = int(np.argmax(prob))
    conf = float(np.max(prob))
    signal = {{0: "DOWN", 1: "HOLD", 2: "UP"}}[klass]
    if conf < float(p.get("min_confidence", 0.5)):
        signal = "HOLD"
    return {{"signal": signal, "confidence": round(conf, 4), "metadata": {{"p_sell": round(float(prob[0]), 4), "p_hold": round(float(prob[1]), 4), "p_buy": round(float(prob[2]), 4), "model": MODEL_NAME, "symbol": SYMBOL}}}}
''',
    "breakout": '''import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


SYMBOL = "{symbol}"
MODEL_NAME = "{slug}"


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
    p = config.get("parameters", {{}})
    df = _prep(data)
    x = _features(df)
    horizon = int(p.get("horizon", 4))
    threshold = float(p.get("threshold", {threshold}))
    fwd = df["close"].pct_change(horizon).shift(-horizon)
    y = pd.Series(1, index=df.index)
    y[fwd > threshold] = 2
    y[fwd < -threshold] = 0
    y = y.reindex(x.index).fillna(1).astype(int)
    clf = RandomForestClassifier(n_estimators=300, max_depth=7, min_samples_leaf=6, class_weight="balanced_subsample", random_state=42, n_jobs=-1)
    clf.fit(x.values.astype(np.float32), y.values)
    pred = clf.predict(x.values.astype(np.float32))
    return {{"model": clf, "features": list(x.columns), "symbol": SYMBOL}}, {{"training_bars": int(len(x)), "feature_count": int(x.shape[1]), "buy_signals": int((pred == 2).sum()), "sell_signals": int((pred == 0).sum()), "hold_signals": int((pred == 1).sum())}}


def predict(model, market_data, config):
    p = config.get("parameters", {{}})
    candles = market_data.get("candles", [])
    if len(candles) < int(p.get("lookback", {lookback})):
        return {{"signal": "HOLD", "confidence": 0.0, "metadata": {{"reason": "not_enough_candles", "model": MODEL_NAME}}}}
    df = _prep(pd.DataFrame(candles, columns=["open", "high", "low", "close", "volume"]))
    row = _features(df).tail(1)
    if row.empty:
        return {{"signal": "HOLD", "confidence": 0.0, "metadata": {{"reason": "no_features", "model": MODEL_NAME}}}}
    prob = model["model"].predict_proba(row[model["features"]].values.astype(np.float32))[0]
    klass = int(np.argmax(prob))
    conf = float(np.max(prob))
    signal = {{0: "DOWN", 1: "HOLD", 2: "UP"}}[klass]
    if conf < float(p.get("min_confidence", 0.5)):
        signal = "HOLD"
    return {{"signal": signal, "confidence": round(conf, 4), "metadata": {{"p_sell": round(float(prob[0]), 4), "p_hold": round(float(prob[1]), 4), "p_buy": round(float(prob[2]), 4), "model": MODEL_NAME, "symbol": SYMBOL}}}}
''',
    "mean_reversion": '''import numpy as np
import pandas as pd


SYMBOL = "{symbol}"
MODEL_NAME = "{slug}"


def _prep(data):
    df = data.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    if "volume" not in df.columns:
        df["volume"] = 1.0
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)


def _zscore(close, window):
    mean = close.rolling(window).mean()
    std = close.rolling(window).std()
    return (close - mean) / (std + 1e-9)


def train(data, config):
    p = config.get("parameters", {{}})
    return {{"window": int(p.get("window", 48)), "entry_z": float(p.get("entry_z", 1.4)), "exit_z": float(p.get("exit_z", 0.3)), "symbol": SYMBOL}}, {{"model_family": "rule_based_mean_reversion", "training_bars": int(len(data))}}


def predict(model, market_data, config):
    p = config.get("parameters", {{}})
    candles = market_data.get("candles", [])
    lookback = int(p.get("lookback", {lookback}))
    if len(candles) < lookback:
        return {{"signal": "HOLD", "confidence": 0.0, "metadata": {{"reason": "not_enough_candles", "model": MODEL_NAME}}}}
    df = _prep(pd.DataFrame(candles, columns=["open", "high", "low", "close", "volume"]))
    window = int(model.get("window", p.get("window", 48)))
    z = float(_zscore(df["close"], window).iloc[-1])
    entry_z = float(model.get("entry_z", p.get("entry_z", 1.4)))
    signal = "HOLD"
    if z <= -entry_z:
        signal = "UP"
    elif z >= entry_z:
        signal = "DOWN"
    confidence = min(abs(z) / max(entry_z, 1e-9), 1.0)
    return {{"signal": signal, "confidence": round(float(confidence), 4), "metadata": {{"zscore": round(z, 4), "window": window, "model": MODEL_NAME, "symbol": SYMBOL}}}}
''',
    "atr_breakout": '''import numpy as np
import pandas as pd


SYMBOL = "{symbol}"
MODEL_NAME = "{slug}"


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


def train(data, config):
    p = config.get("parameters", {{}})
    return {{"atr_window": int(p.get("atr_window", 14)), "breakout_window": int(p.get("breakout_window", 36)), "atr_mult": float(p.get("atr_mult", 0.25)), "symbol": SYMBOL}}, {{"model_family": "rule_based_atr_breakout", "training_bars": int(len(data))}}


def predict(model, market_data, config):
    p = config.get("parameters", {{}})
    candles = market_data.get("candles", [])
    lookback = int(p.get("lookback", {lookback}))
    if len(candles) < lookback:
        return {{"signal": "HOLD", "confidence": 0.0, "metadata": {{"reason": "not_enough_candles", "model": MODEL_NAME}}}}
    df = _prep(pd.DataFrame(candles, columns=["open", "high", "low", "close", "volume"]))
    atr_window = int(model.get("atr_window", p.get("atr_window", 14)))
    breakout_window = int(model.get("breakout_window", p.get("breakout_window", 36)))
    atr_mult = float(model.get("atr_mult", p.get("atr_mult", 0.25)))
    atr = _atr(df, atr_window)
    close = float(df["close"].iloc[-1])
    upper = float(df["high"].rolling(breakout_window).max().shift().iloc[-1] + atr.iloc[-1] * atr_mult)
    lower = float(df["low"].rolling(breakout_window).min().shift().iloc[-1] - atr.iloc[-1] * atr_mult)
    signal = "HOLD"
    distance = 0.0
    if close > upper:
        signal = "UP"
        distance = (close - upper) / max(atr.iloc[-1], 1e-9)
    elif close < lower:
        signal = "DOWN"
        distance = (lower - close) / max(atr.iloc[-1], 1e-9)
    confidence = min(max(distance, 0.0), 1.0)
    return {{"signal": signal, "confidence": round(float(confidence), 4), "metadata": {{"upper": round(upper, 6), "lower": round(lower, 6), "close": round(close, 6), "model": MODEL_NAME, "symbol": SYMBOL}}}}
''',
}


def default_threshold(symbol: str, family: str) -> float:
    if symbol.endswith("USDT"):
        return 0.004 if family != "logistic" else 0.003
    if symbol in {"XAUUSD", "XAGUSD", "XPTUSD", "WTICOUSD", "BRENTUSD", "NATGAS", "COPPERUSD"}:
        return 0.003
    if symbol in {"NAS100", "US30", "SPX500", "GER40", "UK100"}:
        return 0.0025
    if "JPY" in symbol:
        return 0.0018
    if symbol.endswith("MXN"):
        return 0.003
    return 0.0012


def default_lookback(timeframe: str, family: str) -> int:
    if timeframe == "1m":
        return 120
    if timeframe in {"5m", "15m"}:
        return 100
    if timeframe == "30m":
        return 120
    if timeframe == "4h":
        return 180
    return 140


def readme(slug: str, symbol: str, timeframe: str, model_family: str, purpose: str) -> str:
    return f"""# {slug}

{purpose} for PyP Quant Mode.

This is an educational starter template for `{symbol}` on the `{timeframe}` timeframe. It implements the PyP Quant contract:

```python
train(data, config)
predict(model, market_data, config)
```

Use it as a baseline, then validate with PPE before any live deployment.

## Model

- Symbol: `{symbol}`
- Timeframe: `{timeframe}`
- Family: `{model_family}`
- Output: `UP`, `DOWN`, or `HOLD`

## PyP Links

- Quant docs: https://pyp.stanl.ink/docs/quant/what-is-quant-mode
- Quant landing page: https://pyp.stanl.ink/for-quant-traders
- Create project: https://pyp.stanlink.online/projects/quant/new

## Risk

This is not financial advice and is not a verified profitable strategy.
"""


def config(slug: str, symbol: str, timeframe: str, model_family: str, family: str) -> dict:
    threshold = default_threshold(symbol, family)
    lookback = default_lookback(timeframe, family)
    requirements = ["numpy", "pandas"]
    if family == "logistic":
        requirements.append("scikit-learn")
    if family in {"forest", "breakout"}:
        requirements.append("scikit-learn")
    return {
        "name": slug,
        "description": f"{model_family} PyP Quant template for {symbol} {timeframe}.",
        "symbol": symbol,
        "timeframe": timeframe,
        "model_family": model_family,
        "artifact_target": "joblib" if family in {"logistic", "forest", "breakout"} else "python",
        "parameters": {
            "lookback": lookback,
            "horizon": 4 if timeframe in {"1m", "5m", "15m", "30m"} else 6,
            "threshold": threshold,
            "min_confidence": 0.48 if family == "logistic" else 0.5,
            "sl_percent": 0.4 if symbol.endswith("USDT") else 0.25,
            "tp_percent": 0.8 if symbol.endswith("USDT") else 0.5,
        },
        "requirements": requirements,
        "disclaimer": "Educational template only. Not financial advice.",
    }


def main() -> None:
    created = 0
    for slug, symbol, timeframe, model_family, purpose, family in SPECS:
        target = TEMPLATE_ROOT / slug
        if target.exists():
            continue
        target.mkdir(parents=True)
        threshold = default_threshold(symbol, family)
        lookback = default_lookback(timeframe, family)
        (target / "README.md").write_text(readme(slug, symbol, timeframe, model_family, purpose), encoding="utf-8")
        (target / "quant.config.json").write_text(json.dumps(config(slug, symbol, timeframe, model_family, family), indent=2) + "\n", encoding="utf-8")
        strategy = STRATEGIES[family].format(slug=slug, symbol=symbol, threshold=threshold, lookback=lookback)
        (target / "strategy.py").write_text(strategy, encoding="utf-8")
        created += 1
    print(f"Created {created} templates")


if __name__ == "__main__":
    main()
