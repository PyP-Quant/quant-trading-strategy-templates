# XAUUSD ATR Precision Breakout

Custom Python precision-breakout baseline for XAUUSD.

This project avoids ML on purpose. It is useful as a transparent benchmark to compare against heavier gold models like XGBoost or LightGBM.

The default gate is tuned for the `1m` XAUUSD chart and small accounts. It prioritizes controlled churn over forcing oversized trades. The continuation paths are available as parameters, but they are disabled by default because one-day lab tests showed the stricter breakout profile had cleaner drawdown.

It includes:

- hard ATR breakout entries
- optional near-breakout pressure entries
- optional trend-continuation entries after range pressure
- EMA trend alignment
- short momentum measured in ATR units

Current defaults:

- `lookback`: `96`
- `timeframe`: `1m`
- `breakout_window`: `45`
- `atr_mult`: `0.02`
- `near_breakout_atr`: `0.18`
- `pullback_atr`: `0.35`
- `min_momentum_atr`: `0`
- `fast_ema`: `8`
- `slow_ema`: `34`
- `enable_near_breakout`: `false`
- `enable_continuation`: `false`
- assigned runtime: Modal, because this starter uses `pandas`

For more activity, enable `enable_near_breakout` first. Enable `enable_continuation` only after PPE confirms the drawdown remains acceptable.
