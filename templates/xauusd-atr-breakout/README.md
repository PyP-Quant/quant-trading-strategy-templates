# XAUUSD ATR Breakout Continuation

Custom Python breakout + continuation baseline for XAUUSD.

This project avoids ML on purpose. It is useful as a transparent benchmark to compare against heavier gold models like XGBoost or LightGBM.

The default gate is intentionally responsive so PPE produces more events than a strict long-range breakout filter. It combines:

- hard ATR breakout entries
- near-breakout pressure entries
- trend-continuation entries after range pressure
- EMA trend alignment
- short momentum measured in ATR units

Current defaults:

- `lookback`: `96`
- `breakout_window`: `12`
- `atr_mult`: `0.05`
- `near_breakout_atr`: `0.18`
- `pullback_atr`: `0.35`
- `min_momentum_atr`: `0.08`
- `fast_ema`: `8`
- `slow_ema`: `34`
- assigned runtime: Modal, because this starter uses `pandas`

If it overtrades, raise `near_breakout_atr` more carefully than `atr_mult`: `atr_mult` controls hard breakout distance, while `near_breakout_atr` and `pullback_atr` control the extra continuation paths.
