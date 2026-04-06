# XAUUSD ATR Breakout

Custom Python breakout baseline for XAUUSD.

This project avoids ML on purpose. It is useful as a transparent baseline to compare against heavier gold models like XGBoost or LightGBM.

The default gate is intentionally responsive so PPE produces more events than a strict long-range breakout filter:

- `breakout_window`: `12`
- `atr_mult`: `0.05`
- assigned runtime: Modal, because this starter uses `pandas`

If it overtrades, raise `atr_mult` first. If it is still too quiet, shorten `breakout_window`.
