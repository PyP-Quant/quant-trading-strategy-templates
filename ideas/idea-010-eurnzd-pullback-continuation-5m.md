# Idea 010: EURNZD Pullback continuation

## Summary

Document a PyP Quant strategy candidate for EURNZD on 5m candles using a LightGBM approach focused on pullback continuation.

## Market Hypothesis

EURNZD may show repeatable behavior when pullback continuation conditions align with volatility, trend, and recent structure filters. The first implementation should stay conservative and treat this as an educational research candidate, not a production trading recommendation.

## Candidate Features

- Recent return over 3, 6, and 12 bars.
- ATR-normalized candle range and close location value.
- Rolling volatility percentile over 50 and 200 bars.
- Distance from EMA 20, EMA 50, and EMA 200.
- Session flag or market-hours bucket where applicable.
- Prior swing high and swing low distance.

## Label Design

- Predict whether the forward 5m move exceeds an ATR-normalized threshold.
- Use neutral labels when the forward return is too small to justify risk.
- Tune the horizon separately for trend, range, and scalp variants.

## Risk Controls

- Require minimum confidence before emitting UP or DOWN.
- Block signals during abnormal spread, missing candles, or low-liquidity windows.
- Cap exposure per instrument and stop after a daily drawdown limit.
- Validate stop-loss and take-profit levels with PPE simulation before live use.

## Backtest Checklist

- Compare against HOLD and simple moving-average baselines.
- Run walk-forward splits with no future leakage.
- Inspect trade count, profit factor, max drawdown, win rate, and average adverse excursion.
- Review behavior by session, volatility regime, and weekday.

## Implementation Notes

This idea is intentionally Markdown-only. A future template can add `strategy.py`, `quant.config.json`, and a focused README once PPE results justify turning the idea into executable code.

Closes #19
