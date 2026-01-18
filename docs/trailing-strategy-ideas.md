# Trailing TP Ideas

These are strategy ideas to reduce whipsaws when a token dips quickly and rips back.

1) Double-confirm drawdown
   - Trigger only if the drawdown holds for X seconds or two consecutive polls.

2) Volatility-based trailing (ATR-style)
   - Expand trailing distance when volatility is high, tighten when stable.

3) Step-up trailing
   - Increase trailing buffer as gains rise (e.g., +20% -> 12%, +50% -> 15%).

4) Partial exits
   - Sell a portion on first trigger, keep the rest on a wider trailing stop.

5) Minimum-hold after arm
   - Once armed, require price to stay above a floor gain for a minimum time.

6) Peak cooldown
   - After a new peak, ignore drawdown triggers for a short cooldown window.

7) Re-arm logic
   - If a sell triggers and price reclaims the old peak quickly, re-arm or re-enter.

8) Volume/velocity filter (if available)
   - Only trigger trailing when momentum slows or reversal signals appear.
