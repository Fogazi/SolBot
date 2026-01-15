# Handoff Notes

Update this file when you stop working so the next session can pick up quickly.

## Current State
- Bot tracks multiple wallets via `wallets` in config, shows wallet in output.
- BUY/SELL lines use green/red circle emoji (TTY only).
- Output includes post-transaction SOL balance.
- Price alerts: records BUY price via Jupiter Price API V3 and alerts on -10%/+20%.
- Price watch persists to `price_watch.json`; cleared on SELL and on trailing TP trigger.
- Trailing TP alerts: arms at +20% gain, tracks peak, triggers on 10% drawdown.

## Decisions / Context
- Price API V3 used for alerts (Ultra is swap execution, not price feed).
- Config additions: `price_url`, `price_poll_sec`, `price_watch_path`, `alert_*`, `trailing_*`.
- Current `config.example.json` contains real values (wallets + API key) and `.gitignore` is empty by user choice.

## Next Steps
1) Restart bot and verify trailing TP alerts with a live trade.
2) Update `config.json` with `trailing_start_pct`/`trailing_drawdown_pct` if needed.
3) Optionally re-add `.gitignore` entries for `config.json` and `price_watch.json` before pushing.

## Issues / Risks
- Python compile check failed due to `python` not on PATH; use `py -3.14 -m py_compile bot.py` if needed.
- With `.gitignore` empty, secrets in `config.json` will be pushed to GitHub.
