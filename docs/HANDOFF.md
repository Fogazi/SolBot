# Handoff Notes

Update this file when you stop working so the next session can pick up quickly.

## Current State
- Bot tracks multiple wallets; trade output reordered and simplified.
- BUY/SELL trade lines use green/red circle emoji; SELL blocks include PnL with emoji.
- Trailing TP: arms at +20%, tracks peak, triggers on 10% drawdown.
- Trailing status logs show current % and sell trigger %; include estimated MC at stop.
- Stop-loss auto-sells with two levels (-10%, -15%) and retries until success; inflight auto-clears after `sell_retry_sec`.
- Price watch persists to `price_watch.json` and only clears on on-chain SELL detection; if sell fails with balance zero, entry is removed and logged.
- Auto-sell uses Jupiter Ultra (`/ultra/v1/order` + `/execute`) and Ultra holdings for balance.
- Multi-wallet auto-sell supported via `keypair_envs` mapping; per-wallet trading can be toggled via `trade_control.json`.
- Drawdown monitor logs `[LOSS]` at -5% (configurable).
- README updated with `.env`, wallet aliases, trade control, stats usage, and timezone notes.
- Trailing confirm debounce added: `trailing_confirm_sec` (0 to disable); logs confirm start/hit.
- Price polling now logs effective poll rate (`[PRICE]`) based on rate-limit + batch size.
- Discord webhook support for SELL logs (config `discord_webhook_url` / `DISCORD_WEBHOOK_URL`).
- Backfill commands added: `backfill`, `backfill-open`, `backfill-buy`, `backfill-buys`, `recent-buys`, `rebuild-trade-log`.

## Decisions / Context
- Price API V3 used for alerts (Ultra is swap execution, not price feed).
- Swap path is Jupiter Ultra; API key read from `.env` via `jupiter_api_key_env`.
- `.env` added and ignored in `.gitignore`.
- `config.example.json` now uses placeholders and documents new sell/stop-loss fields.

## Next Steps
1) Ensure `keypair_envs` covers wallets that should auto-sell.
2) Verify stop-loss + trailing TP behavior with live trades.
3) Adjust `drawdown_status_*` / `stop_loss_levels_pct` as needed.

## Issues / Risks
- RPC 403s were observed; Ultra holdings now used for sell balance, but RPC is still used for tx parsing.
- `config.json` may still contain secrets; `.env` is now ignored but `config.json` is not.
- Review feedback pending: per-sell PnL uses tx SOL delta (bad for multi-sell tx); `stats --tz` arg order parsing bug.
- `rebuild-trade-log` is destructive; use with a large `--since`/`--limit` or add backups.
