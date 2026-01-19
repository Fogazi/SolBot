# Handoff Notes

Update this file when you stop working so the next session can pick up quickly.

## Current State
- Paper-trading mode added: ignores on-chain SELL detection, logs paper sells, removes watch entries on paper exit.
- Buy-hold window added: `buy_hold_sec` delays all sells for N seconds; `buy_hold_stop_loss_pct` still triggers during hold.
- Ratcheting stops added via `ratchet_levels` (trigger/stop pairs) with `[RATCHET]` arm logs.
- Trailing logic changed to peak percent-points drawdown (e.g., 220pp drop from peak gain).
- RPC endpoints can now be read from `.env` via `rpc_http_env`/`rpc_ws_env`/`rpc_http_backfill_env`.
- Price watch now stores `buy_ts` and ratchet state for persistence.
- `price_watch.json` is no longer tracked in git; runtime files are ignored.
- Dev config set to paper trading + hold + ratchet levels; tracking only the `copy` wallet.

## Decisions / Context
- Paper branch pushed: `paper-trader` contains all new strategy work.
- RPC secrets moved out of `config.json` into `.env` (see `RPC_HTTP`, `RPC_WS`, `RPC_HTTP_BACKFILL`).

## Next Steps
1) Decide which paper-trading changes to port to live (ratchet, hold stop, new trailing).
2) If desired, add a dedicated `config.paper.json` to keep live config clean.
3) Validate paper-trader behavior with real buys and compare against live exits.

## Issues / Risks
- Paper trading skips on-chain SELL detection entirely, so paper exits may diverge from live.
- Trailing with large drawdown (220pp) provides little protection below 220% peak unless other stops hit.
