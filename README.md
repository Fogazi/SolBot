# SolBot - Wallet Trade Monitor

This bot watches a Solana wallet and prints a trade summary when it detects a buy or sell. It uses a Solana RPC for transactions and the Jupiter Ultra API for token metadata (including marketcap when provided).

## Setup

1) Create a config:

```bash
Copy-Item config.example.json config.json
```

2) Create a `.env` (example):

```bash
SOLANA_KEYPAIR_1=YOUR_PRIVATE_KEY_FOR_WALLET_1
SOLANA_KEYPAIR_2=YOUR_PRIVATE_KEY_FOR_WALLET_2
JUPITER_API_KEY=YOUR_JUPITER_API_KEY
```

3) Edit `config.json`:
   - Add wallet aliases + list of wallets (`wallet_aliases`, `wallets`)
   - Map each wallet alias to an env key (`keypair_envs`)
   - Add your RPC HTTP + WSS endpoints
   - Set `jupiter_api_key_env` to match your `.env`

4) Install dependencies:

```bash
python -m pip install -r requirements.txt
```

5) Run:

```bash
python bot.py
```

## Output fields

For each detected trade:
- Wallet
- Token name + ticker
- CA (mint address)
- Marketcap (USD)
- SOL used (buy) or SOL received (sell)
- PnL % (sell)
- Wallet SOL balance (post-transaction)

## Notes

- Marketcap is read from the Jupiter token response (field `mcap`) when available.
- SOL used/received is the wallet's net SOL delta for the transaction (fees included).
- Jupiter Ultra endpoints can change; if needed, update `jupiter_tokens_url` and `swap_url` in `config.json`.
- `jupiter_tokens_url` supports list URLs or dynamic URLs that include `{ids}`, `{mint}`, or `{query}`.

## Price Watch

When a BUY is detected, the bot records the current Jupiter Price API (USD) and
tracks trailing TP and stop-loss logic against that buy price. The watch list is
persisted to `price_watch.json` by default.

Config options:
- `price_url`: Price API endpoint (default `https://api.jup.ag/price/v3?ids={ids}`)
- `price_poll_sec`: Polling interval in seconds
- `price_watch_path`: File path for persisting buy prices and alert state
- `trailing_start_pct`: Percent gain to arm the trailing TP (default 20)
- `trailing_drawdown_pct`: Percent drawdown from peak to trigger (default 10)

## Wallet Aliases + Trading Control

- Use `wallet_aliases` to map friendly names to addresses, and list aliases in `wallets`.
- `wallet_trading_enabled` lets you disable auto-trading per wallet while still tracking buys/sells.
- You can toggle trading while the bot runs by writing `trade_control.json` (polled every `trade_control_poll_sec`).

Example `trade_control.json`:

```json
{
  "wallet_trading_enabled": {
    "main": false,
    "alt": true
  }
}
```

CLI helper:

```bash
python bot.py trading --wallet main --off
python bot.py trading --wallet alt --on
```

## Trade Log + Stats

Trades are stored locally in `trade_log.jsonl` (config `trade_log_path`) for fast PnL stats
without RPC calls.

Examples:

```bash
python bot.py stats --since 12h
python bot.py stats --from "2026-01-18 03:00:00" --to "2026-01-18 12:00:00"
python bot.py stats --since 7d --tz 570
```

## Timezone

Console timestamps use `tz_offset_minutes` (default `570` for GMT+9.5).

## Project Handoff

To keep continuity between machines, update `docs/HANDOFF.md` when you stop work.
Start each new session by asking Codex to read it.

## CLI Commands

Trading toggle (runtime control via `trade_control.json`):
```bash
python bot.py trading --wallet <alias|address> --on
python bot.py trading --wallet <alias|address> --off
```

Stats (from local trade log):
```bash
python bot.py stats --since 12h
python bot.py stats --from "2026-01-18 03:00:00" --to "2026-01-18 12:00:00"
python bot.py stats --since 7d --tz 570
```

Open positions (from trade log + price watch comparison):
```bash
python bot.py positions
```

Backfill missing sells for open positions:
```bash
python bot.py backfill-open --since 2h --limit 500
```

Backfill missing sells for a specific wallet + mint:
```bash
python bot.py backfill --wallet <alias|address> --mint <mint> --since 2h --limit 500
```

Backfill missing buys for a specific wallet + mint:
```bash
python bot.py backfill-buy --wallet <alias|address> --mint <mint> --since 2h --limit 500
```

Backfill missing buys for all wallets (auto-detect):
```bash
python bot.py backfill-buys --since 2h --limit 50
python bot.py backfill-buys --wallet <alias|address> --since 2h --limit 50
```

List recent buys to find a mint:
```bash
python bot.py recent-buys --wallet <alias|address> --since 2h --limit 50
```

Rebuild trade log from chain (destructive):
```bash
python bot.py rebuild-trade-log --since 6h --limit 2000
python bot.py rebuild-trade-log --since 6h --limit 2000 --no-watch
```

Discord webhook test:
```bash
python bot.py webhook-test
python bot.py webhook-test --message "Hello from SolBot"
```
