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

## Price Alerts

When a BUY is detected, the bot records the current Jupiter Price API (USD) and
alerts if the price drops 10% or rises 20% from that buy price.
The watch list is persisted to `price_watch.json` by default.
Trailing TP alerts can be enabled to arm at +20% and trigger on a 10% drawdown
from the peak.

Config options:
- `price_url`: Price API endpoint (default `https://api.jup.ag/price/v3?ids={ids}`)
- `price_poll_sec`: Polling interval in seconds
- `price_watch_path`: File path for persisting buy prices and alert state
- `alert_drop_pct`: Percent drop threshold (default 10)
- `alert_rise_pct`: Percent rise threshold (default 20)
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

## Sync To Another PC (GitHub)

Use these steps on a second Windows PC to pick up the project:

1) Install Git:
```
winget install --id Git.Git -e
```

2) Clone the repo (replace placeholders):
```
git clone https://github.com/<YOUR_USERNAME>/<YOUR_REPO>.git
cd <YOUR_REPO>
```

3) Create local config:
```
Copy-Item config.example.json config.json
```

4) Edit `config.json` and run:
```
python -m pip install -r requirements.txt
python bot.py
```
