# SolBot - Wallet Trade Monitor

This bot watches a Solana wallet and prints a trade summary when it detects a buy or sell. It uses a Solana RPC for transactions and the Jupiter Ultra API for token metadata (including marketcap when provided).

## Setup

1) Create a config:

```bash
Copy-Item config.example.json config.json
```

2) Edit `config.json`:
   - Add your wallet public key (use `wallet` for one or `wallets` for many)
   - Add your RPC HTTP + WSS endpoints
   - Add your Jupiter API key in `jupiter_headers`

3) Install dependencies:

```bash
python -m pip install -r requirements.txt
```

4) Run:

```bash
python bot.py
```

## Output fields

For each detected trade:
- Wallet
- Token Name
- Ticker
- CA (mint address)
- Marketcap (USD)
- Quantity bought/sold
- SOL used (buy) or SOL received (sell)

## Notes

- Marketcap is read from the Jupiter token response (field `mcap`) when available.
- SOL used/received is the wallet's net SOL delta for the transaction (fees included).
- Jupiter Ultra endpoints can change; if needed, update `jupiter_tokens_url` in `config.json`.
- `jupiter_tokens_url` supports list URLs or dynamic URLs that include `{ids}`, `{mint}`, or `{query}`.

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
