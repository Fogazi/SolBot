import asyncio
import json
import sys
import time
from collections import deque
from decimal import Decimal, ROUND_HALF_UP, getcontext
from pathlib import Path

import httpx
import websockets

getcontext().prec = 28

STABLE_MINTS = {
    # USDC
    "EPjFWdd5AufqSSqeM2q1xzybapC8G4wEGGkZwyTDt1v",
    # USDT
    "Es9vMFrzaCER1FhVtq4D4Kc7YbM7U4oT8wM9KcJ5fYh",
}

JUPITER_TOKENS_URL_DEFAULT = "https://api.jup.ag/ultra/v1/search?query={query}"
JUPITER_REFRESH_SEC_DEFAULT = 600
GREEN_CIRCLE = "\U0001F7E2"
RED_CIRCLE = "\U0001F534"


def load_config(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def key_str(key) -> str:
    if isinstance(key, str):
        return key
    return key.get("pubkey", "")


def aggregate_token_balances(token_balances, owner: str):
    totals = {}
    decimals = {}
    if not token_balances:
        return totals, decimals
    for entry in token_balances:
        if entry.get("owner") != owner:
            continue
        mint = entry.get("mint")
        ui = entry.get("uiTokenAmount", {})
        amount = int(ui.get("amount", "0"))
        totals[mint] = totals.get(mint, 0) + amount
        if mint not in decimals:
            decimals[mint] = int(ui.get("decimals", 0))
    return totals, decimals


def diff_token_balances(pre, post, owner: str):
    pre_totals, pre_decimals = aggregate_token_balances(pre, owner)
    post_totals, post_decimals = aggregate_token_balances(post, owner)

    mints = set(pre_totals.keys()) | set(post_totals.keys())
    for mint in mints:
        decimals = post_decimals.get(mint, pre_decimals.get(mint, 0))
        pre_amt = pre_totals.get(mint, 0)
        post_amt = post_totals.get(mint, 0)
        delta = post_amt - pre_amt
        if delta != 0:
            yield mint, delta, decimals


def format_amount(raw_amount: int, decimals: int) -> str:
    ui = Decimal(raw_amount) / (Decimal(10) ** decimals)
    return f"{ui.normalize():f}"


def format_usd(value: Decimal) -> str:
    return f"{value.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP):f}"


def sol_delta(meta: dict, message: dict, wallet: str) -> Decimal:
    keys = message.get("accountKeys", [])
    wallet_index = None
    for i, k in enumerate(keys):
        if key_str(k) == wallet:
            wallet_index = i
            break
    if wallet_index is None:
        return Decimal(0)
    pre = meta.get("preBalances", [])[wallet_index]
    post = meta.get("postBalances", [])[wallet_index]
    return Decimal(post - pre) / Decimal(1_000_000_000)


class RpcClient:
    def __init__(self, http_url: str, timeout_sec: int = 20):
        self.http_url = http_url
        self.client = httpx.AsyncClient(timeout=timeout_sec)
        self._request_id = 1

    async def call(self, method: str, params: list):
        payload = {
            "jsonrpc": "2.0",
            "id": self._request_id,
            "method": method,
            "params": params,
        }
        self._request_id += 1
        resp = await self.client.post(self.http_url, json=payload)
        resp.raise_for_status()
        data = resp.json()
        if "error" in data:
            raise RuntimeError(data["error"])
        return data.get("result")

    async def close(self):
        await self.client.aclose()


class JupiterTokenCache:
    def __init__(self, url: str, refresh_sec: int, headers: dict, timeout_sec: int = 20):
        self.url = url
        self.refresh_sec = refresh_sec
        self.headers = headers
        self.client = httpx.AsyncClient(timeout=timeout_sec)
        self.data = {}
        self.last_refresh = 0.0
        if "{mint}" in url or "{query}" in url:
            self.mode = "single"
        elif "{ids}" in url:
            self.mode = "batch"
        else:
            self.mode = "list"

    def get(self, mint: str):
        return self.data.get(mint)

    @staticmethod
    def _parse_decimal(value):
        if value is None:
            return None
        try:
            return Decimal(str(value))
        except Exception:
            return None

    @staticmethod
    def _extract_tokens(payload):
        if isinstance(payload, dict):
            data = payload.get("tokens") or payload.get("data")
            if isinstance(data, list):
                return data
            if isinstance(data, dict):
                return [data]
            return [payload]
        if isinstance(payload, list):
            return payload
        return []

    @staticmethod
    def _parse_token_entry(token: dict):
        mint = token.get("address") or token.get("mint") or token.get("id")
        if not mint:
            return None
        name = token.get("name") or ""
        symbol = token.get("symbol") or ""
        mcap = token.get("mcap")
        return mint, {"name": name, "symbol": symbol, "mcap": mcap}

    def _build_url(self, ids=None, mint=None):
        if self.mode == "single":
            return self.url.format(mint=mint, query=mint)
        if self.mode == "batch":
            joined = ",".join(ids or [])
            return self.url.format(ids=joined)
        return self.url

    async def refresh(self):
        if self.mode != "list":
            return
        resp = await self.client.get(self._build_url(), headers=self.headers)
        resp.raise_for_status()
        payload = resp.json()
        new_data = {}
        for token in self._extract_tokens(payload):
            if not isinstance(token, dict):
                continue
            parsed = self._parse_token_entry(token)
            if not parsed:
                continue
            mint, meta = parsed
            meta["mcap"] = self._parse_decimal(meta.get("mcap"))
            new_data[mint] = meta
        if new_data:
            self.data = new_data
            self.last_refresh = time.time()

    async def fetch_batch(self, ids):
        if not ids:
            return
        url = self._build_url(ids=ids)
        resp = await self.client.get(url, headers=self.headers)
        resp.raise_for_status()
        payload = resp.json()
        for token in self._extract_tokens(payload):
            if not isinstance(token, dict):
                continue
            parsed = self._parse_token_entry(token)
            if not parsed:
                continue
            mint, meta = parsed
            meta["mcap"] = self._parse_decimal(meta.get("mcap"))
            self.data[mint] = meta

    async def fetch_one(self, mint):
        url = self._build_url(mint=mint)
        resp = await self.client.get(url, headers=self.headers)
        resp.raise_for_status()
        payload = resp.json()
        tokens = self._extract_tokens(payload)
        match = None
        for token in tokens:
            if not isinstance(token, dict):
                continue
            token_id = token.get("address") or token.get("mint") or token.get("id")
            if token_id == mint:
                match = token
                break
        if match is None and tokens:
            match = tokens[0] if isinstance(tokens[0], dict) else None
        if match:
            parsed = self._parse_token_entry(match)
            if parsed:
                _, meta = parsed
                meta["mcap"] = self._parse_decimal(meta.get("mcap"))
                self.data[mint] = meta
                return

    async def ensure(self, mints):
        missing = [mint for mint in mints if mint not in self.data]
        if not missing:
            return
        if self.mode == "list":
            return
        if self.mode == "batch":
            await self.fetch_batch(missing)
            return
        for mint in missing:
            await self.fetch_one(mint)

    async def run(self):
        if self.mode != "list":
            return
        while True:
            try:
                await self.refresh()
                await asyncio.sleep(self.refresh_sec)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                print(f"token list refresh error: {exc}")
                await asyncio.sleep(min(self.refresh_sec, 30))

    async def close(self):
        await self.client.aclose()


async def fetch_transaction(rpc: RpcClient, signature: str, commitment: str, retries: int = 3):
    for attempt in range(retries):
        result = await rpc.call(
            "getTransaction",
            [
                signature,
                {
                    "encoding": "jsonParsed",
                    "commitment": commitment,
                    "maxSupportedTransactionVersion": 0,
                },
            ],
        )
        if result:
            return result
        await asyncio.sleep(1 + attempt)
    return None


def summarize_trades(signature: str, slot: int, block_time: int, wallet: str, tx: dict):
    meta = tx.get("meta", {})
    message = tx.get("transaction", {}).get("message", {})
    pre_tokens = meta.get("preTokenBalances")
    post_tokens = meta.get("postTokenBalances")

    token_deltas = list(diff_token_balances(pre_tokens, post_tokens, wallet))
    if not token_deltas:
        return None

    sol_change = sol_delta(meta, message, wallet)
    trades = []

    for mint, delta_raw, decimals in token_deltas:
        if mint in STABLE_MINTS:
            continue
        side = "BUY" if delta_raw > 0 else "SELL"
        trades.append(
            {
                "mint": mint,
                "side": side,
                "amount_raw": abs(delta_raw),
                "decimals": decimals,
            }
        )

    if not trades:
        return None

    return {
        "signature": signature,
        "slot": slot,
        "block_time": block_time,
        "sol_change": sol_change,
        "trades": trades,
    }


def print_trade_summary(summary: dict):
    ts = summary["block_time"]
    if ts:
        ts_str = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(ts))
    else:
        ts_str = "unknown-time"

    sol_change = summary["sol_change"]

    print("")
    print(f"[TX] {ts_str} | slot {summary['slot']}")
    print(f"signature: {summary['signature']}")
    print(f"wallet sol delta: {sol_change:.9f} SOL")

    for trade in summary["trades"]:
        side = trade["side"]
        if sys.stdout.isatty():
            icon = GREEN_CIRCLE if side == "BUY" else RED_CIRCLE
        else:
            icon = "+" if side == "BUY" else "-"
        side_label = f"{icon} {side}"
        print(f"trade: {side_label}")
        print(f"token: {trade.get('name') or 'unknown'}")
        print(f"ticker: {trade.get('symbol') or 'unknown'}")
        print(f"CA: {trade['mint']}")
        if trade.get("marketcap") is not None:
            print(f"marketcap: {format_usd(trade['marketcap'])} USD")
        else:
            print("marketcap: unknown")
        amount_ui = format_amount(trade["amount_raw"], trade["decimals"])
        print(f"quantity: {amount_ui}")
        if trade["side"] == "BUY":
            sol_used = max(-sol_change, Decimal(0))
            print(f"sol used: {sol_used:.9f} SOL")
        else:
            sol_received = max(sol_change, Decimal(0))
            print(f"sol received: {sol_received:.9f} SOL")


async def listen_for_trades(config: dict):
    wallet = config["wallet"]
    ws_url = config["rpc_ws"]
    commitment = config.get("commitment", "confirmed")
    token_list_url = config.get("jupiter_tokens_url", JUPITER_TOKENS_URL_DEFAULT)
    refresh_sec = int(config.get("jupiter_refresh_sec", JUPITER_REFRESH_SEC_DEFAULT))
    headers = config.get("jupiter_headers") or {}
    if not isinstance(headers, dict):
        headers = {}

    rpc = RpcClient(config["rpc_http"])
    token_cache = JupiterTokenCache(token_list_url, refresh_sec, headers)
    seen = deque(maxlen=2000)

    try:
        await token_cache.refresh()
    except Exception as exc:
        print(f"token list refresh error: {exc}")

    refresh_task = None
    if token_cache.mode == "list":
        refresh_task = asyncio.create_task(token_cache.run())

    backoff = 1
    while True:
        try:
            async with websockets.connect(ws_url, ping_interval=20, ping_timeout=20) as ws:
                sub_msg = {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "logsSubscribe",
                    "params": [
                        {"mentions": [wallet]},
                        {"commitment": commitment},
                    ],
                }
                await ws.send(json.dumps(sub_msg))
                await ws.recv()

                backoff = 1
                async for raw in ws:
                    msg = json.loads(raw)
                    params = msg.get("params", {})
                    result = params.get("result", {})
                    value = result.get("value", {})

                    signature = value.get("signature")
                    err = value.get("err")
                    slot = result.get("context", {}).get("slot")
                    if err is not None or not signature:
                        continue
                    if signature in seen:
                        continue
                    seen.append(signature)

                    tx = await fetch_transaction(rpc, signature, commitment)
                    if not tx:
                        continue
                    summary = summarize_trades(
                        signature=signature,
                        slot=slot or 0,
                        block_time=tx.get("blockTime", 0),
                        wallet=wallet,
                        tx=tx,
                    )
                    if not summary:
                        continue

                    mints = list({trade["mint"] for trade in summary["trades"]})
                    try:
                        await token_cache.ensure(mints)
                    except Exception as exc:
                        print(f"token metadata error: {exc}")

                    for trade in summary["trades"]:
                        mint = trade["mint"]
                        meta = token_cache.get(mint) or {}
                        trade["name"] = meta.get("name")
                        trade["symbol"] = meta.get("symbol")
                        trade["marketcap"] = meta.get("mcap")

                    print_trade_summary(summary)
        except (websockets.WebSocketException, httpx.HTTPError, RuntimeError) as exc:
            print(f"connection error: {exc}. reconnecting in {backoff}s")
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 30)
        except asyncio.CancelledError:
            break
        except Exception as exc:
            print(f"unexpected error: {exc}. reconnecting in {backoff}s")
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 30)

    if refresh_task:
        refresh_task.cancel()
    await token_cache.close()
    await rpc.close()


def main():
    config_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("config.json")
    try:
        config = load_config(config_path)
    except Exception as exc:
        print(f"Failed to load config: {exc}")
        sys.exit(1)

    missing = [k for k in ("wallet", "rpc_http", "rpc_ws") if k not in config]
    if missing:
        print(f"Missing config keys: {', '.join(missing)}")
        sys.exit(1)

    print("Starting wallet trade detector...")
    print(f"wallet: {config['wallet']}")
    print(f"http: {config['rpc_http']}")
    print(f"ws: {config['rpc_ws']}")

    try:
        asyncio.run(listen_for_trades(config))
    except KeyboardInterrupt:
        print("Shutting down.")


if __name__ == "__main__":
    main()
