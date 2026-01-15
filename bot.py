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
PRICE_URL_DEFAULT = "https://api.jup.ag/price/v3?ids={ids}"
PRICE_POLL_SEC_DEFAULT = 30
ALERT_DROP_PCT_DEFAULT = Decimal("10")
ALERT_RISE_PCT_DEFAULT = Decimal("20")
TRAILING_START_PCT_DEFAULT = Decimal("20")
TRAILING_DRAWDOWN_PCT_DEFAULT = Decimal("10")
PRICE_WATCH_PATH_DEFAULT = "price_watch.json"
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


def parse_decimal(value):
    if value is None:
        return None
    try:
        return Decimal(str(value))
    except Exception:
        return None


def chunked(items, size: int):
    for i in range(0, len(items), size):
        yield items[i : i + size]


def serialize_price_watch(price_watch: dict) -> dict:
    entries = []
    for (wallet, mint), entry in price_watch.items():
        buy_price = entry.get("buy_price")
        if buy_price is None:
            continue
        entries.append(
            {
                "wallet": wallet,
                "mint": mint,
                "buy_price": str(buy_price),
                "alerted_down": bool(entry.get("alerted_down", False)),
                "alerted_up": bool(entry.get("alerted_up", False)),
                "trailing_active": bool(entry.get("trailing_active", False)),
                "trailing_peak": (
                    str(entry.get("trailing_peak"))
                    if entry.get("trailing_peak") is not None
                    else None
                ),
                "name": entry.get("name"),
                "symbol": entry.get("symbol"),
            }
        )
    return {"version": 1, "entries": entries}


def load_price_watch(path: Path, wallets: set[str]) -> dict:
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"price watch load error: {exc}")
        return {}

    if isinstance(raw, dict):
        entries = raw.get("entries", [])
    elif isinstance(raw, list):
        entries = raw
    else:
        return {}

    price_watch = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        wallet = entry.get("wallet")
        mint = entry.get("mint")
        if not wallet or not mint:
            continue
        if wallets and wallet not in wallets:
            continue
        buy_price = parse_decimal(entry.get("buy_price"))
        if buy_price is None or buy_price <= 0:
            continue
        price_watch[(wallet, mint)] = {
            "buy_price": buy_price,
            "alerted_down": bool(entry.get("alerted_down", False)),
            "alerted_up": bool(entry.get("alerted_up", False)),
            "trailing_active": bool(entry.get("trailing_active", False)),
            "trailing_peak": parse_decimal(entry.get("trailing_peak")),
            "name": entry.get("name"),
            "symbol": entry.get("symbol"),
        }
        if price_watch[(wallet, mint)]["trailing_active"] and price_watch[(wallet, mint)][
            "trailing_peak"
        ] is None:
            price_watch[(wallet, mint)]["trailing_peak"] = buy_price
    return price_watch


def save_price_watch(path: Path, price_watch: dict):
    payload = serialize_price_watch(price_watch)
    try:
        if path.parent and not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=True),
            encoding="utf-8",
        )
    except Exception as exc:
        print(f"price watch save error: {exc}")


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


def sol_balance(meta: dict, message: dict, wallet: str):
    keys = message.get("accountKeys", [])
    wallet_index = None
    for i, k in enumerate(keys):
        if key_str(k) == wallet:
            wallet_index = i
            break
    if wallet_index is None:
        return None
    post_balances = meta.get("postBalances", [])
    if wallet_index >= len(post_balances):
        return None
    return Decimal(post_balances[wallet_index]) / Decimal(1_000_000_000)


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


class JupiterPriceClient:
    def __init__(self, url: str, headers: dict, timeout_sec: int = 20):
        self.url = url
        self.headers = headers
        self.client = httpx.AsyncClient(timeout=timeout_sec)

    def _build_url(self, ids: list[str]):
        joined = ",".join(ids)
        if "{ids}" in self.url:
            return self.url.format(ids=joined)
        if "?" in self.url:
            return f"{self.url}&ids={joined}"
        return f"{self.url}?ids={joined}"

    async def fetch_prices(self, ids: list[str]) -> dict:
        if not ids:
            return {}
        result = {}
        for batch in chunked(ids, 50):
            url = self._build_url(batch)
            resp = await self.client.get(url, headers=self.headers)
            resp.raise_for_status()
            payload = resp.json()
            if not isinstance(payload, dict):
                continue
            for mint, info in payload.items():
                if not isinstance(info, dict):
                    continue
                price = parse_decimal(info.get("usdPrice"))
                if price is not None:
                    result[mint] = price
        return result

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
    sol_post = sol_balance(meta, message, wallet)
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
        "wallet": wallet,
        "sol_change": sol_change,
        "sol_balance": sol_post,
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
    print(f"wallet: {summary['wallet']}")
    print(f"wallet sol delta: {sol_change:.9f} SOL")
    sol_balance_post = summary.get("sol_balance")
    if sol_balance_post is not None:
        print(f"wallet sol balance: {sol_balance_post:.9f} SOL")

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


async def run_price_alerts(
    price_client,
    price_watch: dict,
    price_lock: asyncio.Lock,
    price_watch_path: Path,
    drop_pct: Decimal,
    rise_pct: Decimal,
    poll_sec: int,
    trailing_start_pct: Decimal,
    trailing_drawdown_pct: Decimal,
):
    if poll_sec <= 0:
        return
    drop_threshold = (drop_pct * Decimal(100)).quantize(
        Decimal("0.01"), rounding=ROUND_HALF_UP
    )
    rise_threshold = (rise_pct * Decimal(100)).quantize(
        Decimal("0.01"), rounding=ROUND_HALF_UP
    )
    trailing_start_threshold = (trailing_start_pct * Decimal(100)).quantize(
        Decimal("0.01"), rounding=ROUND_HALF_UP
    )
    trailing_drawdown_threshold = (trailing_drawdown_pct * Decimal(100)).quantize(
        Decimal("0.01"), rounding=ROUND_HALF_UP
    )
    while True:
        try:
            await asyncio.sleep(poll_sec)
            async with price_lock:
                watch_snapshot = {
                    key: {
                        "buy_price": entry.get("buy_price"),
                        "alerted_down": entry.get("alerted_down", False),
                        "alerted_up": entry.get("alerted_up", False),
                        "trailing_active": entry.get("trailing_active", False),
                        "trailing_peak": entry.get("trailing_peak"),
                        "name": entry.get("name"),
                        "symbol": entry.get("symbol"),
                    }
                    for key, entry in price_watch.items()
                }
            if not watch_snapshot:
                continue

            mints = sorted({mint for (_, mint) in watch_snapshot.keys()})
            prices = await price_client.fetch_prices(mints)
            if not prices:
                continue

            alerts = []
            pending_updates = {}
            pending_removals = {}
            for (wallet, mint), entry in watch_snapshot.items():
                price = prices.get(mint)
                if price is None:
                    continue
                buy_price = entry.get("buy_price")
                if buy_price is None or buy_price <= 0:
                    continue
                change = (price - buy_price) / buy_price
                key = (wallet, mint)

                if trailing_start_pct > 0 and trailing_drawdown_pct > 0:
                    trailing_active = bool(entry.get("trailing_active", False))
                    trailing_peak = entry.get("trailing_peak")
                    if trailing_peak is None and trailing_active:
                        trailing_peak = buy_price

                    if not trailing_active and change >= trailing_start_pct:
                        trailing_active = True
                        trailing_peak = price
                        alerts.append(
                            {
                                "type": "TRAILING_ARMED",
                                "wallet": wallet,
                                "mint": mint,
                                "entry": entry,
                                "price": price,
                                "buy_price": buy_price,
                            }
                        )
                        pending_updates.setdefault(key, {"buy_price": buy_price, "changes": {}})[
                            "changes"
                        ].update(
                            {
                                "trailing_active": True,
                                "trailing_peak": price,
                            }
                        )

                    if trailing_active:
                        if trailing_peak is None:
                            trailing_peak = price
                            pending_updates.setdefault(
                                key, {"buy_price": buy_price, "changes": {}}
                            )["changes"]["trailing_peak"] = price
                        elif price > trailing_peak:
                            trailing_peak = price
                            pending_updates.setdefault(
                                key, {"buy_price": buy_price, "changes": {}}
                            )["changes"]["trailing_peak"] = price

                        stop_price = trailing_peak * (Decimal(1) - trailing_drawdown_pct)
                        if price <= stop_price:
                            alerts.append(
                                {
                                    "type": "TRAILING_TP",
                                    "wallet": wallet,
                                    "mint": mint,
                                    "entry": entry,
                                    "price": price,
                                    "buy_price": buy_price,
                                    "peak_price": trailing_peak,
                                    "stop_price": stop_price,
                                }
                            )
                            pending_removals[key] = buy_price
                            pending_updates.pop(key, None)
                            continue

                if change <= -drop_pct and not entry.get("alerted_down"):
                    alerts.append(
                        {
                            "type": "DROP",
                            "wallet": wallet,
                            "mint": mint,
                            "entry": entry,
                            "price": price,
                            "buy_price": buy_price,
                            "change": change,
                        }
                    )
                    pending_updates.setdefault(key, {"buy_price": buy_price, "changes": {}})[
                        "changes"
                    ]["alerted_down"] = True
                if change >= rise_pct and not entry.get("alerted_up"):
                    alerts.append(
                        {
                            "type": "RISE",
                            "wallet": wallet,
                            "mint": mint,
                            "entry": entry,
                            "price": price,
                            "buy_price": buy_price,
                            "change": change,
                        }
                    )
                    pending_updates.setdefault(key, {"buy_price": buy_price, "changes": {}})[
                        "changes"
                    ]["alerted_up"] = True

            if pending_updates or pending_removals:
                async with price_lock:
                    for (wallet, mint), payload in pending_updates.items():
                        entry = price_watch.get((wallet, mint))
                        if not entry:
                            continue
                        if entry.get("buy_price") != payload["buy_price"]:
                            continue
                        entry.update(payload["changes"])
                    for (wallet, mint), buy_price in pending_removals.items():
                        entry = price_watch.get((wallet, mint))
                        if entry and entry.get("buy_price") == buy_price:
                            price_watch.pop((wallet, mint), None)
                    save_price_watch(price_watch_path, price_watch)

            if alerts:
                ts_str = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
                for alert in alerts:
                    kind = alert["type"]
                    wallet = alert["wallet"]
                    mint = alert["mint"]
                    entry = alert["entry"]
                    price = alert["price"]
                    buy_price = alert["buy_price"]
                    name = entry.get("name") or "unknown"
                    symbol = entry.get("symbol") or "unknown"
                    print("")
                    if kind in ("DROP", "RISE"):
                        change_pct = (alert["change"] * Decimal(100)).quantize(
                            Decimal("0.01"), rounding=ROUND_HALF_UP
                        )
                        target = drop_threshold if kind == "DROP" else rise_threshold
                        print(f"[ALERT] {ts_str} | {kind}")
                        print(f"wallet: {wallet}")
                        print(f"token: {name}")
                        print(f"ticker: {symbol}")
                        print(f"CA: {mint}")
                        print(f"buy price: {format_usd(buy_price)} USD")
                        print(f"current price: {format_usd(price)} USD")
                        print(f"change: {change_pct}% (threshold {target}%)")
                    elif kind == "TRAILING_ARMED":
                        print(f"[ALERT] {ts_str} | TRAILING TP ARMED")
                        print(f"wallet: {wallet}")
                        print(f"token: {name}")
                        print(f"ticker: {symbol}")
                        print(f"CA: {mint}")
                        print(f"buy price: {format_usd(buy_price)} USD")
                        print(f"current price: {format_usd(price)} USD")
                        print(
                            f"trail start: {trailing_start_threshold}% "
                            f"(drawdown {trailing_drawdown_threshold}%)"
                        )
                    elif kind == "TRAILING_TP":
                        peak_price = alert["peak_price"]
                        stop_price = alert["stop_price"]
                        print(f"[ALERT] {ts_str} | TRAILING TP SELL")
                        print(f"wallet: {wallet}")
                        print(f"token: {name}")
                        print(f"ticker: {symbol}")
                        print(f"CA: {mint}")
                        print(f"buy price: {format_usd(buy_price)} USD")
                        print(f"peak price: {format_usd(peak_price)} USD")
                        print(f"stop price: {format_usd(stop_price)} USD")
                        print(f"current price: {format_usd(price)} USD")
                        print(f"drawdown: {trailing_drawdown_threshold}%")
        except asyncio.CancelledError:
            break
        except Exception as exc:
            print(f"price alert error: {exc}")
            await asyncio.sleep(min(poll_sec, 30))


def normalize_wallets(config: dict) -> list[str]:
    wallets = config.get("wallets")
    if wallets is None:
        wallet = config.get("wallet")
        wallets = [wallet] if wallet else []
    elif isinstance(wallets, str):
        wallets = [wallets]
    else:
        wallets = [w for w in wallets if w]
    return wallets


async def subscribe_wallets(ws, wallets: list[str], commitment: str):
    wallet_by_sub_id = {}
    pending = []
    request_id = 1

    for wallet in wallets:
        sub_msg = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": "logsSubscribe",
            "params": [
                {"mentions": [wallet]},
                {"commitment": commitment},
            ],
        }
        await ws.send(json.dumps(sub_msg))
        while True:
            raw = await ws.recv()
            msg = json.loads(raw)
            if msg.get("id") == request_id:
                if "error" in msg:
                    raise RuntimeError(msg["error"])
                sub_id = msg.get("result")
                if sub_id is not None:
                    wallet_by_sub_id[sub_id] = wallet
                break
            pending.append(msg)
        request_id += 1

    return wallet_by_sub_id, pending


async def listen_for_trades(config: dict):
    wallets = normalize_wallets(config)
    ws_url = config["rpc_ws"]
    commitment = config.get("commitment", "confirmed")
    token_list_url = config.get("jupiter_tokens_url", JUPITER_TOKENS_URL_DEFAULT)
    price_url = config.get("price_url", PRICE_URL_DEFAULT)
    price_watch_path = Path(config.get("price_watch_path", PRICE_WATCH_PATH_DEFAULT))
    refresh_sec = int(config.get("jupiter_refresh_sec", JUPITER_REFRESH_SEC_DEFAULT))
    price_poll_sec = int(config.get("price_poll_sec", PRICE_POLL_SEC_DEFAULT))
    drop_pct = parse_decimal(config.get("alert_drop_pct", ALERT_DROP_PCT_DEFAULT))
    rise_pct = parse_decimal(config.get("alert_rise_pct", ALERT_RISE_PCT_DEFAULT))
    trailing_start_pct = parse_decimal(
        config.get("trailing_start_pct", TRAILING_START_PCT_DEFAULT)
    )
    trailing_drawdown_pct = parse_decimal(
        config.get("trailing_drawdown_pct", TRAILING_DRAWDOWN_PCT_DEFAULT)
    )
    drop_ratio = (drop_pct or ALERT_DROP_PCT_DEFAULT) / Decimal(100)
    rise_ratio = (rise_pct or ALERT_RISE_PCT_DEFAULT) / Decimal(100)
    trailing_start_ratio = (
        (trailing_start_pct or TRAILING_START_PCT_DEFAULT) / Decimal(100)
    )
    trailing_drawdown_ratio = (
        (trailing_drawdown_pct or TRAILING_DRAWDOWN_PCT_DEFAULT) / Decimal(100)
    )
    if trailing_start_ratio <= 0 or trailing_drawdown_ratio <= 0:
        trailing_start_ratio = Decimal(0)
        trailing_drawdown_ratio = Decimal(0)
    headers = config.get("jupiter_headers") or {}
    if not isinstance(headers, dict):
        headers = {}

    rpc = RpcClient(config["rpc_http"])
    token_cache = JupiterTokenCache(token_list_url, refresh_sec, headers)
    price_client = JupiterPriceClient(price_url, headers)
    price_watch = load_price_watch(price_watch_path, set(wallets))
    price_lock = asyncio.Lock()
    seen = deque(maxlen=2000)

    if price_watch:
        print(f"loaded {len(price_watch)} price watch entries")

    try:
        await token_cache.refresh()
    except Exception as exc:
        print(f"token list refresh error: {exc}")

    refresh_task = None
    if token_cache.mode == "list":
        refresh_task = asyncio.create_task(token_cache.run())

    price_task = None
    if price_poll_sec > 0:
        price_task = asyncio.create_task(
            run_price_alerts(
                price_client,
                price_watch,
                price_lock,
                price_watch_path,
                drop_ratio,
                rise_ratio,
                price_poll_sec,
                trailing_start_ratio,
                trailing_drawdown_ratio,
            )
        )

    backoff = 1
    while True:
        try:
            async with websockets.connect(ws_url, ping_interval=20, ping_timeout=20) as ws:
                wallet_by_sub_id, pending = await subscribe_wallets(ws, wallets, commitment)

                backoff = 1
                async def handle_message(msg: dict):
                    params = msg.get("params")
                    if not isinstance(params, dict):
                        return
                    result = params.get("result", {})
                    value = result.get("value", {})

                    signature = value.get("signature")
                    err = value.get("err")
                    slot = result.get("context", {}).get("slot")
                    if err is not None or not signature:
                        return

                    sub_id = params.get("subscription")
                    wallet_hint = wallet_by_sub_id.get(sub_id)
                    wallets_to_check = [wallet_hint] if wallet_hint else wallets
                    pending_wallets = [
                        wallet
                        for wallet in wallets_to_check
                        if (signature, wallet) not in seen
                    ]
                    if not pending_wallets:
                        return

                    tx = await fetch_transaction(rpc, signature, commitment)
                    if not tx:
                        return

                    summaries = []
                    for wallet in pending_wallets:
                        seen.append((signature, wallet))
                        summary = summarize_trades(
                            signature=signature,
                            slot=slot or 0,
                            block_time=tx.get("blockTime", 0),
                            wallet=wallet,
                            tx=tx,
                        )
                        if summary:
                            summaries.append(summary)
                    if not summaries:
                        return

                    mints = list(
                        {trade["mint"] for summary in summaries for trade in summary["trades"]}
                    )
                    try:
                        await token_cache.ensure(mints)
                    except Exception as exc:
                        print(f"token metadata error: {exc}")

                    for summary in summaries:
                        for trade in summary["trades"]:
                            mint = trade["mint"]
                            meta = token_cache.get(mint) or {}
                            trade["name"] = meta.get("name")
                            trade["symbol"] = meta.get("symbol")
                            trade["marketcap"] = meta.get("mcap")

                    sell_keys = []
                    for summary in summaries:
                        for trade in summary["trades"]:
                            if trade["side"] == "SELL":
                                sell_keys.append((summary["wallet"], trade["mint"]))
                    if sell_keys:
                        async with price_lock:
                            for key in sell_keys:
                                price_watch.pop(key, None)
                            save_price_watch(price_watch_path, price_watch)

                    buy_requests = []
                    for summary in summaries:
                        for trade in summary["trades"]:
                            if trade["side"] == "BUY":
                                buy_requests.append((summary["wallet"], trade))
                    if buy_requests:
                        buy_mints = sorted({trade["mint"] for _, trade in buy_requests})
                        try:
                            prices = await price_client.fetch_prices(buy_mints)
                        except Exception as exc:
                            print(f"price fetch error: {exc}")
                            prices = {}
                        if prices:
                            async with price_lock:
                                for wallet, trade in buy_requests:
                                    mint = trade["mint"]
                                    price = prices.get(mint)
                                    if price is None:
                                        continue
                                    price_watch[(wallet, mint)] = {
                                        "buy_price": price,
                                        "alerted_down": False,
                                        "alerted_up": False,
                                        "trailing_active": False,
                                        "trailing_peak": None,
                                        "name": trade.get("name"),
                                        "symbol": trade.get("symbol"),
                                    }
                                save_price_watch(price_watch_path, price_watch)

                    for summary in summaries:
                        print_trade_summary(summary)

                for msg in pending:
                    await handle_message(msg)

                async for raw in ws:
                    msg = json.loads(raw)
                    await handle_message(msg)
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
    if price_task:
        price_task.cancel()
    await token_cache.close()
    await price_client.close()
    await rpc.close()


def main():
    config_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("config.json")
    try:
        config = load_config(config_path)
    except Exception as exc:
        print(f"Failed to load config: {exc}")
        sys.exit(1)

    wallets = normalize_wallets(config)
    missing = [k for k in ("rpc_http", "rpc_ws") if k not in config]
    if missing or not wallets:
        missing_keys = missing[:]
        if not wallets:
            missing_keys.append("wallet or wallets")
        print(f"Missing config keys: {', '.join(missing_keys)}")
        sys.exit(1)

    print("Starting wallet trade detector...")
    print(f"wallets: {', '.join(wallets)}")
    print(f"http: {config['rpc_http']}")
    print(f"ws: {config['rpc_ws']}")

    try:
        asyncio.run(listen_for_trades(config))
    except KeyboardInterrupt:
        print("Shutting down.")


if __name__ == "__main__":
    main()
