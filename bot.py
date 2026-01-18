import asyncio
import base64
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
import re
import math
from collections import deque
from decimal import Decimal, ROUND_HALF_UP, ROUND_DOWN, getcontext
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
SWAP_URL_DEFAULT = "https://api.jup.ag/ultra/v1"
PRICE_POLL_SEC_DEFAULT = 30
TRAILING_START_PCT_DEFAULT = Decimal("20")
TRAILING_DRAWDOWN_PCT_DEFAULT = Decimal("10")
TRAILING_STATUS_SEC_DEFAULT = 5
TRAILING_CONFIRM_SEC_DEFAULT = 0
TAKE_PROFIT_PCT_DEFAULT = Decimal("20")
STOP_LOSS_LEVELS_PCT_DEFAULT = [Decimal("10"), Decimal("15")]
DRAW_DOWN_STATUS_PCT_DEFAULT = Decimal("5")
DRAW_DOWN_STATUS_SEC_DEFAULT = 5
SELL_PCT_DEFAULT = Decimal("100")
SELL_SLIPPAGE_PCT_DEFAULT = Decimal("30")
SELL_PRIORITY_FEE_SOL_DEFAULT = Decimal("0.001")
SELL_RETRY_SEC_DEFAULT = 30
SELL_INFLIGHT_TIMEOUT_SEC_DEFAULT = 60
SELL_CONFIRM_DELAY_SEC_DEFAULT = 5
SELL_CONFIRM_MAX_ATTEMPTS_DEFAULT = 3
AUTO_SELL_ENABLED_DEFAULT = False
ENV_PATH_DEFAULT = ".env"
KEYPAIR_ENV_DEFAULT = "SOLANA_KEYPAIR"
PRICE_WATCH_PATH_DEFAULT = "price_watch.json"
TRADE_LOG_PATH_DEFAULT = "trade_log.jsonl"
TZ_OFFSET_MINUTES_DEFAULT = 570
PRICE_WATCH_SYNC_SEC_DEFAULT = 60
PRICE_MAX_RPM_DEFAULT = 60
PRICE_BATCH_SIZE_DEFAULT = 50
WS_PING_INTERVAL_DEFAULT = 20
WS_PING_TIMEOUT_DEFAULT = 20
WS_IDLE_RECONNECT_SEC_DEFAULT = 0
GREEN_CIRCLE = "\U0001F7E2"
RED_CIRCLE = "\U0001F534"
WSOL_MINT = "So11111111111111111111111111111111111111112"


def trend_icon(change: Decimal) -> str:
    return GREEN_CIRCLE if change >= 0 else RED_CIRCLE


def resolve_alias_dict_keys(mapping, alias_map: dict) -> dict:
    if not isinstance(mapping, dict):
        return {}
    alias_map = alias_map or {}
    resolved = {}
    for key, value in mapping.items():
        resolved_key = alias_map.get(key, key)
        resolved[resolved_key] = value
    return resolved


def format_ts(epoch_sec: int | float | None, offset_min: int) -> str:
    if not epoch_sec:
        return "unknown-time"
    tz = timezone(timedelta(minutes=offset_min))
    dt = datetime.fromtimestamp(float(epoch_sec), tz=tz)
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def append_trade_log(path: Path, entry: dict):
    try:
        if path.parent and not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(to_jsonable(entry), ensure_ascii=True) + "\n")
    except Exception as exc:
        print(f"trade log write error: {exc}")


async def send_discord_webhook(
    client: httpx.AsyncClient, url: str, content: str
):
    try:
        resp = await client.post(url, json={"content": content})
        resp.raise_for_status()
    except Exception as exc:
        print(f"webhook error: {exc}")


def format_trade_message(summary: dict, trade: dict, tz_offset_min: int) -> str:
    lines = []
    side = trade.get("side") or "UNKNOWN"
    name = trade.get("name") or "unknown"
    symbol = trade.get("symbol") or "unknown"
    lines.append(f"token: {name} ({symbol})")
    sol_change = summary.get("sol_change") or Decimal(0)
    if side == "SELL":
        sol_received = max(sol_change, Decimal(0))
        lines.append(f"sol received: {sol_received:.9f} SOL")
        pnl_pct = trade.get("pnl_pct")
        if pnl_pct is not None:
            pnl_fmt = pnl_pct.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
            lines.append(f"pnl: {pnl_fmt:+f}%")
    else:
        sol_used = max(-sol_change, Decimal(0))
        lines.append(f"sol used: {sol_used:.9f} SOL")
    sol_balance_post = summary.get("sol_balance")
    if sol_balance_post is not None:
        lines.append(f"wallet balance: {sol_balance_post:.9f} SOL")
    return "\n".join(lines)


def load_trade_controls(
    path: Path, alias_map: dict, tracked_wallets: set[str] | None = None
):
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    mapping = raw.get("wallet_trading_enabled", raw) if isinstance(raw, dict) else raw
    if not isinstance(mapping, dict):
        return {}
    resolved = resolve_alias_dict_keys(mapping, alias_map)
    if tracked_wallets:
        resolved = {k: v for k, v in resolved.items() if k in tracked_wallets}
    return {k: bool(v) for k, v in resolved.items()}


def write_trade_controls(path: Path, mapping: dict):
    payload = {
        "updated_ts": int(time.time()),
        "wallet_trading_enabled": mapping,
    }
    try:
        if path.parent and not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=True),
            encoding="utf-8",
        )
    except Exception as exc:
        print(f"trade control write error: {exc}")


def parse_duration(value: str) -> int | None:
    match = re.match(r"^(\d+)([smhd])$", value.strip().lower())
    if not match:
        return None
    amount = int(match.group(1))
    unit = match.group(2)
    if unit == "s":
        return amount
    if unit == "m":
        return amount * 60
    if unit == "h":
        return amount * 3600
    if unit == "d":
        return amount * 86400
    return None


def parse_datetime_value(value: str, tz_offset_min: int) -> int | None:
    try:
        dt = datetime.fromisoformat(value)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone(timedelta(minutes=tz_offset_min)))
    return int(dt.timestamp())


def resolve_commitment(value: str | None) -> str:
    if not value:
        return "confirmed"
    normalized = str(value).strip().lower()
    if normalized in {"confirmed", "finalized"}:
        return normalized
    return "confirmed"


def iter_trade_log(path: Path):
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def print_trade_stats(
    log_path: Path, since: int | None, until: int | None, tz_offset_min: int
):
    total = 0
    buys = 0
    sells = 0
    sol_spent = Decimal(0)
    sol_received = Decimal(0)
    pnl_sum = Decimal(0)
    wins = 0
    losses = 0
    wallets = set()
    tokens = set()
    positions_all = {}

    for entry in iter_trade_log(log_path):
        ts = entry.get("ts") or entry.get("detected_ts")
        if not ts:
            continue
        ts = int(ts)
        if until and ts > until:
            continue
        wallet = entry.get("wallet")
        mint = entry.get("mint")
        side = entry.get("side")
        key = (wallet, mint) if wallet and mint else None
        buy_used = None

        if side == "BUY":
            sol_used = parse_decimal(entry.get("sol_used"))
            if sol_used is not None and key:
                positions_all.setdefault(key, deque()).append(sol_used)
        elif side == "SELL":
            if key:
                stack = positions_all.get(key)
                if stack:
                    buy_used = stack.popleft()
            if buy_used is None:
                buy_used = parse_decimal(entry.get("buy_sol_used"))

        if since and ts < since:
            continue
        total += 1
        if side == "BUY":
            buys += 1
        elif side == "SELL":
            sells += 1
            sol_recv = parse_decimal(entry.get("sol_received"))
            if sol_recv is not None and buy_used is not None:
                sol_received += sol_recv
                sol_spent += buy_used
            pnl_pct = parse_decimal(entry.get("pnl_pct"))
            if (
                pnl_pct is None
                and sol_recv is not None
                and buy_used is not None
                and buy_used > 0
            ):
                pnl_pct = ((sol_recv - buy_used) / buy_used) * Decimal(100)
            if pnl_pct is not None:
                pnl_sum += pnl_pct
                if pnl_pct > 0:
                    wins += 1
                elif pnl_pct < 0:
                    losses += 1
        if wallet:
            wallets.add(wallet)
        if mint:
            tokens.add(mint)

    if total == 0:
        print("No trades found for the selected period.")
        return

    net_sol = sol_received - sol_spent
    open_positions = sum(len(stack) for stack in positions_all.values())
    start_str = format_ts(since, tz_offset_min) if since else "all-time"
    end_str = format_ts(until, tz_offset_min) if until else "now"

    print(f"Period: {start_str} -> {end_str}")
    print(f"Trades: {total} (buys {buys}, sells {sells})")
    print(f"Wallets: {len(wallets)} | Tokens: {len(tokens)}")
    print(f"SOL spent: {sol_spent:.9f}")
    print(f"SOL received: {sol_received:.9f}")
    print(f"Net SOL: {net_sol:.9f}")
    if sol_spent > 0:
        total_pct = (net_sol / sol_spent) * Decimal(100)
        total_fmt = total_pct.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        print(f"Total PnL (net): {total_fmt:+f}%")
    print(f"Open positions: {open_positions}")
    print(f"Win/Loss (sells): {wins}/{losses}")


def load_trade_log_entries(path: Path) -> list[dict]:
    return list(iter_trade_log(path))


def write_trade_log_entries(path: Path, entries: list[dict]) -> bool:
    tmp_path = Path(str(path) + ".tmp")
    try:
        if path.parent and not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
        with tmp_path.open("w", encoding="utf-8") as f:
            for entry in entries:
                f.write(json.dumps(to_jsonable(entry), ensure_ascii=True) + "\n")
        tmp_path.replace(path)
        return True
    except Exception as exc:
        print(f"trade log write error: {exc}")
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass
        return False


def to_jsonable(value):
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, dict):
        return {k: to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    return value


def log_entry_ts(entry: dict) -> int:
    return int(entry.get("ts") or entry.get("detected_ts") or 0)


def pop_open_position(positions: dict, key: tuple[str, str]):
    queue = positions.get(key)
    if isinstance(queue, dict):
        queue = queue.get("buys")
        if queue is None:
            return None
        positions[key] = queue
    if not queue:
        return None
    value = queue.popleft()
    if not queue:
        positions.pop(key, None)
    if isinstance(value, dict):
        return parse_decimal(value.get("sol_used"))
    return parse_decimal(value)


def build_trade_log_index(entries: list[dict]):
    positions = {}
    sell_sigs = {}
    meta = {}
    for entry in entries:
        wallet = entry.get("wallet")
        mint = entry.get("mint")
        side = entry.get("side")
        if not wallet or not mint or side not in ("BUY", "SELL"):
            continue
        key = (wallet, mint)
        meta_slot = meta.setdefault(
            key, {"name": None, "symbol": None, "marketcap": None}
        )
        if entry.get("name"):
            meta_slot["name"] = entry.get("name")
        if entry.get("symbol"):
            meta_slot["symbol"] = entry.get("symbol")
        if entry.get("marketcap"):
            meta_slot["marketcap"] = entry.get("marketcap")
        if side == "SELL":
            sig = entry.get("signature")
            if sig:
                sell_sigs.setdefault(key, set()).add(sig)
        if side == "BUY":
            slot = positions.setdefault(key, {"buys": deque()})
            slot["buys"].append(
                {
                    "ts": entry.get("ts") or entry.get("detected_ts"),
                    "sol_used": parse_decimal(entry.get("sol_used")),
                }
            )
        elif side == "SELL":
            slot = positions.get(key)
            if slot and slot["buys"]:
                slot["buys"].popleft()
                if not slot["buys"]:
                    positions.pop(key, None)
    return positions, sell_sigs, meta


def sync_price_watch_with_log(watch_path: Path, log_entries: list[dict]) -> int:
    if not watch_path.exists():
        return 0
    watch = load_price_watch(watch_path, set())
    return sync_price_watch_with_log_entries(watch, log_entries, watch_path)


def sync_price_watch_with_log_entries(
    price_watch: dict, log_entries: list[dict], watch_path: Path
) -> int:
    positions, _, _ = build_trade_log_index(log_entries)
    open_keys = set(positions.keys())
    removed = 0
    for key in list(price_watch.keys()):
        if key not in open_keys:
            price_watch.pop(key, None)
            removed += 1
    if removed:
        save_price_watch(watch_path, price_watch)
    return removed


def print_open_positions(
    log_path: Path,
    tz_offset_min: int,
    watch_path: Path | None = None,
):
    entries = load_trade_log_entries(log_path)
    positions, _, meta = build_trade_log_index(entries)

    if not positions:
        print("No open positions found in trade log.")
        return

    items = []
    for (wallet, mint), data in positions.items():
        buys = data.get("buys") or deque()
        if not buys:
            continue
        meta_slot = meta.get((wallet, mint), {})
        last_ts = None
        total_sol_used = Decimal(0)
        sol_count = 0
        for buy in buys:
            ts_val = buy.get("ts")
            if ts_val:
                last_ts = ts_val if last_ts is None else max(last_ts, ts_val)
            sol_used = buy.get("sol_used")
            if sol_used is not None:
                total_sol_used += sol_used
                sol_count += 1
        items.append(
            {
                "wallet": wallet,
                "mint": mint,
                "name": meta_slot.get("name") or "unknown",
                "symbol": meta_slot.get("symbol") or "unknown",
                "open_buys": len(buys),
                "total_sol_used": total_sol_used if sol_count else None,
                "last_ts": last_ts,
            }
        )

    items.sort(
        key=lambda item: item["last_ts"] or 0,
        reverse=True,
    )
    open_slots = sum(item["open_buys"] for item in items)
    print(f"Open positions (buys not sold): {open_slots}")
    print(f"Distinct wallet+mint: {len(items)}")
    for item in items:
        print("")
        print(f"wallet: {item['wallet']}")
        print(f"token: {item['name']} ({item['symbol']})")
        print(f"mint: {item['mint']}")
        print(f"open buys: {item['open_buys']}")
        if item["total_sol_used"] is not None:
            print(f"sol used (open): {item['total_sol_used']:.9f} SOL")
        if item["last_ts"]:
            print(f"last buy: {format_ts(item['last_ts'], tz_offset_min)}")

    if watch_path and watch_path.exists():
        watch = load_price_watch(watch_path, set())
        open_keys = {(item["wallet"], item["mint"]) for item in items}
        watch_keys = set(watch.keys())
        missing_in_watch = sorted(open_keys - watch_keys)
        extra_in_watch = sorted(watch_keys - open_keys)
        print("")
        print(f"Price watch entries: {len(watch_keys)}")
        if missing_in_watch:
            print(f"Missing in price watch: {len(missing_in_watch)}")
            for wallet, mint in missing_in_watch:
                print(f"- {wallet} | {mint}")
        if extra_in_watch:
            print(f"Extra in price watch: {len(extra_in_watch)}")
            for wallet, mint in extra_in_watch:
                entry = watch.get((wallet, mint), {})
                symbol = entry.get("symbol") or entry.get("name") or "unknown"
                print(f"- {wallet} | {symbol} | {mint}")


def load_env_file(path: Path):
    if not path.exists():
        return
    try:
        raw = path.read_text(encoding="utf-8")
    except Exception as exc:
        print(f"env load error: {exc}")
        return
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def parse_keypair_value(raw: str):
    from solders.keypair import Keypair

    raw = raw.strip().strip('"').strip("'")
    if not raw:
        return None
    path = Path(raw)
    if path.exists():
        raw = path.read_text(encoding="utf-8").strip()
    if raw.startswith("["):
        return Keypair.from_json(raw)
    return Keypair.from_base58_string(raw)


def load_keypair(env_path: Path, keypair_env: str, keypair_path: str | None):
    load_env_file(env_path)
    candidates = [keypair_env] if keypair_env else []
    for fallback in ("SOLANA_KEYPAIR", "KEYPAIR"):
        if fallback not in candidates:
            candidates.append(fallback)
    value = None
    for name in candidates:
        env_value = os.getenv(name)
        if env_value:
            value = env_value
            break
    if not value and keypair_path:
        value = keypair_path
    if not value:
        return None
    try:
        return parse_keypair_value(value)
    except Exception as exc:
        print(f"keypair load error: {exc}")
        return None


def load_keypairs(
    env_path: Path,
    wallets: list[str],
    keypair_env: str | None,
    keypair_path: str | None,
    keypair_envs: dict | None,
    keypair_paths: dict | None,
):
    load_env_file(env_path)
    keypairs = {}
    keypair_envs = keypair_envs if isinstance(keypair_envs, dict) else {}
    keypair_paths = keypair_paths if isinstance(keypair_paths, dict) else {}

    default_keypair = None
    if keypair_env or keypair_path:
        default_keypair = load_keypair(env_path, keypair_env or "", keypair_path)

    for wallet in wallets:
        value = None
        env_name = keypair_envs.get(wallet)
        if env_name:
            value = os.getenv(env_name)
        if not value:
            value = keypair_paths.get(wallet)
        if not value and default_keypair is not None:
            keypair = default_keypair
        else:
            if not value:
                continue
            try:
                keypair = parse_keypair_value(value)
            except Exception as exc:
                print(f"keypair load error for {wallet}: {exc}")
                continue

        if keypair is None:
            continue
        pubkey = str(keypair.pubkey())
        if pubkey != wallet:
            print(f"keypair wallet mismatch: {wallet} != {pubkey}")
            continue
        keypairs[wallet] = keypair
    return keypairs


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


def format_mcap_short(value: Decimal | None) -> str | None:
    if value is None:
        return None
    value = Decimal(value)
    thousand = Decimal(1000)
    million = Decimal(1_000_000)
    billion = Decimal(1_000_000_000)
    if value >= billion:
        return f"{(value / billion).quantize(Decimal('0.1'), rounding=ROUND_HALF_UP):f}B"
    if value >= million:
        return f"{(value / million).quantize(Decimal('0.1'), rounding=ROUND_HALF_UP):f}M"
    if value >= thousand:
        if value < Decimal(100_000):
            return f"{(value / thousand).quantize(Decimal('0.1'), rounding=ROUND_HALF_UP):f}k"
        return f"{(value / thousand).quantize(Decimal('1'), rounding=ROUND_HALF_UP):f}k"
    return f"{value.quantize(Decimal('0.1'), rounding=ROUND_HALF_UP):f}"


def parse_decimal_list(values):
    if values is None:
        return []
    if isinstance(values, (list, tuple)):
        raw_list = values
    else:
        raw_list = [values]
    parsed = []
    for item in raw_list:
        value = parse_decimal(item)
        if value is None:
            continue
        parsed.append(value)
    return parsed


def format_percent(value: Decimal) -> str:
    return f"{value.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP):f}"


def stop_loss_reason_for_level(level_ratio: Decimal) -> str:
    return f"STOP_LOSS_{format_percent(level_ratio * Decimal(100))}"


def parse_stop_loss_level(reason: str | None):
    if not reason or not reason.startswith("STOP_LOSS_"):
        return None
    raw = reason[len("STOP_LOSS_") :]
    level_pct = parse_decimal(raw)
    if level_pct is None:
        return None
    return level_pct / Decimal(100)


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
                "trailing_active": bool(entry.get("trailing_active", False)),
                "trailing_peak": (
                    str(entry.get("trailing_peak"))
                    if entry.get("trailing_peak") is not None
                    else None
                ),
                "trailing_below_since": entry.get("trailing_below_since"),
                "trailing_confirmed": bool(entry.get("trailing_confirmed", False)),
                "buy_sol_used": (
                    str(entry.get("buy_sol_used"))
                    if entry.get("buy_sol_used") is not None
                    else None
                ),
                "name": entry.get("name"),
                "symbol": entry.get("symbol"),
                "marketcap": (
                    str(entry.get("marketcap"))
                    if entry.get("marketcap") is not None
                    else None
                ),
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
            "trailing_active": bool(entry.get("trailing_active", False)),
            "trailing_peak": parse_decimal(entry.get("trailing_peak")),
            "trailing_below_since": entry.get("trailing_below_since"),
            "trailing_confirmed": bool(entry.get("trailing_confirmed", False)),
            "buy_sol_used": parse_decimal(entry.get("buy_sol_used")),
            "name": entry.get("name"),
            "symbol": entry.get("symbol"),
            "marketcap": parse_decimal(entry.get("marketcap")),
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


def clamp_decimal(value: Decimal, minimum: Decimal, maximum: Decimal) -> Decimal:
    if value < minimum:
        return minimum
    if value > maximum:
        return maximum
    return value


async def fetch_token_balance_raw(rpc: "RpcClient", owner: str, mint: str):
    result = await rpc.call(
        "getTokenAccountsByOwner",
        [owner, {"mint": mint}, {"encoding": "jsonParsed"}],
    )
    total = 0
    decimals = None
    for item in result.get("value", []):
        info = (
            item.get("account", {})
            .get("data", {})
            .get("parsed", {})
            .get("info", {})
        )
        token_amount = info.get("tokenAmount", {})
        amount = int(token_amount.get("amount", "0"))
        if decimals is None:
            decimals = int(token_amount.get("decimals", 0))
        total += amount
    return total, decimals


async def fetch_token_balance_ultra(
    swap_client: "UltraSwapClient", owner: str, mint: str
):
    holdings = await swap_client.fetch_holdings(owner)
    tokens = holdings.get("tokens")
    if not isinstance(tokens, dict):
        return None, None
    entries = tokens.get(mint) or []
    if not isinstance(entries, list):
        return None, None
    total = 0
    decimals = None
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        amount = entry.get("amount")
        if amount is None:
            continue
        total += int(amount)
        if decimals is None:
            decimals = entry.get("decimals")
    return total, decimals


def sign_ultra_transaction(keypair, tx_b64: str) -> str:
    from solders.transaction import VersionedTransaction

    raw = base64.b64decode(tx_b64)
    tx = VersionedTransaction.from_bytes(raw)
    signed = VersionedTransaction(tx.message, [keypair])
    return base64.b64encode(bytes(signed)).decode("ascii")


async def execute_take_profit_sell(
    rpc: "RpcClient",
    swap_client: "UltraSwapClient",
    keypair,
    wallet: str,
    mint: str,
    sell_ratio: Decimal,
    slippage_bps: int,
    priority_fee_lamports: int,
):
    pubkey = str(keypair.pubkey())
    if wallet != pubkey:
        raise RuntimeError(f"wallet {wallet} does not match keypair {pubkey}")
    total_raw, _ = await fetch_token_balance_ultra(swap_client, wallet, mint)
    if total_raw is None and rpc is not None:
        total_raw, _ = await fetch_token_balance_raw(rpc, wallet, mint)
    if total_raw <= 0:
        raise RuntimeError("token balance is zero")
    sell_ratio = clamp_decimal(sell_ratio, Decimal(0), Decimal(1))
    sell_amount = int(
        (Decimal(total_raw) * sell_ratio).to_integral_value(rounding=ROUND_DOWN)
    )
    if sell_amount <= 0:
        raise RuntimeError("sell amount is zero")

    order = await swap_client.fetch_order(
        input_mint=mint,
        output_mint=WSOL_MINT,
        amount=sell_amount,
        slippage_bps=slippage_bps,
        priority_fee_lamports=priority_fee_lamports,
        taker=wallet,
    )
    order_tx = order.get("transaction")
    request_id = order.get("requestId")
    if not order_tx or not request_id:
        raise RuntimeError("order response missing transaction or requestId")
    signed_tx = sign_ultra_transaction(keypair, order_tx)
    response = await swap_client.execute_order(signed_tx, request_id)
    status = response.get("status")
    signature = response.get("signature")
    if status and status.lower() != "success":
        raise RuntimeError(status)
    if not signature:
        raise RuntimeError("missing signature from execute response")
    return signature


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
    def __init__(
        self, url: str, headers: dict, timeout_sec: int = 20, batch_size: int = 50
    ):
        self.url = url
        self.headers = headers
        self.batch_size = max(1, int(batch_size))
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
        for batch in chunked(ids, self.batch_size):
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


class UltraSwapClient:
    def __init__(self, url: str, headers: dict, timeout_sec: int = 20):
        self.url = url.rstrip("/")
        self.headers = headers
        self.client = httpx.AsyncClient(timeout=timeout_sec)

    async def fetch_order(
        self,
        input_mint: str,
        output_mint: str,
        amount: int,
        slippage_bps: int,
        priority_fee_lamports: int,
        taker: str,
    ) -> dict:
        params = {
            "inputMint": input_mint,
            "outputMint": output_mint,
            "amount": amount,
            "taker": taker,
        }
        if slippage_bps > 0:
            params["slippageBps"] = slippage_bps
        if priority_fee_lamports > 0:
            params["priorityFeeLamports"] = priority_fee_lamports
        resp = await self.client.get(
            f"{self.url}/order", params=params, headers=self.headers
        )
        resp.raise_for_status()
        payload = resp.json()
        if isinstance(payload, dict) and payload.get("error"):
            raise RuntimeError(payload["error"])
        return payload

    async def execute_order(self, signed_tx: str, request_id: str):
        payload = {"signedTransaction": signed_tx, "requestId": request_id}
        headers = {"Content-Type": "application/json", **self.headers}
        resp = await self.client.post(
            f"{self.url}/execute", json=payload, headers=headers
        )
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, dict) and data.get("error"):
            raise RuntimeError(data["error"])
        return data

    async def fetch_holdings(self, owner: str) -> dict:
        resp = await self.client.get(f"{self.url}/holdings/{owner}", headers=self.headers)
        resp.raise_for_status()
        payload = resp.json()
        if isinstance(payload, dict) and payload.get("error"):
            raise RuntimeError(payload["error"])
        return payload

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


async def fetch_signatures_for_address(
    rpc: RpcClient,
    address: str,
    limit: int = 100,
    before: str | None = None,
):
    params = {"limit": limit}
    if before:
        params["before"] = before
    result = await rpc.call("getSignaturesForAddress", [address, params])
    if not isinstance(result, list):
        return []
    return result


async def collect_missing_sell_entries(
    rpc: RpcClient,
    wallet: str,
    mint: str,
    commitment: str,
    since: int | None,
    limit: int,
    logged_sell_sigs: set[str],
    meta_name: str | None,
    meta_symbol: str | None,
    meta_mcap: str | None,
):
    new_entries = []
    before = None
    remaining = max(1, limit)
    stop = False
    while remaining > 0 and not stop:
        try:
            batch = await fetch_signatures_for_address(
                rpc,
                wallet,
                limit=min(1000, remaining),
                before=before,
            )
        except httpx.HTTPError as exc:
            print(f"RPC error fetching signatures for {wallet}: {exc}")
            break
        if not batch:
            break
        for sig_info in batch:
            sig = sig_info.get("signature")
            if not sig or sig in logged_sell_sigs:
                continue
            block_time = sig_info.get("blockTime")
            if since and block_time and block_time < since:
                stop = True
                break
            try:
                tx = await fetch_transaction(rpc, sig, commitment)
            except httpx.HTTPError as exc:
                print(f"RPC error fetching transaction {sig}: {exc}")
                continue
            if not tx:
                continue
            summary = summarize_trades(
                signature=sig,
                slot=tx.get("slot") or sig_info.get("slot") or 0,
                block_time=tx.get("blockTime") or block_time or 0,
                wallet=wallet,
                tx=tx,
            )
            if not summary:
                continue
            for trade in summary.get("trades", []):
                if trade.get("mint") != mint or trade.get("side") != "SELL":
                    continue
                sol_change = parse_decimal(summary.get("sol_change")) or Decimal(0)
                sol_received = max(sol_change, Decimal(0))
                amount_raw = trade.get("amount_raw")
                decimals = trade.get("decimals")
                amount_ui = (
                    format_amount(amount_raw, decimals)
                    if amount_raw is not None and decimals is not None
                    else None
                )
                entry = {
                    "ts": summary.get("block_time"),
                    "detected_ts": int(time.time()),
                    "signature": summary.get("signature"),
                    "slot": summary.get("slot"),
                    "wallet": summary.get("wallet"),
                    "side": trade.get("side"),
                    "mint": trade.get("mint"),
                    "name": meta_name or "unknown",
                    "symbol": meta_symbol or "unknown",
                    "marketcap": meta_mcap,
                    "amount_raw": amount_raw,
                    "decimals": decimals,
                    "amount_ui": amount_ui,
                    "sol_change": str(summary.get("sol_change"))
                    if summary.get("sol_change") is not None
                    else None,
                    "sol_used": None,
                    "sol_received": str(sol_received),
                    "buy_sol_used": None,
                    "sol_balance": str(summary.get("sol_balance"))
                    if summary.get("sol_balance") is not None
                    else None,
                    "buy_price": None,
                    "pnl_pct": None,
                }
                new_entries.append(entry)
        before = batch[-1].get("signature")
        remaining -= len(batch)
    return new_entries


async def collect_missing_buy_entries(
    rpc: RpcClient,
    wallet: str,
    mint: str,
    commitment: str,
    since: int | None,
    limit: int,
    logged_buy_sigs: set[str],
    meta_name: str | None,
    meta_symbol: str | None,
    meta_mcap: str | None,
):
    new_entries = []
    before = None
    remaining = max(1, limit)
    stop = False
    while remaining > 0 and not stop:
        try:
            batch = await fetch_signatures_for_address(
                rpc,
                wallet,
                limit=min(1000, remaining),
                before=before,
            )
        except httpx.HTTPError as exc:
            print(f"RPC error fetching signatures for {wallet}: {exc}")
            break
        if not batch:
            break
        for sig_info in batch:
            sig = sig_info.get("signature")
            if not sig or sig in logged_buy_sigs:
                continue
            block_time = sig_info.get("blockTime")
            if since and block_time and block_time < since:
                stop = True
                break
            try:
                tx = await fetch_transaction(rpc, sig, commitment)
            except httpx.HTTPError as exc:
                print(f"RPC error fetching transaction {sig}: {exc}")
                continue
            if not tx:
                continue
            summary = summarize_trades(
                signature=sig,
                slot=tx.get("slot") or sig_info.get("slot") or 0,
                block_time=tx.get("blockTime") or block_time or 0,
                wallet=wallet,
                tx=tx,
            )
            if not summary:
                continue
            for trade in summary.get("trades", []):
                if trade.get("mint") != mint or trade.get("side") != "BUY":
                    continue
                sol_change = parse_decimal(summary.get("sol_change")) or Decimal(0)
                sol_used = max(-sol_change, Decimal(0))
                amount_raw = trade.get("amount_raw")
                decimals = trade.get("decimals")
                amount_ui = (
                    format_amount(amount_raw, decimals)
                    if amount_raw is not None and decimals is not None
                    else None
                )
                entry = {
                    "ts": summary.get("block_time"),
                    "detected_ts": int(time.time()),
                    "signature": summary.get("signature"),
                    "slot": summary.get("slot"),
                    "wallet": summary.get("wallet"),
                    "side": trade.get("side"),
                    "mint": trade.get("mint"),
                    "name": meta_name or "unknown",
                    "symbol": meta_symbol or "unknown",
                    "marketcap": meta_mcap,
                    "amount_raw": amount_raw,
                    "decimals": decimals,
                    "amount_ui": amount_ui,
                    "sol_change": str(summary.get("sol_change"))
                    if summary.get("sol_change") is not None
                    else None,
                    "sol_used": str(sol_used),
                    "sol_received": None,
                    "buy_sol_used": str(sol_used),
                    "sol_balance": str(summary.get("sol_balance"))
                    if summary.get("sol_balance") is not None
                    else None,
                    "buy_price": None,
                    "pnl_pct": None,
                }
                new_entries.append(entry)
        before = batch[-1].get("signature")
        remaining -= len(batch)
    return new_entries


async def collect_missing_buys_for_wallet(
    rpc: RpcClient,
    token_cache: "JupiterTokenCache",
    wallet: str,
    commitment: str,
    since: int | None,
    limit: int,
    logged_buy_keys: set[tuple[str, str, str]],
):
    new_entries = []
    before = None
    remaining = max(1, limit * 5)
    found = 0
    stop = False
    while remaining > 0 and not stop:
        try:
            batch = await fetch_signatures_for_address(
                rpc,
                wallet,
                limit=min(1000, remaining),
                before=before,
            )
        except httpx.HTTPError as exc:
            print(f"RPC error fetching signatures for {wallet}: {exc}")
            break
        if not batch:
            break
        for sig_info in batch:
            sig = sig_info.get("signature")
            if not sig:
                continue
            block_time = sig_info.get("blockTime")
            if since and block_time and block_time < since:
                stop = True
                break
            try:
                tx = await fetch_transaction(rpc, sig, commitment)
            except httpx.HTTPError as exc:
                print(f"RPC error fetching transaction {sig}: {exc}")
                continue
            if not tx:
                continue
            summary = summarize_trades(
                signature=sig,
                slot=tx.get("slot") or sig_info.get("slot") or 0,
                block_time=tx.get("blockTime") or block_time or 0,
                wallet=wallet,
                tx=tx,
            )
            if not summary:
                continue
            buys = [t for t in summary.get("trades", []) if t.get("side") == "BUY"]
            if not buys:
                continue
            mints = [t.get("mint") for t in buys if t.get("mint")]
            if mints:
                try:
                    await token_cache.ensure(mints)
                except Exception:
                    pass
            sol_used = max(-summary.get("sol_change", Decimal(0)), Decimal(0))
            for trade in buys:
                mint = trade.get("mint")
                if not mint:
                    continue
                key = (wallet, sig, mint)
                if key in logged_buy_keys:
                    continue
                meta = token_cache.get(mint) or {}
                name = meta.get("name") or trade.get("name") or "unknown"
                symbol = meta.get("symbol") or trade.get("symbol") or "unknown"
                amount_raw = trade.get("amount_raw")
                decimals = trade.get("decimals")
                amount_ui = (
                    format_amount(amount_raw, decimals)
                    if amount_raw is not None and decimals is not None
                    else None
                )
                entry = {
                    "ts": summary.get("block_time"),
                    "detected_ts": int(time.time()),
                    "signature": summary.get("signature"),
                    "slot": summary.get("slot"),
                    "wallet": summary.get("wallet"),
                    "side": trade.get("side"),
                    "mint": mint,
                    "name": name,
                    "symbol": symbol,
                    "marketcap": meta.get("mcap"),
                    "amount_raw": amount_raw,
                    "decimals": decimals,
                    "amount_ui": amount_ui,
                    "sol_change": str(summary.get("sol_change"))
                    if summary.get("sol_change") is not None
                    else None,
                    "sol_used": str(sol_used),
                    "sol_received": None,
                    "buy_sol_used": str(sol_used),
                    "sol_balance": str(summary.get("sol_balance"))
                    if summary.get("sol_balance") is not None
                    else None,
                    "buy_price": None,
                    "pnl_pct": None,
                }
                new_entries.append(entry)
                logged_buy_keys.add(key)
                found += 1
                if found >= limit:
                    stop = True
                    break
            if stop:
                break
        before = batch[-1].get("signature")
        remaining -= len(batch)
    return new_entries


async def rebuild_trade_log_from_chain(
    rpc: RpcClient,
    token_cache: "JupiterTokenCache",
    wallets: list[str],
    commitment: str,
    since: int | None,
    limit: int,
):
    entries = []
    for wallet in wallets:
        before = None
        remaining = max(1, limit)
        stop = False
        while remaining > 0 and not stop:
            try:
                batch = await fetch_signatures_for_address(
                    rpc,
                    wallet,
                    limit=min(1000, remaining),
                    before=before,
                )
            except httpx.HTTPError as exc:
                print(f"RPC error fetching signatures for {wallet}: {exc}")
                break
            if not batch:
                break
            for sig_info in batch:
                sig = sig_info.get("signature")
                if not sig:
                    continue
                block_time = sig_info.get("blockTime")
                if since and block_time and block_time < since:
                    stop = True
                    break
                try:
                    tx = await fetch_transaction(rpc, sig, commitment)
                except httpx.HTTPError as exc:
                    print(f"RPC error fetching transaction {sig}: {exc}")
                    continue
                if not tx:
                    continue
                summary = summarize_trades(
                    signature=sig,
                    slot=tx.get("slot") or sig_info.get("slot") or 0,
                    block_time=tx.get("blockTime") or block_time or 0,
                    wallet=wallet,
                    tx=tx,
                )
                if not summary:
                    continue
                mints = [t.get("mint") for t in summary.get("trades", []) if t.get("mint")]
                if mints:
                    try:
                        await token_cache.ensure(mints)
                    except Exception:
                        pass
                sol_change = parse_decimal(summary.get("sol_change")) or Decimal(0)
                sol_used = max(-sol_change, Decimal(0))
                sol_received = max(sol_change, Decimal(0))
                for trade in summary.get("trades", []):
                    mint = trade.get("mint")
                    if not mint:
                        continue
                    meta = token_cache.get(mint) or {}
                    name = meta.get("name") or trade.get("name") or "unknown"
                    symbol = meta.get("symbol") or trade.get("symbol") or "unknown"
                    amount_raw = trade.get("amount_raw")
                    decimals = trade.get("decimals")
                    amount_ui = (
                        format_amount(amount_raw, decimals)
                        if amount_raw is not None and decimals is not None
                        else None
                    )
                    entry = {
                        "ts": summary.get("block_time"),
                        "detected_ts": int(time.time()),
                        "signature": summary.get("signature"),
                        "slot": summary.get("slot"),
                        "wallet": summary.get("wallet"),
                        "side": trade.get("side"),
                        "mint": mint,
                        "name": name,
                        "symbol": symbol,
                        "marketcap": meta.get("mcap"),
                        "amount_raw": amount_raw,
                        "decimals": decimals,
                        "amount_ui": amount_ui,
                        "sol_change": str(summary.get("sol_change"))
                        if summary.get("sol_change") is not None
                        else None,
                        "sol_used": str(sol_used) if trade.get("side") == "BUY" else None,
                        "sol_received": str(sol_received)
                        if trade.get("side") == "SELL"
                        else None,
                        "buy_sol_used": str(sol_used)
                        if trade.get("side") == "BUY"
                        else None,
                        "sol_balance": str(summary.get("sol_balance"))
                        if summary.get("sol_balance") is not None
                        else None,
                        "buy_price": None,
                        "pnl_pct": None,
                    }
                    entries.append(entry)
            before = batch[-1].get("signature")
            remaining -= len(batch)
    return entries


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


def print_trade_summary(
    summary: dict, trade_log_path: Path | None, tz_offset_min: int
):
    ts = summary["block_time"]
    ts_str = format_ts(ts, tz_offset_min)

    sol_balance_post = summary.get("sol_balance")

    for trade in summary["trades"]:
        print("")
        side = trade["side"]
        if sys.stdout.isatty():
            icon = GREEN_CIRCLE if side == "BUY" else RED_CIRCLE
        else:
            icon = "+" if side == "BUY" else "-"
        side_label = f"{icon} {side}"
        print(f"trade: {side_label}")
        name = trade.get("name") or "unknown"
        symbol = trade.get("symbol") or "unknown"
        print(f"token: {name} ({symbol})")
        if trade.get("marketcap") is not None:
            print(f"marketcap: {format_usd(trade['marketcap'])} USD")
        else:
            print("marketcap: unknown")
        print(f"CA: {trade['mint']}")
        print(f"wallet: {summary['wallet']}")
        if trade["side"] == "BUY":
            sol_used = max(-summary["sol_change"], Decimal(0))
            print(f"sol used: {sol_used:.9f} SOL")
        else:
            sol_received = max(summary["sol_change"], Decimal(0))
            print(f"sol received: {sol_received:.9f} SOL")
            pnl_pct = trade.get("pnl_pct")
            if pnl_pct is not None:
                pnl_fmt = pnl_pct.quantize(
                    Decimal("0.01"), rounding=ROUND_HALF_UP
                )
                icon = trend_icon(pnl_pct)
                print(f"pnl: {icon} {pnl_fmt:+f}%")
        if sol_balance_post is not None:
            print(f"wallet sol balance: {sol_balance_post:.9f} SOL")
        print(f"timestamp: {ts_str} | slot {summary['slot']}")

        if trade_log_path:
            amount_ui = format_amount(trade["amount_raw"], trade["decimals"])
            entry = {
                "ts": ts,
                "detected_ts": int(time.time()),
                "signature": summary.get("signature"),
                "slot": summary.get("slot"),
                "wallet": summary.get("wallet"),
                "side": trade.get("side"),
                "mint": trade.get("mint"),
                "name": trade.get("name"),
                "symbol": trade.get("symbol"),
                "marketcap": (
                    str(trade.get("marketcap"))
                    if trade.get("marketcap") is not None
                    else None
                ),
                "amount_raw": trade.get("amount_raw"),
                "decimals": trade.get("decimals"),
                "amount_ui": amount_ui,
                "sol_change": str(summary.get("sol_change"))
                if summary.get("sol_change") is not None
                else None,
                "sol_used": str(sol_used) if trade["side"] == "BUY" else None,
                "sol_received": str(sol_received)
                if trade["side"] == "SELL"
                else None,
                "buy_sol_used": (
                    str(trade.get("buy_sol_used"))
                    if trade.get("buy_sol_used") is not None
                    else (
                        str(trade.get("sol_used"))
                        if trade.get("sol_used") is not None
                        else None
                    )
                ),
                "sol_balance": str(sol_balance_post)
                if sol_balance_post is not None
                else None,
                "buy_price": (
                    str(trade.get("buy_price"))
                    if trade.get("buy_price") is not None
                    else None
                ),
                "pnl_pct": (
                    str(trade.get("pnl_pct"))
                    if trade.get("pnl_pct") is not None
                    else None
                ),
            }
            append_trade_log(trade_log_path, entry)


async def run_price_alerts(
    price_client,
    price_watch: dict,
    price_lock: asyncio.Lock,
    price_watch_path: Path,
    poll_sec: int,
    trailing_start_pct: Decimal,
    trailing_drawdown_pct: Decimal,
    trailing_status_sec: int,
    auto_sell_enabled: bool,
    take_profit_pct: Decimal,
    stop_loss_levels: list[Decimal],
    drawdown_status_pct: Decimal,
    drawdown_status_sec: int,
    sell_ratio: Decimal,
    sell_slippage_bps: int,
    sell_priority_fee_lamports: int,
    sell_retry_sec: int,
    swap_client: "UltraSwapClient | None",
    rpc: "RpcClient | None",
    keypairs: dict,
    tz_offset_min: int,
    trading_enabled_wallets: set[str],
    trade_control_path: Path | None,
    trade_control_poll_sec: int,
    wallet_aliases: dict,
    tracked_wallets: set[str],
    trade_log_path: Path | None,
    price_watch_sync_sec: int,
    price_max_rpm: int,
    price_batch_size: int,
    sell_inflight_timeout_sec: int,
    trailing_confirm_sec: int,
    sell_confirm_delay_sec: int,
    sell_confirm_max_attempts: int,
):
    if poll_sec <= 0:
        return
    trailing_start_threshold = (trailing_start_pct * Decimal(100)).quantize(
        Decimal("0.01"), rounding=ROUND_HALF_UP
    )
    trailing_drawdown_threshold = (trailing_drawdown_pct * Decimal(100)).quantize(
        Decimal("0.01"), rounding=ROUND_HALF_UP
    )
    take_profit_threshold = (take_profit_pct * Decimal(100)).quantize(
        Decimal("0.01"), rounding=ROUND_HALF_UP
    )
    stop_loss_levels = sorted(
        {level for level in stop_loss_levels if level is not None and level > 0}
    )
    drawdown_status_ratio = (
        drawdown_status_pct if drawdown_status_pct is not None else Decimal(0)
    ) / Decimal(100)
    if drawdown_status_ratio < 0:
        drawdown_status_ratio = Decimal(0)
    if auto_sell_enabled and (
        swap_client is None or rpc is None or not keypairs
    ):
        print("auto sell disabled: missing swap client or keypair")
        auto_sell_enabled = False
    last_status = {}
    last_drawdown_status = {}
    control_overrides = {}
    last_control_check = 0.0
    last_control_mtime = None
    base_trading_enabled_wallets = set(trading_enabled_wallets)
    last_effective_trading = None
    last_watch_sync = 0.0
    last_poll_info = None
    next_sleep = poll_sec
    while True:
        try:
            await asyncio.sleep(next_sleep)
            if trade_control_path and trade_control_poll_sec > 0:
                now = time.monotonic()
                if now - last_control_check >= trade_control_poll_sec:
                    last_control_check = now
                    try:
                        if trade_control_path.exists():
                            mtime = trade_control_path.stat().st_mtime
                            if mtime != last_control_mtime:
                                control_overrides = load_trade_controls(
                                    trade_control_path,
                                    wallet_aliases,
                                    tracked_wallets,
                                )
                                last_control_mtime = mtime
                        else:
                            if last_control_mtime is not None:
                                control_overrides = {}
                                last_control_mtime = None
                    except Exception:
                        pass
            trading_enabled_wallets = set(base_trading_enabled_wallets)
            if control_overrides:
                for wallet, enabled in control_overrides.items():
                    if enabled:
                        trading_enabled_wallets.add(wallet)
                    else:
                        trading_enabled_wallets.discard(wallet)
            effective_trading = {
                wallet: wallet in trading_enabled_wallets for wallet in tracked_wallets
            }
            if last_effective_trading is None:
                last_effective_trading = effective_trading
            else:
                ts_str = format_ts(time.time(), tz_offset_min)
                for wallet, enabled in effective_trading.items():
                    if last_effective_trading.get(wallet) != enabled:
                        state = "enabled" if enabled else "disabled"
                        print(f"[CONTROL] {ts_str} | wallet {wallet} | trading {state}")
                last_effective_trading = effective_trading
            if trade_log_path and price_watch_sync_sec > 0:
                now = time.monotonic()
                if now - last_watch_sync >= price_watch_sync_sec:
                    last_watch_sync = now
                    entries = load_trade_log_entries(trade_log_path)
                    async with price_lock:
                        removed = sync_price_watch_with_log_entries(
                            price_watch, entries, price_watch_path
                        )
                    if removed:
                        ts_str = format_ts(time.time(), tz_offset_min)
                        print(
                            f"[WATCH] {ts_str} | removed {removed} stale entries"
                        )
            async with price_lock:
                watch_snapshot = {
                    key: {
                        "buy_price": entry.get("buy_price"),
                    "trailing_active": entry.get("trailing_active", False),
                    "trailing_peak": entry.get("trailing_peak"),
                    "trailing_below_since": entry.get("trailing_below_since"),
                    "trailing_confirmed": entry.get("trailing_confirmed", False),
                    "name": entry.get("name"),
                    "symbol": entry.get("symbol"),
                    "marketcap": entry.get("marketcap"),
                    "buy_sol_used": entry.get("buy_sol_used"),
                    "sell_inflight": entry.get("sell_inflight", False),
                    "sell_last_attempt": entry.get("sell_last_attempt", 0),
                    "sell_last_reason": entry.get("sell_last_reason"),
                    "sell_skip_reason": entry.get("sell_skip_reason"),
                    "sell_confirm_due": entry.get("sell_confirm_due"),
                    "sell_confirm_attempts": entry.get("sell_confirm_attempts", 0),
                    "sell_skip_logged_at": entry.get("sell_skip_logged_at", 0),
                    }
                    for key, entry in price_watch.items()
                }
            if not watch_snapshot:
                next_sleep = poll_sec
                continue

            if (
                swap_client
                and sell_confirm_delay_sec > 0
                and sell_confirm_max_attempts > 0
            ):
                confirm_candidates = []
                for (wallet, mint), entry in watch_snapshot.items():
                    if not entry.get("sell_inflight"):
                        continue
                    due = entry.get("sell_confirm_due")
                    if due is None or status_now < float(due):
                        continue
                    confirm_candidates.append(
                        {
                            "wallet": wallet,
                            "mint": mint,
                            "symbol": entry.get("symbol") or entry.get("name") or mint,
                            "attempts": int(entry.get("sell_confirm_attempts", 0)),
                        }
                    )
                for candidate in confirm_candidates:
                    try:
                        total_raw, _ = await fetch_token_balance_ultra(
                            swap_client, candidate["wallet"], candidate["mint"]
                        )
                    except Exception as exc:
                        print(f"sell confirm error: {exc}")
                        total_raw = None
                    async with price_lock:
                        entry = price_watch.get((candidate["wallet"], candidate["mint"]))
                        if not entry:
                            continue
                        if total_raw is not None and total_raw <= 0:
                            price_watch.pop(
                                (candidate["wallet"], candidate["mint"]), None
                            )
                            save_price_watch(price_watch_path, price_watch)
                            ts_str = format_ts(time.time(), tz_offset_min)
                            print(
                                f"[SELL] {ts_str} | {candidate['symbol']} | "
                                "confirmed (balance zero)"
                            )
                            continue
                        attempts = candidate["attempts"] + 1
                        if attempts >= sell_confirm_max_attempts:
                            entry["sell_confirm_due"] = None
                            entry["sell_confirm_attempts"] = attempts
                        else:
                            entry["sell_confirm_attempts"] = attempts
                            entry["sell_confirm_due"] = (
                                time.monotonic() + sell_confirm_delay_sec
                            )

            mints = sorted({mint for (_, mint) in watch_snapshot.keys()})
            batches = 1
            if price_batch_size > 0:
                batches = max(1, math.ceil(len(mints) / price_batch_size))
            rate_limit_sec = 0
            if price_max_rpm > 0:
                rate_limit_sec = max(
                    1, math.ceil((batches * 60) / price_max_rpm)
                )
            effective_poll_sec = max(poll_sec, rate_limit_sec)
            poll_info = (len(mints), batches, effective_poll_sec)
            if poll_info != last_poll_info:
                last_poll_info = poll_info
                ts_str = format_ts(time.time(), tz_offset_min)
                print(
                    f"[PRICE] {ts_str} | tracked {len(mints)} | "
                    f"batches {batches} | poll {effective_poll_sec}s"
                )
            prices = await price_client.fetch_prices(mints)
            if not prices:
                next_sleep = effective_poll_sec
                continue

            alerts = []
            pending_updates = {}
            status_updates = []
            drawdown_updates = []
            status_enabled = trailing_status_sec > 0
            status_now = time.monotonic()
            sell_candidates = []
            trailing_enabled = trailing_start_pct > 0 and trailing_drawdown_pct > 0
            sell_enabled = auto_sell_enabled and sell_ratio > 0
            for (wallet, mint), entry in watch_snapshot.items():
                price = prices.get(mint)
                if price is None:
                    continue
                buy_price = entry.get("buy_price")
                if buy_price is None or buy_price <= 0:
                    continue
                change = (price - buy_price) / buy_price
                key = (wallet, mint)
                trailing_active = bool(entry.get("trailing_active", False))
                trailing_peak = entry.get("trailing_peak")

                trailing_stop_hit = False
                if trailing_enabled:
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
                                "trailing_below_since": None,
                                "trailing_confirmed": False,
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
                        below_since = entry.get("trailing_below_since")
                        confirmed = bool(entry.get("trailing_confirmed", False))
                        if price <= stop_price:
                            if entry.get("sell_inflight"):
                                trailing_stop_hit = False
                            elif confirmed:
                                trailing_stop_hit = True
                            elif trailing_confirm_sec > 0:
                                now_epoch = time.time()
                                if below_since is None:
                                    pending_updates.setdefault(
                                        key, {"buy_price": buy_price, "changes": {}}
                                    )["changes"]["trailing_below_since"] = now_epoch
                                    symbol = entry.get("symbol") or entry.get("name") or mint
                                    change_pct = (change * Decimal(100)).quantize(
                                        Decimal("0.01"), rounding=ROUND_HALF_UP
                                    )
                                    stop_pct = (
                                        (stop_price - buy_price) / buy_price * Decimal(100)
                                    ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
                                    ts_str = format_ts(time.time(), tz_offset_min)
                                    print(
                                        f"[TRAIL] {ts_str} | {symbol} | "
                                        f"confirm start | {change_pct:+f}% | "
                                        f"stop {stop_pct:+f}%"
                                    )
                                elif now_epoch - float(below_since) >= trailing_confirm_sec:
                                    symbol = entry.get("symbol") or entry.get("name") or mint
                                    ts_str = format_ts(time.time(), tz_offset_min)
                                    print(
                                        f"[TRAIL] {ts_str} | {symbol} | "
                                        "confirm hit"
                                    )
                                    trailing_stop_hit = True
                                    pending_updates.setdefault(
                                        key, {"buy_price": buy_price, "changes": {}}
                                    )["changes"].update(
                                        {
                                            "trailing_confirmed": True,
                                            "trailing_below_since": None,
                                        }
                                    )
                            else:
                                trailing_stop_hit = True
                                pending_updates.setdefault(
                                    key, {"buy_price": buy_price, "changes": {}}
                                )["changes"]["trailing_confirmed"] = True
                        elif below_since is not None or confirmed:
                            pending_updates.setdefault(
                                key, {"buy_price": buy_price, "changes": {}}
                            )["changes"].update(
                                {
                                    "trailing_below_since": None,
                                    "trailing_confirmed": False,
                                }
                            )
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
                            trailing_stop_hit = True

                if status_enabled and trailing_active and not entry.get("sell_inflight"):
                    last_ts = last_status.get(key, 0)
                    if status_now - last_ts >= trailing_status_sec:
                        symbol = entry.get("symbol") or entry.get("name") or mint
                        effective_peak = trailing_peak or price
                        stop_price = effective_peak * (
                            Decimal(1) - trailing_drawdown_pct
                        )
                        stop_gain = (stop_price - buy_price) / buy_price
                        base_mcap = entry.get("marketcap")
                        stop_mcap = None
                        current_mcap = None
                        if base_mcap is not None:
                            stop_mcap = base_mcap * (Decimal(1) + stop_gain)
                            current_mcap = base_mcap * (Decimal(1) + change)
                        status_updates.append(
                            {
                                "symbol": symbol,
                                "change": change,
                                "stop_gain": stop_gain,
                                "stop_mcap": stop_mcap,
                                "current_mcap": current_mcap,
                            }
                        )
                        last_status[key] = status_now

                if drawdown_status_ratio > 0 and not entry.get("sell_inflight"):
                    if change <= -drawdown_status_ratio:
                        last_ts = last_drawdown_status.get(key, 0)
                        if status_now - last_ts >= drawdown_status_sec:
                            symbol = entry.get("symbol") or entry.get("name") or mint
                            drawdown_updates.append(
                                {
                                    "symbol": symbol,
                                    "change": change,
                                }
                            )
                            last_drawdown_status[key] = status_now
                    else:
                        last_drawdown_status.pop(key, None)

                sell_reason = None
                if trailing_enabled and trailing_stop_hit:
                    sell_reason = "TRAILING_TP"
                else:
                    stop_loss_level = None
                    if stop_loss_levels and change < 0:
                        triggered = [
                            level for level in stop_loss_levels if change <= -level
                        ]
                        if triggered:
                            triggered.sort()
                            last_level = parse_stop_loss_level(
                                entry.get("sell_last_reason")
                            )
                            if last_level is not None:
                                for level in triggered:
                                    if level > last_level:
                                        stop_loss_level = level
                                        break
                            if stop_loss_level is None:
                                stop_loss_level = triggered[0]
                    if stop_loss_level is not None:
                        sell_reason = stop_loss_reason_for_level(stop_loss_level)
                    elif (
                        not trailing_enabled
                        and take_profit_pct > 0
                        and change >= take_profit_pct
                    ):
                        sell_reason = "TAKE_PROFIT"
                if sell_enabled and sell_reason:
                    last_attempt = entry.get("sell_last_attempt", 0)
                    last_reason = entry.get("sell_last_reason")
                    is_stop_loss = bool(
                        sell_reason and sell_reason.startswith("STOP_LOSS_")
                    )
                    cooldown_ok = (
                        sell_retry_sec <= 0
                        or status_now - last_attempt >= sell_retry_sec
                        or sell_reason != last_reason
                        or is_stop_loss
                    )
                    skip_reason = None
                    sell_inflight = bool(entry.get("sell_inflight"))
                    inflight_timed_out = (
                        sell_inflight
                        and sell_inflight_timeout_sec > 0
                        and last_attempt
                        and status_now - last_attempt >= sell_inflight_timeout_sec
                    )
                    if inflight_timed_out:
                        pending_updates.setdefault(
                            key, {"buy_price": buy_price, "changes": {}}
                        )["changes"]["sell_inflight"] = False
                        sell_inflight = False

                    if wallet not in trading_enabled_wallets:
                        skip_reason = "trading_disabled"
                    elif wallet not in keypairs:
                        skip_reason = "missing_keypair"
                    elif sell_inflight:
                        skip_reason = "sell_inflight"
                    elif not cooldown_ok:
                        skip_reason = "cooldown"

                    if skip_reason:
                        last_skip_reason = entry.get("sell_skip_reason")
                        last_skip_log = entry.get("sell_skip_logged_at", 0)
                        if (
                            last_skip_reason != skip_reason
                            or status_now - last_skip_log >= 30
                        ):
                            pending_updates.setdefault(
                                key, {"buy_price": buy_price, "changes": {}}
                            )["changes"].update(
                                {
                                    "sell_skip_reason": skip_reason,
                                    "sell_skip_logged_at": status_now,
                                }
                            )
                            symbol = entry.get("symbol") or entry.get("name") or mint
                            change_pct = (change * Decimal(100)).quantize(
                                Decimal("0.01"), rounding=ROUND_HALF_UP
                            )
                            print("")
                            print(
                                f"[SELL] {symbol} | skipped | {skip_reason} | "
                                f"{sell_reason} | {change_pct:+f}%"
                            )
                        continue

                    if cooldown_ok:
                        symbol = entry.get("symbol") or entry.get("name") or mint
                        sell_candidates.append(
                            {
                                "wallet": wallet,
                                "mint": mint,
                                "symbol": symbol,
                                "change": change,
                                "reason": sell_reason,
                            }
                        )

            if pending_updates:
                async with price_lock:
                    for (wallet, mint), payload in pending_updates.items():
                        entry = price_watch.get((wallet, mint))
                        if not entry:
                            continue
                        if entry.get("buy_price") != payload["buy_price"]:
                            continue
                        entry.update(payload["changes"])
                    save_price_watch(price_watch_path, price_watch)

            if prices:
                async with price_lock:
                    for (wallet, mint), entry in price_watch.items():
                        price = prices.get(mint)
                        if price is None:
                            continue
                        entry["last_price"] = price
                        entry["last_price_ts"] = time.time()

            if sell_candidates:
                async with price_lock:
                    for candidate in sell_candidates:
                        key = (candidate["wallet"], candidate["mint"])
                        entry = price_watch.get(key)
                        if entry and not entry.get("sell_inflight"):
                            entry["sell_inflight"] = True
                            entry["sell_last_attempt"] = status_now
                            entry["sell_last_reason"] = candidate.get("reason")

            if status_enabled:
                active_keys = {
                    key
                    for key, entry in watch_snapshot.items()
                    if entry.get("trailing_active")
                }
                for key in list(last_status.keys()):
                    if key not in active_keys:
                        last_status.pop(key, None)
            for key in list(last_drawdown_status.keys()):
                if key not in watch_snapshot:
                    last_drawdown_status.pop(key, None)

            if alerts:
                ts_str = format_ts(time.time(), tz_offset_min)
                for alert in alerts:
                    kind = alert["type"]
                    wallet = alert["wallet"]
                    mint = alert["mint"]
                    entry = alert["entry"]
                    price = alert["price"]
                    buy_price = alert["buy_price"]
                    name = entry.get("name") or "unknown"
                    symbol = entry.get("symbol") or "unknown"
                    if kind == "TRAILING_ARMED":
                        change_pct = ((price - buy_price) / buy_price * Decimal(100)).quantize(
                            Decimal("0.01"), rounding=ROUND_HALF_UP
                        )
                        print(
                            f"[TRAIL] {ts_str} | {symbol} | armed at {change_pct:+f}%"
                        )
                        continue

            if status_updates:
                ts_str = format_ts(time.time(), tz_offset_min)
                for update in status_updates:
                    change_pct = (update["change"] * Decimal(100)).quantize(
                        Decimal("0.01"), rounding=ROUND_HALF_UP
                    )
                    stop_pct = (update["stop_gain"] * Decimal(100)).quantize(
                        Decimal("0.01"), rounding=ROUND_HALF_UP
                    )
                    mcap_label = format_mcap_short(update.get("stop_mcap"))
                    current_label = format_mcap_short(update.get("current_mcap"))
                    if mcap_label and current_label:
                        print(
                            f"[TRAIL] {ts_str} | {update['symbol']} | "
                            f"{change_pct:+f}% | sell at {stop_pct:+f}% | "
                            f"mc now {current_label} | stop {mcap_label}"
                        )
                    elif mcap_label:
                        print(
                            f"[TRAIL] {ts_str} | {update['symbol']} | "
                            f"{change_pct:+f}% | sell at {stop_pct:+f}% | "
                            f"mc stop {mcap_label}"
                        )
                    else:
                        print(
                            f"[TRAIL] {ts_str} | {update['symbol']} | "
                            f"{change_pct:+f}% | sell at {stop_pct:+f}%"
                        )

            if drawdown_updates:
                ts_str = format_ts(time.time(), tz_offset_min)
                for update in drawdown_updates:
                    change_pct = (update["change"] * Decimal(100)).quantize(
                        Decimal("0.01"), rounding=ROUND_HALF_UP
                    )
                    print(f"[LOSS] {ts_str} | {update['symbol']} | {change_pct:+f}%")

            if sell_candidates and swap_client and rpc and keypairs:
                for candidate in sell_candidates:
                    ts_str = format_ts(time.time(), tz_offset_min)
                    change_pct = (candidate["change"] * Decimal(100)).quantize(
                        Decimal("0.01"), rounding=ROUND_HALF_UP
                    )
                    reason = candidate.get("reason")
                    if reason == "TRAILING_TP":
                        reason_label = (
                            f"trail {trailing_start_threshold}%/"
                            f"{trailing_drawdown_threshold}%"
                        )
                    elif reason and reason.startswith("STOP_LOSS_"):
                        reason_label = f"sl -{reason[len('STOP_LOSS_'):]}%"
                    else:
                        reason_label = f"tp {take_profit_threshold}%"
                    keypair = keypairs.get(candidate["wallet"])
                    if keypair is None:
                        continue
                    try:
                        await execute_take_profit_sell(
                            rpc=rpc,
                            swap_client=swap_client,
                            keypair=keypair,
                            wallet=candidate["wallet"],
                            mint=candidate["mint"],
                            sell_ratio=sell_ratio,
                            slippage_bps=sell_slippage_bps,
                            priority_fee_lamports=sell_priority_fee_lamports,
                        )
                        print("")
                        print(
                            f"[SELL] {ts_str} | {candidate['symbol']} | "
                            f"{reason_label} | {change_pct:+f}% | sent"
                        )
                        if sell_confirm_delay_sec > 0 and sell_ratio >= 1:
                            async with price_lock:
                                entry = price_watch.get(
                                    (candidate["wallet"], candidate["mint"])
                                )
                                if entry:
                                    entry["sell_confirm_attempts"] = 0
                                    entry["sell_confirm_due"] = (
                                        time.monotonic() + sell_confirm_delay_sec
                                    )
                    except Exception as exc:
                        print("")
                        print(
                            f"[SELL] {ts_str} | {candidate['symbol']} | "
                            f"{reason_label} | {change_pct:+f}% | error: {exc}"
                        )
                        async with price_lock:
                            entry = price_watch.get(
                                (candidate["wallet"], candidate["mint"])
                            )
                            if entry:
                                if "token balance is zero" in str(exc).lower():
                                    price_watch.pop(
                                        (candidate["wallet"], candidate["mint"]),
                                        None,
                                    )
                                    save_price_watch(price_watch_path, price_watch)
                                    print(
                                        f"[WARN] {ts_str} | {candidate['symbol']} | "
                                        "removed from price watch (balance zero)"
                                    )
                                else:
                                    entry["sell_inflight"] = False

            next_sleep = effective_poll_sec
            if status_enabled and any(
                entry.get("trailing_active") for entry in watch_snapshot.values()
            ):
                next_sleep = max(
                    next_sleep, min(poll_sec, trailing_status_sec)
                )
        except asyncio.CancelledError:
            break
        except Exception as exc:
            msg = str(exc).strip()
            if msg:
                print(f"price alert error: {type(exc).__name__}: {msg}")
            else:
                print(f"price alert error: {type(exc).__name__}: {exc!r}")
            await asyncio.sleep(min(poll_sec, 30))


def normalize_wallets(config: dict, alias_map: dict | None = None) -> list[str]:
    alias_map = alias_map or {}
    wallets = config.get("wallets")
    if wallets is None:
        wallet = config.get("wallet")
        wallets = [wallet] if wallet else []
    elif isinstance(wallets, str):
        wallets = [wallets]
    else:
        wallets = [w for w in wallets if w]
    resolved = []
    for wallet in wallets:
        if not wallet:
            continue
        resolved.append(alias_map.get(wallet, wallet))
    return list(dict.fromkeys(resolved))


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
    wallet_aliases = config.get("wallet_aliases") or {}
    if not isinstance(wallet_aliases, dict):
        wallet_aliases = {}
    wallets = normalize_wallets(config, wallet_aliases)
    wallet_trading_enabled = resolve_alias_dict_keys(
        config.get("wallet_trading_enabled") or {}, wallet_aliases
    )
    trading_enabled_wallets = set(wallets)
    for wallet, enabled in wallet_trading_enabled.items():
        if wallet in trading_enabled_wallets and not bool(enabled):
            trading_enabled_wallets.discard(wallet)
    ws_url = config["rpc_ws"]
    commitment = resolve_commitment(config.get("commitment"))
    ws_ping_interval = int(
        config.get("ws_ping_interval", WS_PING_INTERVAL_DEFAULT)
    )
    ws_ping_timeout = int(
        config.get("ws_ping_timeout", WS_PING_TIMEOUT_DEFAULT)
    )
    ws_idle_reconnect_sec = int(
        config.get("ws_idle_reconnect_sec", WS_IDLE_RECONNECT_SEC_DEFAULT)
    )
    token_list_url = config.get("jupiter_tokens_url", JUPITER_TOKENS_URL_DEFAULT)
    price_url = config.get("price_url", PRICE_URL_DEFAULT)
    swap_url = config.get("swap_url", SWAP_URL_DEFAULT)
    price_watch_path = Path(config.get("price_watch_path", PRICE_WATCH_PATH_DEFAULT))
    trade_log_path = Path(config.get("trade_log_path", TRADE_LOG_PATH_DEFAULT))
    refresh_sec = int(config.get("jupiter_refresh_sec", JUPITER_REFRESH_SEC_DEFAULT))
    price_poll_sec = int(config.get("price_poll_sec", PRICE_POLL_SEC_DEFAULT))
    price_max_rpm = int(config.get("price_max_rpm", PRICE_MAX_RPM_DEFAULT))
    price_batch_size = int(
        config.get("price_batch_size", PRICE_BATCH_SIZE_DEFAULT)
    )
    trailing_start_pct = parse_decimal(
        config.get("trailing_start_pct", TRAILING_START_PCT_DEFAULT)
    )
    trailing_drawdown_pct = parse_decimal(
        config.get("trailing_drawdown_pct", TRAILING_DRAWDOWN_PCT_DEFAULT)
    )
    trailing_status_sec = int(
        config.get("trailing_status_sec", TRAILING_STATUS_SEC_DEFAULT)
    )
    trailing_confirm_sec = int(
        config.get("trailing_confirm_sec", TRAILING_CONFIRM_SEC_DEFAULT)
    )
    auto_sell_enabled = bool(
        config.get("auto_sell_enabled", AUTO_SELL_ENABLED_DEFAULT)
    )
    take_profit_pct = parse_decimal(
        config.get("take_profit_pct", TAKE_PROFIT_PCT_DEFAULT)
    )
    stop_loss_levels_pct = parse_decimal_list(
        config.get("stop_loss_levels_pct", STOP_LOSS_LEVELS_PCT_DEFAULT)
    )
    drawdown_status_pct = parse_decimal(
        config.get("drawdown_status_pct", DRAW_DOWN_STATUS_PCT_DEFAULT)
    )
    drawdown_status_sec = int(
        config.get("drawdown_status_sec", DRAW_DOWN_STATUS_SEC_DEFAULT)
    )
    tz_offset_min = int(config.get("tz_offset_minutes", TZ_OFFSET_MINUTES_DEFAULT))
    sell_pct = parse_decimal(config.get("sell_pct", SELL_PCT_DEFAULT))
    sell_slippage_pct = parse_decimal(
        config.get("sell_slippage_pct", SELL_SLIPPAGE_PCT_DEFAULT)
    )
    sell_priority_fee_sol = parse_decimal(
        config.get("sell_priority_fee_sol", SELL_PRIORITY_FEE_SOL_DEFAULT)
    )
    sell_retry_sec = int(config.get("sell_retry_sec", SELL_RETRY_SEC_DEFAULT))
    sell_inflight_timeout_sec = int(
        config.get("sell_inflight_timeout_sec", SELL_INFLIGHT_TIMEOUT_SEC_DEFAULT)
    )
    sell_confirm_delay_sec = int(
        config.get("sell_confirm_delay_sec", SELL_CONFIRM_DELAY_SEC_DEFAULT)
    )
    sell_confirm_max_attempts = int(
        config.get("sell_confirm_max_attempts", SELL_CONFIRM_MAX_ATTEMPTS_DEFAULT)
    )
    trade_control_path = Path(
        config.get("trade_control_path", "trade_control.json")
    )
    trade_control_poll_sec = int(config.get("trade_control_poll_sec", 5))
    price_watch_sync_sec = int(
        config.get("price_watch_sync_sec", PRICE_WATCH_SYNC_SEC_DEFAULT)
    )
    env_path = Path(config.get("env_path", ENV_PATH_DEFAULT))
    load_env_file(env_path)
    keypair_env = config.get("keypair_env", KEYPAIR_ENV_DEFAULT)
    keypair_path = config.get("keypair_path")
    keypair_envs = resolve_alias_dict_keys(
        config.get("keypair_envs"), wallet_aliases
    )
    keypair_paths = resolve_alias_dict_keys(
        config.get("keypair_paths"), wallet_aliases
    )
    api_key_env = config.get("jupiter_api_key_env", "JUPITER_API_KEY")
    webhook_url = config.get("discord_webhook_url")
    webhook_env = config.get("discord_webhook_env", "DISCORD_WEBHOOK_URL")
    webhook_send_buys = bool(config.get("webhook_send_buys", False))
    webhook_send_sells = bool(config.get("webhook_send_sells", True))
    if not webhook_url and webhook_env:
        webhook_url = os.getenv(webhook_env)
    trailing_start_ratio = (
        (trailing_start_pct or TRAILING_START_PCT_DEFAULT) / Decimal(100)
    )
    trailing_drawdown_ratio = (
        (trailing_drawdown_pct or TRAILING_DRAWDOWN_PCT_DEFAULT) / Decimal(100)
    )
    take_profit_ratio = (
        (take_profit_pct or TAKE_PROFIT_PCT_DEFAULT) / Decimal(100)
    )
    stop_loss_ratios = [
        level / Decimal(100) for level in stop_loss_levels_pct if level > 0
    ]
    sell_ratio = (sell_pct or SELL_PCT_DEFAULT) / Decimal(100)
    sell_slippage_bps = int(
        (
            (sell_slippage_pct or SELL_SLIPPAGE_PCT_DEFAULT) * Decimal(100)
        ).to_integral_value(rounding=ROUND_HALF_UP)
    )
    sell_priority_fee_lamports = int(
        (
            (sell_priority_fee_sol or SELL_PRIORITY_FEE_SOL_DEFAULT)
            * Decimal(1_000_000_000)
        ).to_integral_value(rounding=ROUND_HALF_UP)
    )
    if trailing_start_ratio <= 0 or trailing_drawdown_ratio <= 0:
        trailing_start_ratio = Decimal(0)
        trailing_drawdown_ratio = Decimal(0)
    if sell_ratio <= 0 or (
        trailing_start_ratio <= 0
        and take_profit_ratio <= 0
        and not stop_loss_ratios
    ):
        auto_sell_enabled = False
    headers = config.get("jupiter_headers") or {}
    if not isinstance(headers, dict):
        headers = {}
    env_api_key = os.getenv(api_key_env) if api_key_env else None
    if env_api_key and not headers.get("x-api-key"):
        headers["x-api-key"] = env_api_key

    rpc = RpcClient(config["rpc_http"])
    token_cache = JupiterTokenCache(token_list_url, refresh_sec, headers)
    price_client = JupiterPriceClient(
        price_url, headers, batch_size=price_batch_size
    )
    webhook_client = None
    if webhook_url:
        webhook_client = httpx.AsyncClient(timeout=10)
    keypairs = {}
    if auto_sell_enabled:
        keypairs = load_keypairs(
            env_path=env_path,
            wallets=wallets,
            keypair_env=keypair_env,
            keypair_path=keypair_path,
            keypair_envs=keypair_envs,
            keypair_paths=keypair_paths,
        )
        if not keypairs:
            print("auto sell disabled: keypair not found")
            auto_sell_enabled = False
    swap_client = UltraSwapClient(swap_url, headers) if auto_sell_enabled else None
    price_watch = load_price_watch(price_watch_path, set(wallets))
    price_lock = asyncio.Lock()
    seen = deque(maxlen=2000)
    positions_live = {}
    if trade_log_path and trade_log_path.exists():
        entries = load_trade_log_entries(trade_log_path)
        positions_live, _, _ = build_trade_log_index(entries)
        positions_live = {
            key: slot.get("buys") for key, slot in positions_live.items()
        }
        removed = sync_price_watch_with_log_entries(
            price_watch, entries, price_watch_path
        )
        if removed:
            print(f"price watch cleaned: {removed} stale entries removed")

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
                price_poll_sec,
                trailing_start_ratio,
                trailing_drawdown_ratio,
                trailing_status_sec,
                auto_sell_enabled,
                take_profit_ratio,
                stop_loss_ratios,
                drawdown_status_pct,
                drawdown_status_sec,
                sell_ratio,
                sell_slippage_bps,
                sell_priority_fee_lamports,
                sell_retry_sec,
                swap_client,
                rpc,
                keypairs,
                tz_offset_min,
                trading_enabled_wallets,
                trade_control_path,
                trade_control_poll_sec,
                wallet_aliases,
                set(wallets),
                trade_log_path,
                price_watch_sync_sec,
                price_max_rpm,
                price_batch_size,
                sell_inflight_timeout_sec,
                trailing_confirm_sec,
                sell_confirm_delay_sec,
                sell_confirm_max_attempts,
            )
        )

    backoff = 1
    while True:
        try:
            async with websockets.connect(
                ws_url,
                ping_interval=ws_ping_interval,
                ping_timeout=ws_ping_timeout,
            ) as ws:
                wallet_by_sub_id, pending = await subscribe_wallets(ws, wallets, commitment)
                print(
                    f"ws connected ({commitment}); subscribed {len(wallets)} wallets"
                )

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
                    sell_pnls = {}
                    sell_sol_received = {}
                    sell_buy_sol_used = {}
                    for summary in summaries:
                        for trade in summary["trades"]:
                            if trade["side"] == "SELL":
                                key = (summary["wallet"], trade["mint"])
                                sell_keys.append(key)
                                sell_sol_received[key] = max(
                                    summary["sol_change"], Decimal(0)
                                )
                    if sell_keys:
                        async with price_lock:
                            for key in sell_keys:
                                entry = price_watch.get(key)
                                pnl_pct = None
                                buy_sol_used_fallback = pop_open_position(
                                    positions_live, key
                                )
                                if entry:
                                    buy_sol_used = entry.get("buy_sol_used")
                                    sol_received = sell_sol_received.get(key)
                                    if buy_sol_used and sol_received is not None:
                                        pnl_pct = (
                                            (sol_received - buy_sol_used)
                                            / buy_sol_used
                                        ) * Decimal(100)
                                        sell_buy_sol_used[key] = buy_sol_used
                                    elif buy_sol_used_fallback is not None:
                                        sell_buy_sol_used[key] = buy_sol_used_fallback
                                elif buy_sol_used_fallback is not None:
                                    sell_buy_sol_used[key] = buy_sol_used_fallback
                                    sol_received = sell_sol_received.get(key)
                                    if sol_received is not None:
                                        pnl_pct = (
                                            (sol_received - buy_sol_used_fallback)
                                            / buy_sol_used_fallback
                                        ) * Decimal(100)
                                sell_pnls[key] = pnl_pct
                                price_watch.pop(key, None)
                            save_price_watch(price_watch_path, price_watch)

                    buy_requests = []
                    for summary in summaries:
                        sol_used = max(-summary["sol_change"], Decimal(0))
                        for trade in summary["trades"]:
                            if trade["side"] == "BUY":
                                trade["sol_used"] = sol_used
                                positions_live.setdefault(
                                    (summary["wallet"], trade["mint"]),
                                    deque(),
                                ).append(sol_used)
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
                                        "trailing_active": False,
                                        "trailing_peak": None,
                                        "buy_sol_used": trade.get("sol_used"),
                                        "name": trade.get("name"),
                                        "symbol": trade.get("symbol"),
                                        "marketcap": trade.get("marketcap"),
                                    }
                                    trade["buy_price"] = price
                                save_price_watch(price_watch_path, price_watch)

                    for summary in summaries:
                        for trade in summary["trades"]:
                            if trade["side"] == "SELL":
                                key = (summary["wallet"], trade["mint"])
                                trade["pnl_pct"] = sell_pnls.get(key)
                                trade["buy_sol_used"] = sell_buy_sol_used.get(key)
                        print_trade_summary(summary, trade_log_path, tz_offset_min)
                        if webhook_url and webhook_client:
                            for trade in summary["trades"]:
                                side = trade.get("side")
                                if side == "BUY" and not webhook_send_buys:
                                    continue
                                if side == "SELL" and not webhook_send_sells:
                                    continue
                                message = format_trade_message(
                                    summary, trade, tz_offset_min
                                )
                                asyncio.create_task(
                                    send_discord_webhook(
                                        webhook_client, webhook_url, message
                                    )
                                )

                for msg in pending:
                    await handle_message(msg)

                if ws_idle_reconnect_sec > 0:
                    while True:
                        try:
                            raw = await asyncio.wait_for(
                                ws.recv(), timeout=ws_idle_reconnect_sec
                            )
                        except asyncio.TimeoutError:
                            print(
                                f"ws idle >{ws_idle_reconnect_sec}s; reconnecting"
                            )
                            break
                        msg = json.loads(raw)
                        await handle_message(msg)
                else:
                    async for raw in ws:
                        msg = json.loads(raw)
                        await handle_message(msg)
        except (websockets.WebSocketException, httpx.HTTPError, RuntimeError) as exc:
            err = exc
            err_payload = err.args[0] if err.args else None
            msg = ""
            if isinstance(err_payload, dict):
                msg = str(err_payload.get("message", ""))
            else:
                msg = str(err)
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
    if webhook_client:
        await webhook_client.aclose()
    if swap_client:
        await swap_client.close()
    await rpc.close()


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "trading":
        args = sys.argv[2:]
        wallet_arg = None
        state = None
        config_path = Path("config.json")
        control_path = None
        i = 0
        while i < len(args):
            arg = args[i]
            if arg == "--wallet" and i + 1 < len(args):
                wallet_arg = args[i + 1]
                i += 2
                continue
            if arg == "--on":
                state = True
                i += 1
                continue
            if arg == "--off":
                state = False
                i += 1
                continue
            if arg == "--config" and i + 1 < len(args):
                config_path = Path(args[i + 1])
                i += 2
                continue
            if arg in ("--control", "--control-path") and i + 1 < len(args):
                control_path = Path(args[i + 1])
                i += 2
                continue
            i += 1

        if wallet_arg is None or state is None:
            print("Usage: bot.py trading --wallet <alias|address> --on|--off")
            return

        config = {}
        if config_path.exists():
            try:
                config = load_config(config_path)
            except Exception:
                config = {}
        wallet_aliases = config.get("wallet_aliases") or {}
        if not isinstance(wallet_aliases, dict):
            wallet_aliases = {}
        resolved_wallet = wallet_aliases.get(wallet_arg, wallet_arg)
        if control_path is None:
            control_path = Path(
                config.get("trade_control_path", "trade_control.json")
                if isinstance(config, dict)
                else "trade_control.json"
            )

        current = load_trade_controls(control_path, wallet_aliases)
        current[resolved_wallet] = state
        write_trade_controls(control_path, current)
        status = "enabled" if state else "disabled"
        print(f"Trading {status} for {resolved_wallet}")
        return

    if len(sys.argv) > 1 and sys.argv[1] == "stats":
        args = sys.argv[2:]
        log_path = Path(TRADE_LOG_PATH_DEFAULT)
        since = None
        until = None
        tz_offset_min = TZ_OFFSET_MINUTES_DEFAULT
        i = 0
        while i < len(args):
            arg = args[i]
            if arg in ("--log", "--log-path") and i + 1 < len(args):
                log_path = Path(args[i + 1])
                i += 2
                continue
            if arg == "--since" and i + 1 < len(args):
                duration = parse_duration(args[i + 1])
                if duration is None:
                    since = parse_datetime_value(args[i + 1], tz_offset_min)
                else:
                    since = int(time.time()) - duration
                i += 2
                continue
            if arg == "--from" and i + 1 < len(args):
                since = parse_datetime_value(args[i + 1], tz_offset_min)
                i += 2
                continue
            if arg == "--to" and i + 1 < len(args):
                until = parse_datetime_value(args[i + 1], tz_offset_min)
                i += 2
                continue
            if arg == "--tz" and i + 1 < len(args):
                tz_offset_min = int(args[i + 1])
                i += 2
                continue
            i += 1
        if not log_path.exists():
            print(f"Trade log not found: {log_path}")
            return
        print_trade_stats(log_path, since, until, tz_offset_min)
        return

    if len(sys.argv) > 1 and sys.argv[1] == "backfill":
        args = sys.argv[2:]
        config_path = Path("config.json")
        wallet_arg = None
        mint_arg = None
        limit = 200
        since = None
        dry_run = False
        rpc_override = None
        i = 0
        while i < len(args):
            arg = args[i]
            if arg == "--wallet" and i + 1 < len(args):
                wallet_arg = args[i + 1]
                i += 2
                continue
            if arg == "--mint" and i + 1 < len(args):
                mint_arg = args[i + 1]
                i += 2
                continue
            if arg == "--limit" and i + 1 < len(args):
                limit = int(args[i + 1])
                i += 2
                continue
            if arg == "--since" and i + 1 < len(args):
                duration = parse_duration(args[i + 1])
                if duration is None:
                    since = parse_datetime_value(args[i + 1], TZ_OFFSET_MINUTES_DEFAULT)
                else:
                    since = int(time.time()) - duration
                i += 2
                continue
            if arg == "--config" and i + 1 < len(args):
                config_path = Path(args[i + 1])
                i += 2
                continue
            if arg in ("--rpc", "--rpc-http") and i + 1 < len(args):
                rpc_override = args[i + 1]
                i += 2
                continue
            if arg == "--dry-run":
                dry_run = True
                i += 1
                continue
            i += 1

        if not wallet_arg or not mint_arg:
            print("Usage: bot.py backfill --wallet <alias|address> --mint <mint> [--limit N] [--since <time>] [--dry-run]")
            return

        config = {}
        if config_path.exists():
            try:
                config = load_config(config_path)
            except Exception:
                config = {}
        wallet_aliases = config.get("wallet_aliases") or {}
        if not isinstance(wallet_aliases, dict):
            wallet_aliases = {}
        wallet = wallet_aliases.get(wallet_arg, wallet_arg)
        rpc_http = rpc_override or config.get("rpc_http_backfill") or config.get("rpc_http")
        commitment = resolve_commitment(config.get("commitment"))
        log_path = Path(
            config.get("trade_log_path", TRADE_LOG_PATH_DEFAULT)
            if isinstance(config, dict)
            else TRADE_LOG_PATH_DEFAULT
        )
        watch_path = Path(
            config.get("price_watch_path", PRICE_WATCH_PATH_DEFAULT)
            if isinstance(config, dict)
            else PRICE_WATCH_PATH_DEFAULT
        )
        watch_path = Path(
            config.get("price_watch_path", PRICE_WATCH_PATH_DEFAULT)
            if isinstance(config, dict)
            else PRICE_WATCH_PATH_DEFAULT
        )
        tz_offset_min = int(
            config.get("tz_offset_minutes", TZ_OFFSET_MINUTES_DEFAULT)
            if isinstance(config, dict)
            else TZ_OFFSET_MINUTES_DEFAULT
        )
        if not rpc_http:
            print("Missing rpc_http in config.")
            return
        if not log_path.exists():
            print(f"Trade log not found: {log_path}")
            return

        entries = load_trade_log_entries(log_path)
        positions, sell_sigs, meta = build_trade_log_index(entries)
        logged_sell_sigs = sell_sigs.get((wallet, mint_arg), set())
        meta_slot = meta.get((wallet, mint_arg), {})
        meta_name = meta_slot.get("name")
        meta_symbol = meta_slot.get("symbol")
        meta_mcap = meta_slot.get("marketcap")
        last_buy_ts = None
        open_slot = positions.get((wallet, mint_arg))
        if open_slot:
            for buy in open_slot.get("buys", []):
                ts_val = buy.get("ts")
                if ts_val:
                    last_buy_ts = max(last_buy_ts or 0, int(ts_val))

        if since is None:
            since = last_buy_ts
        elif last_buy_ts:
            since = max(since, last_buy_ts)

        async def run_backfill():
            rpc = RpcClient(rpc_http)
            new_entries = await collect_missing_sell_entries(
                rpc=rpc,
                wallet=wallet,
                mint=mint_arg,
                commitment=commitment,
                since=since,
                limit=limit,
                logged_sell_sigs=logged_sell_sigs,
                meta_name=meta_name,
                meta_symbol=meta_symbol,
                meta_mcap=meta_mcap,
            )
            await rpc.close()

            if not new_entries:
                print("No missing sell entries found.")
                return

            new_entries.sort(key=log_entry_ts)
            print(f"Backfill candidates: {len(new_entries)}")
            for entry in new_entries:
                ts_str = format_ts(entry.get("ts"), tz_offset_min)
                print(
                    f"- {entry.get('signature')} | {entry.get('mint')} | "
                    f"{entry.get('sol_received')} SOL | {ts_str}"
                )

            if dry_run:
                return

            entries.extend(new_entries)
            entries.sort(key=log_entry_ts)
            if not write_trade_log_entries(log_path, entries):
                print("Trade log not updated (write failed).")
                return
            print(f"Trade log updated: {log_path}")
            removed = sync_price_watch_with_log(watch_path, entries)
            if removed:
                print(f"Price watch cleaned: {removed} stale entries removed")

        asyncio.run(run_backfill())
        return

    if len(sys.argv) > 1 and sys.argv[1] == "backfill-buy":
        args = sys.argv[2:]
        config_path = Path("config.json")
        wallet_arg = None
        mint_arg = None
        limit = 200
        since = None
        dry_run = False
        rpc_override = None
        i = 0
        while i < len(args):
            arg = args[i]
            if arg == "--wallet" and i + 1 < len(args):
                wallet_arg = args[i + 1]
                i += 2
                continue
            if arg == "--mint" and i + 1 < len(args):
                mint_arg = args[i + 1]
                i += 2
                continue
            if arg == "--limit" and i + 1 < len(args):
                limit = int(args[i + 1])
                i += 2
                continue
            if arg == "--since" and i + 1 < len(args):
                duration = parse_duration(args[i + 1])
                if duration is None:
                    since = parse_datetime_value(args[i + 1], TZ_OFFSET_MINUTES_DEFAULT)
                else:
                    since = int(time.time()) - duration
                i += 2
                continue
            if arg == "--config" and i + 1 < len(args):
                config_path = Path(args[i + 1])
                i += 2
                continue
            if arg in ("--rpc", "--rpc-http") and i + 1 < len(args):
                rpc_override = args[i + 1]
                i += 2
                continue
            if arg == "--dry-run":
                dry_run = True
                i += 1
                continue
            i += 1

        if not wallet_arg or not mint_arg:
            print("Usage: bot.py backfill-buy --wallet <alias|address> --mint <mint> [--limit N] [--since <time>] [--dry-run]")
            return

        config = {}
        if config_path.exists():
            try:
                config = load_config(config_path)
            except Exception:
                config = {}
        wallet_aliases = config.get("wallet_aliases") or {}
        if not isinstance(wallet_aliases, dict):
            wallet_aliases = {}
        wallet = wallet_aliases.get(wallet_arg, wallet_arg)
        rpc_http = rpc_override or config.get("rpc_http_backfill") or config.get("rpc_http")
        commitment = resolve_commitment(config.get("commitment"))
        log_path = Path(
            config.get("trade_log_path", TRADE_LOG_PATH_DEFAULT)
            if isinstance(config, dict)
            else TRADE_LOG_PATH_DEFAULT
        )
        watch_path = Path(
            config.get("price_watch_path", PRICE_WATCH_PATH_DEFAULT)
            if isinstance(config, dict)
            else PRICE_WATCH_PATH_DEFAULT
        )
        price_url = config.get("price_url", PRICE_URL_DEFAULT)
        price_batch_size = int(
            config.get("price_batch_size", PRICE_BATCH_SIZE_DEFAULT)
            if isinstance(config, dict)
            else PRICE_BATCH_SIZE_DEFAULT
        )
        env_path = Path(config.get("env_path", ENV_PATH_DEFAULT))
        api_key_env = config.get("jupiter_api_key_env", "JUPITER_API_KEY")
        if not rpc_http:
            print("Missing rpc_http in config.")
            return
        if not log_path.exists():
            print(f"Trade log not found: {log_path}")
            return

        load_env_file(env_path)
        headers = config.get("jupiter_headers") or {}
        if not isinstance(headers, dict):
            headers = {}
        env_api_key = os.getenv(api_key_env) if api_key_env else None
        if env_api_key and not headers.get("x-api-key"):
            headers["x-api-key"] = env_api_key

        entries = load_trade_log_entries(log_path)
        positions, sell_sigs, meta = build_trade_log_index(entries)
        logged_buy_sigs = set()
        for entry in entries:
            if entry.get("side") == "BUY" and entry.get("wallet") == wallet and entry.get("mint") == mint_arg:
                sig = entry.get("signature")
                if sig:
                    logged_buy_sigs.add(sig)
        meta_slot = meta.get((wallet, mint_arg), {})
        meta_name = meta_slot.get("name")
        meta_symbol = meta_slot.get("symbol")
        meta_mcap = meta_slot.get("marketcap")

        async def run_backfill_buy():
            rpc = RpcClient(rpc_http)
            new_entries = await collect_missing_buy_entries(
                rpc=rpc,
                wallet=wallet,
                mint=mint_arg,
                commitment=commitment,
                since=since,
                limit=limit,
                logged_buy_sigs=logged_buy_sigs,
                meta_name=meta_name,
                meta_symbol=meta_symbol,
                meta_mcap=meta_mcap,
            )
            await rpc.close()

            if not new_entries:
                print("No missing buy entries found.")
                return

            new_entries.sort(key=log_entry_ts)
            print(f"Backfill candidates: {len(new_entries)}")
            for entry in new_entries:
                ts_str = format_ts(entry.get("ts"), TZ_OFFSET_MINUTES_DEFAULT)
                print(
                    f"- {entry.get('wallet')} | {entry.get('mint')} | "
                    f"{entry.get('sol_used')} SOL | {ts_str}"
                )

            if dry_run:
                return

            entries.extend(new_entries)
            entries.sort(key=log_entry_ts)
            if not write_trade_log_entries(log_path, entries):
                print("Trade log not updated (write failed).")
                return
            print(f"Trade log updated: {log_path}")

            price_client = JupiterPriceClient(
                price_url, headers, batch_size=price_batch_size
            )
            prices = await price_client.fetch_prices([mint_arg])
            await price_client.close()
            price = prices.get(mint_arg)
            if price is None:
                print("Price watch not updated: no price available")
                return

            watch = load_price_watch(watch_path, set())
            newest = max(new_entries, key=log_entry_ts)
            buy_sol_used = parse_decimal(newest.get("sol_used"))
            watch[(wallet, mint_arg)] = {
                "buy_price": price,
                "trailing_active": False,
                "trailing_peak": None,
                "trailing_below_since": None,
                "buy_sol_used": buy_sol_used,
                "name": meta_name or "unknown",
                "symbol": meta_symbol or "unknown",
                "marketcap": parse_decimal(meta_mcap),
            }
            save_price_watch(watch_path, watch)
            print("Price watch updated from backfill (current price).")

        asyncio.run(run_backfill_buy())
        return

    if len(sys.argv) > 1 and sys.argv[1] == "recent-buys":
        args = sys.argv[2:]
        config_path = Path("config.json")
        wallet_arg = None
        limit = 50
        since = None
        rpc_override = None
        i = 0
        while i < len(args):
            arg = args[i]
            if arg == "--wallet" and i + 1 < len(args):
                wallet_arg = args[i + 1]
                i += 2
                continue
            if arg == "--limit" and i + 1 < len(args):
                limit = int(args[i + 1])
                i += 2
                continue
            if arg == "--since" and i + 1 < len(args):
                duration = parse_duration(args[i + 1])
                if duration is None:
                    since = parse_datetime_value(args[i + 1], TZ_OFFSET_MINUTES_DEFAULT)
                else:
                    since = int(time.time()) - duration
                i += 2
                continue
            if arg == "--config" and i + 1 < len(args):
                config_path = Path(args[i + 1])
                i += 2
                continue
            if arg in ("--rpc", "--rpc-http") and i + 1 < len(args):
                rpc_override = args[i + 1]
                i += 2
                continue
            i += 1

        if not wallet_arg:
            print("Usage: bot.py recent-buys --wallet <alias|address> [--limit N] [--since <time>]")
            return

        config = {}
        if config_path.exists():
            try:
                config = load_config(config_path)
            except Exception:
                config = {}
        wallet_aliases = config.get("wallet_aliases") or {}
        if not isinstance(wallet_aliases, dict):
            wallet_aliases = {}
        wallet = wallet_aliases.get(wallet_arg, wallet_arg)
        rpc_http = rpc_override or config.get("rpc_http_backfill") or config.get("rpc_http")
        commitment = resolve_commitment(config.get("commitment"))
        token_list_url = config.get("jupiter_tokens_url", JUPITER_TOKENS_URL_DEFAULT)
        refresh_sec = int(config.get("jupiter_refresh_sec", JUPITER_REFRESH_SEC_DEFAULT))
        tz_offset_min = int(
            config.get("tz_offset_minutes", TZ_OFFSET_MINUTES_DEFAULT)
            if isinstance(config, dict)
            else TZ_OFFSET_MINUTES_DEFAULT
        )
        env_path = Path(config.get("env_path", ENV_PATH_DEFAULT))
        api_key_env = config.get("jupiter_api_key_env", "JUPITER_API_KEY")

        if not rpc_http:
            print("Missing rpc_http in config.")
            return

        load_env_file(env_path)
        headers = config.get("jupiter_headers") or {}
        if not isinstance(headers, dict):
            headers = {}
        env_api_key = os.getenv(api_key_env) if api_key_env else None
        if env_api_key and not headers.get("x-api-key"):
            headers["x-api-key"] = env_api_key

        async def run_recent_buys():
            rpc = RpcClient(rpc_http)
            token_cache = JupiterTokenCache(token_list_url, refresh_sec, headers)
            results = []
            before = None
            remaining = max(1, limit * 5)
            stop = False
            while remaining > 0 and not stop:
                try:
                    batch = await fetch_signatures_for_address(
                        rpc,
                        wallet,
                        limit=min(1000, remaining),
                        before=before,
                    )
                except httpx.HTTPError as exc:
                    print(f"RPC error fetching signatures for {wallet}: {exc}")
                    break
                if not batch:
                    break
                for sig_info in batch:
                    sig = sig_info.get("signature")
                    if not sig:
                        continue
                    block_time = sig_info.get("blockTime")
                    if since and block_time and block_time < since:
                        stop = True
                        break
                    try:
                        tx = await fetch_transaction(rpc, sig, commitment)
                    except httpx.HTTPError as exc:
                        print(f"RPC error fetching transaction {sig}: {exc}")
                        continue
                    if not tx:
                        continue
                    summary = summarize_trades(
                        signature=sig,
                        slot=tx.get("slot") or sig_info.get("slot") or 0,
                        block_time=tx.get("blockTime") or block_time or 0,
                        wallet=wallet,
                        tx=tx,
                    )
                    if not summary:
                        continue
                    buys = [t for t in summary.get("trades", []) if t.get("side") == "BUY"]
                    if not buys:
                        continue
                    mints = [t.get("mint") for t in buys if t.get("mint")]
                    if mints:
                        try:
                            await token_cache.ensure(mints)
                        except Exception:
                            pass
                    sol_used = max(-summary.get("sol_change", Decimal(0)), Decimal(0))
                    for trade in buys:
                        mint = trade.get("mint")
                        meta = token_cache.get(mint) or {}
                        name = meta.get("name") or trade.get("name") or "unknown"
                        symbol = meta.get("symbol") or trade.get("symbol") or "unknown"
                        results.append(
                            {
                                "ts": summary.get("block_time"),
                                "signature": summary.get("signature"),
                                "mint": mint,
                                "name": name,
                                "symbol": symbol,
                                "sol_used": sol_used,
                            }
                        )
                        if len(results) >= limit:
                            stop = True
                            break
                    if stop:
                        break
                before = batch[-1].get("signature")
                remaining -= len(batch)

            await token_cache.close()
            await rpc.close()

            if not results:
                print("No recent buys found.")
                return

            results.sort(key=lambda item: item["ts"] or 0, reverse=True)
            print(f"Recent buys for {wallet}:")
            for item in results:
                ts_str = format_ts(item.get("ts"), tz_offset_min)
                print(
                    f"- {item['symbol']} | {item['name']} | {item['mint']} | "
                    f"{item['sol_used']:.9f} SOL | {ts_str} | {item['signature']}"
                )

        asyncio.run(run_recent_buys())
        return

    if len(sys.argv) > 1 and sys.argv[1] == "backfill-buys":
        args = sys.argv[2:]
        config_path = Path("config.json")
        wallet_arg = None
        limit = 50
        since = None
        dry_run = False
        rpc_override = None
        i = 0
        while i < len(args):
            arg = args[i]
            if arg == "--wallet" and i + 1 < len(args):
                wallet_arg = args[i + 1]
                i += 2
                continue
            if arg == "--limit" and i + 1 < len(args):
                limit = int(args[i + 1])
                i += 2
                continue
            if arg == "--since" and i + 1 < len(args):
                duration = parse_duration(args[i + 1])
                if duration is None:
                    since = parse_datetime_value(args[i + 1], TZ_OFFSET_MINUTES_DEFAULT)
                else:
                    since = int(time.time()) - duration
                i += 2
                continue
            if arg == "--config" and i + 1 < len(args):
                config_path = Path(args[i + 1])
                i += 2
                continue
            if arg in ("--rpc", "--rpc-http") and i + 1 < len(args):
                rpc_override = args[i + 1]
                i += 2
                continue
            if arg == "--dry-run":
                dry_run = True
                i += 1
                continue
            i += 1

        config = {}
        if config_path.exists():
            try:
                config = load_config(config_path)
            except Exception:
                config = {}
        wallet_aliases = config.get("wallet_aliases") or {}
        if not isinstance(wallet_aliases, dict):
            wallet_aliases = {}
        wallets = normalize_wallets(config, wallet_aliases)
        if wallet_arg:
            wallets = [wallet_aliases.get(wallet_arg, wallet_arg)]
        rpc_http = rpc_override or config.get("rpc_http_backfill") or config.get("rpc_http")
        commitment = resolve_commitment(config.get("commitment"))
        token_list_url = config.get("jupiter_tokens_url", JUPITER_TOKENS_URL_DEFAULT)
        refresh_sec = int(config.get("jupiter_refresh_sec", JUPITER_REFRESH_SEC_DEFAULT))
        log_path = Path(
            config.get("trade_log_path", TRADE_LOG_PATH_DEFAULT)
            if isinstance(config, dict)
            else TRADE_LOG_PATH_DEFAULT
        )
        watch_path = Path(
            config.get("price_watch_path", PRICE_WATCH_PATH_DEFAULT)
            if isinstance(config, dict)
            else PRICE_WATCH_PATH_DEFAULT
        )
        price_url = config.get("price_url", PRICE_URL_DEFAULT)
        price_batch_size = int(
            config.get("price_batch_size", PRICE_BATCH_SIZE_DEFAULT)
            if isinstance(config, dict)
            else PRICE_BATCH_SIZE_DEFAULT
        )
        env_path = Path(config.get("env_path", ENV_PATH_DEFAULT))
        api_key_env = config.get("jupiter_api_key_env", "JUPITER_API_KEY")
        tz_offset_min = int(
            config.get("tz_offset_minutes", TZ_OFFSET_MINUTES_DEFAULT)
            if isinstance(config, dict)
            else TZ_OFFSET_MINUTES_DEFAULT
        )

        if not wallets:
            print("No wallets configured.")
            return
        if not rpc_http:
            print("Missing rpc_http in config.")
            return
        if not log_path.exists():
            print(f"Trade log not found: {log_path}")
            return

        load_env_file(env_path)
        headers = config.get("jupiter_headers") or {}
        if not isinstance(headers, dict):
            headers = {}
        env_api_key = os.getenv(api_key_env) if api_key_env else None
        if env_api_key and not headers.get("x-api-key"):
            headers["x-api-key"] = env_api_key

        entries = load_trade_log_entries(log_path)
        logged_buy_keys = set()
        for entry in entries:
            if entry.get("side") == "BUY":
                sig = entry.get("signature")
                mint = entry.get("mint")
                wallet = entry.get("wallet")
                if sig and mint and wallet:
                    logged_buy_keys.add((wallet, sig, mint))

        async def run_backfill_buys():
            rpc = RpcClient(rpc_http)
            token_cache = JupiterTokenCache(token_list_url, refresh_sec, headers)
            new_entries = []
            for wallet in wallets:
                batch_entries = await collect_missing_buys_for_wallet(
                    rpc=rpc,
                    token_cache=token_cache,
                    wallet=wallet,
                    commitment=commitment,
                    since=since,
                    limit=limit,
                    logged_buy_keys=logged_buy_keys,
                )
                if batch_entries:
                    new_entries.extend(batch_entries)
            await token_cache.close()
            await rpc.close()

            if not new_entries:
                print("No missing buy entries found.")
                return

            new_entries.sort(key=log_entry_ts)
            print(f"Backfill candidates: {len(new_entries)}")
            for entry in new_entries:
                ts_str = format_ts(entry.get("ts"), tz_offset_min)
                print(
                    f"- {entry.get('wallet')} | {entry.get('symbol')} | "
                    f"{entry.get('mint')} | {entry.get('sol_used')} SOL | {ts_str}"
                )

            if dry_run:
                return

            entries.extend(new_entries)
            entries.sort(key=log_entry_ts)
            if not write_trade_log_entries(log_path, entries):
                print("Trade log not updated (write failed).")
                return
            print(f"Trade log updated: {log_path}")

            positions, _, meta = build_trade_log_index(entries)
            watch = load_price_watch(watch_path, set())
            missing_keys = [key for key in positions.keys() if key not in watch]
            if not missing_keys:
                return
            price_client = JupiterPriceClient(
                price_url, headers, batch_size=price_batch_size
            )
            mints = sorted({mint for (_, mint) in missing_keys})
            prices = await price_client.fetch_prices(mints)
            await price_client.close()
            for (wallet, mint) in missing_keys:
                price = prices.get(mint)
                if price is None:
                    continue
                buys = positions.get((wallet, mint), {}).get("buys") or deque()
                if not buys:
                    continue
                last_buy = buys[-1]
                buy_sol_used = parse_decimal(last_buy.get("sol_used"))
                meta_slot = meta.get((wallet, mint), {})
                watch[(wallet, mint)] = {
                    "buy_price": price,
                    "trailing_active": False,
                    "trailing_peak": None,
                    "trailing_below_since": None,
                    "buy_sol_used": buy_sol_used,
                    "name": meta_slot.get("name") or "unknown",
                    "symbol": meta_slot.get("symbol") or "unknown",
                    "marketcap": parse_decimal(meta_slot.get("marketcap")),
                }
            save_price_watch(watch_path, watch)
            print("Price watch updated from backfill (current prices).")

        asyncio.run(run_backfill_buys())
        return

    if len(sys.argv) > 1 and sys.argv[1] == "rebuild-trade-log":
        args = sys.argv[2:]
        config_path = Path("config.json")
        wallet_arg = None
        since = None
        limit = 1000
        dry_run = False
        rpc_override = None
        rebuild_watch = True
        i = 0
        while i < len(args):
            arg = args[i]
            if arg == "--wallet" and i + 1 < len(args):
                wallet_arg = args[i + 1]
                i += 2
                continue
            if arg == "--since" and i + 1 < len(args):
                duration = parse_duration(args[i + 1])
                if duration is None:
                    since = parse_datetime_value(args[i + 1], TZ_OFFSET_MINUTES_DEFAULT)
                else:
                    since = int(time.time()) - duration
                i += 2
                continue
            if arg == "--limit" and i + 1 < len(args):
                limit = int(args[i + 1])
                i += 2
                continue
            if arg == "--config" and i + 1 < len(args):
                config_path = Path(args[i + 1])
                i += 2
                continue
            if arg in ("--rpc", "--rpc-http") and i + 1 < len(args):
                rpc_override = args[i + 1]
                i += 2
                continue
            if arg == "--no-watch":
                rebuild_watch = False
                i += 1
                continue
            if arg == "--dry-run":
                dry_run = True
                i += 1
                continue
            i += 1

        config = {}
        if config_path.exists():
            try:
                config = load_config(config_path)
            except Exception:
                config = {}
        wallet_aliases = config.get("wallet_aliases") or {}
        if not isinstance(wallet_aliases, dict):
            wallet_aliases = {}
        wallets = normalize_wallets(config, wallet_aliases)
        if wallet_arg:
            wallets = [wallet_aliases.get(wallet_arg, wallet_arg)]
        rpc_http = rpc_override or config.get("rpc_http_backfill") or config.get("rpc_http")
        commitment = resolve_commitment(config.get("commitment"))
        token_list_url = config.get("jupiter_tokens_url", JUPITER_TOKENS_URL_DEFAULT)
        refresh_sec = int(config.get("jupiter_refresh_sec", JUPITER_REFRESH_SEC_DEFAULT))
        log_path = Path(
            config.get("trade_log_path", TRADE_LOG_PATH_DEFAULT)
            if isinstance(config, dict)
            else TRADE_LOG_PATH_DEFAULT
        )
        watch_path = Path(
            config.get("price_watch_path", PRICE_WATCH_PATH_DEFAULT)
            if isinstance(config, dict)
            else PRICE_WATCH_PATH_DEFAULT
        )
        price_url = config.get("price_url", PRICE_URL_DEFAULT)
        price_batch_size = int(
            config.get("price_batch_size", PRICE_BATCH_SIZE_DEFAULT)
            if isinstance(config, dict)
            else PRICE_BATCH_SIZE_DEFAULT
        )
        env_path = Path(config.get("env_path", ENV_PATH_DEFAULT))
        api_key_env = config.get("jupiter_api_key_env", "JUPITER_API_KEY")

        if not wallets:
            print("No wallets configured.")
            return
        if not rpc_http:
            print("Missing rpc_http in config.")
            return

        load_env_file(env_path)
        headers = config.get("jupiter_headers") or {}
        if not isinstance(headers, dict):
            headers = {}
        env_api_key = os.getenv(api_key_env) if api_key_env else None
        if env_api_key and not headers.get("x-api-key"):
            headers["x-api-key"] = env_api_key

        async def run_rebuild():
            rpc = RpcClient(rpc_http)
            token_cache = JupiterTokenCache(token_list_url, refresh_sec, headers)
            new_entries = await rebuild_trade_log_from_chain(
                rpc=rpc,
                token_cache=token_cache,
                wallets=wallets,
                commitment=commitment,
                since=since,
                limit=limit,
            )
            await token_cache.close()
            await rpc.close()

            if not new_entries:
                print("No trades found for rebuild.")
                return

            new_entries.sort(key=log_entry_ts)
            print(f"Rebuild candidates: {len(new_entries)}")
            if dry_run:
                return

            if not write_trade_log_entries(log_path, new_entries):
                print("Trade log not updated (write failed).")
                return
            print(f"Trade log rebuilt: {log_path}")

            if not rebuild_watch:
                return

            positions, _, meta = build_trade_log_index(new_entries)
            price_client = JupiterPriceClient(
                price_url, headers, batch_size=price_batch_size
            )
            mints = sorted({mint for (_, mint) in positions.keys()})
            prices = await price_client.fetch_prices(mints)
            await price_client.close()
            watch = {}
            for (wallet, mint), data in positions.items():
                price = prices.get(mint)
                if price is None:
                    continue
                buys = data.get("buys") or deque()
                if not buys:
                    continue
                last_buy = buys[-1]
                buy_sol_used = parse_decimal(last_buy.get("sol_used"))
                meta_slot = meta.get((wallet, mint), {})
                watch[(wallet, mint)] = {
                    "buy_price": price,
                    "trailing_active": False,
                    "trailing_peak": None,
                    "trailing_below_since": None,
                    "buy_sol_used": buy_sol_used,
                    "name": meta_slot.get("name") or "unknown",
                    "symbol": meta_slot.get("symbol") or "unknown",
                    "marketcap": parse_decimal(meta_slot.get("marketcap")),
                }
            save_price_watch(watch_path, watch)
            print("Price watch rebuilt from open positions (current prices).")

        asyncio.run(run_rebuild())
        return

    if len(sys.argv) > 1 and sys.argv[1] == "webhook-test":
        args = sys.argv[2:]
        config_path = Path("config.json")
        message = "SolBot webhook test"
        i = 0
        while i < len(args):
            arg = args[i]
            if arg == "--config" and i + 1 < len(args):
                config_path = Path(args[i + 1])
                i += 2
                continue
            if arg == "--message" and i + 1 < len(args):
                message = args[i + 1]
                i += 2
                continue
            i += 1

        config = {}
        if config_path.exists():
            try:
                config = load_config(config_path)
            except Exception:
                config = {}

        env_path = Path(config.get("env_path", ENV_PATH_DEFAULT))
        load_env_file(env_path)
        webhook_url = config.get("discord_webhook_url")
        webhook_env = config.get("discord_webhook_env", "DISCORD_WEBHOOK_URL")
        if not webhook_url and webhook_env:
            webhook_url = os.getenv(webhook_env)
        if not webhook_url:
            print("No webhook URL configured.")
            return

        async def run_webhook_test():
            client = httpx.AsyncClient(timeout=10)
            await send_discord_webhook(client, webhook_url, message)
            await client.aclose()
            print("Webhook test sent.")

        asyncio.run(run_webhook_test())
        return

    if len(sys.argv) > 1 and sys.argv[1] == "backfill-open":
        args = sys.argv[2:]
        config_path = Path("config.json")
        limit = 200
        since = None
        dry_run = False
        rpc_override = None
        i = 0
        while i < len(args):
            arg = args[i]
            if arg == "--limit" and i + 1 < len(args):
                limit = int(args[i + 1])
                i += 2
                continue
            if arg == "--since" and i + 1 < len(args):
                duration = parse_duration(args[i + 1])
                if duration is None:
                    since = parse_datetime_value(args[i + 1], TZ_OFFSET_MINUTES_DEFAULT)
                else:
                    since = int(time.time()) - duration
                i += 2
                continue
            if arg == "--config" and i + 1 < len(args):
                config_path = Path(args[i + 1])
                i += 2
                continue
            if arg in ("--rpc", "--rpc-http") and i + 1 < len(args):
                rpc_override = args[i + 1]
                i += 2
                continue
            if arg == "--dry-run":
                dry_run = True
                i += 1
                continue
            i += 1

        config = {}
        if config_path.exists():
            try:
                config = load_config(config_path)
            except Exception:
                config = {}
        rpc_http = rpc_override or config.get("rpc_http_backfill") or config.get("rpc_http")
        commitment = resolve_commitment(config.get("commitment"))
        log_path = Path(
            config.get("trade_log_path", TRADE_LOG_PATH_DEFAULT)
            if isinstance(config, dict)
            else TRADE_LOG_PATH_DEFAULT
        )
        tz_offset_min = int(
            config.get("tz_offset_minutes", TZ_OFFSET_MINUTES_DEFAULT)
            if isinstance(config, dict)
            else TZ_OFFSET_MINUTES_DEFAULT
        )
        if not rpc_http:
            print("Missing rpc_http in config.")
            return
        if not log_path.exists():
            print(f"Trade log not found: {log_path}")
            return

        entries = load_trade_log_entries(log_path)
        positions, sell_sigs, meta = build_trade_log_index(entries)
        if not positions:
            print("No open positions found in trade log.")
            return

        async def run_backfill_open():
            rpc = RpcClient(rpc_http)
            new_entries = []
            for (wallet, mint), data in positions.items():
                last_buy_ts = None
                for buy in data.get("buys", []):
                    ts_val = buy.get("ts")
                    if ts_val:
                        last_buy_ts = max(last_buy_ts or 0, int(ts_val))
                since_for_key = last_buy_ts
                if since is not None:
                    since_for_key = max(since or 0, since_for_key or 0) or since_for_key
                meta_slot = meta.get((wallet, mint), {})
                batch_entries = await collect_missing_sell_entries(
                    rpc=rpc,
                    wallet=wallet,
                    mint=mint,
                    commitment=commitment,
                    since=since_for_key,
                    limit=limit,
                    logged_sell_sigs=sell_sigs.get((wallet, mint), set()),
                    meta_name=meta_slot.get("name"),
                    meta_symbol=meta_slot.get("symbol"),
                    meta_mcap=meta_slot.get("marketcap"),
                )
                if batch_entries:
                    new_entries.extend(batch_entries)
            await rpc.close()

            if not new_entries:
                print("No missing sell entries found.")
                return

            new_entries.sort(key=log_entry_ts)
            print(f"Backfill candidates: {len(new_entries)}")
            for entry in new_entries:
                ts_str = format_ts(entry.get("ts"), tz_offset_min)
                print(
                    f"- {entry.get('wallet')} | {entry.get('mint')} | "
                    f"{entry.get('sol_received')} SOL | {ts_str}"
                )

            if dry_run:
                return

            entries.extend(new_entries)
            entries.sort(key=log_entry_ts)
            if not write_trade_log_entries(log_path, entries):
                print("Trade log not updated (write failed).")
                return
            print(f"Trade log updated: {log_path}")

        asyncio.run(run_backfill_open())
        return

    if len(sys.argv) > 1 and sys.argv[1] == "positions":
        args = sys.argv[2:]
        config_path = Path("config.json")
        log_path = None
        watch_path = None
        tz_offset_min = TZ_OFFSET_MINUTES_DEFAULT
        i = 0
        while i < len(args):
            arg = args[i]
            if arg == "--config" and i + 1 < len(args):
                config_path = Path(args[i + 1])
                i += 2
                continue
            if arg in ("--log", "--log-path") and i + 1 < len(args):
                log_path = Path(args[i + 1])
                i += 2
                continue
            if arg in ("--watch", "--watch-path") and i + 1 < len(args):
                watch_path = Path(args[i + 1])
                i += 2
                continue
            if arg == "--tz" and i + 1 < len(args):
                tz_offset_min = int(args[i + 1])
                i += 2
                continue
            i += 1

        config = {}
        if config_path.exists():
            try:
                config = load_config(config_path)
            except Exception:
                config = {}
        if log_path is None:
            log_path = Path(
                config.get("trade_log_path", TRADE_LOG_PATH_DEFAULT)
                if isinstance(config, dict)
                else TRADE_LOG_PATH_DEFAULT
            )
        if watch_path is None:
            watch_path = Path(
                config.get("price_watch_path", PRICE_WATCH_PATH_DEFAULT)
                if isinstance(config, dict)
                else PRICE_WATCH_PATH_DEFAULT
            )
        if tz_offset_min == TZ_OFFSET_MINUTES_DEFAULT and isinstance(config, dict):
            tz_offset_min = int(
                config.get("tz_offset_minutes", TZ_OFFSET_MINUTES_DEFAULT)
            )

        if not log_path.exists():
            print(f"Trade log not found: {log_path}")
            return
        print_open_positions(log_path, tz_offset_min, watch_path)
        return

    config_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("config.json")
    try:
        config = load_config(config_path)
    except Exception as exc:
        print(f"Failed to load config: {exc}")
        sys.exit(1)

    wallet_aliases = config.get("wallet_aliases") or {}
    if not isinstance(wallet_aliases, dict):
        wallet_aliases = {}
    wallets = normalize_wallets(config, wallet_aliases)
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
