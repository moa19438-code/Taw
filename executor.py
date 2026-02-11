from __future__ import annotations
from typing import Dict, Any, Tuple, List
import json
from datetime import datetime, timezone, timedelta

from config import (
    AUTO_TRADE, EXECUTE_TRADES, ALLOW_LIVE_TRADING,
    RISK_PER_TRADE_PCT, MAX_DAILY_TRADES, TP_R_MULT, SL_ATR_MULT,
    SKIP_OPEN_MINUTES, SKIP_CLOSE_MINUTES,
    USE_MARKET_FILTER, MARKET_SYMBOL, MARKET_SMA_FAST, MARKET_SMA_SLOW,
    BLOCK_IF_POSITION_OPEN, BLOCK_IF_ORDER_OPEN
)
from alpaca_client import account, place_bracket_order, clock, positions, open_orders, bars
from indicators import sma
from storage import log_order, last_orders

def _today_utc() -> str:
    return datetime.now(timezone.utc).date().isoformat()

def _count_today_orders() -> int:
    orders = last_orders(500)
    today = _today_utc()
    return sum(1 for o in orders if o.get("ts","").startswith(today))

def _parse_ts(ts: str) -> datetime:
    # Alpaca returns RFC3339 like 2026-02-11T13:30:00Z
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))

def _within_time_window() -> Tuple[bool, str]:
    c = clock()
    if not c.get("is_open"):
        return False, "Market closed"
    ts = _parse_ts(c["timestamp"])
    nxt_close = _parse_ts(c["next_close"])
    nxt_open  = _parse_ts(c["next_open"])  # not used but for sanity
    # When market open, we can infer today's open by using next_open of today? Alpaca doesn't always give last_open.
    # Approx: use next_close date and assume open at 13:30 UTC (NYSE) can be wrong on DST; so instead,
    # we skip using inferred open and only skip close window reliably via next_close - minutes.
    # For open window, we rely on clock "timestamp" and an approximate 1st bar time using bars API on MARKET_SYMBOL.
    # We'll compute open time from the first 1Min bar of the day for MARKET_SYMBOL.
    try:
        # get today's 1Min bars for MARKET_SYMBOL and take first timestamp
        end = ts
        start = ts.replace(hour=0, minute=0, second=0, microsecond=0)
        data = bars([MARKET_SYMBOL], start=start, end=end, timeframe="1Min", limit=2000)
        blist = (data.get("bars", {}) or {}).get(MARKET_SYMBOL, [])
        if blist:
            first_bar_ts = _parse_ts(blist[0]["t"])
            minutes_from_open = (ts - first_bar_ts).total_seconds() / 60.0
            if minutes_from_open < SKIP_OPEN_MINUTES:
                return False, f"Skipping first {SKIP_OPEN_MINUTES}m after open"
    except Exception:
        # If we can't compute open accurately, don't block.
        pass

    minutes_to_close = (nxt_close - ts).total_seconds() / 60.0
    if minutes_to_close < SKIP_CLOSE_MINUTES:
        return False, f"Skipping last {SKIP_CLOSE_MINUTES}m before close"
    return True, "OK"

def _market_ok() -> Tuple[bool, str]:
    if not USE_MARKET_FILTER:
        return True, "Market filter off"
    try:
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=400)
    except Exception:
        end = datetime.now(timezone.utc)
        start = end
    try:
        data = bars([MARKET_SYMBOL], start=start, end=end, timeframe="1Day", limit=260)
        blist = (data.get("bars", {}) or {}).get(MARKET_SYMBOL, [])
        closes = [float(b["c"]) for b in blist if "c" in b]
        if len(closes) < max(MARKET_SMA_FAST, MARKET_SMA_SLOW) + 5:
            return True, "Market filter: insufficient data"
        fast = sma(closes, MARKET_SMA_FAST)
        slow = sma(closes, MARKET_SMA_SLOW)
        last = closes[-1]
        if fast is None or slow is None:
            return True, "Market filter: insufficient SMA"
        if last > slow and fast > slow:
            return True, f"{MARKET_SYMBOL} bullish (close>{MARKET_SMA_SLOW}SMA & {MARKET_SMA_FAST}>{MARKET_SMA_SLOW})"
        return False, f"{MARKET_SYMBOL} not bullish (filter blocks longs)"
    except Exception as e:
        return True, f"Market filter error (ignored): {e}"

def _has_position(symbol: str) -> bool:
    try:
        if not BLOCK_IF_POSITION_OPEN:
            return False
        pos = positions()
        for p in pos or []:
            if str(p.get("symbol","")).upper() == symbol.upper():
                qty = float(p.get("qty", 0) or 0)
                if abs(qty) > 0:
                    return True
        return False
    except Exception:
        return False

def _has_open_order(symbol: str) -> bool:
    try:
        if not BLOCK_IF_ORDER_OPEN:
            return False
        od = open_orders(status="open", limit=500)
        for o in od or []:
            if str(o.get("symbol","")).upper() == symbol.upper():
                return True
        return False
    except Exception:
        return False

def compute_qty(equity: float, last_price: float, atr: float) -> float:
    risk_amt = equity * (RISK_PER_TRADE_PCT / 100.0)
    stop_dist = max(atr * SL_ATR_MULT, 0.01)
    qty = risk_amt / stop_dist
    max_notional = equity * 0.20
    qty = min(qty, max_notional / max(last_price, 0.01))
    return max(0.0, float(int(qty)))

def maybe_trade(picks: List[Dict[str, Any]]) -> List[str]:
    logs: List[str] = []
    if not AUTO_TRADE:
        return ["AUTO_TRADE=false (scan only)"]

    if not (EXECUTE_TRADES and ALLOW_LIVE_TRADING):
        return ["Trade safety latches are OFF (EXECUTE_TRADES & ALLOW_LIVE_TRADING must be true)"]

    # Market open + time filters
    try:
        ok, reason = _within_time_window()
        if not ok:
            return [f"Cannot trade now: {reason}"]
    except Exception as e:
        return [f"Cannot trade now: clock error ({e})"]

    # Market regime filter
    ok_mkt, mkt_reason = _market_ok()
    if not ok_mkt:
        return [f"Blocked by market filter: {mkt_reason}"]
    logs.append(mkt_reason)

    if _count_today_orders() >= MAX_DAILY_TRADES:
        return [f"Daily limit reached ({MAX_DAILY_TRADES})"]

    acct = account()
    equity = float(acct.get("equity", 0.0))

    for p in picks:
        if _count_today_orders() >= MAX_DAILY_TRADES:
            logs.append("Stopped: daily limit reached")
            break

        sym = p["symbol"]
        last_price = float(p["last_close"])
        atr = float(p["atr"])

        if _has_position(sym):
            logs.append(f"{sym}: skip (position already open)")
            continue
        if _has_open_order(sym):
            logs.append(f"{sym}: skip (open order exists)")
            continue

        qty = compute_qty(equity, last_price, atr)
        if qty <= 0:
            logs.append(f"{sym}: qty=0 (skip)")
            continue

        side = "buy"  # Long-only
        stop_price = max(last_price - (atr * SL_ATR_MULT), 0.01)
        r = last_price - stop_price
        take_profit = last_price + (r * TP_R_MULT)

        payload = {
            "symbol": sym,
            "side": side,
            "qty": qty,
            "take_profit": take_profit,
            "stop_loss": stop_price,
        }

        try:
            resp = place_bracket_order(sym, side, qty, take_profit, stop_price)
            oid = resp.get("id", "")
            log_order(sym, side, qty, "bracket", json.dumps(payload), oid, "ok", "submitted")
            logs.append(f"{sym}: order submitted qty={qty} TP={take_profit:.2f} SL={stop_price:.2f}")
        except Exception as e:
            log_order(sym, side, qty, "bracket", json.dumps(payload), "", "error", str(e))
            logs.append(f"{sym}: order failed ({e})")

    return logs
