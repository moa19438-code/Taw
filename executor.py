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
from indicators import sma, atr
from storage import log_order, last_orders, get_all_settings, parse_bool, parse_int, parse_float


def _today_utc() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def _count_today_orders() -> int:
    orders = last_orders(500)
    today = _today_utc()
    return sum(1 for o in orders if o.get("ts", "").startswith(today))


def _parse_ts(ts: str) -> datetime:
    # Alpaca returns RFC3339 like 2026-02-11T13:30:00Z
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


def _within_time_window() -> Tuple[bool, str]:
    c = clock()
    if not c.get("is_open"):
        return False, "Market closed"

    ts = _parse_ts(c["timestamp"])
    nxt_close = _parse_ts(c["next_close"])
    _nxt_open = _parse_ts(c["next_open"])  # not used but for sanity

    # Try to compute first bar time for MARKET_SYMBOL to skip early minutes after open.
    try:
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

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=400)

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
            if str(p.get("symbol", "")).upper() == symbol.upper():
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
            if str(o.get("symbol", "")).upper() == symbol.upper():
                return True
        return False
    except Exception:
        return False


def compute_qty(equity: float, last_price: float, atr: float, risk_pct: float, sl_atr_mult: float) -> float:
    """
    Compute position size from equity + ATR stop distance.
    risk_pct is in percent (e.g., 0.25 means 0.25% of equity).
    """
    risk_amt = equity * (risk_pct / 100.0)
    stop_dist = max(atr * sl_atr_mult, 0.01)
    qty = risk_amt / stop_dist

    # Cap notional exposure to 20% of equity
    max_notional = equity * 0.20
    qty = min(qty, max_notional / max(last_price, 0.01))

    # Use whole shares
    return max(0.0, float(int(qty)))


def maybe_trade(picks: List[Dict[str, Any]]) -> List[str]:
    logs: List[str] = []
    settings = get_all_settings()

    auto_trade = parse_bool(settings.get("AUTO_TRADE"), AUTO_TRADE)
    max_daily = parse_int(settings.get("MAX_DAILY_TRADES"), MAX_DAILY_TRADES)

    # Fix: provide defaults (no UnboundLocalError)
    risk_pct = parse_float(settings.get("risk_pct"), RISK_PER_TRADE_PCT)
    tp_r = parse_float(settings.get("tp_r"), TP_R_MULT)
    sl_atr_mult = parse_float(settings.get("sl_atr_mult"), SL_ATR_MULT)

    if not auto_trade:
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

    if _count_today_orders() >= max_daily:
        return [f"Daily limit reached ({max_daily})"]

    acct = account()
    equity = float(acct.get("equity", 0.0))

    for p in picks:
        if _count_today_orders() >= max_daily:
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

        qty = compute_qty(equity, last_price, atr, risk_pct=risk_pct, sl_atr_mult=sl_atr_mult)
        if qty <= 0:
            logs.append(f"{sym}: qty=0 (skip)")
            continue

        side = "buy"  # Long-only
        stop_price = max(last_price - (atr * sl_atr_mult), 0.01)
        r = last_price - stop_price
        take_profit = last_price + (r * tp_r)

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



def trade_symbol(symbol: str, *, side: str = "buy", risk_pct: float | None = None, tp_r: float | None = None, sl_atr_mult: float | None = None) -> List[str]:
    """Trade a single symbol (long-only by default).

    Intended for external signals (e.g., TradingView alerts) where you want
    to execute one symbol on-demand using the same risk settings + safety latches.

    Returns logs (human readable).
    """
    logs: List[str] = []
    settings = get_all_settings()

    auto_trade = parse_bool(settings.get("AUTO_TRADE"), AUTO_TRADE)
    max_daily = parse_int(settings.get("MAX_DAILY_TRADES"), MAX_DAILY_TRADES)

    eff_risk = risk_pct if risk_pct is not None else parse_float(settings.get("risk_pct"), RISK_PER_TRADE_PCT)
    eff_tp_r = tp_r if tp_r is not None else parse_float(settings.get("tp_r"), TP_R_MULT)
    eff_sl_atr = sl_atr_mult if sl_atr_mult is not None else parse_float(settings.get("sl_atr_mult"), SL_ATR_MULT)

    if not auto_trade:
        return ["AUTO_TRADE=false (execution disabled)"]

    if not (EXECUTE_TRADES and ALLOW_LIVE_TRADING):
        return ["Trade safety latches are OFF (EXECUTE_TRADES & ALLOW_LIVE_TRADING must be true)"]

    # Market open + time filters
    try:
        ok, reason = _within_time_window()
        if not ok:
            return [f"Cannot trade now: {reason}"]
    except Exception as e:
        return [f"Cannot trade now: clock error ({e})"]

    # Market regime filter (optional)
    ok_mkt, mkt_reason = _market_ok()
    if not ok_mkt:
        return [f"Blocked by market filter: {mkt_reason}"]
    logs.append(mkt_reason)

    if _count_today_orders() >= max_daily:
        return [f"Daily limit reached ({max_daily})"]

    sym = (symbol or "").upper().strip()
    if not sym:
        return ["Empty symbol"]

    if _has_position(sym):
        return [f"{sym}: skip (position already open)"]
    if _has_open_order(sym):
        return [f"{sym}: skip (open order exists)"]

    # Fetch recent daily bars to compute last close + ATR
    try:
        now = datetime.now(timezone.utc)
        start = now - timedelta(days=180)
        data = bars([sym], start=start, end=now, timeframe="1Day", limit=120)
        blist = (data.get("bars", {}) or {}).get(sym, [])
        if not blist or len(blist) < 30:
            return [f"{sym}: not enough bars to compute ATR"]
        closes = [float(b.get("c", 0)) for b in blist if "c" in b]
        highs  = [float(b.get("h", 0)) for b in blist if "h" in b]
        lows   = [float(b.get("l", 0)) for b in blist if "l" in b]
        if len(closes) < 30 or len(highs) != len(lows) or len(lows) != len(closes):
            return [f"{sym}: invalid bars data"]
        last_price = closes[-1]
        a14 = atr(highs, lows, closes, 14)
        if a14 is None or a14 <= 0:
            return [f"{sym}: ATR unavailable"]
    except Exception as e:
        return [f"{sym}: bars fetch failed ({e})"]

    acct = account()
    equity = float(acct.get("equity", 0.0))

    qty = compute_qty(equity, last_price, a14, risk_pct=eff_risk, sl_atr_mult=eff_sl_atr)
    if qty <= 0:
        return [f"{sym}: qty=0 (equity={equity:.2f} price={last_price:.2f})"]

    side = (side or "buy").lower().strip()
    if side not in ("buy", "sell"):
        side = "buy"

    if side != "buy":
        return [f"{sym}: only long trades are supported حاليا (side={side})"]

    stop_price = max(last_price - (a14 * eff_sl_atr), 0.01)
    r = last_price - stop_price
    take_profit = last_price + (r * eff_tp_r)

    payload = {
        "symbol": sym,
        "side": side,
        "qty": qty,
        "take_profit": take_profit,
        "stop_loss": stop_price,
        "risk_pct": eff_risk,
        "tp_r": eff_tp_r,
        "sl_atr_mult": eff_sl_atr,
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
