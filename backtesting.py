
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from alpaca_client import bars
from indicators import ema, rsi, atr, macd

@dataclass
class Trade:
    symbol: str
    entry_ts: str
    exit_ts: str
    entry: float
    exit: float
    qty: int
    side: str
    pnl: float
    r_mult: float
    outcome: str  # "tp"|"sl"|"time"|"exit"

def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default

def _position_size(equity: float, risk_pct: float, entry: float, sl: float) -> int:
    # risk_pct is percent of equity, e.g. 1.0
    risk_amount = max(0.0, equity * (risk_pct / 100.0))
    risk_per_share = max(entry - sl, 1e-6)
    qty = int(risk_amount / risk_per_share) if risk_per_share > 0 else 0
    return max(qty, 0)

def run_backtest_symbol(
    symbol: str,
    start: datetime,
    end: datetime,
    *,
    capital: float = 10000.0,
    risk_per_trade_pct: float = 1.0,
    sl_atr_mult: float = 1.5,
    tp_r_mult: float = 1.8,
    max_holding_days: int = 7,
    cooldown_days: int = 2,
) -> Dict[str, Any]:
    """
    Simple, transparent daily-bar backtest:
      Entry condition (long):
        - EMA20 > EMA50
        - Close > EMA200
        - RSI14 between 50..70
        - MACD hist > 0
      Entry at next day's open (approximated by next day's close if open not available).
      Stop = entry - ATR14 * sl_atr_mult
      Take profit = entry + (entry - stop) * tp_r_mult
      Exit on TP/SL, or time-based exit at max_holding_days.
    """
    symbol = (symbol or "").upper().strip()
    if not symbol:
        return {"error": "empty_symbol"}

    # pull enough lookback to compute EMA200
    pull_start = start - timedelta(days=365)
    data = bars([symbol], start=pull_start, end=end, timeframe="1Day", limit=1000)
    bmap = data.get("bars", {}) if isinstance(data, dict) else {}
    blist = bmap.get(symbol) or []
    if len(blist) < 260:
        return {"error": "not_enough_bars", "bars": len(blist)}

    # Build series
    ts = [str(b.get("t")) for b in blist]
    closes = [_safe_float(b.get("c")) for b in blist]
    highs  = [_safe_float(b.get("h")) for b in blist]
    lows   = [_safe_float(b.get("l")) for b in blist]

    # Map to index range within [start,end]
    def _to_dt(s: str) -> Optional[datetime]:
        try:
            # Alpaca bar time is ISO; can end with Z
            ss = s.replace("Z", "+00:00")
            return datetime.fromisoformat(ss)
        except Exception:
            return None

    dts = [_to_dt(x) for x in ts]
    idxs = [i for i, d in enumerate(dts) if d and d >= start and d <= end]
    if len(idxs) < 40:
        return {"error": "range_too_small", "in_range": len(idxs)}

    equity = float(capital)
    peak = float(capital)
    max_dd = 0.0

    trades: List[Trade] = []
    in_pos = False
    entry_i = -1
    entry_price = 0.0
    sl = 0.0
    tp = 0.0
    qty = 0
    cooldown_until_i = -1

    # Helper: drawdown tracking
    def _update_dd(eq: float) -> None:
        nonlocal peak, max_dd
        if eq > peak:
            peak = eq
        dd = (peak - eq) / peak if peak > 0 else 0.0
        if dd > max_dd:
            max_dd = dd

    for i in idxs:
        if i < 210:
            continue  # need EMA200/ATR history

        if i <= cooldown_until_i:
            continue

        # compute indicators up to i (inclusive)
        c_slice = closes[: i + 1]
        h_slice = highs[: i + 1]
        l_slice = lows[: i + 1]

        e20 = ema(c_slice, 20)
        e50 = ema(c_slice, 50)
        e200 = ema(c_slice, 200)
        r14 = rsi(c_slice, 14)
        a14 = atr(h_slice, l_slice, c_slice, 14)
        m = macd(c_slice, 12, 26, 9)

        if None in (e20, e50, e200, r14, a14, m):
            continue

        _, _, hist = m
        close = c_slice[-1]

        # Manage open position first
        if in_pos:
            hold_days = i - entry_i
            lo = lows[i]
            hi = highs[i]

            exit_reason = None
            exit_price = None

            # Worst-first: stop then tp within same day (conservative for longs)
            if lo <= sl:
                exit_reason = "sl"
                exit_price = sl
            elif hi >= tp:
                exit_reason = "tp"
                exit_price = tp
            elif hold_days >= max_holding_days:
                exit_reason = "time"
                exit_price = close

            if exit_reason and exit_price is not None:
                pnl = (exit_price - entry_price) * qty
                r = (exit_price - entry_price) / max(entry_price - sl, 1e-6)
                equity += pnl
                _update_dd(equity)
                trades.append(Trade(
                    symbol=symbol,
                    entry_ts=ts[entry_i],
                    exit_ts=ts[i],
                    entry=round(entry_price, 4),
                    exit=round(exit_price, 4),
                    qty=int(qty),
                    side="buy",
                    pnl=round(pnl, 2),
                    r_mult=round(r, 3),
                    outcome=exit_reason,
                ))
                in_pos = False
                entry_i = -1
                qty = 0
                cooldown_until_i = i + max(0, cooldown_days)
            continue

        # Entry signal
        cond = (e20 > e50) and (close > e200) and (50.0 <= float(r14) <= 70.0) and (float(hist) > 0)
        if not cond:
            continue

        # Entry next bar (approx): use current close as entry to keep deterministic
        entry_price = float(close)

        # Stops/TP
        atr_val = float(a14) if float(a14) > 0 else max(entry_price * 0.01, 0.5)
        sl = max(0.01, entry_price - (atr_val * float(sl_atr_mult)))
        risk_per_share = max(entry_price - sl, 1e-6)
        tp = entry_price + (risk_per_share * float(tp_r_mult))

        qty = _position_size(equity, float(risk_per_trade_pct), entry_price, sl)
        if qty < 1:
            continue

        in_pos = True
        entry_i = i

    # compute stats
    wins = sum(1 for t in trades if t.pnl > 0)
    losses = sum(1 for t in trades if t.pnl <= 0)
    winrate = (wins / len(trades)) if trades else 0.0
    pnl_total = sum(t.pnl for t in trades)
    avg_r = (sum(t.r_mult for t in trades) / len(trades)) if trades else 0.0

    return {
        "symbol": symbol,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "capital_start": float(capital),
        "capital_end": round(equity, 2),
        "net_pnl": round(pnl_total, 2),
        "trades": [t.__dict__ for t in trades],
        "stats": {
            "trades": len(trades),
            "wins": wins,
            "losses": losses,
            "winrate": round(winrate, 3),
            "avg_r": round(avg_r, 3),
            "max_drawdown": round(max_dd, 4),
        },
        "params": {
            "risk_per_trade_pct": float(risk_per_trade_pct),
            "sl_atr_mult": float(sl_atr_mult),
            "tp_r_mult": float(tp_r_mult),
            "max_holding_days": int(max_holding_days),
            "cooldown_days": int(cooldown_days),
        }
    }
