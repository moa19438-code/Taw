from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from core.alpaca_client import bars
from core.indicators import ema, rsi, atr, macd


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
    # Optional diagnostics (non-breaking for existing consumers)
    tp1_hit: bool = False
    tp1_price: float | None = None
    tp2_price: float | None = None
    trail_used: bool = False
    partial_pct: float | None = None


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _position_size(equity: float, risk_pct: float, entry: float, sl: float) -> int:
    """Position sizing by fixed % equity risk."""
    risk_amount = max(0.0, equity * (risk_pct / 100.0))
    risk_per_share = max(abs(entry - sl), 1e-6)
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
    tp2_r_mult: float = 1.8,
    # Win-rate focused exits (1D): partial take profit + trailing stop
    tp1_r_mult: float = 1.0,
    partial_pct: float = 0.5,
    trail_atr_mult: float = 1.2,
    trail_after_tp1: bool = True,
    move_sl_to_be_after_tp1: bool = True,
    max_holding_days: int = 12,
    cooldown_days: int = 2,
) -> Dict[str, Any]:
    """Daily-bar backtest with practical 1D exits.

    Entry (LONG) baseline:
      - EMA20 > EMA50
      - Close > EMA200
      - RSI14 between 50..70
      - MACD hist > 0

    Exits (LONG):
      - SL = entry - ATR14 * sl_atr_mult
      - TP1 = entry + R * tp1_r_mult (optional partial)
      - TP2 = entry + R * tp2_r_mult (final target)
      - Optional trailing stop (ATR-based), typically activated after TP1.

    Notes:
      - Uses OHLC daily bars; intra-day order is unknown, so we use a conservative ordering:
        Stop-loss triggers before take-profit if both touched in same bar.
    """
    symbol = (symbol or "").upper().strip()
    if not symbol:
        return {"error": "empty_symbol"}

    pull_start = start - timedelta(days=365)
    data = bars([symbol], start=pull_start, end=end, timeframe="1Day", limit=1000)
    bmap = data.get("bars", {}) if isinstance(data, dict) else {}
    blist = bmap.get(symbol) or []
    if len(blist) < 260:
        return {"error": "not_enough_bars", "bars": len(blist)}

    ts = [str(b.get("t")) for b in blist]
    closes = [_safe_float(b.get("c")) for b in blist]
    highs = [_safe_float(b.get("h")) for b in blist]
    lows = [_safe_float(b.get("l")) for b in blist]

    def _to_dt(s: str) -> Optional[datetime]:
        try:
            return datetime.fromisoformat((s or "").replace("Z", "+00:00"))
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

    # position state
    in_pos = False
    entry_i = -1
    entry_price = 0.0
    base_sl = 0.0
    tp1 = None  # type: ignore[assignment]
    tp2 = 0.0
    qty_total = 0
    qty_left = 0
    qty_tp1 = 0
    tp1_hit = False
    trail_sl = None  # type: ignore[assignment]
    trail_used = False
    cooldown_until_i = -1

    def _update_dd(eq: float) -> None:
        nonlocal peak, max_dd
        if eq > peak:
            peak = eq
        dd = (peak - eq) / peak if peak > 0 else 0.0
        if dd > max_dd:
            max_dd = dd

    def _risk_per_share() -> float:
        return max(abs(entry_price - base_sl), 1e-6)

    for i in idxs:
        if i < 210:
            continue
        if i <= cooldown_until_i:
            continue

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
        close = float(c_slice[-1])
        lo = float(lows[i])
        hi = float(highs[i])

        atr_val = float(a14) if float(a14) > 0 else max(close * 0.01, 0.5)

        # ---- Manage open position (LONG only in this baseline backtest) ----
        if in_pos:
            hold_days = i - entry_i

            # Update trailing stop (end-of-day style) using today's close
            if trail_after_tp1:
                active_trail = bool(tp1_hit)
            else:
                active_trail = True

            if trail_atr_mult > 0 and active_trail:
                new_trail = close - (atr_val * float(trail_atr_mult))
                if trail_sl is None:
                    trail_sl = new_trail
                else:
                    trail_sl = max(float(trail_sl), float(new_trail))
                trail_used = True

            # Effective SL: max(base_sl, trail_sl) for LONG
            eff_sl = float(base_sl)
            if trail_sl is not None:
                eff_sl = max(eff_sl, float(trail_sl))

            exit_reason = None
            exit_price = None

            # Conservative intraday ordering: SL first
            if lo <= eff_sl:
                exit_reason = "sl" if eff_sl == base_sl else "exit"
                exit_price = eff_sl
            else:
                # Partial TP1
                if (tp1 is not None) and (not tp1_hit) and hi >= float(tp1) and qty_tp1 > 0:
                    tp1_hit = True
                    realized = (float(tp1) - entry_price) * qty_tp1
                    equity += realized
                    qty_left -= qty_tp1
                    _update_dd(equity)

                    # optionally move SL to breakeven after TP1
                    if move_sl_to_be_after_tp1:
                        base_sl = max(base_sl, entry_price)

                # Final TP2
                if qty_left > 0 and hi >= float(tp2):
                    exit_reason = "tp"
                    exit_price = float(tp2)
                elif hold_days >= int(max_holding_days):
                    exit_reason = "time"
                    exit_price = close

            if exit_reason and exit_price is not None:
                pnl = (exit_price - entry_price) * qty_left
                equity += pnl
                _update_dd(equity)

                # Total pnl includes any partial realized earlier
                # Estimate total pnl by (equity delta) is hard here; record full-trade pnl as:
                # (exit remainder pnl) + (tp1 pnl if hit)
                tp1_pnl = 0.0
                if tp1_hit and tp1 is not None and qty_tp1 > 0:
                    tp1_pnl = (float(tp1) - entry_price) * qty_tp1
                total_pnl = pnl + tp1_pnl

                r = (exit_price - entry_price) / _risk_per_share()
                if tp1_hit and tp1 is not None and qty_tp1 > 0:
                    # Add partial R contribution (approx)
                    r += ((float(tp1) - entry_price) / _risk_per_share()) * (qty_tp1 / max(qty_total, 1))

                trades.append(
                    Trade(
                        symbol=symbol,
                        entry_ts=ts[entry_i],
                        exit_ts=ts[i],
                        entry=round(entry_price, 4),
                        exit=round(exit_price, 4),
                        qty=int(qty_total),
                        side="buy",
                        pnl=round(total_pnl, 2),
                        r_mult=round(float(r), 3),
                        outcome=exit_reason,
                        tp1_hit=bool(tp1_hit),
                        tp1_price=(round(float(tp1), 4) if tp1 is not None else None),
                        tp2_price=round(float(tp2), 4),
                        trail_used=bool(trail_used),
                        partial_pct=float(partial_pct),
                    )
                )

                # reset state
                in_pos = False
                entry_i = -1
                entry_price = 0.0
                base_sl = 0.0
                tp1 = None
                tp2 = 0.0
                qty_total = 0
                qty_left = 0
                qty_tp1 = 0
                tp1_hit = False
                trail_sl = None
                trail_used = False
                cooldown_until_i = i + max(0, int(cooldown_days))
            continue

        # ---- Entry signal ----
        cond = (float(e20) > float(e50)) and (close > float(e200)) and (50.0 <= float(r14) <= 70.0) and (float(hist) > 0)
        if not cond:
            continue

        entry_price = float(close)
        base_sl = max(0.01, entry_price - (atr_val * float(sl_atr_mult)))
        rps = max(entry_price - base_sl, 1e-6)

        # targets
        tp2 = entry_price + (rps * float(tp2_r_mult))
        tp1 = None
        if partial_pct and partial_pct > 0 and tp1_r_mult and tp1_r_mult > 0:
            tp1 = entry_price + (rps * float(tp1_r_mult))

        qty_total = _position_size(equity, float(risk_per_trade_pct), entry_price, base_sl)
        if qty_total < 1:
            continue

        qty_left = qty_total
        tp1_hit = False
        trail_sl = None
        trail_used = False

        # partial sizing
        pp = float(partial_pct) if partial_pct is not None else 0.0
        pp = max(0.0, min(0.95, pp))
        qty_tp1 = int(round(qty_total * pp)) if (tp1 is not None and pp > 0) else 0
        qty_tp1 = min(qty_tp1, qty_total - 1) if qty_total > 1 else 0  # keep at least 1 share for runner
        in_pos = True
        entry_i = i

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
            "tp1_r_mult": float(tp1_r_mult),
            "tp2_r_mult": float(tp2_r_mult),
            "partial_pct": float(partial_pct),
            "trail_atr_mult": float(trail_atr_mult),
            "trail_after_tp1": bool(trail_after_tp1),
            "move_sl_to_be_after_tp1": bool(move_sl_to_be_after_tp1),
            "max_holding_days": int(max_holding_days),
            "cooldown_days": int(cooldown_days),
        },
    }
