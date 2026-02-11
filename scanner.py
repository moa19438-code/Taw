from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime, timedelta, timezone

from config import (
    UNIVERSE_MAX,
    MIN_PRICE,
    MAX_PRICE,
    MIN_AVG_DOLLAR_VOL,
    LOOKBACK_DAYS,
    TOP_N,
    SYMBOL_BATCH,
)
from alpaca_client import list_assets, bars
from indicators import sma, ema, rsi, atr
from storage import get_all_settings, parse_int, parse_float


@dataclass
class Candidate:
    symbol: str
    score: float
    last_close: float
    avg_dollar_vol: float
    atr: float
    rsi14: float
    trend: str
    notes: str
    daily_ok: bool
    weekly_ok: bool
    monthly_ok: bool


def _chunks(lst: List[str], n: int) -> List[List[str]]:
    return [lst[i : i + n] for i in range(0, len(lst), n)]


def _ema_series(values: List[float], period: int) -> Optional[List[float]]:
    if period <= 0 or len(values) < period:
        return None
    k = 2 / (period + 1)
    out: List[float] = []
    e = values[0]
    out.append(e)
    for v in values[1:]:
        e = v * k + e * (1 - k)
        out.append(e)
    return out


def build_universe() -> List[str]:
    assets = list_assets(limit=5000)
    syms = []
    for a in assets:
        sym = a.get("symbol")
        if not sym:
            continue
        # Basic symbol hygiene
        if "." in sym or "/" in sym:
            continue
        syms.append(sym)

    settings = get_all_settings()
    umax = parse_int(settings.get("UNIVERSE_MAX"), UNIVERSE_MAX)
    return syms[:umax]


def scan_universe() -> List[Candidate]:
    symbols = build_universe()
    return scan_universe_from_symbols(symbols)


def scan_universe_with_meta() -> Tuple[List[Candidate], int]:
    symbols = build_universe()
    picks = scan_universe_from_symbols(symbols)
    return picks, len(symbols)


def _plan_allows(c: Candidate, plan_mode: str) -> bool:
    plan_mode = (plan_mode or "daily").lower()
    if plan_mode in ("daily",):
        return c.daily_ok
    if plan_mode in ("weekly",):
        return c.weekly_ok
    if plan_mode in ("monthly",):
        return c.monthly_ok
    if plan_mode in ("daily_weekly", "dw"):
        return c.daily_ok or c.weekly_ok
    if plan_mode in ("weekly_monthly", "wm"):
        return c.weekly_ok or c.monthly_ok
    return True


def scan_universe_from_symbols(symbols: List[str]) -> List[Candidate]:
    settings = get_all_settings()
    top_n = parse_int(settings.get("TOP_N"), TOP_N)
    scan_mode = (settings.get("SCAN_MODE") or "mixed").lower()
    plan_mode = (settings.get("PLAN_MODE") or "daily").lower()

    # Market proxy for relative-strength scoring (SPY).
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=max(LOOKBACK_DAYS * 3, 220))

    spy_ret20 = None
    try:
        spy_data = bars(["SPY"], start=start, end=end, timeframe="1Day", limit=220)
        spy_b = (spy_data.get("bars", {}) or {}).get("SPY", [])
        spy_closes = [float(b["c"]) for b in spy_b if "c" in b]
        if len(spy_closes) >= 21:
            spy_ret20 = (spy_closes[-1] / spy_closes[-21]) - 1.0
    except Exception:
        spy_ret20 = None

    results: List[Candidate] = []

    for batch in _chunks(symbols, SYMBOL_BATCH):
        data = bars(batch, start=start, end=end, timeframe="1Day", limit=LOOKBACK_DAYS + 220)
        bars_by_symbol: Dict[str, List[Dict[str, Any]]] = data.get("bars", {}) if isinstance(data, dict) else {}

        for sym, blist in bars_by_symbol.items():
            if not blist or len(blist) < 60:
                continue

            closes = [float(b["c"]) for b in blist if "c" in b]
            highs = [float(b["h"]) for b in blist if "h" in b]
            lows = [float(b["l"]) for b in blist if "l" in b]
            vols = [float(b["v"]) for b in blist if "v" in b]
            if len(closes) < 60 or len(vols) < 60:
                continue

            last = closes[-1]
            if last < MIN_PRICE or last > MAX_PRICE:
                continue

            # Liquidity: avg dollar volume (20D)
            adv = sum([closes[i] * vols[i] for i in range(-20, 0)]) / 20.0
            if adv < MIN_AVG_DOLLAR_VOL:
                continue

            # Core indicators (pro-level basics)
            e20 = ema(closes, 20)
            e50 = ema(closes, 50)
            e200 = ema(closes, 200)
            r14 = rsi(closes, 14)
            a14 = atr(highs, lows, closes, 14)
            if e20 is None or e50 is None or r14 is None or a14 is None:
                continue

            atr_pct = a14 / max(last, 0.01)
            vavg20 = sum(vols[-20:]) / 20.0
            vol_ratio = (vols[-1] / vavg20) if vavg20 > 0 else 0.0

            # MACD (12,26,9) – simplified but effective
            e12s = _ema_series(closes, 12)
            e26s = _ema_series(closes, 26)
            macd_ok = False
            macd_hist = 0.0
            if e12s and e26s and len(e12s) == len(e26s):
                macd_line = [a - b for a, b in zip(e12s, e26s)]
                sig = _ema_series(macd_line, 9)
                if sig:
                    macd_hist = macd_line[-1] - sig[-1]
                    macd_ok = macd_hist > 0

            # Relative strength vs SPY (20D)
            rs20 = None
            if len(closes) >= 21:
                ret20 = (closes[-1] / closes[-21]) - 1.0
                if spy_ret20 is not None:
                    rs20 = ret20 - spy_ret20
                else:
                    rs20 = ret20

            # Pattern logic: breakout vs pullback
            hi20 = max(highs[-20:])
            lo20 = min(lows[-20:])
            near_hi20 = last >= 0.99 * hi20
            uptrend = (e20 > e50) and (e200 is None or last > e200)
            strong_trend = (e200 is not None) and (e20 > e50 > e200) and (last > e200)

            # Pullback: price not far above EMA20/EMA50 and trend up
            dist_e20 = (last - e20) / e20
            dist_e50 = (last - e50) / e50
            pullback_zone = uptrend and (dist_e20 <= 0.03) and (dist_e50 <= 0.08)

            # Breakout: near highs + volume confirmation
            breakout_ok = near_hi20 and uptrend and (vol_ratio >= 1.2)

            # Apply scan_mode gating (medium+strict blended)
            mode_ok = True
            mode_note = []
            if scan_mode == "breakout":
                mode_ok = breakout_ok
                mode_note.append("Mode=breakout")
            elif scan_mode == "pullback":
                mode_ok = pullback_zone and (40 <= r14 <= 65)
                mode_note.append("Mode=pullback")
            elif scan_mode == "mixed":
                # blended: accept either, but score prefers stronger confirmations
                mode_ok = breakout_ok or (pullback_zone and (38 <= r14 <= 70))
                mode_note.append("Mode=mixed")

            if not mode_ok:
                continue

            # Daily/weekly/monthly "ok" flags (simple proxy on daily data)
            daily_ok = bool(e20 > e50)
            weekly_ok = bool((e200 is not None) and (e50 > e200) and (last > e200))
            monthly_ok = bool((e200 is not None) and (e20 > e200) and (last > e200))

            c_tmp = Candidate(
                symbol=sym,
                score=0.0,
                last_close=last,
                avg_dollar_vol=adv,
                atr=a14,
                rsi14=r14,
                trend="up" if uptrend else "down",
                notes="",
                daily_ok=daily_ok,
                weekly_ok=weekly_ok,
                monthly_ok=monthly_ok,
            )

            if not _plan_allows(c_tmp, plan_mode):
                continue

            # Scoring (professional-style, but simple & robust)
            score = 0.0
            notes: List[str] = []

            # Trend structure
            if strong_trend:
                score += 2.5
                notes.append("Strong trend (EMA20>EMA50>EMA200)")
            elif uptrend:
                score += 1.8
                notes.append("Uptrend (EMA20>EMA50)")
            else:
                score += 0.5
                notes.append("Trend weak")

            # RSI band
            if 45 <= r14 <= 65:
                score += 1.8
                notes.append(f"RSI {r14:.0f} (healthy)")
            elif 35 <= r14 < 45:
                score += 1.0
                notes.append(f"RSI {r14:.0f} (pullback)")
            elif r14 > 70:
                score += 0.5
                notes.append(f"RSI {r14:.0f} (hot)")
            else:
                score += 0.4
                notes.append(f"RSI {r14:.0f} (low)")

            # Volatility fit (ATR%)
            if 0.012 <= atr_pct <= 0.06:
                score += 1.6
                notes.append(f"ATR% {atr_pct*100:.1f} (good)")
            elif atr_pct < 0.012:
                score += 0.7
                notes.append("ATR low")
            else:
                score += 0.9
                notes.append("ATR high")

            # Volume confirmation
            if vol_ratio >= 1.8:
                score += 1.6
                notes.append(f"Vol spike x{vol_ratio:.1f}")
            elif vol_ratio >= 1.2:
                score += 1.0
                notes.append(f"Vol ok x{vol_ratio:.1f}")
            else:
                score += 0.3
                notes.append("Vol low")

            # MACD momentum
            if macd_ok:
                score += 1.0
                notes.append("MACD +")
            else:
                score += 0.2
                notes.append("MACD -")

            # Relative strength
            if rs20 is not None:
                if rs20 > 0.03:
                    score += 1.2
                    notes.append("RS20 strong")
                elif rs20 > 0.0:
                    score += 0.7
                    notes.append("RS20 ok")
                else:
                    score += 0.1
                    notes.append("RS20 weak")

            # Mode-specific bonus (blended strict/medium)
            if breakout_ok:
                score += 1.3
                notes.append("Breakout setup")
            if pullback_zone:
                score += 0.9
                notes.append("Pullback zone")

            # Penalty for huge daily gaps (risk)
            if len(closes) >= 2:
                gap = abs(closes[-1] - closes[-2]) / max(closes[-2], 0.01)
                if gap > 0.12:
                    score -= 1.5
                    notes.append("Big gap")

            notes.extend(mode_note)

            results.append(
                Candidate(
                    symbol=sym,
                    score=score,
                    last_close=last,
                    avg_dollar_vol=adv,
                    atr=a14,
                    rsi14=r14,
                    trend="up" if uptrend else "down",
                    notes=", ".join(notes),
                    daily_ok=daily_ok,
                    weekly_ok=weekly_ok,
                    monthly_ok=monthly_ok,
                )
            )

    results.sort(key=lambda x: x.score, reverse=True)
    return results[: max(1, top_n)]
