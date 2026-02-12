from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple
from datetime import datetime, timedelta, timezone

from config import (
    UNIVERSE_MAX, MIN_PRICE, MAX_PRICE, MIN_AVG_DOLLAR_VOL,
    LOOKBACK_DAYS, TOP_N, SYMBOL_BATCH
)
from alpaca_client import list_assets, bars
from indicators import sma, rsi, atr
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
    return [lst[i:i+n] for i in range(0, len(lst), n)]

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
    umax = parse_int(settings.get('UNIVERSE_MAX'), UNIVERSE_MAX)
    return syms[:umax]

def scan_universe() -> List[Candidate]:
    symbols = build_universe()
    return scan_universe_from_symbols(symbols)


def scan_universe_with_meta() -> Tuple[List[Candidate], int]:
    symbols = build_universe()
    picks = scan_universe_from_symbols(symbols)
    return picks, len(symbols)

def get_symbol_features(symbol: str) -> Dict[str, Any]:
    """Fetch recent daily bars for one symbol and compute features for AI analysis.

    Returns a dict of numeric indicators + a short notes string.
    If data is insufficient, returns {"error": "..."}.
    """
    symbol = (symbol or "").upper().strip()
    if not symbol:
        return {"error": "Empty symbol"}

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=max(LOOKBACK_DAYS * 2, 120))

    data = bars([symbol], start=start, end=end, timeframe="1Day", limit=LOOKBACK_DAYS + 20)
    bars_by_symbol: Dict[str, List[Dict[str, Any]]] = data.get("bars", {}) if isinstance(data, dict) else {}
    blist = bars_by_symbol.get(symbol) or []
    if len(blist) < 30:
        return {"error": "Not enough bars"}

    closes = [float(b.get("c", 0)) for b in blist if "c" in b]
    highs  = [float(b.get("h", 0)) for b in blist if "h" in b]
    lows   = [float(b.get("l", 0)) for b in blist if "l" in b]
    vols   = [float(b.get("v", 0)) for b in blist if "v" in b]
    if len(closes) < 30 or len(highs) < 30 or len(lows) < 30:
        return {"error": "Not enough data"}

    last = closes[-1]
    s20 = sma(closes, 20)
    s50 = sma(closes, 50)
    s100 = sma(closes, 100)
    s200 = sma(closes, 200)
    r14 = rsi(closes, 14)
    a14 = atr(highs, lows, closes, 14)

    atr_pct = (a14 / last) if (a14 and last) else None
    hi20 = max(highs[-20:]) if len(highs) >= 20 else None
    vavg20 = (sum(vols[-20:]) / 20.0) if len(vols) >= 20 else None
    vol_spike = bool(vavg20 and vols[-1] >= 1.5 * vavg20)

    notes = []
    if s20 is not None and s50 is not None and s20 > s50:
        notes.append("SMA20>SMA50")
    if s200 is not None and last > s200:
        notes.append("Above SMA200")
    if r14 is not None:
        notes.append(f"RSI {r14:.0f}")
    if atr_pct is not None:
        notes.append(f"ATR% {atr_pct*100:.1f}")
    if hi20 is not None and last >= 0.98 * hi20:
        notes.append("Near 20D high")
    if vol_spike:
        notes.append("Vol spike")

    return {
        "price": round(last, 4),
        "SMA20": round(s20, 4) if s20 is not None else None,
        "SMA50": round(s50, 4) if s50 is not None else None,
        "SMA100": round(s100, 4) if s100 is not None else None,
        "SMA200": round(s200, 4) if s200 is not None else None,
        "RSI14": round(r14, 2) if r14 is not None else None,
        "ATR14": round(a14, 4) if a14 is not None else None,
        "ATR%": round(atr_pct * 100, 2) if atr_pct is not None else None,
        "Near 20D high": bool(hi20 is not None and last >= 0.98 * hi20),
        "Vol spike": bool(vol_spike),
        "notes": ", ".join(notes),
    }


def scan_universe_from_symbols(symbols: List[str]) -> List[Candidate]:
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=max(LOOKBACK_DAYS * 2, 120))
    results: List[Candidate] = []
    for batch in _chunks(symbols, SYMBOL_BATCH):
        data = bars(batch, start=start, end=end, timeframe="1Day", limit=LOOKBACK_DAYS + 20)
        bars_by_symbol: Dict[str, List[Dict[str, Any]]] = data.get("bars", {}) if isinstance(data, dict) else {}
        for sym, blist in bars_by_symbol.items():
            if not blist or len(blist) < 30:
                continue
            closes = [float(b["c"]) for b in blist if "c" in b]
            highs  = [float(b["h"]) for b in blist if "h" in b]
            lows   = [float(b["l"]) for b in blist if "l" in b]
            vols   = [float(b["v"]) for b in blist if "v" in b]
            if len(closes) < 30 or len(vols) < 30:
                continue
            last = closes[-1]
            if last < MIN_PRICE or last > MAX_PRICE:
                continue
            adv = sum([closes[i] * vols[i] for i in range(-20, 0)]) / 20.0
            if adv < MIN_AVG_DOLLAR_VOL:
                continue
            s20 = sma(closes, 20)
            s50 = sma(closes, 50)
            s100 = sma(closes, 100)
            s200 = sma(closes, 200)
            r14 = rsi(closes, 14)
            a14 = atr(highs, lows, closes, 14)
            if s20 is None or s50 is None or r14 is None or a14 is None:
                continue
            score = 0.0
            notes = []
            if s20 > s50:
                score += 2.0
                notes.append("SMA20>SMA50")
                trend = "up"
            else:
                trend = "down"
            if s200 is not None and last > s200:
                score += 1.0
                notes.append("Above SMA200")
            if 45 <= r14 <= 70:
                score += 2.0
                notes.append(f"RSI {r14:.0f}")
            elif r14 > 70:
                score += 0.5
                notes.append("RSI hot")
            else:
                score += 0.5
                notes.append("RSI low")
            atr_pct = a14 / last
            if 0.012 <= atr_pct <= 0.06:
                score += 2.0
                notes.append(f"ATR% {atr_pct*100:.1f}")
            elif atr_pct < 0.012:
                score += 0.5
                notes.append("ATR low")
            else:
                score += 0.8
                notes.append("ATR high")
            hi20 = max(highs[-20:])
            if last >= 0.98 * hi20:
                score += 2.0
                notes.append("Near 20D high")
            vavg20 = sum(vols[-20:]) / 20.0
            if vavg20 > 0 and vols[-1] >= 1.5 * vavg20:
                score += 1.5
                notes.append("Vol spike")
            if len(closes) >= 2:
                gap = abs(closes[-1] - closes[-2]) / closes[-2]
                if gap > 0.12:
                    score -= 1.5
                    notes.append("Big gap")

            daily_ok = bool(s20 > s50)
            weekly_ok = bool((s200 is not None) and (s50 is not None) and (s50 > s200) and (last > s200))
            monthly_ok = bool((s200 is not None) and (s100 is not None) and (s100 > s200) and (last > s200))

            results.append(Candidate(
                symbol=sym,
                score=score,
                last_close=last,
                avg_dollar_vol=adv,
                atr=a14,
                rsi14=r14,
                trend=trend,
                notes=", ".join(notes),
                daily_ok=daily_ok,
                weekly_ok=weekly_ok,
                monthly_ok=monthly_ok
            ))
    results.sort(key=lambda x: x.score, reverse=True)
    settings = get_all_settings()
    top_n = parse_int(settings.get('TOP_N'), TOP_N)
    return results[:max(1, top_n)]
