from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple
from datetime import datetime, timedelta, timezone

from core.config import (
    UNIVERSE_MAX, MIN_PRICE, MAX_PRICE, MIN_AVG_DOLLAR_VOL,
    LOOKBACK_DAYS, TOP_N, SYMBOL_BATCH
)
from core.alpaca_client import list_assets, bars
from core.indicators import sma, ema, rsi, atr, macd, bollinger_bands, adx, stochastic, obv, vwap
from core.candlestick_patterns import classify_last_patterns
from core.storage import get_all_settings, parse_int, parse_float, get_watchlist

@dataclass
class Candidate:
    symbol: str
    side: str  # 'buy' (long) or 'sell' (short)
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

    # Optional: use manual watchlist only (set USE_WATCHLIST=1 in settings table)
    s = get_all_settings()
    use_wl = str(s.get('USE_WATCHLIST','0')).strip().lower() in ('1','true','yes','y','on')
    wl = get_watchlist() if use_wl else []
    if wl:
        return wl

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
    """Fetch recent daily bars for one symbol and compute richer technical features.

    This is used by the /ai command (Gemini prompt). It intentionally stays "lightweight"
    and relies only on daily bars (no intraday).
    """
    symbol = (symbol or "").upper().strip()
    if not symbol:
        return {"error": "Empty symbol"}

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=max(LOOKBACK_DAYS * 2, 180))

    data = bars([symbol], start=start, end=end, timeframe="1Day", limit=LOOKBACK_DAYS + 60)
    bars_by_symbol: Dict[str, List[Dict[str, Any]]] = data.get("bars", {}) if isinstance(data, dict) else {}
    blist = bars_by_symbol.get(symbol) or []
    if len(blist) < 60:
        return {"error": "Not enough bars"}

    closes = [float(b.get("c", 0)) for b in blist if "c" in b]
    highs  = [float(b.get("h", 0)) for b in blist if "h" in b]
    lows   = [float(b.get("l", 0)) for b in blist if "l" in b]
    vols   = [float(b.get("v", 0)) for b in blist if "v" in b]
    if len(closes) < 60 or len(highs) < 60 or len(lows) < 60 or len(vols) < 60:
        return {"error": "Not enough data"}

    last = closes[-1]

    # Moving averages
    s20 = sma(closes, 20)
    s50 = sma(closes, 50)
    s100 = sma(closes, 100)
    s200 = sma(closes, 200)

    e20 = ema(closes, 20)
    e50 = ema(closes, 50)
    e200 = ema(closes, 200)

    # Momentum / trend strength
    r14 = rsi(closes, 14)
    a14 = atr(highs, lows, closes, 14)
    macd_vals = macd(closes, 12, 26, 9)
    bb = bollinger_bands(closes, 20, 2.0)
    adx_vals = adx(highs, lows, closes, 14)
    stoch_vals = stochastic(highs, lows, closes, 14, 3)

    # Volume
    vavg20 = (sum(vols[-20:]) / 20.0) if len(vols) >= 20 else None
    vol_spike = bool(vavg20 and vols[-1] >= 1.5 * vavg20)
    obv_val = obv(closes, vols)

    # Liquidity / spread risk (heuristic)
    avg_dollar_vol = (vavg20 * last) if (vavg20 is not None and last) else None
    if avg_dollar_vol is None:
        liquidity = None
        spread_risk = None
    else:
        # Classify liquidity using average $ volume (20 bars)
        if avg_dollar_vol >= 20_000_000:
            liquidity = "GOOD"
            spread_risk = "LOW"
        elif avg_dollar_vol >= 5_000_000:
            liquidity = "OK"
            spread_risk = "MED"
        else:
            liquidity = "BAD"
            spread_risk = "HIGH"

    # Candlestick pattern on last 3 daily candles
    pat_bars = [{"o": float(b.get("o", 0)), "h": float(b.get("h", 0)), "l": float(b.get("l", 0)), "c": float(b.get("c", 0))} for b in blist[-3:]]
    pat = classify_last_patterns(pat_bars)


    # Price context
    atr_pct = (a14 / last) if (a14 and last) else None
    hi20 = max(highs[-20:]) if len(highs) >= 20 else None
    near_20d_high = bool(hi20 is not None and last >= 0.98 * hi20)

    vwap20 = vwap(highs, lows, closes, vols, 20)

    notes = []
    if e20 is not None and e50 is not None and e20 > e50:
        notes.append("EMA20>EMA50")
    if e200 is not None and last > e200:
        notes.append("Above EMA200")
    if r14 is not None:
        notes.append(f"RSI {r14:.0f}")
    if macd_vals is not None:
        _, _, hist = macd_vals
        notes.append(f"MACD hist {hist:.3f}")
    if adx_vals is not None:
        adxv, pdi, mdi = adx_vals
        notes.append(f"ADX {adxv:.0f} (+DI {pdi:.0f}/-DI {mdi:.0f})")
    if bb is not None:
        _, _, _, pctb = bb
        notes.append(f"BB% {pctb:.2f}")
    if stoch_vals is not None:
        k, d = stoch_vals
        notes.append(f"Stoch {k:.0f}/{d:.0f}")
    if atr_pct is not None:
        notes.append(f"ATR% {atr_pct*100:.1f}")
    if near_20d_high:
        notes.append("Near 20D high")
    if vol_spike:
        notes.append("Vol spike")
    if liquidity:
        notes.append(f"Liquidity {liquidity}")
    if pat.get("pattern"):
        notes.append(f"{pat.get('pattern')} {pat.get('strength') or ''}".strip())
    if vwap20 is not None and last > vwap20:
        notes.append("Above VWAP20")

    return {
        "price": round(last, 4),
        "SMA20": round(s20, 4) if s20 is not None else None,
        "SMA50": round(s50, 4) if s50 is not None else None,
        "SMA100": round(s100, 4) if s100 is not None else None,
        "SMA200": round(s200, 4) if s200 is not None else None,
        "EMA20": round(e20, 4) if e20 is not None else None,
        "EMA50": round(e50, 4) if e50 is not None else None,
        "EMA200": round(e200, 4) if e200 is not None else None,
        "RSI14": round(r14, 2) if r14 is not None else None,
        "ATR14": round(a14, 4) if a14 is not None else None,
        "ATR%": round(atr_pct * 100, 2) if atr_pct is not None else None,
        "MACD": {
            "macd": round(macd_vals[0], 5),
            "signal": round(macd_vals[1], 5),
            "hist": round(macd_vals[2], 5),
        } if macd_vals is not None else None,
        "Bollinger": {
            "mid": round(bb[0], 4),
            "upper": round(bb[1], 4),
            "lower": round(bb[2], 4),
            "pct_b": round(bb[3], 4),
        } if bb is not None else None,
        "ADX14": {
            "adx": round(adx_vals[0], 3),
            "+di": round(adx_vals[1], 3),
            "-di": round(adx_vals[2], 3),
        } if adx_vals is not None else None,
        "Stochastic": {
            "%K": round(stoch_vals[0], 3),
            "%D": round(stoch_vals[1], 3),
        } if stoch_vals is not None else None,
        "VWAP20": round(vwap20, 4) if vwap20 is not None else None,
        "OBV": round(obv_val, 2) if obv_val is not None else None,
        "Near 20D high": bool(near_20d_high),
        "Vol spike": bool(vol_spike),
        "notes": ", ".join(notes),
    }


def get_symbol_features_m5(symbol: str) -> Dict[str, Any]:
    """Fetch recent 5-min bars for one symbol and compute lightweight intraday features.

    Used for fast M5 / hybrid predictions. Keeps it lightweight to avoid slowing scans.
    """
    symbol = (symbol or "").upper().strip()
    if not symbol:
        return {"error": "Empty symbol"}

    end = datetime.now(timezone.utc)
    # last ~7 days is usually enough to compute EMA/RSI/ATR on 5m
    start = end - timedelta(days=7)

    data = bars([symbol], start=start, end=end, timeframe="5Min", limit=1000)
    bars_by_symbol: Dict[str, List[Dict[str, Any]]] = data.get("bars", {}) if isinstance(data, dict) else {}
    blist = bars_by_symbol.get(symbol) or []
    if len(blist) < 120:
        return {"error": "Not enough bars"}

    closes = [float(b.get("c", 0)) for b in blist if "c" in b]
    highs  = [float(b.get("h", 0)) for b in blist if "h" in b]
    lows   = [float(b.get("l", 0)) for b in blist if "l" in b]
    vols   = [float(b.get("v", 0)) for b in blist if "v" in b]
    if len(closes) < 120 or len(highs) < 120 or len(lows) < 120 or len(vols) < 120:
        return {"error": "Not enough data"}

    last = closes[-1]

    # Fast moving averages suitable for 5m
    e20 = ema(closes, 20)
    e50 = ema(closes, 50)
    r14 = rsi(closes, 14)
    a14 = atr(highs, lows, closes, 14)
    atr_pct = (a14 / last) if (a14 and last) else None

    vavg20 = (sum(vols[-20:]) / 20.0) if len(vols) >= 20 else None
    vol_spike = bool(vavg20 and vols[-1] >= 1.5 * vavg20)

    notes = []
    if e20 is not None and e50 is not None and e20 > e50:
        notes.append("EMA20>EMA50")
    if r14 is not None:
        notes.append(f"RSI {r14:.0f}")
    if vol_spike:
        notes.append("Vol spike")
    if liquidity:
        notes.append(f"Liquidity {liquidity}")
    if pat.get("pattern"):
        notes.append(f"{pat.get('pattern')} {pat.get('strength') or ''}".strip())

    return {
        "tf": "5Min",
        "last": round(float(last), 4),
        "ema20": (round(float(e20), 4) if e20 is not None else None),
        "ema50": (round(float(e50), 4) if e50 is not None else None),
        "rsi14": (round(float(r14), 2) if r14 is not None else None),
        "atr14": (round(float(a14), 4) if a14 is not None else None),
        "atr_pct": (round(float(atr_pct), 4) if atr_pct is not None else None),
        "vol_spike": bool(vol_spike),
        "avg_dollar_vol_5m": (round(float(avg_dollar_vol), 2) if avg_dollar_vol is not None else None),
        "liquidity": liquidity,
        "spread_risk": spread_risk,
        "pattern": pat.get("pattern"),
        "pattern_strength": pat.get("strength"),
        "pattern_bias": pat.get("bias"),
        "notes": ",".join(notes),
    }


def scan_universe_from_symbols(symbols: List[str]) -> List[Candidate]:
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=max(LOOKBACK_DAYS * 2, 180))

    results: List[Candidate] = []

    for batch in _chunks(symbols, SYMBOL_BATCH):
        data = bars(batch, start=start, end=end, timeframe="1Day", limit=LOOKBACK_DAYS + 60)
        bars_by_symbol: Dict[str, List[Dict[str, Any]]] = data.get("bars", {}) if isinstance(data, dict) else {}

        for sym, blist in bars_by_symbol.items():
            if not blist or len(blist) < 60:
                continue

            closes = [float(b["c"]) for b in blist if "c" in b]
            highs  = [float(b["h"]) for b in blist if "h" in b]
            lows   = [float(b["l"]) for b in blist if "l" in b]
            vols   = [float(b["v"]) for b in blist if "v" in b]

            if len(closes) < 60 or len(vols) < 60 or len(highs) != len(lows) or len(lows) != len(closes) or len(closes) != len(vols):
                continue

            last = closes[-1]
            if last < MIN_PRICE or last > MAX_PRICE:
                continue

            # Average $ volume (liquidity filter)
            adv = sum([closes[i] * vols[i] for i in range(-20, 0)]) / 20.0
            if adv < MIN_AVG_DOLLAR_VOL:
                continue

            # Core indicators
            e20 = ema(closes, 20)
            e50 = ema(closes, 50)
            e200 = ema(closes, 200)

            s100 = sma(closes, 100)
            s200 = sma(closes, 200)

            r14 = rsi(closes, 14)
            a14 = atr(highs, lows, closes, 14)
            m = macd(closes, 12, 26, 9)
            bb = bollinger_bands(closes, 20, 2.0)
            adx_vals = adx(highs, lows, closes, 14)
            stoch_vals = stochastic(highs, lows, closes, 14, 3)
            vwap20 = vwap(highs, lows, closes, vols, 20)

            if e20 is None or e50 is None or r14 is None or a14 is None or m is None or bb is None or adx_vals is None:
                continue

            macd_line, macd_sig, macd_hist = m
            bb_mid, bb_up, bb_lo, bb_pctb = bb
            adx_v, pdi, mdi = adx_vals

            # Volume features
            vavg20 = sum(vols[-20:]) / 20.0
            vol_spike = bool(vavg20 > 0 and vols[-1] >= 1.5 * vavg20)
            obv_val = obv(closes, vols)
            obv_prev = obv(closes[:-5], vols[:-5]) if len(closes) >= 10 else None
            obv_rising = bool(obv_val is not None and obv_prev is not None and obv_val > obv_prev)

            # Price context
            atr_pct = a14 / last if last else 0.0
            hi20 = max(highs[-20:])
            near_20d_high = bool(last >= 0.98 * hi20)

            # ===== Scoring =====
            score = 0.0
            notes = []

            # Trend (multi-speed EMAs)
            if e20 > e50:
                score += 2.0
                notes.append("EMA20>EMA50")
                trend = "up"
            else:
                score -= 0.5
                notes.append("EMA20<EMA50")
                trend = "down"

            if e200 is not None and last > e200:
                score += 1.5
                notes.append("Above EMA200")
            elif e200 is not None:
                score -= 0.5
                notes.append("Below EMA200")

            if e200 is not None and e50 > e200:
                score += 1.0
                notes.append("EMA50>EMA200")

            # Momentum (RSI + MACD + Stoch)
            if 50 <= r14 <= 70:
                score += 2.0
                notes.append(f"RSI {r14:.0f}")
            elif 40 <= r14 < 50:
                score += 1.0
                notes.append(f"RSI {r14:.0f} (ok)")
            elif r14 > 70:
                score += 0.5
                notes.append("RSI hot")
            else:
                score -= 0.5
                notes.append("RSI weak")

            if macd_hist > 0:
                score += 1.5
                notes.append("MACD+")
            else:
                score -= 0.5
                notes.append("MACD-")

            if stoch_vals is not None:
                k, d = stoch_vals
                if k > d and 40 <= k <= 85:
                    score += 0.8
                    notes.append("Stoch up")
                elif k < 20:
                    score += 0.3
                    notes.append("Stoch oversold")

            # Trend strength (ADX)
            if adx_v >= 30:
                score += 1.5
                notes.append(f"ADX {adx_v:.0f}")
            elif adx_v >= 20:
                score += 1.0
                notes.append(f"ADX {adx_v:.0f}")
            else:
                score += 0.2
                notes.append("ADX low")

            if pdi > mdi:
                score += 0.4
                notes.append("+DI>-DI")
            else:
                score -= 0.2
                notes.append("-DI>=+DI")

            # Volatility sanity (ATR%)
            if 0.012 <= atr_pct <= 0.06:
                score += 1.0
                notes.append(f"ATR% {atr_pct*100:.1f}")
            elif atr_pct < 0.012:
                score += 0.2
                notes.append("ATR low")
            elif atr_pct > 0.10:
                score -= 0.5
                notes.append("ATR very high")
            else:
                score += 0.5
                notes.append("ATR high")

            # Breakout / positioning
            if near_20d_high:
                score += 1.5
                notes.append("Near 20D high")

            if bb_pctb >= 0.8:
                score += 0.8
                notes.append("BB strong")
            if bb_pctb > 1.05:
                score -= 0.3
                notes.append("BB extended")

            if vwap20 is not None:
                if last > vwap20:
                    score += 0.6
                    notes.append("Above VWAP20")
                else:
                    score -= 0.2
                    notes.append("Below VWAP20")

            # Volume confirmation
            if vol_spike:
                score += 1.0
                notes.append("Vol spike")
            if obv_rising:
                score += 0.5
                notes.append("OBV rising")

            # Penalize extreme gap days
            if len(closes) >= 2 and closes[-2] != 0:
                gap = abs(closes[-1] - closes[-2]) / closes[-2]
                if gap > 0.12:
                    score -= 1.5
                    notes.append("Big gap")

            # --- Directional scoring (Long + Short) ---
            long_score = float(score)
            long_notes = list(notes)

            # Build a mirrored score for short setups
            short_score = 0.0
            short_notes: List[str] = []
            try:
                short_score = 0.0

                # Trend (bearish)
                if e20 is not None and e50 is not None and e20 < e50:
                    short_score += 2.0
                    short_notes.append("EMA20<EMA50")
                if e50 is not None and e200 is not None and e50 < e200 and last < e200:
                    short_score += 2.0
                    short_notes.append("Below EMA200")

                # Momentum
                if macd_vals is not None:
                    _, _, hist = macd_vals
                    if hist < 0:
                        short_score += 1.5
                        short_notes.append("MACD-")
                if r14 is not None:
                    if r14 < 45:
                        short_score += 1.5
                        short_notes.append("RSI weak")
                    elif r14 > 65:
                        short_score -= 1.0
                        short_notes.append("RSI high (bad for short)")

                # Trend strength
                if adx_vals is not None:
                    adx14, di_p, di_m = adx_vals
                    if adx14 is not None and di_m is not None and di_p is not None and adx14 > 18 and di_m > di_p:
                        short_score += 1.2
                        short_notes.append("ADX bear")

                # Bands / mean reversion (avoid overextended down)
                if bb is not None:
                    _, _, _, pct_b = bb
                    if pct_b < -0.05:
                        short_score -= 0.6
                        short_notes.append("BB very low")
                    elif pct_b > 0.85:
                        short_score += 0.6
                        short_notes.append("BB high (pullback short)")

                # VWAP: prefer below for short
                if vwap20 is not None:
                    if last < vwap20:
                        short_score += 0.6
                        short_notes.append("Below VWAP20")
                    else:
                        short_score -= 0.2
                        short_notes.append("Above VWAP20")

                # Volume confirmation
                if vol_spike:
                    short_score += 0.8
                    short_notes.append("Vol spike")
                if obv_rising is False:
                    short_score += 0.3
                    short_notes.append("OBV falling")

                # Penalize extreme gap days (same)
                if len(closes) >= 2 and closes[-2] != 0:
                    gap = abs(closes[-1] - closes[-2]) / closes[-2]
                    if gap > 0.12:
                        short_score -= 1.5
                        short_notes.append("Big gap")
            except Exception:
                pass

            # Timeframe alignment flags are computed per-direction
            long_daily_ok = bool(e20 > e50) if (e20 is not None and e50 is not None) else False
            long_weekly_ok = bool((e200 is not None) and (e50 is not None) and (e50 > e200) and (last > e200))
            long_monthly_ok = bool((s200 is not None) and (s100 is not None) and (s100 > s200) and (last > s200))

            short_daily_ok = bool(e20 < e50) if (e20 is not None and e50 is not None) else False
            short_weekly_ok = bool((e200 is not None) and (e50 is not None) and (e50 < e200) and (last < e200))
            short_monthly_ok = bool((s200 is not None) and (s100 is not None) and (s100 < s200) and (last < s200))

            # Choose direction: keep the stronger setup. Require some minimum conviction.
            # You can tune thresholds via settings later.
            chosen_side = "buy"
            chosen_score = long_score
            chosen_notes = long_notes
            chosen_daily_ok, chosen_weekly_ok, chosen_monthly_ok = long_daily_ok, long_weekly_ok, long_monthly_ok
            chosen_trend = trend

            if short_score > long_score + 0.75:
                chosen_side = "sell"
                chosen_score = short_score
                chosen_notes = short_notes
                chosen_daily_ok, chosen_weekly_ok, chosen_monthly_ok = short_daily_ok, short_weekly_ok, short_monthly_ok
                chosen_trend = "Bear"

            results.append(Candidate(
                symbol=sym,
                side=chosen_side,
                score=float(chosen_score),
                last_close=last,
                avg_dollar_vol=adv,
                atr=a14,
                rsi14=r14,
                trend=chosen_trend,
                notes=", ".join(chosen_notes),
                daily_ok=bool(chosen_daily_ok),
                weekly_ok=bool(chosen_weekly_ok),
                monthly_ok=bool(chosen_monthly_ok)
            ))

    results.sort(key=lambda x: x.score, reverse=True)
    settings = get_all_settings()
    top_n = parse_int(settings.get('TOP_N'), TOP_N)
    return results[:max(1, top_n)]
