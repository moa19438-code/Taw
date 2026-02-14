from __future__ import annotations
from typing import Dict, List, Optional

def _body(o: float, c: float) -> float:
    return abs(c - o)

def _range(h: float, l: float) -> float:
    return max(1e-9, h - l)

def classify_last_patterns(bars: List[Dict]) -> Dict[str, Optional[str]]:
    """Classify a small set of candlestick reversal patterns using the last 3 candles.

    bars: list of dicts with keys o,h,l,c (float). Uses the last candles in the list.
    Returns:
      {"pattern": <name or None>, "strength": "WEAK|MED|STRONG|None", "bias":"BULL|BEAR|NEUTRAL"}
    """
    if not bars or len(bars) < 2:
        return {"pattern": None, "strength": None, "bias": "NEUTRAL"}

    # Use last 3 candles if available
    c1 = bars[-1]
    c0 = bars[-2]
    c_1 = bars[-3] if len(bars) >= 3 else None

    o1,h1,l1,cl1 = float(c1["o"]), float(c1["h"]), float(c1["l"]), float(c1["c"])
    o0,h0,l0,cl0 = float(c0["o"]), float(c0["h"]), float(c0["l"]), float(c0["c"])

    r1 = _range(h1,l1)
    b1 = _body(o1,cl1)
    upper1 = h1 - max(o1,cl1)
    lower1 = min(o1,cl1) - l1

    r0 = _range(h0,l0)
    b0 = _body(o0,cl0)

    is_green1 = cl1 > o1
    is_red1 = cl1 < o1
    is_green0 = cl0 > o0
    is_red0 = cl0 < o0

    # Engulfing
    bull_engulf = is_red0 and is_green1 and (cl1 >= o0) and (o1 <= cl0)
    bear_engulf = is_green0 and is_red1 and (o1 >= cl0) and (cl1 <= o0)

    # Hammer / Shooting star (single candle)
    hammer = (lower1 >= 2.0 * b1) and (upper1 <= 0.35 * b1) and (b1 >= 0.15 * r1) and is_green1
    inv_hammer = (upper1 >= 2.0 * b1) and (lower1 <= 0.35 * b1) and (b1 >= 0.15 * r1) and is_red1

    shooting_star = (upper1 >= 2.0 * b1) and (lower1 <= 0.35 * b1) and (b1 >= 0.15 * r1) and is_red1

    # Morning/Evening star (3 candles - simplified)
    morning_star = False
    evening_star = False
    if c_1 is not None:
        o_1,h_1,l_1,cl_1 = float(c_1["o"]), float(c_1["h"]), float(c_1["l"]), float(c_1["c"])
        is_red_1 = cl_1 < o_1
        is_green_1 = cl_1 > o_1
        # star = small body in middle
        mid_small = b0 <= 0.45 * r0
        # morning star: red, small, then strong green closing above midpoint of first
        morning_star = is_red_1 and mid_small and is_green1 and (cl1 >= (o_1 + cl_1)/2.0)
        evening_star = is_green_1 and mid_small and is_red1 and (cl1 <= (o_1 + cl_1)/2.0)

    # Inside bar (last candle inside previous)
    inside = (h1 <= h0) and (l1 >= l0)

    # Strength heuristic: bigger body relative to range
    strength = None
    pattern = None
    bias = "NEUTRAL"

    def strength_from_body(body: float, rng: float) -> str:
        ratio = body / max(1e-9, rng)
        if ratio >= 0.7:
            return "STRONG"
        if ratio >= 0.45:
            return "MED"
        return "WEAK"

    if bull_engulf:
        pattern = "Bullish Engulfing"
        bias = "BULL"
        strength = "STRONG" if (b1 >= b0) else "MED"
    elif bear_engulf:
        pattern = "Bearish Engulfing"
        bias = "BEAR"
        strength = "STRONG" if (b1 >= b0) else "MED"
    elif morning_star:
        pattern = "Morning Star"
        bias = "BULL"
        strength = "MED"
    elif evening_star:
        pattern = "Evening Star"
        bias = "BEAR"
        strength = "MED"
    elif hammer:
        pattern = "Hammer"
        bias = "BULL"
        strength = strength_from_body(b1, r1)
    elif shooting_star:
        pattern = "Shooting Star"
        bias = "BEAR"
        strength = strength_from_body(b1, r1)
    elif inside:
        pattern = "Inside Bar"
        bias = "NEUTRAL"
        strength = "WEAK"

    return {"pattern": pattern, "strength": strength, "bias": bias}
