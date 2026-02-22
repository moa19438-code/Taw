
from __future__ import annotations

from typing import Any, Dict, Tuple, List

def classify_setup(f: Dict[str, Any], side: str = "buy") -> Tuple[str, List[str]]:
    """Classify the primary setup for 1D tactical trading.

    Returns (setup_name, notes).
    setup_name in: BREAKOUT | PULLBACK | GAP | MIXED
    """
    side = (side or "buy").lower().strip()
    notes: List[str] = []

    price = float(f.get("price") or 0.0)
    ema20 = f.get("ema20")
    ema50 = f.get("ema50")
    ema200 = f.get("ema200")
    rsi14 = f.get("rsi14")
    adx14 = f.get("adx14")
    vol_spike = f.get("vol_spike")
    near_high20 = f.get("near_high20")
    gap_pct = f.get("gap_pct")
    atr_pct = f.get("atr_pct")

    def _trend_up() -> bool:
        try:
            return (price > float(ema20) > float(ema50) > float(ema200))
        except Exception:
            return False

    # Gap+Go: noticeable gap and volume
    try:
        gp = float(gap_pct) if gap_pct is not None else 0.0
        if abs(gp) >= 0.03 and bool(vol_spike):
            if side == "buy" and gp > 0:
                notes.append(f"Gap +{gp*100:.1f}% مع حجم مرتفع")
                return "GAP", notes
            if side == "sell" and gp < 0:
                notes.append(f"Gap {gp*100:.1f}% مع حجم مرتفع")
                return "GAP", notes
    except Exception:
        pass

    # Breakout: near 20D high + volume spike + reasonable trend
    if bool(near_high20) and bool(vol_spike):
        notes.append("قريب من قمة 20 يوم + حجم تداول مرتفع")
        return "BREAKOUT", notes

    # Pullback: trend up and price near EMA20/50 with RSI not overbought
    if _trend_up():
        try:
            p_ema20 = abs(price - float(ema20)) / max(price, 1e-6)
        except Exception:
            p_ema20 = 9e9
        if p_ema20 <= 0.02 and (rsi14 is None or float(rsi14) <= 65):
            notes.append("تصحيح إلى EMA20 داخل اتجاه صاعد")
            return "PULLBACK", notes

    # Mixed fallback
    if atr_pct is not None:
        try:
            notes.append(f"ATR%≈{float(atr_pct)*100:.1f}%")
        except Exception:
            pass
    if adx14 is not None:
        try:
            notes.append(f"ADX≈{float(adx14):.1f}")
        except Exception:
            pass
    return "MIXED", notes
