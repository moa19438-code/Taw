
from __future__ import annotations

from typing import Any, Dict, Tuple


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def estimate_loss_probability(features: Dict[str, Any], score: float | int | None = None) -> Tuple[float, list[str]]:
    """تقدير احتمالية الخسارة (تقريبي/قابل للمعايرة) بناءً على خصائص الصفقة.

    الهدف: إعطاء رقم عملي يُستخدم كفلتر وحجم صفقة (Position Sizing) قبل إرسال الإشارة.
    هذا *ليس* ضمانًا، وإنما تقدير مبني على خصائص السوق/الصفقة.

    Returns:
        (loss_prob بين 0 و 1, أسباب مختصرة)
    """
    f = features or {}
    reasons: list[str] = []

    # Base: بدون معلومات = احتمال خسارة أعلى من 50% قليلًا
    p = 0.48

    try:
        sc = float(score) if score is not None else None
        if sc is not None:
            if sc >= 92:
                p -= 0.14; reasons.append("Score عالي جدًا")
            elif sc >= 88:
                p -= 0.11; reasons.append("Score عالي")
            elif sc >= 82:
                p -= 0.08; reasons.append("Score جيد")
            elif sc >= 75:
                p -= 0.04; reasons.append("Score مقبول")
            else:
                p += 0.06; reasons.append("Score منخفض")
    except Exception:
        pass

    # Market regime (SPY) if available
    try:
        risk = str(f.get("market_risk") or f.get("risk") or "").upper()
        if risk == "OFF":
            p += 0.15; reasons.append("السوق العام Risk-OFF")
        elif risk == "ON":
            p -= 0.03; reasons.append("السوق العام داعم")
    except Exception:
        pass

    # Weekly alignment (for D1)
    try:
        w_ok = None
        # if weekly indicators exist, infer alignment quickly
        w_close = f.get("w_close")
        w_ema20 = f.get("w_ema20")
        w_ema50 = f.get("w_ema50")
        if w_close is not None and w_ema20 is not None and w_ema50 is not None:
            w_ok = float(w_close) > float(w_ema20) > float(w_ema50)
        if w_ok is True:
            p -= 0.08; reasons.append("تأكيد أسبوعي داعم")
        elif w_ok is False:
            p += 0.10; reasons.append("تأكيد أسبوعي ضعيف/معاكس")
    except Exception:
        pass

    # Chop / trend strength
    try:
        adx = f.get("adx14")
        if adx is not None:
            a = float(adx)
            if a < 12:
                p += 0.10; reasons.append("ADX منخفض جدًا (سوق متذبذب)")
            elif a < 15:
                p += 0.06; reasons.append("ADX منخفض (تذبذب)")
            elif a >= 25:
                p -= 0.04; reasons.append("ADX قوي (ترند واضح)")
    except Exception:
        pass

    # Volatility
    try:
        ap = f.get("atr_pct")
        if ap is not None:
            v = float(ap)
            if v > 0.10:
                p += 0.10; reasons.append("تذبذب عالي جدًا (ATR%)")
            elif v > 0.08:
                p += 0.07; reasons.append("تذبذب عالي (ATR%)")
            elif v < 0.006:
                p += 0.04; reasons.append("تذبذب منخفض جدًا (خطر فشل الاختراق)")
            elif v < 0.010:
                p += 0.02; reasons.append("تذبذب منخفض")
    except Exception:
        pass

    # Late entry / overextension if available
    try:
        ext = f.get("ext_vs_ema20")
        if ext is not None:
            e = float(ext)
            if e > 0.03:
                p += 0.06; reasons.append("دخول متأخر (ممتد عن EMA20)")
            elif e > 0.02:
                p += 0.03; reasons.append("ممتد قليلًا عن EMA20")
    except Exception:
        pass

    # Volume breakout context
    try:
        if bool(f.get("vol_spike")):
            p -= 0.04; reasons.append("فوليوم داعم")
        if bool(f.get("near_high20")):
            p -= 0.03; reasons.append("سياق اختراق (قرب قمة 20D)")
    except Exception:
        pass

    p = clamp(p, 0.05, 0.85)
    return float(p), reasons
