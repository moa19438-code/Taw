from __future__ import annotations

import os
import json
import re




from typing import Any, Dict, List, Tuple, Optional

from scanner import get_symbol_features
from ai_analyzer import gemini_predict_direction


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def score_signal(symbol: str, side: str = "buy") -> Tuple[int, List[str], Dict[str, Any]]:
    """Deterministic (non-learning) scoring model to *filter* signals.

    Returns:
        score (0-100), reasons, features
    """
    side = (side or "buy").lower().strip()
    f = get_symbol_features(symbol)
    if f.get("error"):
        return 0, [f"data_error: {f['error']}"], f

    # Pull common fields with defaults
    price = float(f.get("price") or 0.0)

    ema20 = f.get("ema20")
    ema50 = f.get("ema50")
    ema200 = f.get("ema200")
    macd_hist = f.get("macd_hist")
    rsi14 = f.get("rsi14")
    adx14 = f.get("adx14")
    di_plus = f.get("di_plus")
    di_minus = f.get("di_minus")
    stoch_k = f.get("stoch_k")
    stoch_d = f.get("stoch_d")
    bb_pct_b = f.get("bb_pct_b")
    vwap20 = f.get("vwap20")
    atr_pct = f.get("atr_pct")
    vol_spike = f.get("vol_spike")
    obv_slope = f.get("obv_slope")
    near_high20 = f.get("near_high20")

    score = 50.0
    reasons: List[str] = []

    def bump(cond: bool, pts: float, why: str):
        nonlocal score
        if cond:
            score += pts
            reasons.append(f"+{pts:g} {why}")

    def penalize(cond: bool, pts: float, why: str):
        nonlocal score
        if cond:
            score -= pts
            reasons.append(f"-{pts:g} {why}")

    # --- Trend (up to ~25 pts) ---
    if side == "buy":
        bump(ema20 is not None and price > float(ema20), 6, "price>EMA20")
        bump(ema50 is not None and price > float(ema50), 6, "price>EMA50")
        bump(ema200 is not None and price > float(ema200), 4, "price>EMA200")
        bump(ema20 is not None and ema50 is not None and float(ema20) > float(ema50), 5, "EMA20>EMA50")
        bump(ema50 is not None and ema200 is not None and float(ema50) > float(ema200), 4, "EMA50>EMA200")
    else:
        # For sell signals (if you ever use them): invert some checks
        bump(ema20 is not None and price < float(ema20), 6, "price<EMA20")
        bump(ema50 is not None and price < float(ema50), 6, "price<EMA50")
        bump(ema200 is not None and price < float(ema200), 4, "price<EMA200")
        bump(ema20 is not None and ema50 is not None and float(ema20) < float(ema50), 5, "EMA20<EMA50")
        bump(ema50 is not None and ema200 is not None and float(ema50) < float(ema200), 4, "EMA50<EMA200")

    # --- Momentum (up to ~25 pts) ---
    if macd_hist is not None:
        bump((side == "buy" and float(macd_hist) > 0) or (side != "buy" and float(macd_hist) < 0), 7, "MACD hist confirms")
    if rsi14 is not None:
        r = float(rsi14)
        # Prefer healthy RSI (avoid too oversold/overbought for momentum entries)
        if side == "buy":
            bump(48 <= r <= 70, 8, "RSI in momentum zone")
            penalize(r > 75, 6, "RSI overbought risk")
            penalize(r < 35, 4, "RSI weak / falling knife risk")
        else:
            bump(30 <= r <= 52, 8, "RSI in short zone")
            penalize(r < 25, 6, "RSI oversold risk")
            penalize(r > 70, 4, "RSI too strong for short")
    if adx14 is not None and di_plus is not None and di_minus is not None:
        a = float(adx14)
        if side == "buy":
            bump(a >= 18 and float(di_plus) > float(di_minus), 10, "ADX trend strength + DI+>DI-")
        else:
            bump(a >= 18 and float(di_minus) > float(di_plus), 10, "ADX trend strength + DI->DI+")
        penalize(a < 12, 5, "weak trend (ADX low)")

    if stoch_k is not None and stoch_d is not None:
        k = float(stoch_k); d = float(stoch_d)
        if side == "buy":
            bump(k > d and k >= 50, 4, "stoch bullish")
            penalize(k > 85, 3, "stoch overbought")
        else:
            bump(k < d and k <= 50, 4, "stoch bearish")
            penalize(k < 15, 3, "stoch oversold")

    # --- Volume/participation (up to ~15 pts) ---
    bump(bool(vol_spike), 6, "volume spike")
    if obv_slope is not None:
        bump((side == "buy" and float(obv_slope) > 0) or (side != "buy" and float(obv_slope) < 0), 5, "OBV confirms")
    if vwap20 is not None:
        bump((side == "buy" and price > float(vwap20)) or (side != "buy" and price < float(vwap20)), 4, "price vs VWAP20")

    # --- Volatility / quality gates (penalties only) ---
    if atr_pct is not None:
        ap = float(atr_pct)
        penalize(ap > 8, 7, "too volatile (ATR%)")
        penalize(ap < 0.6, 4, "too quiet (ATR%)")

    if bb_pct_b is not None:
        pb = float(bb_pct_b)
        # For longs, avoid buying at extreme upper band unless it's a breakout near highs.
        if side == "buy":
            penalize(pb > 0.95 and not near_high20, 6, "extended near upper BB")
        else:
            penalize(pb < 0.05 and not near_high20, 6, "extended near lower BB")

    # --- Breakout bonus ---
    bump(bool(near_high20), 6, "near 20D high (breakout context)")

    score = _clamp(score, 0.0, 100.0)
    return int(round(score)), reasons, f


def should_alert(symbol: str, side: str, min_score: int = 70) -> Tuple[bool, int, List[str], Dict[str, Any]]:
    score, reasons, features = score_signal(symbol, side)
    ok = score >= int(min_score)
    return ok, score, reasons, features


def _safe_json_extract(text: str) -> dict:
    text = (text or "").strip()
    if not text:
        return {}
    # try direct json
    try:
        return json.loads(text)
    except Exception:
        pass
    # find first {...} block
    m = re.search(r"\{.*\}", text, flags=re.S)
    if m:
        blob = m.group(0)
        try:
            return json.loads(blob)
        except Exception:
            return {}
    return {}

def decide_signal(symbol: str, features: dict, horizon: str = "M5") -> dict:
    """Return a trading-like decision for manual execution (ENTER/SKIP).

    Uses Gemini if available (GEMINI_API_KEY set). Always falls back to deterministic scoring.
    Output:
      {
        "decision": "ENTER"|"SKIP",
        "direction": "LONG"|"SHORT"|"NEUTRAL",
        "confidence": 0-100,
        "reasons": [..],
        "risks": [..],
        "liquidity": "GOOD|OK|BAD|None",
        "spread_risk": "LOW|MED|HIGH|None",
        "model": <gemini model or None>,
      }
    """
    horizon = (horizon or "M5").upper()
    # Deterministic baseline
    base_side = "buy"
    # Map current direction hint from features if exists
    hint_bias = (features.get("pattern_bias") or "").upper()
    if hint_bias == "BEAR":
        base_side = "sell"
    score, base_reasons, _ = score_signal(symbol, side=base_side)

    liq = features.get("liquidity")
    spread_risk = features.get("spread_risk")

    # If no Gemini key, return deterministic decision
    has_key = bool((os.getenv("GEMINI_API_KEY") or "").strip())
    if not has_key:
        decision = "ENTER" if score >= 75 and spread_risk != "HIGH" else "SKIP"
        direction = "LONG" if base_side == "buy" else "SHORT"
        conf = int(max(0, min(100, score)))
        return {
            "decision": decision,
            "direction": direction,
            "confidence": conf,
            "reasons": base_reasons[:3],
            "risks": (["Spread/Liquidity risk"] if spread_risk == "HIGH" else []),
            "liquidity": liq,
            "spread_risk": spread_risk,
            "model": None,
        }

    # Gemini: ask for UP/DOWN/NEUTRAL plus reasons/risks
    g = gemini_predict_direction(symbol, features, horizon=horizon)
    raw = (g.get("raw") or "").strip()
    parsed = _safe_json_extract(raw)
    direction_map = {"UP": "LONG", "DOWN": "SHORT", "NEUTRAL": "NEUTRAL"}
    gdir = direction_map.get(str(parsed.get("direction","")).upper(), "NEUTRAL")
    conf = parsed.get("confidence")
    try:
        conf = int(float(conf))
    except Exception:
        conf = None

    reasons = parsed.get("reasons") if isinstance(parsed.get("reasons"), list) else []
    risks = parsed.get("risks") if isinstance(parsed.get("risks"), list) else []

    # Decision rule: need decent confidence + not high spread risk
    if conf is None:
        conf = int(score)
    if spread_risk == "HIGH":
        decision = "SKIP"
        risks = (risks or []) + ["High spread/liquidity risk"]
    else:
        decision = "ENTER" if (conf >= 70 and gdir in ("LONG","SHORT")) else "SKIP"

    # Blend deterministic reasons (keeps stability)
    blended_reasons = []
    for r in reasons[:3]:
        if isinstance(r, str) and r.strip():
            blended_reasons.append(r.strip())
    for r in base_reasons:
        if len(blended_reasons) >= 3:
            break
        if r not in blended_reasons:
            blended_reasons.append(r)

    return {
        "decision": decision,
        "direction": gdir if gdir != "NEUTRAL" else ("LONG" if base_side=="buy" else "SHORT"),
        "confidence": int(max(0, min(100, conf))),
        "reasons": blended_reasons[:3],
        "risks": [x for x in risks[:3] if isinstance(x,str)],
        "liquidity": liq,
        "spread_risk": spread_risk,
        "model": g.get("model"),
    }
