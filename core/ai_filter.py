from __future__ import annotations

import os
import json
import re
from typing import Any, Dict, List, Tuple

from core.scanner import get_symbol_features
from core.features_store import normalize_features
from core.market_regime import get_market_regime
from core.ai_analyzer import gemini_predict_direction


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def score_signal(symbol: str, side: str = "buy") -> Tuple[int, List[str], Dict[str, Any]]:
    """Deterministic scoring model to *filter* signals.

    Returns:
        (score 0-100, reasons, normalized_features)
    """

    side = (side or "buy").lower().strip()

    # Always normalize to avoid silent scoring bugs (mismatched feature keys)
    f = normalize_features(get_symbol_features(symbol))
    if f.get("error"):
        return 0, [f"data_error: {f['error']}"], f

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

    # Weekly confirmation (optional for 1D signals)
    w_close = f.get("w_close")
    w_ema20 = f.get("w_ema20")
    w_ema50 = f.get("w_ema50")
    w_rsi14 = f.get("w_rsi14")

    score = 50.0
    reasons: List[str] = []

    def bump(cond: bool, pts: float, why: str) -> None:
        nonlocal score
        if cond:
            score += pts
            reasons.append(f"+{pts:g} {why}")

    def penalize(cond: bool, pts: float, why: str) -> None:
        nonlocal score
        if cond:
            score -= pts
            reasons.append(f"-{pts:g} {why}")

    # --- Market regime (index filter) ---
    try:
        reg = get_market_regime()
        f["market_risk"] = reg.get("risk")
        f["market_reason"] = reg.get("reason")
        penalize(reg.get("risk") == "OFF", 18, "Market risk OFF (SPY weak)")
    except Exception:
        # If market filter fails, we continue without it.
        pass

    # --- Weekly confirmation (designed for 1D signals) ---
    try:
        if w_close is not None and w_ema20 is not None and w_ema50 is not None:
            if side == "buy":
                bump(float(w_close) > float(w_ema20) and float(w_ema20) > float(w_ema50), 6, "Weekly trend aligned")
                penalize(float(w_close) < float(w_ema50), 12, "Weekly trend against (avoid long)")
            else:
                bump(float(w_close) < float(w_ema20) and float(w_ema20) < float(w_ema50), 6, "Weekly downtrend aligned")
                penalize(float(w_close) > float(w_ema50), 12, "Weekly trend against (avoid short)")

        if w_rsi14 is not None:
            wr = float(w_rsi14)
            if side == "buy":
                penalize(wr < 45, 4, "Weekly RSI weak")
            else:
                penalize(wr > 55, 4, "Weekly RSI strong")
    except Exception:
        pass

    # --- Trend (up to ~25 pts) ---
    if side == "buy":
        bump(ema20 is not None and price > float(ema20), 6, "price>EMA20")
        bump(ema50 is not None and price > float(ema50), 6, "price>EMA50")
        bump(ema200 is not None and price > float(ema200), 4, "price>EMA200")
        bump(ema20 is not None and ema50 is not None and float(ema20) > float(ema50), 5, "EMA20>EMA50")
        bump(ema50 is not None and ema200 is not None and float(ema50) > float(ema200), 4, "EMA50>EMA200")
    else:
        bump(ema20 is not None and price < float(ema20), 6, "price<EMA20")
        bump(ema50 is not None and price < float(ema50), 6, "price<EMA50")
        bump(ema200 is not None and price < float(ema200), 4, "price<EMA200")
        bump(ema20 is not None and ema50 is not None and float(ema20) < float(ema50), 5, "EMA20<EMA50")
        bump(ema50 is not None and ema200 is not None and float(ema50) < float(ema200), 4, "EMA50<EMA200")

    # --- Momentum (up to ~25 pts) ---
    if macd_hist is not None:
        bump(
            (side == "buy" and float(macd_hist) > 0) or (side != "buy" and float(macd_hist) < 0),
            7,
            "MACD hist confirms",
        )

    if rsi14 is not None:
        r = float(rsi14)
        if side == "buy":
            bump(48 <= r <= 70, 8, "RSI in momentum zone")
            penalize(r > 75, 6, "RSI overbought risk")
            penalize(r < 35, 4, "RSI weak / falling knife risk")
        else:
            bump(30 <= r <= 52, 8, "RSI in short zone")
            penalize(r < 25, 6, "RSI oversold risk")
            penalize(r > 70, 4, "RSI too strong for short")

    # --- Entry quality: avoid late / overextended entries on 1D ---
    if ema20 is not None and price and atr_pct is not None:
        try:
            ext = (price / float(ema20)) - 1.0
            # Allow more room if breakout context is strong
            limit = 0.025 + (0.010 if near_high20 else 0.0) + (0.005 if vol_spike else 0.0)
            penalize(side == "buy" and ext > limit, 6, "overextended vs EMA20 (late entry)")
            penalize(side != "buy" and ext < -limit, 6, "overextended vs EMA20 (late entry)")
        except Exception:
            pass

    # --- Trend strength ---
    if adx14 is not None and di_plus is not None and di_minus is not None:
        try:
            a = float(adx14)
            if side == "buy":
                bump(a >= 18 and float(di_plus) > float(di_minus), 10, "ADX trend strength + DI+>DI-")
            else:
                bump(a >= 18 and float(di_minus) > float(di_plus), 10, "ADX trend strength + DI->DI+")
            penalize(a < 12, 5, "weak trend (ADX low)")
        except Exception:
            pass

    # --- Stochastic ---
    if stoch_k is not None and stoch_d is not None:
        try:
            k = float(stoch_k)
            d = float(stoch_d)
            if side == "buy":
                bump(k > d and k >= 50, 4, "stoch bullish")
                penalize(k > 85, 3, "stoch overbought")
            else:
                bump(k < d and k <= 50, 4, "stoch bearish")
                penalize(k < 15, 3, "stoch oversold")
        except Exception:
            pass

    # --- Chop filter (1D): avoid low-trend environments unless breakout is clear ---
    if adx14 is not None:
        try:
            a = float(adx14)
            penalize(a < 15 and not near_high20 and not vol_spike, 7, "chop filter (ADX low, no breakout)")
        except Exception:
            pass

    # --- Volume/participation (up to ~15 pts) ---
    bump(bool(vol_spike), 6, "volume spike")
    if obv_slope is not None:
        try:
            bump(
                (side == "buy" and float(obv_slope) > 0) or (side != "buy" and float(obv_slope) < 0),
                5,
                "OBV confirms",
            )
        except Exception:
            pass

    if vwap20 is not None:
        try:
            bump((side == "buy" and price > float(vwap20)) or (side != "buy" and price < float(vwap20)), 4, "price vs VWAP20")
        except Exception:
            pass

    # --- Volatility / quality gates (penalties only) ---
    if atr_pct is not None:
        try:
            ap = float(atr_pct)
            penalize(ap > 0.08, 7, "too volatile (ATR%)")  # atr_pct is fraction
            penalize(ap < 0.006, 4, "too quiet (ATR%)")
        except Exception:
            pass

    if bb_pct_b is not None:
        try:
            pb = float(bb_pct_b)
            if side == "buy":
                penalize(pb > 0.95 and not near_high20, 6, "extended near upper BB")
            else:
                penalize(pb < 0.05 and not near_high20, 6, "extended near lower BB")
        except Exception:
            pass

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
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, flags=re.S)
    if m:
        blob = m.group(0)
        try:
            return json.loads(blob)
        except Exception:
            return {}
    return {}


def decide_signal(symbol: str, features: dict, horizon: str = "M5") -> dict:
    """Return a manual decision suggestion (ENTER/SKIP).

    Uses Gemini if available (GEMINI_API_KEY set). Always falls back to deterministic scoring.
    """

    horizon = (horizon or "M5").upper()

    base_side = "buy"
    hint_bias = (features.get("pattern_bias") or "").upper()
    if hint_bias == "BEAR":
        base_side = "sell"

    score, base_reasons, _ = score_signal(symbol, side=base_side)

    liq = features.get("liquidity")
    spread_risk = features.get("spread_risk")

    has_key = bool((os.getenv("GEMINI_API_KEY") or "").strip())
    if not has_key:
        decision = "ENTER" if score >= 75 and spread_risk != "HIGH" else "SKIP"
        direction = "LONG" if base_side == "buy" else "SHORT"
        conf = int(max(0, min(100, score)))
        return {
            "decision": decision,
            "direction": direction,
            "confidence": conf,
            "reasons": base_reasons,
            "risks": [],
            "liquidity": liq,
            "spread_risk": spread_risk,
            "model": None,
            "horizon": horizon,
        }

    # Gemini layer: ask for a meta-evaluation (never sole source of truth)
    ai = gemini_predict_direction(symbol, features)
    ai_json = _safe_json_extract(ai.get("raw") if isinstance(ai, dict) else str(ai))

    ai_conf = None
    ai_dir = None
    if isinstance(ai_json, dict):
        ai_conf = ai_json.get("confidence")
        ai_dir = ai_json.get("direction")

    decision = "ENTER" if score >= 78 and spread_risk != "HIGH" else "SKIP"
    direction = "LONG" if base_side == "buy" else "SHORT"

    # If AI strongly disagrees, downgrade decision
    try:
        if isinstance(ai_dir, str) and ai_dir.upper() in {"LONG", "SHORT", "NEUTRAL"}:
            if ai_dir.upper() == "NEUTRAL":
                decision = "SKIP"
                base_reasons.append("AI: neutral")
            else:
                if (ai_dir.upper() == "LONG" and base_side != "buy") or (ai_dir.upper() == "SHORT" and base_side == "buy"):
                    decision = "SKIP" if score < 90 else decision
                    base_reasons.append("AI: disagrees")
        if ai_conf is not None:
            base_reasons.append(f"AI confidence: {ai_conf}")
    except Exception:
        pass

    return {
        "decision": decision,
        "direction": direction,
        "confidence": int(max(0, min(100, score))),
        "reasons": base_reasons,
        "risks": [],
        "liquidity": liq,
        "spread_risk": spread_risk,
        "model": ai.get("model") if isinstance(ai, dict) else None,
        "horizon": horizon,
    }
