from __future__ import annotations

from typing import Any, Dict, Optional


def _to_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def normalize_features(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize (flatten) a feature dictionary into a consistent lower_snake_case schema.

    - Preserves original keys (so existing callers don't break)
    - Adds normalized keys used by scoring/backtesting layers

    Expected normalized keys (when available):
      price,
      ema20/ema50/ema100/ema200,
      sma20/sma50/sma100/sma200,
      rsi14, atr14, atr_pct,
      macd_hist,
      bb_pct_b,
      adx14, di_plus, di_minus,
      stoch_k, stoch_d,
      vwap20,
      vol_spike, obv,
      near_high20,
      w_close, w_ema20, w_ema50, w_rsi14
    """

    if not isinstance(raw, dict):
        return {"error": "features_not_a_dict"}

    f: Dict[str, Any] = dict(raw)  # preserve originals

    # Price
    if "price" in raw:
        f["price"] = _to_float(raw.get("price"))

    # MAs (daily)
    for n in (20, 50, 100, 200):
        cap = f"EMA{n}"
        low = f"ema{n}"
        if low not in f:
            f[low] = _to_float(raw.get(low))
        if f.get(low) is None and cap in raw:
            f[low] = _to_float(raw.get(cap))

        scap = f"SMA{n}"
        slow = f"sma{n}"
        if slow not in f:
            f[slow] = _to_float(raw.get(slow))
        if f.get(slow) is None and scap in raw:
            f[slow] = _to_float(raw.get(scap))

    # RSI / ATR
    if "rsi14" not in f:
        f["rsi14"] = _to_float(raw.get("rsi14"))
    if f.get("rsi14") is None and "RSI14" in raw:
        f["rsi14"] = _to_float(raw.get("RSI14"))

    if "atr14" not in f:
        f["atr14"] = _to_float(raw.get("atr14"))
    if f.get("atr14") is None and "ATR14" in raw:
        f["atr14"] = _to_float(raw.get("ATR14"))

    # ATR% (scanner may return 'ATR%' as percent)
    if "atr_pct" not in f:
        f["atr_pct"] = _to_float(raw.get("atr_pct"))
    if f.get("atr_pct") is None:
        atrpct_percent = _to_float(raw.get("ATR%"))
        if atrpct_percent is not None:
            f["atr_pct"] = atrpct_percent / 100.0

    # Weekly confirmation (optional)
    if "w_close" not in f:
        f["w_close"] = _to_float(raw.get("w_close"))
    if f.get("w_close") is None and "W_CLOSE" in raw:
        f["w_close"] = _to_float(raw.get("W_CLOSE"))

    if "w_ema20" not in f:
        f["w_ema20"] = _to_float(raw.get("w_ema20"))
    if f.get("w_ema20") is None and "W_EMA20" in raw:
        f["w_ema20"] = _to_float(raw.get("W_EMA20"))

    if "w_ema50" not in f:
        f["w_ema50"] = _to_float(raw.get("w_ema50"))
    if f.get("w_ema50") is None and "W_EMA50" in raw:
        f["w_ema50"] = _to_float(raw.get("W_EMA50"))

    if "w_rsi14" not in f:
        f["w_rsi14"] = _to_float(raw.get("w_rsi14"))
    if f.get("w_rsi14") is None and "W_RSI14" in raw:
        f["w_rsi14"] = _to_float(raw.get("W_RSI14"))

    # MACD (scanner may return MACD dict)
    if "macd_hist" not in f:
        f["macd_hist"] = _to_float(raw.get("macd_hist"))
    if f.get("macd_hist") is None:
        macd_obj = raw.get("MACD")
        if isinstance(macd_obj, dict):
            f["macd_hist"] = _to_float(macd_obj.get("hist"))

    # Bollinger pct_b
    if "bb_pct_b" not in f:
        f["bb_pct_b"] = _to_float(raw.get("bb_pct_b"))
    if f.get("bb_pct_b") is None:
        bb_obj = raw.get("Bollinger")
        if isinstance(bb_obj, dict):
            f["bb_pct_b"] = _to_float(bb_obj.get("pct_b"))

    # ADX / DI
    if "adx14" not in f:
        f["adx14"] = _to_float(raw.get("adx14"))
    if "di_plus" not in f:
        f["di_plus"] = _to_float(raw.get("di_plus"))
    if "di_minus" not in f:
        f["di_minus"] = _to_float(raw.get("di_minus"))

    if (f.get("adx14") is None) or (f.get("di_plus") is None) or (f.get("di_minus") is None):
        adx_obj = raw.get("ADX14")
        if isinstance(adx_obj, dict):
            if f.get("adx14") is None:
                f["adx14"] = _to_float(adx_obj.get("adx"))
            if f.get("di_plus") is None:
                f["di_plus"] = _to_float(adx_obj.get("+di"))
            if f.get("di_minus") is None:
                f["di_minus"] = _to_float(adx_obj.get("-di"))

    # Stochastic
    if "stoch_k" not in f:
        f["stoch_k"] = _to_float(raw.get("stoch_k"))
    if "stoch_d" not in f:
        f["stoch_d"] = _to_float(raw.get("stoch_d"))
    if (f.get("stoch_k") is None) or (f.get("stoch_d") is None):
        st = raw.get("Stochastic")
        if isinstance(st, dict):
            if f.get("stoch_k") is None:
                f["stoch_k"] = _to_float(st.get("%K"))
            if f.get("stoch_d") is None:
                f["stoch_d"] = _to_float(st.get("%D"))

    # VWAP
    if "vwap20" not in f:
        f["vwap20"] = _to_float(raw.get("vwap20"))
    if f.get("vwap20") is None and "VWAP20" in raw:
        f["vwap20"] = _to_float(raw.get("VWAP20"))

    # Volume / OBV
    if "vol_spike" not in f:
        if "vol_spike" in raw:
            f["vol_spike"] = bool(raw.get("vol_spike"))
        elif "Vol spike" in raw:
            f["vol_spike"] = bool(raw.get("Vol spike"))
        else:
            f["vol_spike"] = None

    if "obv" not in f:
        f["obv"] = _to_float(raw.get("obv"))
    if f.get("obv") is None and "OBV" in raw:
        f["obv"] = _to_float(raw.get("OBV"))

    # Near 20D high
    if "near_high20" not in f:
        if "near_high20" in raw:
            f["near_high20"] = bool(raw.get("near_high20"))
        elif "Near 20D high" in raw:
            f["near_high20"] = bool(raw.get("Near 20D high"))
        else:
            f["near_high20"] = None

    return f
