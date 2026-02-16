from __future__ import annotations

import json
import math
from typing import Dict, Any, Tuple

# Lightweight online logistic model (no external deps)
# We store weights in storage.settings as JSON.

DEFAULT_FEATURE_KEYS = [
    "trend_strength",   # -1..+1
    "macd_hist",        # roughly -1..+1
    "rsi_norm",         # 0..1
    "adx_norm",         # 0..1
    "vol_spike",        # 0..1+
    "obv_slope",        # -1..+1
    "vwap_pos",         # -1..+1
    "atr_pct",          # 0..1 (scaled)
    "bb_pos",           # 0..1
]

def _sigmoid(x: float) -> float:
    # stable sigmoid
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)

def default_weights() -> Dict[str, float]:
    # Bias + small initial weights (encourage trend + momentum)
    w = {k: 0.0 for k in DEFAULT_FEATURE_KEYS}
    w.update({
        "bias": -0.25,
        "trend_strength": 0.60,
        "macd_hist": 0.55,
        "rsi_norm": 0.20,
        "adx_norm": 0.35,
        "vol_spike": 0.10,
        "obv_slope": 0.25,
        "vwap_pos": 0.25,
        "atr_pct": -0.15,  # penalize too much vol
        "bb_pos": -0.10,   # penalize over-extension
    })
    return w

def parse_weights(s: str | None) -> Dict[str, float]:
    if not s:
        return default_weights()
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            # ensure all keys exist
            base = default_weights()
            for k, v in obj.items():
                if isinstance(v, (int, float)):
                    base[k] = float(v)
            return base
    except Exception:
        pass
    return default_weights()

def dumps_weights(w: Dict[str, float]) -> str:
    return json.dumps({k: float(v) for k, v in w.items()}, ensure_ascii=False)

def featurize(features: Dict[str, Any]) -> Dict[str, float]:
    # Expecting keys from ai_filter.should_alert() "features"
    # Normalize / clamp to sane ranges.
    def clamp(v: float, lo: float, hi: float) -> float:
        return lo if v < lo else hi if v > hi else v

    trend_strength = float(features.get("trend_score", 0.0))  # -1..+1
    macd_hist = float(features.get("macd_hist", 0.0))
    rsi = float(features.get("rsi", 50.0))
    adx = float(features.get("adx", 20.0))
    vol_spike = float(features.get("vol_spike", 0.0))
    obv_slope = float(features.get("obv_slope", 0.0))
    vwap_pos = float(features.get("vwap_pos", 0.0))
    atr_pct = float(features.get("atr_pct", 2.0))
    bb_pos = float(features.get("bb_pos", 0.5))

    x = {
        "trend_strength": clamp(trend_strength, -1.0, 1.0),
        "macd_hist": clamp(macd_hist, -1.0, 1.0),
        "rsi_norm": clamp((rsi - 30.0) / 40.0, 0.0, 1.0),  # 30->0, 70->1
        "adx_norm": clamp((adx - 15.0) / 25.0, 0.0, 1.0),  # 15->0, 40->1
        "vol_spike": clamp(vol_spike / 2.0, 0.0, 2.0),     # 2x vol -> 1
        "obv_slope": clamp(obv_slope, -1.0, 1.0),
        "vwap_pos": clamp(vwap_pos, -1.0, 1.0),
        "atr_pct": clamp(atr_pct / 8.0, 0.0, 1.5),         # 8% -> 1
        "bb_pos": clamp(bb_pos, 0.0, 1.0),
    }
    return x

def predict_prob(x: Dict[str, float], w: Dict[str, float]) -> float:
    s = float(w.get("bias", 0.0))
    for k in DEFAULT_FEATURE_KEYS:
        s += float(w.get(k, 0.0)) * float(x.get(k, 0.0))
    return _sigmoid(s)

def update_online(w: Dict[str, float], x: Dict[str, float], label: int, lr: float = 0.15) -> Dict[str, float]:
    # SGD step on log-loss: w += lr*(y - p)*x
    y = 1.0 if int(label) == 1 else 0.0
    p = predict_prob(x, w)
    g = (y - p)
    w["bias"] = float(w.get("bias", 0.0)) + lr * g
    for k in DEFAULT_FEATURE_KEYS:
        w[k] = float(w.get(k, 0.0)) + lr * g * float(x.get(k, 0.0))
    return w
