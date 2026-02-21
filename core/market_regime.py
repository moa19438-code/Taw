from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, List
import threading

from core.alpaca_client import bars
from core.indicators import ema

_lock = threading.Lock()
_cache: Dict[str, Any] = {"ts": None, "regime": None}

def _utcnow() -> datetime:
    return datetime.now(timezone.utc)

def get_market_regime(ttl_sec: int = 300) -> Dict[str, Any]:
    """Compute a simple US market regime using SPY daily trend.

    Returns dict:
      risk: 'ON' | 'OFF' | 'UNK'
      reason: short text
      ts: ISO timestamp
    """
    with _lock:
        ts = _cache.get("ts")
        if ts and ( _utcnow().timestamp() - float(ts) ) < ttl_sec and _cache.get("regime"):
            return dict(_cache["regime"])

    try:
        end = _utcnow()
        start = end - timedelta(days=220)
        data = bars(["SPY"], start=start, end=end, timeframe="1Day", limit=220)
        # alpaca_client.bars returns {"bars": {"SPY": [...]}}; keep backward compatibility
        if isinstance(data, dict):
            if "bars" in data and isinstance(data.get("bars"), dict):
                blist = data.get("bars", {}).get("SPY") or []
            else:
                blist = data.get("SPY") or []
        else:
            blist = []
        closes: List[float] = []
        for b in blist:
            c = b.get("c") if isinstance(b, dict) else None
            if c is not None:
                closes.append(float(c))
        if len(closes) < 60:
            reg = {"risk":"UNK","reason":"بيانات غير كافية","ts":_utcnow().isoformat()}
        else:
            e20 = ema(closes, 20)[-1]
            e50 = ema(closes, 50)[-1]
            last = closes[-1]
            risk_on = (last >= e50) and (e20 >= e50)
            reg = {
                "risk": "ON" if risk_on else "OFF",
                "reason": "SPY فوق EMA50 و EMA20>=EMA50" if risk_on else "SPY تحت EMA50 أو EMA20<EMA50",
                "ts": _utcnow().isoformat(),
                "spy": {"last": last, "ema20": e20, "ema50": e50},
            }
    except Exception:
        reg = {"risk":"UNK","reason":"تعذر حساب وضع السوق","ts":_utcnow().isoformat()}

    with _lock:
        _cache["ts"] = _utcnow().timestamp()
        _cache["regime"] = dict(reg)
    return reg
