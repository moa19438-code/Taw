
from __future__ import annotations

import os
from typing import Any, Dict, Tuple, List
from datetime import datetime, timezone

from core.alpaca_client import account
from core.storage import get_setting, set_setting


def _get_float_env(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)).strip())
    except Exception:
        return float(default)


def check_drawdown_and_pause() -> Tuple[bool, Dict[str, Any], List[str]]:
    """حماية رأس المال:
    - نحسب Equity الحالي من Alpaca
    - نحدث أعلى قيمة (High Watermark) محفوظة في settings
    - إذا تجاوز السحب (Drawdown) نسبة MAX_DRAWDOWN_PCT → نوقف الإشارات تلقائيًا

    مفاتيح:
      MAX_DRAWDOWN_PCT (default 10)
      TRADING_PAUSED (setting) = "1" / "0"
      EQUITY_HWM (setting) = float
    """
    max_dd = _get_float_env("MAX_DRAWDOWN_PCT", 10.0)
    reasons: List[str] = []
    meta: Dict[str, Any] = {"max_drawdown_pct": max_dd}

    try:
        acc = account() or {}
        eq = float(acc.get("equity") or acc.get("last_equity") or 0.0)
        meta["equity"] = eq
    except Exception as e:
        # If account call fails, don't block trading
        return False, {"error": "account_unavailable", "message": str(e)}, []

    try:
        paused = (get_setting("TRADING_PAUSED", "0") or "0").strip() in {"1", "true", "on", "yes"}
        hwm_raw = (get_setting("EQUITY_HWM", "") or "").strip()
        hwm = float(hwm_raw) if hwm_raw else eq
        if eq > hwm:
            hwm = eq
            set_setting("EQUITY_HWM", str(round(hwm, 6)))
        meta["equity_hwm"] = hwm
        if hwm <= 0:
            return False, meta, []
        dd = (hwm - eq) / hwm * 100.0
        meta["drawdown_pct"] = dd

        if dd >= max_dd:
            if not paused:
                set_setting("TRADING_PAUSED", "1")
                set_setting("TRADING_PAUSED_TS", datetime.now(timezone.utc).isoformat())
            reasons.append(f"تم إيقاف التداول تلقائيًا: السحب {dd:.1f}% تجاوز الحد {max_dd:.1f}%")
            return True, meta, reasons

        # if below threshold, do not auto-unpause. Keep manual control.
        return paused, meta, (["التداول موقوف يدويًا"] if paused else [])
    except Exception:
        return False, meta, []
