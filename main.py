from __future__ import annotations

from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
from typing import Any, Dict, List, Optional, Tuple
import os
import json
import time

import requests
import atexit
import traceback
import threading
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
from flask import Flask, request, jsonify
from ai_analyzer import gemini_analyze
from ai_filter import should_alert
from ml_model import parse_weights, dumps_weights, featurize, predict_prob, update_online
from executor import trade_symbol
from alpaca_client import bars
from backtesting import run_backtest_symbol

from config import (
    RUN_KEY,
    TELEGRAM_BOT_TOKEN,
    TELEGRAM_CHAT_ID,
    TELEGRAM_ADMIN_ID,
    TELEGRAM_CHANNEL_ID,
    SEND_DAILY_SUMMARY,
    LOCAL_TZ,
    TRADINGVIEW_WEBHOOK_KEY,
    AI_FILTER_ENABLED,
    AI_FILTER_MIN_SCORE,
    AI_FILTER_SEND_REJECTS,
    SIGNAL_EVAL_DAYS,
    ML_ENABLED,
    ML_LEARNING_RATE,
)
from storage import (
    init_db,
    ensure_default_settings,
    last_orders,
    log_scan,
    last_scans,
    get_all_settings,
    set_setting,
    parse_int,
    parse_float,
    parse_bool,
    last_signal,
    log_signal,
    pending_signals_for_eval,
    mark_signal_evaluated,
    last_signals,
)
from scanner import scan_universe_with_meta, Candidate, get_symbol_features

app = Flask(__name__)

@app.get("/health")
def health():
    return jsonify({"ok": True, "service": "taw-bot"})

# ===== تنفيذ مهام ثقيلة بدون تعطيل webhook =====
def _run_async(fn, *args, **kwargs):
    t = threading.Thread(target=fn, args=args, kwargs=kwargs, daemon=True)
    t.start()

init_db()
ensure_default_settings()


# ================= Telegram helpers =================
def _tg_send(chat_id: str, text: str, reply_markup: Optional[Dict[str, Any]] = None, silent: bool = False) -> None:
    if not (TELEGRAM_BOT_TOKEN and chat_id):
        return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload: Dict[str, Any] = {"chat_id": chat_id, "text": text, "disable_notification": bool(silent)}
        if reply_markup:
            payload["reply_markup"] = reply_markup
        requests.post(url, json=payload, timeout=5)
    except Exception:
        pass





def _tg_send_async(chat_id: str, text: str, reply_markup: Optional[Dict[str, Any]] = None, silent: bool = False) -> None:
    """Fire-and-forget Telegram send to keep webhook responses snappy."""
    _run_async(_tg_send, chat_id, text, reply_markup, silent)


# --- Telegram callback responsiveness / anti-duplicate ---
_CB_SEEN: Dict[str, float] = {}  # callback_query.id -> ts
_ACTION_SEEN: Dict[str, float] = {}  # f"{chat_id}:{action}" -> ts
_CB_TTL_SEC = int(os.getenv('TG_CB_TTL_SEC', '600'))  # 10 minutes default
_ACTION_DEBOUNCE_SEC = float(os.getenv('TG_ACTION_DEBOUNCE_SEC', '2.5'))

def _tg_answer_callback(callback_id: Optional[str], text: Optional[str] = None, show_alert: bool = False) -> None:
    """Acknowledge Telegram inline button click quickly to avoid retries/spinner."""
    if not (TELEGRAM_BOT_TOKEN and callback_id):
        return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/answerCallbackQuery"
        payload: Dict[str, Any] = {"callback_query_id": callback_id, "show_alert": bool(show_alert)}
        if text:
            payload["text"] = text
        requests.post(url, json=payload, timeout=10)
    except Exception:
        pass

def _seen_and_mark(d: Dict[str, float], key: str, ttl_sec: float) -> bool:
    """Return True if key was seen recently; otherwise mark and return False."""
    now = time.time()
    # cheap cleanup
    if len(d) > 2000:
        for k, ts in list(d.items())[:1000]:
            if now - ts > ttl_sec:
                d.pop(k, None)
    ts = d.get(key)
    if ts is not None and (now - ts) < ttl_sec:
        return True
    d[key] = now
    return False

def send_telegram(text: str, reply_markup: Optional[Dict[str, Any]] = None) -> None:
    """Send *notifications* according to routing settings.

    NOTIFY_ROUTE: dm|group|both
    NOTIFY_SILENT: 1/0 (disable push notifications)
    """
    settings = _settings()
    route = _get_str(settings, "NOTIFY_ROUTE", "dm").lower().strip()
    silent = _get_bool(settings, "NOTIFY_SILENT", True)

    admin_id = str(TELEGRAM_ADMIN_ID or TELEGRAM_CHAT_ID or "").strip()
    channel_id = str(TELEGRAM_CHANNEL_ID or "").strip()

    send_dm = route in ("dm", "both")
    send_group = route in ("group", "both")

    # If channel not configured, fallback to DM
    if send_group and not channel_id:
        send_group = False
        send_dm = True

    if send_group and channel_id:
        _tg_send_async(channel_id, text, reply_markup=reply_markup, silent=silent)

    if send_dm and admin_id:
        _tg_send_async(admin_id, text, reply_markup=reply_markup, silent=silent)
def _admin_id_int() -> int:
    try:
        return int(str(TELEGRAM_ADMIN_ID).strip()) if str(TELEGRAM_ADMIN_ID).strip() else 0
    except Exception:
        return 0


def _is_admin(user_id: Optional[int]) -> bool:
    aid = _admin_id_int()
    if aid <= 0:
        # If not configured, allow (but you should set TELEGRAM_ADMIN_ID in production)
        return True
    return int(user_id or 0) == aid


# ================= Bot settings =================
def _settings() -> Dict[str, str]:
    return get_all_settings()


def _get_str(settings: Dict[str, str], k: str, default: str) -> str:
    v = settings.get(k)
    return v if (v is not None and str(v).strip() != "") else default


def _get_int(settings: Dict[str, str], k: str, default: int) -> int:
    return parse_int(settings.get(k), default)


def _get_float(settings: Dict[str, str], k: str, default: float) -> float:
    return parse_float(settings.get(k), default)


def _get_bool(settings: Dict[str, str], k: str, default: bool) -> bool:
    return parse_bool(settings.get(k), default)


# ================= Market window (Riyadh) =================
def _now_local() -> datetime:
    try:
        return datetime.now(ZoneInfo(LOCAL_TZ))
    except Exception:
        return datetime.now(timezone.utc)


def _parse_hhmm(s: str) -> Tuple[int, int]:
    try:
        hh, mm = s.strip().split(":")
        return int(hh), int(mm)
    except Exception:
        return 0, 0


def _within_notification_window(settings: Dict[str, str]) -> Tuple[bool, str]:
    """
    Window is in LOCAL_TZ (default Asia/Riyadh).
    Supports ranges crossing midnight (e.g. 17:30 -> 00:00).
    """
    now = _now_local()

    # Only weekdays (US market)
    if now.weekday() >= 5:
        return False, "Weekend"

    start_s = _get_str(settings, "WINDOW_START", "17:30")
    end_s = _get_str(settings, "WINDOW_END", "00:00")
    sh, sm = _parse_hhmm(start_s)
    eh, em = _parse_hhmm(end_s)

    t = now.time()
    start_t = t.replace(hour=sh, minute=sm, second=0, microsecond=0)
    end_t = t.replace(hour=eh, minute=em, second=0, microsecond=0)

    if (sh, sm) == (eh, em):
        return True, "Window: all day"

    if (sh, sm) < (eh, em):
        ok = (t >= start_t) and (t < end_t)
    else:
        # crosses midnight
        ok = (t >= start_t) or (t < end_t)

    return (ok, f"Window {start_s}-{end_s} {LOCAL_TZ}")


# ================= Scoring -> strength =================
_STRENGTH_RANK = {"ضعيف": 1, "متوسط": 2, "قوي": 3, "قوي جداً": 4}


def _strength(score: float) -> str:
    if score >= 8.5:
        return "قوي جداً"
    if score >= 7.0:
        return "قوي"
    if score >= 5.0:
        return "متوسط"
    return "ضعيف"


def _mode_matches(c: Candidate, mode: str) -> bool:
    mode = (mode or "daily").lower()
    if mode == "daily":
        return bool(c.daily_ok)
    if mode == "weekly":
        return bool(c.weekly_ok)
    if mode == "monthly":
        return bool(c.monthly_ok)
    if mode == "daily_weekly":
        return bool(c.daily_ok and c.weekly_ok)
    if mode == "weekly_monthly":
        return bool(c.weekly_ok and c.monthly_ok)
    return bool(c.daily_ok)


def _mode_label(mode: str) -> str:
    m = (mode or "daily").lower()
    return {
        "daily": "يومي",
        "weekly": "أسبوعي",
        "monthly": "شهري",
        "daily_weekly": "يومي + أسبوعي",
        "weekly_monthly": "أسبوعي + شهري",
    }.get(m, "يومي")


def _entry_type_label(entry_mode: str) -> str:
    em = (entry_mode or "auto").lower()
    return {"auto": "تلقائي", "market": "سوق", "limit": "محدد"}.get(em, "تلقائي")


def _compute_trade_plan(settings: Dict[str, str], c: Candidate) -> Dict[str, Any]:
    """
    خطة يدوية لتطبيق Sahm (ATR):
    - الدخول: سعر الإغلاق الأخير
    - وقف الخسارة: ATR * SL_ATR_MULT تحت الدخول
    - جني الربح: (المخاطرة R) * TP_R_MULT فوق الدخول
    - الكمية: حسب رأس المال والمخاطرة المتغيرة A+/A/B
    """
    entry = float(c.last_close)

    # إعدادات ATR
    sl_atr_mult = _get_float(settings, "SL_ATR_MULT", 2.0)
    tp_r_mult = _get_float(settings, "TP_R_MULT", 2.0)

    atr_val = float(getattr(c, "atr", 0.0) or 0.0)
    if atr_val <= 0:
        # fallback
        atr_val = max(entry * 0.01, 0.5)

    sl = max(0.01, entry - (atr_val * sl_atr_mult))
    risk_per_share = max(entry - sl, 0.01)
    tp = entry + (risk_per_share * tp_r_mult)

    # تصنيف (A+/A/B) حسب القوة
    st = _strength(float(c.score))
    if st == "قوي جداً":
        grade = "A+"
        risk_pct = _get_float(settings, "RISK_APLUS_PCT", 1.5)
    elif st == "قوي":
        grade = "A"
        risk_pct = _get_float(settings, "RISK_A_PCT", 1.0)
    else:
        grade = "B"
        risk_pct = _get_float(settings, "RISK_B_PCT", 0.5)

    capital = _get_float(settings, "CAPITAL_USD", 800.0)
    risk_amount = max(1.0, capital * (risk_pct / 100.0))

    qty_risk = int(risk_amount / risk_per_share)
    if qty_risk < 1:
        qty_risk = 1

    # حد أقصى لحجم الصفقة (كنسبة من رأس المال)
    pos_pct = _get_float(settings, "POSITION_PCT", 0.20)
    max_notional = max(0.0, capital * pos_pct)
    qty_cap = int(max_notional / max(entry, 0.01)) if max_notional > 0 else qty_risk
    if qty_cap < 1:
        qty_cap = 1

    qty = max(1, min(qty_risk, qty_cap))

    entry_mode = _get_str(settings, "ENTRY_MODE", "auto").lower()

    rr = (tp - entry) / max(entry - sl, 0.01)

    return {
        "entry": round(entry, 2),
        "sl": round(sl, 2),
        "tp": round(tp, 2),
        "qty": int(qty),
        "atr": round(atr_val, 2),
        "sl_atr_mult": sl_atr_mult,
        "tp_r_mult": tp_r_mult,
        "risk_pct": round(risk_pct, 2),
        "risk_amount": round(risk_amount, 2),
        "risk_per_share": round(risk_per_share, 2),
        "rr": round(rr, 2),
        "grade": grade,
        "entry_mode": entry_mode,
        "ml_prob": None,
        "ev_r": None,

    }


def _format_sahm_block(mode_label: str, c: Candidate, plan: Dict[str, Any], ai_score: int | None = None) -> str:
    strength = _strength(float(c.score))
    entry_type = _entry_type_label(plan["entry_mode"])
    # Sahm screen fields (Arabic, as requested)
    return (
        f"🚀 سهم: {c.symbol} | التصنيف: {plan.get('grade','')} | القوة: {strength} | Score: {c.score:.1f}" + (f" | AI: {ai_score}/100" if ai_score is not None else "") + (f" | ML: {int(round(float(plan.get('ml_prob') or 0)*100))}%" if plan.get('ml_prob') is not None else "") + (f" | EV(R): {float(plan.get('ev_r')):.2f}" if plan.get('ev_r') is not None else "") + "\n"
        f"العملية: شراء\n"
        f"النوع: {entry_type}\n"
        f"السعر: {plan['entry']}\n"
        f"الكمية: {plan['qty']}\n"
        f"المخاطرة: {plan.get('risk_pct',0)}% (≈ {plan.get('risk_amount',0)}$) | R/R: {plan.get('rr',0)}\n"
        f"ATR: {plan.get('atr',0)} | SL×ATR: {plan.get('sl_atr_mult',0)} | TP×R: {plan.get('tp_r_mult',0)}\n"
        f"الأمر المرفق: جني الربح/وقف الخسارة\n"
        f"جني الربح:\n"
        f"  سعر الإيقاف: {plan['tp']}\n"
        f"  سعر الأمر: {plan['tp']}\n"
        f"وقف الخسارة:\n"
        f"  سعر الإيقاف: {plan['sl']}\n"
        f"  سعر الأمر: {plan['sl']}\n"
        f"تاريخ الاستحقاق: {mode_label}\n"
        f"ملاحظة: {c.notes}\n"
    )


def _build_menu(settings: Dict[str, str]) -> Dict[str, Any]:
    mode = _get_str(settings, "PLAN_MODE", "daily")
    entry = _get_str(settings, "ENTRY_MODE", "auto")
    auto_notify = _get_bool(settings, "AUTO_NOTIFY", True)
    route = _get_str(settings, "NOTIFY_ROUTE", "dm").lower()
    silent = _get_bool(settings, "NOTIFY_SILENT", True)

    def _route_label(r: str) -> str:
        if r == "group":
            return "القروب فقط"
        if r == "both":
            return "الخاص+القروب"
        return "الخاص فقط"

    return {
        "inline_keyboard": [
            [
                {"text": "🔎 تحليل الآن", "callback_data": "do_analyze"},
                {"text": "⭐ أفضل الفرص", "callback_data": "do_top"},
            ],
            [
                {"text": f"📆 الخطة: {_mode_label(mode)}", "callback_data": "show_modes"},
                {"text": f"🎯 الدخول: {_entry_type_label(entry)}", "callback_data": "show_entry"},
            ],
            [
                {"text": f"🔔 التنبيهات: {'ON' if auto_notify else 'OFF'}", "callback_data": "toggle_notify"},
                {"text": "⚙️ الإعدادات", "callback_data": "show_settings"},
            ],
            [
                {"text": f"📨 الوجهة: {_route_label(route)}", "callback_data": "show_notify_route"},
                {"text": f"🔕 صامت: {'ON' if silent else 'OFF'}", "callback_data": "toggle_silent"},
            ],
        ]
    }

def _build_modes_kb() -> Dict[str, Any]:
    return {
        "inline_keyboard": [
            [
                {"text": "يومي", "callback_data": "set_mode:daily"},
                {"text": "أسبوعي", "callback_data": "set_mode:weekly"},
                {"text": "شهري", "callback_data": "set_mode:monthly"},
            ],
            [
                {"text": "يومي+أسبوعي", "callback_data": "set_mode:daily_weekly"},
                {"text": "أسبوعي+شهري", "callback_data": "set_mode:weekly_monthly"},
            ],
            [{"text": "⬅️ رجوع", "callback_data": "menu"}],
        ]
    }


def _build_entry_kb() -> Dict[str, Any]:
    return {
        "inline_keyboard": [
            [
                {"text": "تلقائي", "callback_data": "set_entry:auto"},
                {"text": "سوق", "callback_data": "set_entry:market"},
                {"text": "محدد", "callback_data": "set_entry:limit"},
            ],
            [{"text": "⬅️ رجوع", "callback_data": "menu"}],
        ]
    }


def _build_notify_route_kb() -> Dict[str, Any]:
    return {
        "inline_keyboard": [
            [
                {"text": "الخاص فقط", "callback_data": "set_notify_route:dm"},
                {"text": "القروب فقط", "callback_data": "set_notify_route:group"},
            ],
            [
                {"text": "الخاص + القروب", "callback_data": "set_notify_route:both"},
            ],
            [{"text": "⬅️ رجوع", "callback_data": "menu"}],
        ]
    }





def _build_settings_kb(settings: Dict[str, str]) -> Dict[str, Any]:
    auto_notify = _get_bool(settings, "AUTO_NOTIFY", True)
    allow_resend = _get_bool(settings, "ALLOW_RESEND_IF_STRONGER", True)
    return {
        "inline_keyboard": [
            [
                {"text": "💰 رأس المال", "callback_data": "show_capital"},
                {"text": "⚖️ المخاطرة", "callback_data": "show_risk"},
            ],
            [
                {"text": "⏱️ وقت الفحص", "callback_data": "show_interval"},
                {"text": "📦 حجم الصفقة", "callback_data": "show_position"},
            ],
            [
                {"text": "📉 وقف الخسارة%", "callback_data": "show_sl"},
                {"text": "📈 جني الربح%", "callback_data": "show_tp"},
            ],
            [
                {"text": "🎛 عدد الفرص", "callback_data": "show_send"},
                {"text": f"🔁 إعادة الإرسال إذا أقوى: {'نعم' if allow_resend else 'لا'}", "callback_data": "toggle_resend"},
            ],
            [
                {"text": f"🔔 التنبيهات: {'ON' if auto_notify else 'OFF'}", "callback_data": "toggle_notify"},
                {"text": "🕒 نافذة السوق", "callback_data": "show_window"},
            ],
            [{"text": "⬅️ رجوع", "callback_data": "menu"}],
        ]
    }



def _build_risk_kb(settings: Dict[str, str]) -> Dict[str, Any]:
    presets = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
    aplus = _get_float(settings, "RISK_APLUS_PCT", 1.5)
    a = _get_float(settings, "RISK_A_PCT", 1.0)
    b = _get_float(settings, "RISK_B_PCT", 0.5)

    rows: List[List[Dict[str, str]]] = []
    rows.append([
        {"text": f"A+ = {aplus}%", "callback_data": "noop"},
        {"text": f"A = {a}%", "callback_data": "noop"},
        {"text": f"B = {b}%", "callback_data": "noop"},
    ])

    rows.append([{"text": f"A+ {p}%", "callback_data": f"set_risk_aplus:{p}"} for p in presets[:3]])
    rows.append([{"text": f"A+ {p}%", "callback_data": f"set_risk_aplus:{p}"} for p in presets[3:]])
    rows.append([{"text": f"A {p}%", "callback_data": f"set_risk_a:{p}"} for p in presets[:3]])
    rows.append([{"text": f"A {p}%", "callback_data": f"set_risk_a:{p}"} for p in presets[3:]])
    rows.append([{"text": f"B {p}%", "callback_data": f"set_risk_b:{p}"} for p in presets[:3]])
    rows.append([{"text": f"B {p}%", "callback_data": f"set_risk_b:{p}"} for p in presets[3:]])

    rows.append([{"text": "⬅️ رجوع", "callback_data": "show_settings"}])
    return {"inline_keyboard": rows}


def _build_interval_kb(settings: Dict[str, str]) -> Dict[str, Any]:
    presets = [10, 15, 20, 30, 60]
    cur = _get_int(settings, "SCAN_INTERVAL_MIN", 20)
    rows: List[List[Dict[str, str]]] = []
    rows.append([{"text": f"الحالي: {cur} دقيقة", "callback_data": "noop"}])
    rows.append([{"text": f"{p} دقيقة", "callback_data": f"set_interval:{p}"} for p in presets[:3]])
    rows.append([{"text": f"{p} دقيقة", "callback_data": f"set_interval:{p}"} for p in presets[3:]])
    rows.append([{"text": "⬅️ رجوع", "callback_data": "show_settings"}])
    return {"inline_keyboard": rows}



def _build_capital_kb() -> Dict[str, Any]:
    presets = [300, 500, 800, 1000, 2000, 5000]
    rows: List[List[Dict[str, str]]] = []
    rows.append([{"text": f"{p}$", "callback_data": f"set_capital:{p}"} for p in presets[:3]])
    rows.append([{"text": f"{p}$", "callback_data": f"set_capital:{p}"} for p in presets[3:]])
    rows.append([{"text": "✍️ قيمة مخصصة", "callback_data": "set_capital_custom"}])
    rows.append([{"text": "⬅️ رجوع", "callback_data": "show_settings"}])
    return {"inline_keyboard": rows}

def _build_position_kb() -> Dict[str, Any]:
    # % of capital used per trade suggestion (manual trading)
    presets = [0.10, 0.15, 0.20, 0.25, 0.30]
    rows = []
    rows.append([{"text": f"{int(p*100)}%", "callback_data": f"set_position:{p}"} for p in presets[:3]])
    rows.append([{"text": f"{int(p*100)}%", "callback_data": f"set_position:{p}"} for p in presets[3:]])
    rows.append([{"text": "⬅️ رجوع", "callback_data": "show_settings"}])
    return {"inline_keyboard": rows}


def _build_sl_kb() -> Dict[str, Any]:
    presets = [2, 3, 4, 5]
    rows = []
    rows.append([{"text": f"{p}%", "callback_data": f"set_sl:{p}"} for p in presets[:2]])
    rows.append([{"text": f"{p}%", "callback_data": f"set_sl:{p}"} for p in presets[2:]])
    rows.append([{"text": "⬅️ رجوع", "callback_data": "show_settings"}])
    return {"inline_keyboard": rows}


def _build_tp_kb() -> Dict[str, Any]:
    # base TP for متوسط/ضعيف; قوي/قوي جداً use TP_PCT_STRONG / TP_PCT_VSTRONG
    presets = [5, 6, 7, 8, 10]
    rows = []
    rows.append([{"text": f"{p}%", "callback_data": f"set_tp:{p}"} for p in presets[:3]])
    rows.append([{"text": f"{p}%", "callback_data": f"set_tp:{p}"} for p in presets[3:]])
    rows.append([{"text": "⬅️ رجوع", "callback_data": "show_settings"}])
    return {"inline_keyboard": rows}


def _build_send_kb() -> Dict[str, Any]:
    # min,max pairs
    pairs = [(5, 7), (7, 10), (10, 15)]
    rows = []
    rows.append([{"text": f"{a}-{b}", "callback_data": f"set_send:{a}:{b}"} for a, b in pairs])
    rows.append([{"text": "⬅️ رجوع", "callback_data": "show_settings"}])
    return {"inline_keyboard": rows}


def _build_window_kb() -> Dict[str, Any]:
    # Common US market windows in Riyadh; you can change later
    presets = [("17:30", "00:00"), ("17:30", "00:30"), ("16:30", "23:30")]
    rows = []
    for a, b in presets:
        rows.append([{"text": f"{a}→{b}", "callback_data": f"set_window:{a}:{b}"}])
    rows.append([{"text": "⬅️ رجوع", "callback_data": "show_settings"}])
    return {"inline_keyboard": rows}

# ================= Core scan/notify logic =================
def _select_and_log_new_candidates(picks: List[Candidate], settings: Dict[str, str]) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    Returns:
      - blocks: list[str] formatted for Telegram
      - logged: list of dicts (symbol, strength, score, entry, sl, tp)
    """
    mode = _get_str(settings, "PLAN_MODE", "daily").lower()
    dedup_hours = _get_int(settings, "DEDUP_HOURS", 6)
    allow_resend_stronger = _get_bool(settings, "ALLOW_RESEND_IF_STRONGER", True)
    max_send = _get_int(settings, "MAX_SEND", 10)
    min_send = _get_int(settings, "MIN_SEND", 7)

    now_utc = datetime.now(timezone.utc)
    cutoff = now_utc - timedelta(hours=dedup_hours)

    mode_label = _mode_label(mode)

    # filter + sort
    candidates = [c for c in picks if _mode_matches(c, mode)]
    candidates.sort(key=lambda x: x.score, reverse=True)

    blocks: List[str] = []
    logged: List[Dict[str, Any]] = []

    for c in candidates:
        if len(blocks) >= max_send:
            break

        st = _strength(float(c.score))
        last = last_signal(c.symbol, mode)
        should_send = False

        if not last:
            should_send = True
        else:
            # check time
            try:
                last_ts = datetime.fromisoformat(str(last["ts"]).replace("Z", "+00:00"))
            except Exception:
                last_ts = datetime(1970, 1, 1, tzinfo=timezone.utc)

            if last_ts < cutoff:
                should_send = True
            else:
                if allow_resend_stronger:
                    prev_rank = _STRENGTH_RANK.get(str(last.get("strength")), 0)
                    cur_rank = _STRENGTH_RANK.get(st, 0)
                    if cur_rank > prev_rank:
                        should_send = True

        if not should_send:
            continue

        ai_score = None
        if AI_FILTER_ENABLED:
            ok_ai, ai_score, _ai_reasons, _ai_features = should_alert(c.symbol, "buy", min_score=AI_FILTER_MIN_SCORE)
            if not ok_ai:
                continue

        plan = _compute_trade_plan(settings, c)
        # ML probability / expected value (اختياري)
        try:
            if ML_ENABLED and AI_FILTER_ENABLED and (_ai_features is not None):
                w = parse_weights(settings.get("ML_WEIGHTS") or "")
                x = featurize(_ai_features)
                p = float(predict_prob(w, x))
                plan["ml_prob"] = round(p, 4)
                # Expected value in R units: p*TP_R - (1-p)*1R
                tp_r = float(plan.get("tp_r_mult") or 0.0)
                plan["ev_r"] = round((p * tp_r) - ((1.0 - p) * 1.0), 3)
        except Exception:
            pass
        blocks.append(_format_sahm_block(mode_label, c, plan, ai_score=ai_score))
        logged.append({
            "symbol": c.symbol,
            "strength": st,
            "score": float(c.score),
            "entry": float(plan["entry"]),
            "sl": float(plan["sl"]),
            "tp": float(plan["tp"]),
            "mode": mode,
            "ai_score": ai_score,
            "reasons": (_ai_reasons if AI_FILTER_ENABLED else None),
            "features": (_ai_features if AI_FILTER_ENABLED else None),
        })

    # ensure at least min_send if possible (even if repeats blocked by dedup)
    if len(blocks) < min_send:
        # fill remaining with highest-ranked not already chosen (but still avoid duplicates within this message)
        chosen = {d["symbol"] for d in logged}
        for c in candidates:
            if len(blocks) >= min_send:
                break
            if c.symbol in chosen:
                continue
            ai_score = None
            if AI_FILTER_ENABLED:
                ok_ai, ai_score, _ai_reasons, _ai_features = should_alert(c.symbol, "buy", min_score=AI_FILTER_MIN_SCORE)
                if not ok_ai:
                    continue

            plan = _compute_trade_plan(settings, c)
            try:
                if ML_ENABLED and AI_FILTER_ENABLED and (_ai_features is not None):
                    w = parse_weights(settings.get("ML_WEIGHTS") or "")
                    x = featurize(_ai_features)
                    p = float(predict_prob(w, x))
                    plan["ml_prob"] = round(p, 4)
                    tp_r = float(plan.get("tp_r_mult") or 0.0)
                    plan["ev_r"] = round((p * tp_r) - ((1.0 - p) * 1.0), 3)
            except Exception:
                pass
            st = _strength(float(c.score))
            blocks.append(_format_sahm_block(mode_label, c, plan, ai_score=ai_score))
            logged.append({
                "symbol": c.symbol,
                "strength": st,
                "score": float(c.score),
                "entry": float(plan["entry"]),
                "sl": float(plan["sl"]),
                "tp": float(plan["tp"]),
                "mode": mode,
            "ai_score": ai_score,
            "reasons": (_ai_reasons if AI_FILTER_ENABLED else None),
            "features": (_ai_features if AI_FILTER_ENABLED else None),
            })

    # persist
    ts = now_utc.isoformat()
    for d in logged:
        log_signal(ts=ts, symbol=d["symbol"], source="scan", side="buy", mode=d["mode"], strength=d["strength"], score=float(d["score"]), entry=float(d["entry"]), sl=d.get("sl"), tp=d.get("tp"), features_json=json.dumps(d.get("features") or {}, ensure_ascii=False), reasons_json=json.dumps(d.get("reasons") or [], ensure_ascii=False), horizon_days=int(_get_int(_settings(), "SIGNAL_EVAL_DAYS", SIGNAL_EVAL_DAYS)))

    return blocks, logged


def _run_scan_and_build_message(settings: Dict[str, str]) -> Tuple[str, int]:
    picks, universe_size = scan_universe_with_meta()
    blocks, _ = _select_and_log_new_candidates(picks, settings)

    if not blocks:
        return "❌ لا توجد فرص جديدة الآن.", universe_size

    header = f"📊 فرص جديدة ({_mode_label(_get_str(settings,'PLAN_MODE','daily'))})\n"
    msg = header + "\n\n".join(blocks)
    return msg, universe_size


# ================= Telegram webhook =================
@app.post("/webhook")
def telegram_webhook():
    try:
        if not TELEGRAM_BOT_TOKEN:
            return jsonify({"ok": True})

        data = request.get_json(silent=True) or {}

        # Handle button clicks
        cb = data.get("callback_query")
        if cb:
            user_id = cb.get("from", {}).get("id")
            chat_id = cb.get("message", {}).get("chat", {}).get("id")
            action = (cb.get("data") or "").strip()
            callback_id = cb.get("id")
            # IMPORTANT: acknowledge callback fast to avoid spinner/retries
            _tg_answer_callback(callback_id)

            # Dedupe: Telegram may deliver the same callback update more than once
            if callback_id and _seen_and_mark(_CB_SEEN, str(callback_id), float(_CB_TTL_SEC)):
                return jsonify({"ok": True})

            # Debounce: prevent accidental double-click from triggering twice
            if chat_id is not None and action:
                if _seen_and_mark(_ACTION_SEEN, f"{chat_id}:{action}", float(_ACTION_DEBOUNCE_SEC)):
                    return jsonify({"ok": True})


            if not _is_admin(user_id):
                _tg_send_async(str(chat_id), "⛔ هذا البوت للأدمن فقط.")
                return jsonify({"ok": True})

            settings = _settings()

            if action == "menu":
                _tg_send_async(str(chat_id), "📌 اختر:", reply_markup=_build_menu(settings))
                return jsonify({"ok": True})

            if action == "show_modes":
                _tg_send_async(str(chat_id), "📆 اختر الخطة الزمنية:", reply_markup=_build_modes_kb())
                return jsonify({"ok": True})

            if action.startswith("set_mode:"):
                mode = action.split(":", 1)[1]
                set_setting("PLAN_MODE", mode)
                settings = _settings()
                _tg_send_async(str(chat_id), f"✅ تم ضبط الخطة: {_mode_label(mode)}", reply_markup=_build_menu(settings))
                return jsonify({"ok": True})

            if action == "show_entry":
                _tg_send_async(str(chat_id), "🎯 اختر نوع الدخول:", reply_markup=_build_entry_kb())
                return jsonify({"ok": True})

            if action.startswith("set_entry:"):
                entry = action.split(":", 1)[1]
                set_setting("ENTRY_MODE", entry)
                settings = _settings()
                _tg_send_async(str(chat_id), f"✅ نوع الدخول: {_entry_type_label(entry)}", reply_markup=_build_menu(settings))
                return jsonify({"ok": True})

            if action == "toggle_notify":
                cur = _get_bool(settings, "AUTO_NOTIFY", True)
                set_setting("AUTO_NOTIFY", "0" if cur else "1")
                settings = _settings()
                _tg_send_async(str(chat_id), "✅ تم تحديث التنبيهات.", reply_markup=_build_settings_kb(settings))
                return jsonify({"ok": True})


            if action == "show_notify_route":
                _tg_send_async(str(chat_id), "📨 اختر وجهة التنبيهات:", reply_markup=_build_notify_route_kb())
                return jsonify({"ok": True})

            if action.startswith("set_notify_route:"):
                route = action.split(":", 1)[1].strip().lower()
                if route not in ("dm", "group", "both"):
                    route = "dm"
                set_setting("NOTIFY_ROUTE", route)
                settings = _settings()
                _tg_send_async(str(chat_id), "✅ تم تحديث الوجهة.", reply_markup=_build_menu(settings))
                return jsonify({"ok": True})

            if action == "toggle_silent":
                cur = _get_bool(settings, "NOTIFY_SILENT", True)
                set_setting("NOTIFY_SILENT", "0" if cur else "1")
                settings = _settings()
                _tg_send_async(str(chat_id), "✅ تم تحديث وضع الصامت.", reply_markup=_build_menu(settings))
                return jsonify({"ok": True})
            if action == "show_settings":
                s = _settings()
                txt = (
                    "⚙️ الإعدادات الحالية:\n"
                    f"- الخطة: {_mode_label(_get_str(s,'PLAN_MODE','daily'))}\n"
                    f"- الدخول: {_entry_type_label(_get_str(s,'ENTRY_MODE','auto'))}\n"
                    f"- SL%: {_get_float(s,'SL_PCT',3.0)}\n"
                    f"- TP% (لضعيف/متوسط): {_get_float(s,'TP_PCT',5.0)}\n"
                    f"- TP قوي: {_get_float(s,'TP_PCT_STRONG',7.0)}\n"
                    f"- TP قوي جداً: {_get_float(s,'TP_PCT_VSTRONG',10.0)}\n"
                    f"- رأس المال: {_get_float(s,'CAPITAL_USD',800.0)}$\n"
                    f"- حجم الصفقة: {_get_float(s,'POSITION_PCT',0.20)*100:.0f}%\n"
                    f"- عدد الفرص: {_get_int(s,'MIN_SEND',7)} إلى {_get_int(s,'MAX_SEND',10)}\n"
                    f"- منع تكرار: {_get_int(s,'DEDUP_HOURS',6)} ساعات\n"
                    f"- إعادة إرسال إذا صار أقوى: {'نعم' if _get_bool(s,'ALLOW_RESEND_IF_STRONGER',True) else 'لا'}\n"
                    f"- نافذة السوق: {_get_str(s,'WINDOW_START','17:30')} إلى {_get_str(s,'WINDOW_END','00:00')} ({LOCAL_TZ})\n"
                )
                _tg_send_async(str(chat_id), txt, reply_markup=_build_settings_kb(s))
                return jsonify({"ok": True})

            if action == "show_capital":
                reply = _build_capital_kb() if "_build_capital_kb" in globals() else {"inline_keyboard":[[{"text":"✍️ قيمة مخصصة","callback_data":"set_capital_custom"}],[{"text":"⬅️ رجوع","callback_data":"show_settings"}]]}
                _tg_send_async(str(chat_id), "💰 اختر رأس المال بالدولار:", reply_markup=reply)
                return jsonify({"ok": True})

            if action == "set_capital_custom":
                from storage import set_user_state
                set_user_state(str(chat_id), "pending", "capital")
                _tg_send_async(str(chat_id), "✍️ أرسل رقم رأس المال بالدولار (مثال: 5000)")
                return jsonify({"ok": True})

            if action.startswith("set_capital:"):
                val = action.split(":", 1)[1]
                set_setting("CAPITAL_USD", val)
                s = _settings()
                _tg_send_async(str(chat_id), f"✅ تم ضبط رأس المال: {val}$", reply_markup=_build_settings_kb(s))
                return jsonify({"ok": True})

            if action == "show_position":
                _tg_send_async(str(chat_id), "📦 اختر نسبة حجم الصفقة من رأس المال:", reply_markup=_build_position_kb())
                return jsonify({"ok": True})

            if action.startswith("set_position:"):
                val = action.split(":", 1)[1]
                set_setting("POSITION_PCT", val)
                s = _settings()
                _tg_send_async(str(chat_id), f"✅ تم ضبط حجم الصفقة: {float(val)*100:.0f}%", reply_markup=_build_settings_kb(s))
                return jsonify({"ok": True})

            if action == "show_sl":
                _tg_send_async(str(chat_id), "📉 اختر وقف الخسارة %:", reply_markup=_build_sl_kb())
                return jsonify({"ok": True})

            if action.startswith("set_sl:"):
                val = action.split(":", 1)[1]
                set_setting("SL_PCT", val)
                s = _settings()
                _tg_send_async(str(chat_id), f"✅ تم ضبط وقف الخسارة: {val}%", reply_markup=_build_settings_kb(s))
                return jsonify({"ok": True})

            if action == "show_tp":
                _tg_send_async(str(chat_id), "📈 اختر جني الربح % (لضعيف/متوسط):", reply_markup=_build_tp_kb())
                return jsonify({"ok": True})

            if action.startswith("set_tp:"):
                val = action.split(":", 1)[1]
                set_setting("TP_PCT", val)
                s = _settings()
                _tg_send_async(str(chat_id), f"✅ تم ضبط جني الربح (لضعيف/متوسط): {val}%", reply_markup=_build_settings_kb(s))
                return jsonify({"ok": True})

            if action == "show_send":
                _tg_send_async(str(chat_id), "🎛 اختر عدد الفرص في كل فحص:", reply_markup=_build_send_kb())
                return jsonify({"ok": True})

            if action.startswith("set_send:"):
                parts = action.split(":")
                if len(parts) == 3:
                    set_setting("MIN_SEND", parts[1])
                    set_setting("MAX_SEND", parts[2])
                s = _settings()
                _tg_send_async(str(chat_id), f"✅ تم ضبط عدد الفرص: {s.get('MIN_SEND','7')} إلى {s.get('MAX_SEND','10')}", reply_markup=_build_settings_kb(s))
                return jsonify({"ok": True})

            if action == "toggle_resend":
                cur = _get_bool(settings, "ALLOW_RESEND_IF_STRONGER", True)
                set_setting("ALLOW_RESEND_IF_STRONGER", "0" if cur else "1")
                s = _settings()
                _tg_send_async(str(chat_id), "✅ تم تحديث خيار إعادة الإرسال.", reply_markup=_build_settings_kb(s))
                return jsonify({"ok": True})

            if action == "show_window":
                _tg_send_async(str(chat_id), "🕒 اختر نافذة السوق (بتوقيت الرياض):", reply_markup=_build_window_kb())
                return jsonify({"ok": True})

            if action.startswith("set_window:"):
                parts = action.split(":")
                if len(parts) == 3:
                    set_setting("WINDOW_START", parts[1])
                    set_setting("WINDOW_END", parts[2])
                s = _settings()
                _tg_send_async(str(chat_id), f"✅ تم ضبط النافذة: {s.get('WINDOW_START','17:30')}→{s.get('WINDOW_END','00:00')}", reply_markup=_build_settings_kb(s))
                return jsonify({"ok": True})

            if action == "noop":
                return jsonify({"ok": True})

            if action == "show_risk":
                _tg_send_async(str(chat_id), "⚖️ اختر نسب المخاطرة حسب التصنيف (A+/A/B):", reply_markup=_build_risk_kb(settings))
                return jsonify({"ok": True})

            if action.startswith("set_risk_aplus:"):
                val = action.split(":", 1)[1]
                set_setting("RISK_APLUS_PCT", val)
                s = _settings()
                _tg_send_async(str(chat_id), f"✅ تم ضبط مخاطرة A+: {val}%", reply_markup=_build_settings_kb(s))
                return jsonify({"ok": True})

            if action.startswith("set_risk_a:"):
                val = action.split(":", 1)[1]
                set_setting("RISK_A_PCT", val)
                s = _settings()
                _tg_send_async(str(chat_id), f"✅ تم ضبط مخاطرة A: {val}%", reply_markup=_build_settings_kb(s))
                return jsonify({"ok": True})

            if action.startswith("set_risk_b:"):
                val = action.split(":", 1)[1]
                set_setting("RISK_B_PCT", val)
                s = _settings()
                _tg_send_async(str(chat_id), f"✅ تم ضبط مخاطرة B: {val}%", reply_markup=_build_settings_kb(s))
                return jsonify({"ok": True})

            if action == "show_interval":
                _tg_send_async(str(chat_id), "⏱️ اختر فترة الفحص:", reply_markup=_build_interval_kb(settings))
                return jsonify({"ok": True})

            if action.startswith("set_interval:"):
                val = action.split(":", 1)[1]
                set_setting("SCAN_INTERVAL_MIN", val)

                # Apply immediately if scheduler already running
                try:
                    if _scheduler is not None:
                        job = _scheduler.get_job("scan_job")
                        if job:
                            job.reschedule(trigger=IntervalTrigger(minutes=max(5, int(val))))
                except Exception:
                    pass

                s = _settings()
                _tg_send_async(str(chat_id), f"✅ تم ضبط فترة الفحص: {val} دقيقة", reply_markup=_build_settings_kb(s))
                return jsonify({"ok": True})

            if action in ("do_analyze", "do_top"):
                settings = _settings()
                _tg_send_async(str(chat_id), "⏳ جاري التحليل...")

                def _job():
                    try:
                        msg, _ = _run_scan_and_build_message(settings)
                        send_telegram(msg)
                    except Exception as e:
                        _tg_send_async(str(chat_id), f"❌ خطأ أثناء الفحص:\n{e}")

                _run_async(_job)
                return jsonify({"ok": True})

            # Unknown action
            _tg_send_async(str(chat_id), "❓ أمر غير معروف.", reply_markup=_build_menu(settings))
            return jsonify({"ok": True})

        # Handle normal messages
        message = data.get("message") or data.get("channel_post")
        if not message:
            return jsonify({"ok": True})

        chat_id = message["chat"]["id"]
        user_id = message.get("from", {}).get("id")
        text = (message.get("text") or "").strip()

        # إدخال مخصص بعد ضغط زر
        from storage import get_user_state, clear_user_state
        pending = get_user_state(str(chat_id), "pending", "")
        if pending == "capital" and text:
            t = text.replace(",", "").strip()
            try:
                val = float(t)
                if val <= 0:
                    raise ValueError("bad")
                set_setting("CAPITAL_USD", str(val))
                clear_user_state(str(chat_id), "pending")
                s = _settings()
                _tg_send_async(str(chat_id), f"✅ تم تحديث رأس المال إلى {val}$", reply_markup=_build_settings_kb(s))
                return jsonify({"ok": True})
            except Exception:
                _tg_send_async(str(chat_id), "❌ رقم غير صحيح. أرسل رقم مثل: 5000")
                return jsonify({"ok": True})

        if not _is_admin(user_id):
            # Ignore silently for channels, but reply in private
            if str(message.get("chat", {}).get("type")) == "private":
                _tg_send_async(str(chat_id), "⛔ هذا البوت للأدمن فقط.")
            return jsonify({"ok": True})

        settings = _settings()

        if text.startswith("/start"):
            _tg_send_async(str(chat_id), "🤖 البوت شغال.\nاكتب /menu للأزرار.", reply_markup=_build_menu(settings))
            return jsonify({"ok": True})

        if text.startswith("/menu"):
            _tg_send_async(str(chat_id), "📌 اختر:", reply_markup=_build_menu(settings))
            return jsonify({"ok": True})

        if text.startswith("/wl"):
            parts = text.strip().split()
            if len(parts) == 1 or (len(parts) >= 2 and parts[1].lower() in ("list","show")):
                wl = get_watchlist()
                if not wl:
                    _tg_send_async(str(chat_id), "📌 الـ Watchlist فاضي.\nاستخدم: /wl add TSLA")
                    return jsonify({"ok": True})
                _tg_send_async(str(chat_id), "📌 Watchlist:\n" + "\n".join(wl))
                return jsonify({"ok": True})

            if len(parts) >= 3 and parts[1].lower() in ("add","+"):
                sym = parts[2].upper()
                add_watchlist(sym)
                _tg_send_async(str(chat_id), f"✅ تم إضافة {sym} للـ Watchlist.")
                return jsonify({"ok": True})

            if len(parts) >= 3 and parts[1].lower() in ("del","remove","rm","-"):
                sym = parts[2].upper()
                remove_watchlist(sym)
                _tg_send_async(str(chat_id), f"✅ تم حذف {sym} من الـ Watchlist.")
                return jsonify({"ok": True})

            _tg_send_async(str(chat_id), "استخدم: /wl أو /wl add TSLA أو /wl del TSLA")
            return jsonify({"ok": True})



        if text.startswith("/analyze"):
            _tg_send_async(str(chat_id), "⏳ جاري التحليل...")

            def _job():
                try:
                    msg, _ = _run_scan_and_build_message(settings)
                    send_telegram(msg)
                except Exception as e:
                    _tg_send_async(str(chat_id), f"❌ خطأ أثناء الفحص:\n{e}")

            _run_async(_job)
            return jsonify({"ok": True})

        if text.startswith("/ai"):
            parts = text.split()
            if len(parts) < 2:
                _tg_send_async(str(chat_id), "اكتب: /ai SYMBOL  مثال: /ai AXTA")
                return jsonify({"ok": True})

            symbol = parts[1].upper().strip()
            _tg_send_async(str(chat_id), f"🧠 جاري تحليل {symbol} بالذكاء الاصطناعي...")

            def _job_ai():
                try:
                    feats = get_symbol_features(symbol)
                    if isinstance(feats, dict) and feats.get("error"):
                        _tg_send_async(str(chat_id), f"❌ {symbol}: {feats['error']}")
                        return
                    ai_text = gemini_analyze(symbol, feats if isinstance(feats, dict) else {"data": str(feats)})
                    _tg_send_async(str(chat_id), f"🧠 تحليل AI لـ {symbol}\n\n{ai_text}")
                except Exception as e:
                    _tg_send_async(str(chat_id), f"❌ خطأ تحليل AI:\n{e}")

            _run_async(_job_ai)
            return jsonify({"ok": True})

        if text.startswith("/settings"):
            _tg_send_async(str(chat_id), "⚙️", reply_markup=_build_menu(settings))
            return jsonify({"ok": True})

        return jsonify({"ok": True})

    except Exception:
        print('WEBHOOK ERROR:')
        print(traceback.format_exc())
        return jsonify({"ok": True})



@app.post("/tradingview")
def tradingview_webhook():
    """TradingView alerts webhook.

    Setup in TradingView alert message as JSON, for example:

    {
      "key": "YOUR_TV_KEY",
      "symbol": "{{ticker}}",
      "side": "buy"
    }

    You can also add optional overrides:
      - risk_pct
      - tp_r
      - sl_atr_mult
    """
    try:
        payload = request.get_json(silent=True) or {}
        key = (payload.get("key") or request.args.get("key") or request.headers.get("X-TV-KEY") or "").strip()
        if not key or key != TRADINGVIEW_WEBHOOK_KEY:
            return jsonify({"ok": False, "error": "unauthorized"}), 401

        symbol = (payload.get("symbol") or payload.get("ticker") or payload.get("s") or "").upper().strip()
        side = (payload.get("side") or payload.get("action") or "buy").lower().strip()


        # Optional overrides
        risk_pct = payload.get("risk_pct")
        tp_r = payload.get("tp_r")
        sl_atr_mult = payload.get("sl_atr_mult")

        def _to_float(x):
            try:
                return float(x)
            except Exception:
                return None

        # ---- AI filter (deterministic, no training) ----
        passed = True
        ai_score = None
        ai_reasons: List[str] = []
        ai_features: Dict[str, Any] = {}
        if AI_FILTER_ENABLED:
            passed, ai_score, ai_reasons, ai_features = should_alert(symbol, side, min_score=AI_FILTER_MIN_SCORE)

        # Build a human-friendly alert message
        lines = [f"📡 TradingView Signal: {symbol} ({side.upper()})"]
        if ai_score is not None:
            lines.append(f"🧠 AI score: {ai_score}/100 (min {AI_FILTER_MIN_SCORE})")

        # Lightweight model probability (learns over time from evaluations)
        model_prob = None
        try:
            s = _settings()
            if ML_ENABLED and _get_bool(s, "ML_ENABLED", True):
                w = parse_weights(_get_str(s, "ML_WEIGHTS", ""))
                x = featurize(ai_features)
                model_prob = predict_prob(x, w)
                lines.append(f"📈 Prob: {model_prob*100:.1f}%")
        except Exception:
            model_prob = None
            # keep reasons short
            for r in ai_reasons[:8]:
                lines.append(r)

        # Add quick indicator snapshot (if available)
        try:
            p = ai_features.get("price")
            if p is not None:
                lines.append(f"Price: {float(p):.2f}")
            rsi = ai_features.get("rsi14")
            macd = ai_features.get("macd_hist")
            adx = ai_features.get("adx14")
            ema20 = ai_features.get("ema20")
            ema50 = ai_features.get("ema50")
            if rsi is not None:
                lines.append(f"RSI14: {float(rsi):.1f}")
            if macd is not None:
                lines.append(f"MACD hist: {float(macd):.4f}")
            if adx is not None:
                lines.append(f"ADX14: {float(adx):.1f}")
            if ema20 is not None and ema50 is not None:
                lines.append(f"EMA20/50: {float(ema20):.2f} / {float(ema50):.2f}")
        except Exception:
            pass

        # Decide whether to notify
        notify = passed or AI_FILTER_SEND_REJECTS
        if notify:
            try:
                tag = "✅ PASSED" if passed else "⛔️ FILTERED"
                send_telegram(tag + "\n" + "\n".join(lines))
            except Exception:
                pass

        # Execution is handled elsewhere (and likely disabled via safety latches)
        logs = []
        if passed:
            try:
                logs = trade_symbol(
                    symbol,
                    side=side,
                    risk_pct=_to_float(risk_pct),
                    tp_r=_to_float(tp_r),
                    sl_atr_mult=_to_float(sl_atr_mult),
                )
            except Exception as e:
                logs = [f"trade_symbol error: {e}"]

        return jsonify({
            "ok": True,
            "symbol": symbol,
            "side": side,
            "ai": {"enabled": AI_FILTER_ENABLED, "passed": passed, "score": ai_score, "min_score": AI_FILTER_MIN_SCORE, "reasons": ai_reasons[:12]},
            "executed": bool(logs) and not (len(logs)==1 and "disabled" in (logs[0] or "").lower()),
            "logs": logs,
        })

    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400


@app.get("/")
def home():
    return jsonify({"ok": True, "service": "us-stocks-scanner-executor"})


@app.get("/status")
def status():
    if request.args.get("key") != RUN_KEY:
        return jsonify({"ok": False, "error": "unauthorized"}), 401
    return jsonify({
        "ok": True,
        "orders_logged": len(last_orders(200)),
        "scans_logged": len(last_scans(200)),
    })




@app.get("/signals")
def signals_route():
    try:
        limit = int(request.args.get("limit", "50"))
    except Exception:
        limit = 50
    rows = last_signals(limit=max(1, min(200, limit)))
    return jsonify({"ok": True, "count": len(rows), "signals": rows})


@app.get("/signals/export")
def signals_export():
    """Export evaluated signals as CSV."""
    rows = last_signals(limit=500)
    # only evaluated rows
    rows = [r for r in rows if int(r.get("evaluated") or 0) == 1]
    # Simple CSV
    cols = ["id","ts","symbol","source","side","score","model_prob","horizon_days","evaluated","eval_ts","return_pct","mfe_pct","mae_pct","label"]
    lines = [",".join(cols)]
    for r in rows:
        line = []
        for c in cols:
            v = r.get(c, "")
            if v is None:
                v = ""
            s = str(v).replace("\n"," ").replace(",",";")
            line.append(s)
        lines.append(",".join(line))
    return ("\n".join(lines), 200, {"Content-Type": "text/csv; charset=utf-8"})
@app.get("/scan")
def scan():
    """
    Used by:
      - Manual testing: /scan?key=RUN_KEY
      - Render cron: /scan?key=RUN_KEY&notify=1
    """
    if request.args.get("key") != RUN_KEY:
        return jsonify({"ok": False, "error": "unauthorized"}), 401

    settings = _settings()

    # Log scan (always)
    picks, universe_size = scan_universe_with_meta()
    top_syms = ",".join([c.symbol for c in picks[:20]])
    ts = datetime.now(timezone.utc).isoformat()
    log_scan(ts, universe_size, top_syms, payload="http:/scan")

    notify = request.args.get("notify") == "1"

    sent = False
    sent_reason = ""

    if notify and _get_bool(settings, "AUTO_NOTIFY", True):
        ok, reason = _within_notification_window(settings)
        if ok:
            try:
                blocks, logged = _select_and_log_new_candidates(picks, settings)
                if blocks:
                    msg = f"📊 فرص جديدة ({_mode_label(_get_str(settings,'PLAN_MODE','daily'))})\n" + "\n\n".join(blocks)
                    send_telegram(msg)
                    sent = True
                    sent_reason = f"sent {len(logged)}"
                else:
                    sent_reason = "no new"
            except Exception as e:
                sent_reason = f"error: {e}"
        else:
            sent_reason = reason
    else:
        sent_reason = "notify=0 or AUTO_NOTIFY=OFF"

    return jsonify({
        "ok": True,
        "universe_size": universe_size,
        "top": [{"symbol": c.symbol, "score": c.score, "last_close": c.last_close, "notes": c.notes} for c in picks[:10]],
        "notify": notify,
        "notify_status": {"sent": sent, "reason": sent_reason},
    })


@app.get("/daily")
def daily():
    if request.args.get("key") != RUN_KEY:
        return jsonify({"ok": False, "error": "unauthorized"}), 401

    now = datetime.now(timezone.utc)
    scans = [s for s in last_scans(200) if _parse_dt(s["ts"]) >= now - timedelta(hours=24)]
    orders = [o for o in last_orders(200) if o.get("ts", "").startswith(now.date().isoformat())]

    msg_lines = [
        f"Daily summary (UTC): {now.date().isoformat()}",
        f"Scans last 24h: {len(scans)}",
        f"Orders today: {len(orders)}",
    ]

    if scans:
        msg_lines.append("Last scan top: " + (scans[0].get("top_symbols", "") or ""))

    if orders:
        msg_lines.append("Recent orders:")
        for o in orders[:5]:
            msg_lines.append(f"- {o['symbol']} {o['side']} qty={o['qty']} {o['status']}")

    msg = "\n".join(msg_lines)
    if SEND_DAILY_SUMMARY or request.args.get("notify") == "1":
        send_telegram(msg)

    return jsonify({"ok": True, "message": msg})


def _parse_dt(s: str) -> datetime:
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return datetime(1970, 1, 1, tzinfo=timezone.utc)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)


# ================= Scheduler (بديل GitHub Actions) =================
_scheduler: Optional[BackgroundScheduler] = None

def _fmt_scan_summary_ar(settings: Dict[str, str], universe_size: int, picks: List[Candidate]) -> str:
    mode = _get_str(settings, "PLAN_MODE", "daily")
    return (
        "⏱ انتهى الفحص — لا توجد فرص جديدة\n\n"
        f"الخطة: {_mode_label(mode)}\n"
        f"حجم الكون: {universe_size}\n"
        f"عدد النتائج: {len(picks)}\n"
        f"وقت الرياض: {_now_local().strftime('%H:%M')}\n"
    )

def _run_scan_and_notify(force_summary: bool=True) -> None:
    s = _settings()
    if not _get_bool(s, "SCHED_ENABLED", True):
        return
    ok, _ = _within_notification_window(s)
    if not ok:
        return

    picks, universe_size = scan_universe_with_meta()
    if not _get_bool(s, "AUTO_NOTIFY", True):
        return

    blocks, _logged = _select_and_log_new_candidates(picks, s)
    if blocks:
        for b in blocks:
            send_telegram(b)
    elif force_summary:
        send_telegram(_fmt_scan_summary_ar(s, universe_size, picks))



def _evaluate_pending_signals() -> None:
    """Evaluate old signals (after horizon) and optionally update lightweight model weights."""
    try:
        now = datetime.now(timezone.utc)
        rows = pending_signals_for_eval(limit=300)
        if not rows:
            return

        s = _settings()
        default_horizon = int(_get_int(s, "SIGNAL_EVAL_DAYS", SIGNAL_EVAL_DAYS))

        weights = parse_weights(_get_str(s, "ML_WEIGHTS", "")) if ML_ENABLED and _get_bool(s, "ML_ENABLED", True) else None
        updated = False

        for r in rows:
            try:
                ts = r.get("ts") or ""
                if not ts:
                    continue
                # normalize ts
                dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                horizon = int(r.get("horizon_days") or default_horizon)
                if (now - dt).days < horizon:
                    continue

                symbol = (r.get("symbol") or "").upper().strip()
                side = (r.get("side") or "buy").lower().strip()

                start = dt
                end = dt + timedelta(days=horizon + 2)

                data = bars([symbol], start=start, end=end, timeframe="1Day", limit=500)
                bars_list = (data.get("bars", {}).get(symbol) or [])
                if len(bars_list) < 2:
                    continue

                entry = float(r.get("entry") or bars_list[0].get("c") or 0.0)
                if entry <= 0:
                    continue

                # Use the last bar close within horizon window
                last_close = float(bars_list[-1].get("c") or entry)

                highs = [float(b.get("h") or b.get("c") or entry) for b in bars_list]
                lows = [float(b.get("l") or b.get("c") or entry) for b in bars_list]
                max_high = max(highs) if highs else entry
                min_low = min(lows) if lows else entry

                if side == "sell":
                    # Profit if price drops
                    ret = (entry - last_close) / entry * 100.0
                    mfe = (entry - min_low) / entry * 100.0
                    mae = (entry - max_high) / entry * 100.0
                else:
                    ret = (last_close - entry) / entry * 100.0
                    mfe = (max_high - entry) / entry * 100.0
                    mae = (min_low - entry) / entry * 100.0  # negative when drawdown

                label = 1 if ret > 0 else 0
                eval_ts = now.isoformat()

                # Update model weights (lightweight online learning)
                if weights is not None:
                    try:
                        fj = r.get("features_json") or ""
                        feats = json.loads(fj) if fj else {}
                        x = featurize(feats)
                        weights = update_online(weights, x, label=label, lr=float(ML_LEARNING_RATE))
                        updated = True
                    except Exception:
                        pass

                mark_signal_evaluated(
                    signal_id=int(r.get("id")),
                    eval_ts=eval_ts,
                    return_pct=float(ret),
                    mfe_pct=float(mfe),
                    mae_pct=float(mae),
                    label=int(label),
                )
            except Exception:
                continue

        if updated and weights is not None:
            set_setting("ML_WEIGHTS", dumps_weights(weights))

    except Exception:
        pass
def _start_scheduler() -> None:
    global _scheduler
    if _scheduler is not None:
        return
    s = _settings()
    interval = _get_int(s, "SCAN_INTERVAL_MIN", 20)
    _scheduler = BackgroundScheduler(timezone=LOCAL_TZ)
    _scheduler.add_job(
        _evaluate_pending_signals,
        IntervalTrigger(hours=6),
        id="eval_job",
        replace_existing=True,
    )
    _scheduler.add_job(
        _run_scan_and_notify,
        IntervalTrigger(minutes=max(5, interval)),
        kwargs={"force_summary": True},
        id="scan_job",
        replace_existing=True,
    )
    _scheduler.start()
    atexit.register(lambda: _scheduler.shutdown(wait=False) if _scheduler else None)

try:
    if os.getenv("ENABLE_SCHEDULER", "1") == "1":
        _start_scheduler()
except Exception:
    pass

# ================= Dashboard (simple UI) =================
_DASH_TEMPLATE = """<!doctype html>
<html lang="ar" dir="rtl">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Taw Bot Dashboard</title>
  <style>
    body{font-family:system-ui,-apple-system,Segoe UI,Roboto,Arial; margin:16px; background:#0b1220; color:#e6e8ee}
    a{color:#8ab4ff}
    .card{background:#121a2b; border:1px solid #1f2a44; border-radius:12px; padding:14px; margin:12px 0}
    .row{display:flex; gap:12px; flex-wrap:wrap}
    .kpi{min-width:220px; flex:1}
    input,select,button{padding:10px; border-radius:10px; border:1px solid #2b3a61; background:#0b1220; color:#e6e8ee}
    button{cursor:pointer}
    table{width:100%; border-collapse:collapse; font-size:14px}
    th,td{padding:8px; border-bottom:1px solid #1f2a44; text-align:right}
    .muted{opacity:.8}
  </style>
</head>
<body>
  <h2>لوحة التحكم - Taw Bot</h2>
  <div class="muted">هذه لوحة خفيفة (بدون قاعدة بيانات إضافية). كل شيء يُجلب من نفس الخدمة.</div>

  <div class="card row" id="kpis"></div>

  <div class="card">
    <h3>آخر الإشارات</h3>
    <div id="signals"></div>
  </div>

  <div class="card">
    <h3>Backtest (تجربة تاريخية)</h3>
    <div class="row">
      <input id="bt_symbol" placeholder="رمز السهم (مثال: AAPL)" value="AAPL"/>
      <select id="bt_days">
        <option value="365">365 يوم</option>
        <option value="730">سنتين</option>
        <option value="1095">3 سنوات</option>
      </select>
      <button onclick="runBacktest()">تشغيل</button>
    </div>
    <pre id="bt_out" style="white-space:pre-wrap"></pre>
  </div>

<script>
const KEY = new URLSearchParams(window.location.search).get('key') || '';

async function getJSON(path){
  const r = await fetch(path + (path.includes('?')?'&':'?') + 'key=' + encodeURIComponent(KEY));
  return await r.json();
}

function esc(s){ return (s||'').toString().replace(/[&<>]/g, c=>({ '&':'&amp;','<':'&lt;','>':'&gt;'}[c])); }

async function load(){
  const sum = await getJSON('/api/summary');
  const k = document.getElementById('kpis');
  const items = [
    ['الوضع', esc(sum.mode||'')],
    ['AUTO_TRADE', esc(sum.auto_trade)],
    ['CAPITAL_USD', esc(sum.capital_usd)],
    ['إشارات (آخر 7 أيام)', esc(sum.signals_7d)],
    ['Winrate (آخر 100)', esc(sum.winrate_100)],
  ];
  k.innerHTML = items.map(([a,b])=>`<div class="kpi card"><div class="muted">${a}</div><div style="font-size:20px;margin-top:6px">${b}</div></div>`).join('');

  const sig = await getJSON('/api/signals?limit=20');
  const div = document.getElementById('signals');
  if(!sig.items || !sig.items.length){
    div.innerHTML = '<div class="muted">لا يوجد بيانات.</div>';
    return;
  }
  const rows = sig.items.map(x=>`
    <tr>
      <td>${esc(x.ts)}</td><td>${esc(x.symbol)}</td><td>${esc(x.mode)}</td><td>${esc(x.strength)}</td>
      <td>${esc(x.entry)}</td><td>${esc(x.sl)}</td><td>${esc(x.tp)}</td>
      <td>${x.ml_prob!=null ? Math.round(x.ml_prob*100)+'%' : ''}</td>
      <td>${x.ev_r!=null ? x.ev_r.toFixed(2) : ''}</td>
    </tr>`).join('');
  div.innerHTML = `<table>
    <thead><tr><th>الوقت</th><th>السهم</th><th>الخطة</th><th>القوة</th><th>دخول</th><th>وقف</th><th>هدف</th><th>ML</th><th>EV(R)</th></tr></thead>
    <tbody>${rows}</tbody></table>`;
}

async function runBacktest(){
  const sym = (document.getElementById('bt_symbol').value||'').trim();
  const days = parseInt(document.getElementById('bt_days').value||'365',10);
  const out = document.getElementById('bt_out');
  out.textContent = '...';
  const res = await getJSON('/api/backtest?symbol=' + encodeURIComponent(sym) + '&days=' + days);
  out.textContent = JSON.stringify(res, null, 2);
}

load();
</script>
</body>
</html>"""

@app.get("/dashboard")
def dashboard():
    key = (request.args.get("key") or "").strip()
    if RUN_KEY and key != RUN_KEY:
        return ("unauthorized", 401)
    return _DASH_TEMPLATE

@app.get("/api/summary")
def api_summary():
    key = (request.args.get("key") or "").strip()
    if RUN_KEY and key != RUN_KEY:
        return jsonify({"error":"unauthorized"}), 401
    s = _settings()
    mode = _get_str(s, "PLAN_MODE", "daily")
    # count signals in last 7d from DB (best-effort)
    items = last_signals(limit=100) or []
    signals_7d = 0
    try:
        from datetime import datetime, timezone, timedelta
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(days=7)
        for it in items:
            try:
                ts = datetime.fromisoformat(str(it.get("ts")).replace("Z","+00:00"))
                if ts >= cutoff:
                    signals_7d += 1
            except Exception:
                pass
    except Exception:
        pass
    # winrate last 100 (requires evaluated results if available; fallback using pnl if present)
    wins=0; n=0
    for it in items:
        # if storage has "outcome" fields in the future, handle; else skip
        o = (it.get("outcome") or "").lower()
        if o:
            n += 1
            if o in ("tp","win","profit"):
                wins += 1
    winrate_100 = (wins/n) if n else None

    return jsonify({
        "mode": mode,
        "auto_trade": _get_bool(s, "AUTO_TRADE", False),
        "capital_usd": _get_float(s, "CAPITAL_USD", 800.0),
        "signals_7d": signals_7d,
        "winrate_100": winrate_100,
    })

@app.get("/api/signals")
def api_signals():
    key = (request.args.get("key") or "").strip()
    if RUN_KEY and key != RUN_KEY:
        return jsonify({"error":"unauthorized"}), 401
    limit = int(request.args.get("limit") or 50)
    items = last_signals(limit=limit) or []
    # attach ml_prob/ev_r if present in reasons/features payloads
    out=[]
    for it in items:
        try:
            feat = json.loads(it.get("features") or "{}") if isinstance(it.get("features"), str) else (it.get("features") or {})
        except Exception:
            feat={}
        ml_prob = None
        ev_r = None
        # if we stored these in reasons json in future, ignore; here compute if ai features exist
        if ML_ENABLED and feat:
            try:
                w = parse_weights((_settings().get("ML_WEIGHTS") or ""))
                x = featurize(feat)
                ml_prob = float(predict_prob(w, x))
                tp_r = float(_get_float(_settings(), "TP_R_MULT", 1.8))
                ev_r = (ml_prob*tp_r) - ((1-ml_prob)*1.0)
            except Exception:
                pass
        out.append({
            "ts": it.get("ts"),
            "symbol": it.get("symbol"),
            "mode": it.get("mode"),
            "strength": it.get("strength"),
            "entry": it.get("entry"),
            "sl": it.get("sl"),
            "tp": it.get("tp"),
            "ml_prob": ml_prob,
            "ev_r": ev_r,
        })
    return jsonify({"items": out})

@app.get("/api/backtest")
def api_backtest():
    key = (request.args.get("key") or "").strip()
    if RUN_KEY and key != RUN_KEY:
        return jsonify({"error":"unauthorized"}), 401
    symbol = (request.args.get("symbol") or "").strip().upper()
    days = int(request.args.get("days") or 365)
    s = _settings()
    capital = float(_get_float(s, "CAPITAL_USD", 10000.0))
    risk = float(_get_float(s, "RISK_A_PCT", 1.0))
    sl_atr = float(_get_float(s, "SL_ATR_MULT", 1.5))
    tp_r = float(_get_float(s, "TP_R_MULT", 1.8))
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    res = run_backtest_symbol(symbol, start, end, capital=capital, risk_per_trade_pct=risk, sl_atr_mult=sl_atr, tp_r_mult=tp_r)
    return jsonify(res)
