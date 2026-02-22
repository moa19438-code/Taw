from __future__ import annotations
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
from typing import Any, Dict, List, Optional, Tuple
import os
import json
import time
import re
import requests
import xml.etree.ElementTree as ET
import atexit
import traceback
import threading
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
from flask import Flask, request, jsonify

# Network timeouts (avoid NameError + keep webhook responsive)
HTTP_TIMEOUT_SEC = float(os.getenv("HTTP_TIMEOUT_SEC", "20"))
from core.admin_dashboard import bp as admin_bp
from core.ai_analyzer import gemini_analyze, gemini_predict_direction
from core.ai_filter import should_alert, decide_signal, score_signal
from core.ml_model import parse_weights, dumps_weights, featurize, predict_prob, update_online
from core.executor import trade_symbol
from core.alpaca_client import bars, clock
from core.backtesting import run_backtest_symbol
from core.config import (
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
from core.storage import (
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
    get_watchlist,
    add_watchlist,
    remove_watchlist,
    log_signal_review,
    last_signal_reviews,
    latest_signal_reviews_since,
    add_paper_trade,
    due_paper_trades,
    mark_paper_trade_notified,
    list_paper_trades_for_chat,
    delete_paper_trade_for_chat,
    cleanup_old_paper_trades,
    list_final_paper_reviews_for_chat,
    open_paper_trades_for_monitor,
    update_paper_trade_monitor_state,
)
from core.scanner import scan_universe_with_meta, Candidate, get_symbol_features, get_symbol_features_m5
from core.setup_classifier import classify_setup
app = Flask(__name__)
app.register_blueprint(admin_bp)
@app.get("/health")
def health():
    return jsonify({"ok": True, "service": "taw-bot"})
# ===== Market hours helpers (cached) =====
_MARKET_CACHE = {"ts": 0.0, "is_open": None, "next_open": None, "next_close": None}
def _market_status_cached(ttl_sec: float = 60.0) -> Dict[str, Any]:
    now = time.time()
    if _MARKET_CACHE["is_open"] is not None and (now - float(_MARKET_CACHE["ts"] or 0)) < ttl_sec:
        return dict(_MARKET_CACHE)
    try:
        c = clock() or {}
        _MARKET_CACHE["ts"] = now
        _MARKET_CACHE["is_open"] = bool(c.get("is_open"))
        _MARKET_CACHE["next_open"] = c.get("next_open")
        _MARKET_CACHE["next_close"] = c.get("next_close")
        return dict(_MARKET_CACHE)
    except Exception:
        # if Alpaca clock fails, assume open to avoid blocking user, but warn in message
        return {"ts": now, "is_open": True, "next_open": None, "next_close": None, "clock_error": True}
def _format_market_status_line(ms: Dict[str, Any]) -> str:
    is_open = ms.get("is_open")
    if is_open:
        return "🟢 السوق الأمريكي: مفتوح الآن"
    nxt = ms.get("next_open")
    if nxt:
        return f"🔴 السوق الأمريكي: مغلق | الافتتاح القادم: {nxt}"
    return "🔴 السوق الأمريكي: مغلق"
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
        requests.post(url, json=payload, timeout=20)
    except Exception:
        pass
# --- Telegram callback responsiveness / anti-duplicate ---
_CB_SEEN: Dict[str, float] = {}  # callback_query.id -> ts
_ACTION_SEEN: Dict[str, float] = {}  # f"{chat_id}:{action}" -> ts
_PICK_IN_PROGRESS: Dict[str, float] = {}  # f"{chat}:{tf}" -> start_ts
_LAST_PAPER_REVIEW_RUN = 0.0
_CB_TTL_SEC = int(os.getenv('TG_CB_TTL_SEC', '600'))  # 10 minutes default
_ACTION_DEBOUNCE_SEC = float(os.getenv('TG_ACTION_DEBOUNCE_SEC', '2.5'))


def _tg_call(method: str, payload: Dict[str, Any]) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
    """Call Telegram API. Returns (ok, description, json)."""
    if not TELEGRAM_BOT_TOKEN:
        return False, "no_token", None
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/{method}"
        r = requests.post(url, json=payload, timeout=float(HTTP_TIMEOUT_SEC))
        try:
            j = r.json()
        except Exception:
            return False, f"http_{r.status_code}", None
        if not j.get("ok"):
            return False, str(j.get("description") or f"http_{r.status_code}"), j
        return True, "ok", j
    except Exception as e:
        return False, str(e), None



def _notify_simple(text: str, settings: Dict[str, str] | None = None, silent: bool = True) -> None:
    """Send a Telegram message using configured route (dm/group/both)."""
    s = settings or get_all_settings()
    route = str(s.get("NOTIFY_ROUTE") or "dm").lower().strip()
    silent_flag = bool(parse_bool(s.get("NOTIFY_SILENT") or "1")) if settings is None else bool(silent)
    # Fallbacks
    dm = TELEGRAM_CHAT_ID or TELEGRAM_ADMIN_ID
    grp = TELEGRAM_CHANNEL_ID
    if route == "both":
        if dm:
            _tg_send(dm, text, silent=silent_flag)
        if grp:
            _tg_send(grp, text, silent=silent_flag)
    elif route == "group":
        if grp:
            _tg_send(grp, text, silent=silent_flag)
        elif dm:
            _tg_send(dm, text, silent=silent_flag)
    else:  # dm
        if dm:
            _tg_send(dm, text, silent=silent_flag)
        elif grp:
            _tg_send(grp, text, silent=silent_flag)

def _tg_edit_text(chat_id: str, message_id: int, text: str, reply_markup: Optional[Dict[str, Any]] = None) -> Tuple[bool, str]:
    if not (TELEGRAM_BOT_TOKEN and chat_id and message_id):
        return False, "missing_params"
    payload: Dict[str, Any] = {"chat_id": chat_id, "message_id": int(message_id), "text": text}
    if reply_markup is not None:
        payload["reply_markup"] = reply_markup
    ok, desc, _ = _tg_call("editMessageText", payload)
    return ok, desc

def _tg_edit_markup(chat_id: str, message_id: int, reply_markup: Optional[Dict[str, Any]] = None) -> Tuple[bool, str]:
    if not (TELEGRAM_BOT_TOKEN and chat_id and message_id):
        return False, "missing_params"
    payload: Dict[str, Any] = {"chat_id": chat_id, "message_id": int(message_id)}
    if reply_markup is not None:
        payload["reply_markup"] = reply_markup
    ok, desc, _ = _tg_call("editMessageReplyMarkup", payload)
    return ok, desc

def _tg_ui(chat_id: str, message_id: Optional[int], text: str, reply_markup: Optional[Dict[str, Any]] = None, silent: bool = False) -> None:
    """BotFather-like UI behavior (Option 1):
    - On button clicks (message_id is present): ALWAYS edit the same message (no new messages).
    - If Telegram says "message is not modified", try updating just the keyboard.
    - If edit is impossible/forbidden, do nothing (so we never spam sendMessage).
    """
    if message_id:
        ok, desc = _tg_edit_text(chat_id, int(message_id), text, reply_markup=reply_markup)
        if ok:
            return
        if "message is not modified" in (desc or "").lower():
            _tg_edit_markup(chat_id, int(message_id), reply_markup=reply_markup)
            return
        # No fallback to sendMessage on callbacks.
        return

    # No message to edit (e.g., /start): send a new message.
    _tg_send(chat_id, text, reply_markup=reply_markup, silent=silent)

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
@app.get("/api/review")
def api_review():
    key = (request.args.get("key") or "").strip()
    if RUN_KEY and key != RUN_KEY:
        return jsonify({"ok": False, "error": "unauthorized"}), 403
    try:
        lookback = int(request.args.get("days") or 2)
    except Exception:
        lookback = 2
    lookback = max(1, min(30, lookback))

    try:
        msg = _review_recent_signals(lookback_days=lookback, limit=80)
        try:
            send_telegram(msg)
        except Exception:
            pass
        return jsonify({"ok": True, "reviewed_days": lookback})
    except Exception as e:
        return jsonify({
            "ok": False,
            "error": "review_failed",
            "message": str(e),
            "hint": "If you just deployed, run /scan first. Reviews need stored signals and market data access."
        }), 500


@app.get("/api/weekly_report")
def api_weekly_report():
    key = (request.args.get("key") or "").strip()
    if RUN_KEY and key != RUN_KEY:
        return jsonify({"ok": False, "error": "unauthorized"}), 403
    try:
        days = int(request.args.get("days") or 7)
    except Exception:
        days = 7
    days = max(1, min(90, days))

    try:
        msg = _weekly_report(days=days)
        try:
            send_telegram(msg)
        except Exception:
            pass
        return jsonify({"ok": True, "days": days})
    except Exception as e:
        return jsonify({
            "ok": False,
            "error": "weekly_report_failed",
            "message": str(e),
            "hint": "This report needs stored signals/reviews. If there is no data yet, run /scan on market days and then /api/review after a couple of days."
        }), 500

# ================= Telegram keyboards =================
def _ikb(rows: List[List[Tuple[str, str]]]) -> Dict[str, Any]:
    """Build Telegram inline keyboard markup from rows of (text, callback_data)."""
    return {
        "inline_keyboard": [
            [{"text": t, "callback_data": d} for (t, d) in row]
            for row in rows
        ]
    }

def _build_menu(settings: Dict[str, str]) -> Dict[str, Any]:
    # Main menu
    return _ikb([
        [("📊 فحص السوق", "do_analyze"), ("⚙️ الإعدادات", "show_settings")],
        [("🔥 أفضل فرص الآن (D1)", "pick_d1"), ("⚡ سكالبينغ (M5)", "pick_m5")],
        [("🧠 1- أفضل EV", "ai_top_ev"), ("🧠 2- أعلى احتمال", "ai_top_prob")],
        [("🧠 3- سكالبينغ M5", "ai_top_m5"), ("🔎 AI سهم معين", "ai_symbol_start")],
        [("📊 إشاراتي", "my_sig_menu"), ("📅 تقرير أسبوعي", "weekly_report")],
        [("🔁 تحديث القائمة", "menu")],
    ])


def _build_pick_kb() -> Dict[str, Any]:
    """Actions for a single pick (manual simulation)."""
    return _ikb([
        [("📝 سجّل كأني دخلت", "paper_log")],
        [("➡️ التالي", "pick_next"), ("⬅️ رجوع", "menu")],
    ])




def _build_my_signals_kb(has_items: bool = True, back_action: str = "menu") -> Dict[str, Any]:
    rows = []
    if has_items:
        rows.append([("🗑 حذف إشارة", "my_sig_delete")])
    rows.append([("🔄 تحديث", "my_sig_refresh"), ("⬅️ رجوع", back_action)])
    return _ikb(rows)


def _build_my_signals_delete_kb(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    rows: List[List[Tuple[str, str]]] = []
    # show up to 12 items as buttons (Telegram UI friendly)
    for r in (items or [])[:12]:
        sym = (r.get("symbol") or "").upper()
        mode = (r.get("mode") or "").upper() or "D1"
        pid = int(r.get("paper_id") or 0)
        label = f"❌ {sym} ({mode})"
        rows.append([(label, f"del_sig:{pid}")])
    rows.append([("⬅️ رجوع", "review_signals")])
    return _ikb(rows)



def _build_my_signals_root_kb() -> Dict[str, Any]:
    """Entry point for user's signals management."""
    return _ikb([
        [("📈 مراجعة الأداء", "my_sig_review"), ("📌 شاراتي المحفوظة", "my_sig_list")],
        [("📊 مراجعات 24 ساعة", "my_sig_24h"), ("📊 داشبورد", "my_sig_dash")],
        [("🗑 حذف الكل", "my_sig_delall")],
        [("⬅️ رجوع", "menu")],
    ])


def _build_my_sig_24h_kb(back_action: str = "my_sig_menu") -> Dict[str, Any]:
    return _ikb([
        [("🔄 تحديث", "my_sig_24h_refresh"), ("⬅️ رجوع", back_action)],
    ])


def _build_my_sig_review_kb(back_action: str = "my_sig_menu") -> Dict[str, Any]:
    return _ikb([
        [("🔄 تحديث", "my_sig_review_refresh"), ("⬅️ رجوع", back_action)],
    ])

def _build_settings_kb(s: Dict[str, str]) -> Dict[str, Any]:
    ai_on = "ON" if _get_bool(s, "AI_PREDICT_ENABLED", False) else "OFF"
    notify_on = "ON" if _get_bool(s, "AUTO_NOTIFY", True) else "OFF"
    silent_on = "ON" if _get_bool(s, "NOTIFY_SILENT", True) else "OFF"
    route = (_get_str(s, "NOTIFY_ROUTE", "dm") or "dm").upper()
    return _ikb([
        [("📆 الخطة الزمنية", "show_modes"), ("🎯 نوع الدخول", "show_entry")],
        [("💰 رأس المال", "show_capital"), ("📦 حجم الصفقة", "show_position")],
        [("📉 وقف الخسارة SL%", "show_sl"), ("📈 جني الربح TP%", "show_tp")],
        [("🎛 عدد الفرص", "show_send"), ("🕒 نافذة السوق", "show_window")],
        [("⏱️ فترة الفحص", "show_interval"), ("⚖️ المخاطرة", "show_risk")],
        [(f"🔔 التنبيهات: {notify_on}", "toggle_notify"), (f"🔕 صامت: {silent_on}", "toggle_silent")],
        [(f"🤖 AI تنبؤ: {ai_on}", "toggle_ai_predict"), (f"📨 الوجهة: {route}", "show_notify_route")],
        [("🧪 فحص ذاتي", "self_check"), ("⬅️ رجوع", "menu")],
    ])

def _build_modes_kb() -> Dict[str, Any]:
    return _ikb([
        [("📅 يومي D1", "set_mode:daily"), ("⏱️ سكالبينغ M5", "set_mode:scalp")],
        [("📈 سونق/سوينغ", "set_mode:swing"), ("⬅️ رجوع", "show_settings")],
    ])

def _build_entry_kb() -> Dict[str, Any]:
    return _ikb([
        [("🧠 تلقائي", "set_entry:auto"), ("✅ كسر/تأكيد", "set_entry:breakout")],
        [("🎯 حد/Limit", "set_entry:limit"), ("⬅️ رجوع", "show_settings")],
    ])

def _build_horizon_kb(s: Dict[str, str]) -> Dict[str, Any]:
    cur = (_get_str(s, "PREDICT_FRAME", "D1") or "D1").upper()
    def lab(v: str) -> str:
        return f"✅ {v}" if v == cur else v
    return _ikb([
        [(lab("D1"), "set_horizon:D1"), (lab("M5"), "set_horizon:M5"), (lab("M5+"), "set_horizon:M5+")],
        [("⬅️ رجوع", "show_settings")],
    ])

def _build_notify_route_kb() -> Dict[str, Any]:
    return _ikb([
        [("📩 خاص (DM)", "set_notify_route:dm"), ("👥 مجموعة", "set_notify_route:group")],
        [("🔁 الاثنين معاً", "set_notify_route:both"), ("⬅️ رجوع", "show_settings")],
    ])

def _build_capital_kb() -> Dict[str, Any]:
    opts = [200, 500, 800, 1000, 2000, 5000, 10000]
    rows = []
    for i in range(0, len(opts), 3):
        row = []
        for v in opts[i:i+3]:
            row.append((f"{v}$", f"set_capital:{v}"))
        rows.append(row)
    rows.append([("✍️ قيمة مخصصة", "set_capital_custom"), ("⬅️ رجوع", "show_settings")])
    return _ikb(rows)

def _build_position_kb() -> Dict[str, Any]:
    opts = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    rows=[]
    for i in range(0, len(opts), 3):
        row=[]
        for v in opts[i:i+3]:
            row.append((f"{int(v*100)}%", f"set_position:{v}"))
        rows.append(row)
    rows.append([("⬅️ رجوع", "show_settings")])
    return _ikb(rows)

def _build_sl_kb() -> Dict[str, Any]:
    opts = [1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
    rows=[]
    for i in range(0, len(opts), 3):
        row=[]
        for v in opts[i:i+3]:
            row.append((f"{v}%", f"set_sl:{v}"))
        rows.append(row)
    rows.append([("⬅️ رجوع", "show_settings")])
    return _ikb(rows)

def _build_tp_kb() -> Dict[str, Any]:
    opts = [3.0, 4.0, 5.0, 6.0, 7.0, 10.0]
    rows=[]
    for i in range(0, len(opts), 3):
        row=[]
        for v in opts[i:i+3]:
            row.append((f"{v}%", f"set_tp:{v}"))
        rows.append(row)
    rows.append([("⬅️ رجوع", "show_settings")])
    return _ikb(rows)

def _build_send_kb() -> Dict[str, Any]:
    presets = [("5-7", (5,7)), ("7-10", (7,10)), ("10-15", (10,15))]
    rows=[[(p[0], f"set_send:{p[1][0]}:{p[1][1]}") for p in presets],
          [("⬅️ رجوع", "show_settings")]]
    return _ikb(rows)

def _build_window_kb() -> Dict[str, Any]:
    # Times are in LOCAL_TZ (Asia/Riyadh). Keep a few common presets.
    presets = [("17:30→00:00", ("17:30","00:00")), ("16:30→23:30", ("16:30","23:30")), ("18:00→01:00", ("18:00","01:00"))]
    rows=[]
    for label,(a,b) in presets:
        rows.append([(label, f"set_window:{a}:{b}")])
    rows.append([("⬅️ رجوع", "show_settings")])
    return _ikb(rows)

def _build_risk_kb(s: Dict[str, str]) -> Dict[str, Any]:
    """Risk per trade presets (as % of capital) by grade A+/A/B."""
    aplus_cur = _get_float(s, "RISK_APLUS_PCT", 1.0)
    a_cur = _get_float(s, "RISK_A_PCT", 0.75)
    b_cur = _get_float(s, "RISK_B_PCT", 0.5)

    def _btn(label, cb):
        return (label, cb)

    def _mark(v, cur):
        try:
            return f"✅ {v}%" if float(v) == float(cur) else f"{v}%"
        except Exception:
            return f"{v}%"

    return _ikb([
        [(_mark(0.5, aplus_cur), "set_risk_aplus:0.5"), (_mark(1.0, aplus_cur), "set_risk_aplus:1.0"), (_mark(1.5, aplus_cur), "set_risk_aplus:1.5")],
        [(_mark(0.5, a_cur), "set_risk_a:0.5"), (_mark(0.75, a_cur), "set_risk_a:0.75"), (_mark(1.0, a_cur), "set_risk_a:1.0")],
        [(_mark(0.25, b_cur), "set_risk_b:0.25"), (_mark(0.5, b_cur), "set_risk_b:0.5"), (_mark(0.75, b_cur), "set_risk_b:0.75")],
        [("⬅️ رجوع", "show_settings")],
    ])

def _build_interval_kb(s: Dict[str, str]) -> Dict[str, Any]:
    cur = int(_get_int(s, "SCAN_INTERVAL_MIN", 15))
    opts = [5, 10, 15, 30, 60]
    rows=[]
    row=[]
    for v in opts:
        t = f"✅ {v}m" if v == cur else f"{v}m"
        row.append((t, f"set_interval:{v}"))
        if len(row)==3:
            rows.append(row); row=[]
    if row: rows.append(row)
    rows.append([("⬅️ رجوع", "show_settings")])
    return _ikb(rows)


# ================= Self-check (buttons / handlers / settings) =================
def _extract_callbacks(markup: Optional[Dict[str, Any]]) -> List[str]:
    out: List[str] = []
    if not markup:
        return out
    try:
        rows = (markup or {}).get("inline_keyboard") or []
        for r in rows:
            for b in (r or []):
                d = (b or {}).get("callback_data")
                if d:
                    out.append(str(d))
    except Exception:
        pass
    return out

def _cb_matches(cb: str, allowed_exact: set[str], allowed_prefixes: List[str]) -> bool:
    if cb in allowed_exact:
        return True
    for p in allowed_prefixes:
        if cb.startswith(p):
            return True
    return False

def _self_check(fix: bool = True) -> Dict[str, Any]:
    """
    Runtime self-check to reduce 'hidden' button logic bugs.
    - Validates that every callback_data used in keyboards has a handler match (exact or prefix).
    - Validates labels cover all option values shown to user.
    - Validates 'Back' buttons in settings submenus return to settings.
    Returns a report dict (errors/warnings/info).
    """
    s = _settings()
    allowed_exact = {
        # menus / pages
        "menu", "show_settings", "show_modes", "show_entry", "show_capital", "show_position",
        "show_sl", "show_tp", "show_send", "show_window", "show_interval", "show_risk",
        "show_notify_route", "show_horizon",
        # actions
        "do_analyze", "do_top", "pick_d1", "pick_m5", "pick_next",
        "ai_top_ev", "ai_top_prob", "ai_top_m5", "ai_symbol_start", "ai_cancel",
	        "review_signals", "weekly_report",
	        # my signals menu
	        "my_sig_menu", "my_sig_review", "my_sig_list", "my_sig_refresh",
	        "my_sig_delete", "my_sig_delall",
        # toggles / misc
        "toggle_notify", "toggle_silent", "toggle_ai_predict", "toggle_resend",
        "set_capital_custom",
        "self_check",
        "noop",
    }
    allowed_prefixes = [
        "ai_pick:",
        "set_mode:", "set_entry:", "set_horizon:",
        "set_notify_route:",
        "set_capital:", "set_position:",
        "set_sl:", "set_tp:", "set_send:",
        "set_window:",
        "set_risk_aplus:", "set_risk_a:", "set_risk_b:",
        "set_interval:",
	        # my signals delete item
	        "del_sig:",
    ]

    markups = [
        ("menu", _build_menu(s)),
        ("settings", _build_settings_kb(s)),
        ("modes", _build_modes_kb()),
        ("entry", _build_entry_kb()),
        ("horizon", _build_horizon_kb(s)),
        ("notify_route", _build_notify_route_kb()),
        ("capital", _build_capital_kb()),
        ("position", _build_position_kb()),
        ("sl", _build_sl_kb()),
        ("tp", _build_tp_kb()),
        ("send", _build_send_kb()),
        ("window", _build_window_kb()),
        ("risk", _build_risk_kb(s)),
        ("interval", _build_interval_kb(s)),
    ]

    callbacks: List[Tuple[str, str]] = []
    for name, mk in markups:
        for cb in _extract_callbacks(mk):
            callbacks.append((name, cb))

    missing: List[Dict[str, str]] = []
    for where, cb in callbacks:
        if not _cb_matches(cb, allowed_exact, allowed_prefixes):
            missing.append({"where": where, "callback": cb})

    # Label coverage checks
    label_issues: List[str] = []
    for v in ["daily", "scalp", "swing"]:
        lab = _mode_label(v)
        if not lab or lab == "يومي":
            label_issues.append(f"mode label missing for '{v}' -> '{lab}'")
    for v in ["auto", "limit", "breakout"]:
        lab = _entry_type_label(v)
        if not lab or lab == "تلقائي":
            # auto is valid 'تلقائي'
            if v != "auto":
                label_issues.append(f"entry label missing for '{v}' -> '{lab}'")

    # Back button checks for settings submenus
    back_issues: List[str] = []
    for name, mk in [("modes", _build_modes_kb()), ("entry", _build_entry_kb())]:
        for cb in _extract_callbacks(mk):
            # detect any back buttons by callback target
            pass
        try:
            rows = (mk or {}).get("inline_keyboard") or []
            for r in rows:
                for b in (r or []):
                    if (b or {}).get("text","").strip().startswith("⬅️"):
                        if (b or {}).get("callback_data") != "show_settings":
                            back_issues.append(f"Back button in {name} should go to show_settings")
        except Exception:
            pass

    ok = (len(missing)==0 and len(label_issues)==0 and len(back_issues)==0)

    return {
        "ok": ok,
        "counts": {
            "callbacks_total": len(callbacks),
            "missing_handlers": len(missing),
            "label_issues": len(label_issues),
            "back_issues": len(back_issues),
        },
        "missing_handlers": missing[:50],
        "label_issues": label_issues,
        "back_issues": back_issues,
        "notes": [
            "هذا الفحص يتحقق من أزرار التليجرام فقط (callback_data) وتغطيتها في الهاندلرز.",
            "إذا ظهرت 'لا توجد فرص' فهذا غالباً بسبب فلترة/منع تكرار وليس خطأ أزرار.",
        ],
    }

def _self_check_text(rep: Dict[str, Any]) -> str:
    c = rep.get("counts") or {}
    lines = []
    lines.append("🧪 تقرير الفحص الذاتي للأزرار")
    lines.append(f"✅ الحالة: {'سليم' if rep.get('ok') else 'يوجد مشاكل'}")
    lines.append(f"• إجمالي الأزرار: {c.get('callbacks_total', 0)}")
    lines.append(f"• أزرار بدون معالج: {c.get('missing_handlers', 0)}")
    lines.append(f"• مشاكل تسميات: {c.get('label_issues', 0)}")
    lines.append(f"• مشاكل رجوع: {c.get('back_issues', 0)}")
    if rep.get("missing_handlers"):
        lines.append("\n🔻 أزرار بدون handler (أول 10):")
        for it in (rep.get("missing_handlers") or [])[:10]:
            lines.append(f"- [{it.get('where')}] {it.get('callback')}")
    if rep.get("label_issues"):
        lines.append("\n🔻 مشاكل التسميات:")
        for it in rep.get("label_issues")[:10]:
            lines.append(f"- {it}")
    if rep.get("back_issues"):
        lines.append("\n🔻 مشاكل أزرار الرجوع:")
        for it in rep.get("back_issues")[:10]:
            lines.append(f"- {it}")
    return "\n".join(lines)

@app.get("/selfcheck")
def http_selfcheck():
    try:
        key = (request.args.get("key") or "").strip()
        if RUN_KEY and key != RUN_KEY:
            return jsonify({"ok": False, "error": "unauthorized"}), 401
        rep = _self_check(fix=False)
        return jsonify(rep)
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

def _fetch_news_headlines(symbol: str, limit: int = 5) -> list[dict]:
    """Fetch latest trading-relevant headlines (best-effort, no key required).

    Uses Google News RSS by default. If NEWSAPI_KEY is set, uses NewsAPI.org.
    Returns list of dicts: {title, source, published, url}
    """
    sym = re.sub(r"[^A-Za-z\.]", "", (symbol or "").upper()).strip()
    if not sym:
        return []
    # 1) Optional NewsAPI (if user provides key)
    newsapi_key = (os.getenv("NEWSAPI_KEY") or "").strip()
    if newsapi_key:
        try:
            q = f"{sym} stock OR shares"
            url = "https://newsapi.org/v2/everything"
            r = requests.get(url, params={
                "q": q,
                "language": "en",
                "sortBy": "publishedAt",
                "pageSize": int(limit),
                "apiKey": newsapi_key,
            }, timeout=12)
            data = r.json() if r.ok else {}
            out = []
            for a in (data.get("articles") or [])[:limit]:
                out.append({
                    "title": a.get("title"),
                    "source": (a.get("source") or {}).get("name"),
                    "published": a.get("publishedAt"),
                    "url": a.get("url"),
                })
            return [x for x in out if x.get("title")]
        except Exception:
            pass

    # 2) Google News RSS (no key)
    try:
        q = f"{sym}%20stock"
        rss = f"https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en"
        r = requests.get(rss, timeout=12, headers={"User-Agent": "Mozilla/5.0"})
        if not r.ok or not (r.text or "").strip():
            return []
        root = ET.fromstring(r.text)
        out = []
        for item in root.findall(".//item")[:limit]:
            title = (item.findtext("title") or "").strip()
            link = (item.findtext("link") or "").strip()
            pub = (item.findtext("pubDate") or "").strip()
            source = None
            s = item.find("source")
            if s is not None and (s.text or "").strip():
                source = s.text.strip()
            if title:
                out.append({"title": title, "source": source, "published": pub, "url": link})
        return out
    except Exception:
        return []

def _build_top10_kb(items: list[dict], title: str = "") -> Dict[str, Any]:
    rows = []
    for it in items[:10]:
        sym = it.get("symbol") or ""
        if not sym:
            continue
        label = it.get("label") or sym
        rows.append([(label, f"ai_pick:{sym}")])
    rows.append([("⬅️ القائمة", "menu")])
    return _ikb(rows)

def _start_ai_symbol_analysis(chat_id: str, symbol: str) -> None:
    symbol = re.sub(r"[^A-Za-z\.]", "", (symbol or "").strip().upper())
    if not symbol:
        _tg_send(str(chat_id), "❌ اكتب رمز صحيح مثل: TSLA")
        return
    _tg_send(str(chat_id), f"🧠 جاري تحليل {symbol}...")
    def _job():
        try:
            s = _settings()
            feats = get_symbol_features(symbol)
            if isinstance(feats, dict) and feats.get("error"):
                _tg_send(str(chat_id), f"❌ {symbol}: {feats['error']}", reply_markup=_build_menu(s))
                return

            # Attach news headlines (trading catalysts)
            news = _fetch_news_headlines(symbol, limit=int(_get_int(s, "AI_NEWS_LIMIT", 5)))
            if news:
                feats["_news"] = [{"title": n.get("title"), "source": n.get("source"), "published": n.get("published") } for n in news]

            # Optional M5 features (for context)
            feats_m5 = None
            try:
                feats_m5 = get_symbol_features_m5(symbol)
            except Exception:
                feats_m5 = None

            # AI filter + score
            side = "buy"
            passed, ai_score, ai_reasons, ai_features = (True, None, [], {})
            if AI_FILTER_ENABLED:
                try:
                    ai_score, ai_reasons, ai_features = score_signal(symbol, side=side)
                except Exception:
                    ai_score, ai_reasons, ai_features = (None, [], {})

            # Build plan for EV/prob (uses same TP/SL settings)
            plan = _build_trade_plan(symbol, side=side, entry=float(feats.get("close") or feats.get("last_close") or 0.0),
                                     atr=float(feats.get("atr") or feats.get("atr14") or 0.0),
                                     settings=s,
                                     score=(float(ai_score)/10.0 if ai_score is not None else None))

            # ML probability / expected value (optional)
            try:
                if ML_ENABLED and AI_FILTER_ENABLED and (ai_features is not None):
                    w = parse_weights(s.get("ML_WEIGHTS") or "")
                    x = featurize(ai_features)
                    p = float(predict_prob(w, x))
                    plan["ml_prob"] = round(p, 4)
                    tp_r = float(plan.get("tp_r_mult") or 0.0)
                    plan["ev_r"] = round((p * tp_r) - ((1.0 - p) * 1.0), 3)
            except Exception:
                pass

            # Gemini analysis (with news)
            gem = None
            try:
                gem = gemini_analyze(symbol, feats)
            except Exception:
                gem = None

            lines = []
            lines.append(f"🧠 تحليل AI للسهم: {symbol}")
            if ai_score is not None:
                lines.append(f"• AI Score: {ai_score}/100")
            if plan.get("ml_prob") is not None:
                lines.append(f"• Probability (ML): {plan['ml_prob']}")
            if plan.get("ev_r") is not None:
                lines.append(f"• EV (R): {plan['ev_r']}")
            if ai_reasons:
                lines.append("\n✅ أهم الأسباب:")
                for r in ai_reasons[:6]:
                    lines.append(f"- {r}")
            if feats_m5 and not feats_m5.get("error"):
                lines.append("\n⚡ لمحة M5:")
                for k in ("last","rsi14","atr14","pattern","liquidity","spread_risk"):
                    if k in feats_m5:
                        lines.append(f"- {k}: {feats_m5.get(k)}")
            if news:
                lines.append("\n🗞️ آخر الأخبار (مختصر):")
                for n in news[:5]:
                    t = n.get("title")
                    src = n.get("source")
                    if t:
                        lines.append(f"- {t}" + (f" ({src})" if src else ""))
            if gem:
                lines.append("\n🤖 Gemini:\n" + str(gem).strip())

            _tg_send(str(chat_id), "\n".join(lines), reply_markup=_build_menu(s))
        except Exception as e:
            _tg_send(str(chat_id), f"❌ خطأ أثناء التحليل: {e}")
    _run_async(_job)

def _build_ai_start_kb() -> Dict[str, Any]:
    return _ikb([
        [("❌ إلغاء", "ai_cancel"), ("⬅️ القائمة", "menu")]
    ])

# --- Fast-pick cache for M5/D1 buttons (precompute in background) ---
_PICK_CACHE: Dict[str, Dict[str, Any]] = {
    "m5": {"ts": 0.0, "items": [], "idx_by_chat": {}},
    "d1": {"ts": 0.0, "items": [], "idx_by_chat": {}},
}
_PICK_LOCK = threading.Lock()
M5_CACHE_MIN = float(os.getenv("M5_CACHE_MIN", "3"))   # refresh every N minutes
D1_CACHE_MIN = float(os.getenv("D1_CACHE_MIN", "60"))  # refresh every N minutes

# --- Market open helper (cached) ---
_MARKET_CACHE = {"ts": 0.0, "open": None, "next_open": None, "next_close": None}
def _is_us_market_open(ttl_sec: int = 30) -> bool:
    """Return True if US equities market is open (using Alpaca clock). Cached for ttl_sec."""
    now = time.time()
    try:
        if _MARKET_CACHE["open"] is not None and (now - _MARKET_CACHE["ts"]) < ttl_sec:
            return bool(_MARKET_CACHE["open"])
    except Exception:
        pass
    try:
        c = clock()
        _MARKET_CACHE["ts"] = now
        _MARKET_CACHE["open"] = bool(c.get("is_open"))
        _MARKET_CACHE["next_open"] = c.get("next_open")
        _MARKET_CACHE["next_close"] = c.get("next_close")
        return bool(_MARKET_CACHE["open"])
    except Exception:
        return True  # fail-open; better to still respond than block the bot

M5_TOP_K = int(os.getenv("M5_TOP_K", "40"))            # compute M5 features only for top K daily picks
M5_RETURN_N = int(os.getenv("M5_RETURN_N", "12"))      # keep N candidates in cache
def _m5_score_from_features(f: Dict[str, Any]) -> Tuple[float, str]:
    """Return (score 0..100, direction) from 5Min features.

    This is intentionally lightweight: we score trend + RSI + volume spike + sane volatility,
    then apply small adjustments for liquidity/spread heuristics + candlestick confirmation.
    """
    ema20 = f.get("ema20")
    ema50 = f.get("ema50")
    rsi14 = f.get("rsi14")
    atr_pct = f.get("atr_pct")
    vol_spike = bool(f.get("vol_spike"))
    liquidity = f.get("liquidity")
    spread_risk = f.get("spread_risk")
    pat_bias = f.get("pattern_bias")
    pat_strength = f.get("pattern_strength")

    score = 0.0

    # Trend
    if isinstance(ema20, (int, float)) and isinstance(ema50, (int, float)):
        if ema20 > ema50:
            score += 40.0
            direction = "LONG"
        elif ema20 < ema50:
            score += 30.0
            direction = "SHORT"
        else:
            direction = "LONG"
    else:
        direction = "LONG"

    # RSI (prefer 'room' for continuation scalps)
    if isinstance(rsi14, (int, float)):
        if 50 <= rsi14 <= 70:
            score += 30.0
        elif 40 <= rsi14 < 50:
            score += 18.0
        elif 70 < rsi14 <= 80:
            score += 12.0
        else:
            score += 6.0

    # Vol spike
    if vol_spike:
        score += 20.0

    # Volatility sanity (avoid ultra-crazy / ultra-dead)
    if isinstance(atr_pct, (int, float)):
        if 0.003 <= atr_pct <= 0.02:
            score += 10.0
        elif 0.02 < atr_pct <= 0.05:
            score += 6.0
        else:
            score += 2.0

    # Liquidity / spread adjustments (heuristic)
    if liquidity == "GOOD":
        score += 5.0
    elif liquidity == "BAD":
        score -= 15.0
    if spread_risk == "HIGH":
        score -= 10.0

    # Candlestick confirmation
    if isinstance(pat_bias, str) and isinstance(pat_strength, str):
        want = "BULL" if direction == "LONG" else "BEAR"
        if pat_bias == want:
            if pat_strength == "STRONG":
                score += 8.0
            elif pat_strength == "MED":
                score += 4.0
            elif pat_strength == "WEAK":
                score += 2.0

    score = max(0.0, min(100.0, score))
    return score, direction
def _format_pick_m5(item: Dict[str, Any]) -> str:
    sym = item.get("symbol")
    direction = item.get("direction", "")
    score = item.get("score")
    last = item.get("last")
    rsi_v = item.get("rsi14")
    atr_v = item.get("atr14")
    notes = item.get("notes", "")
    pat = item.get("pattern")
    liquidity = item.get("liquidity")
    spread_risk = item.get("spread_risk")
    ai = item.get("ai") if isinstance(item.get("ai"), dict) else {}

    # Decision block
    decision = (ai.get("decision") or "").upper() if isinstance(ai, dict) else ""
    conf = ai.get("confidence")
    reasons = ai.get("reasons") or []
    if isinstance(reasons, str):
        reasons = [reasons]

    # Use AI direction if provided
    if ai.get("direction") in ("LONG", "SHORT"):
        direction = ai.get("direction")

    entry = last
    sl = None
    tp = None
    try:
        sl_mult = float(os.getenv("M5_SL_ATR", "1.2"))
        tp_mult = float(os.getenv("M5_TP_ATR", "1.5"))
        if isinstance(atr_v, (int, float)) and isinstance(entry, (int, float)):
            if direction == "SHORT":
                sl = round(entry + (atr_v * sl_mult), 4)
                tp = round(entry - (atr_v * tp_mult), 4)
            else:
                sl = round(entry - (atr_v * sl_mult), 4)
                tp = round(entry + (atr_v * tp_mult), 4)
    except Exception:
        pass

    lines = [
        "⏱️ M5 سهم سريع للتطبيق اليدوي",
        f"🚀 {sym} | {direction} | Score: {float(score):.0f}/100" if score is not None else f"🚀 {sym} | {direction}",
    ]

    if decision in ("ENTER", "SKIP"):
        if decision == "ENTER":
            conf_txt = f" (ثقة {int(conf)}%)" if isinstance(conf, (int, float)) else ""
            lines.append(f"قرار AI: ✅ ادخل{conf_txt}")
        else:
            conf_txt = f" (ثقة {int(conf)}%)" if isinstance(conf, (int, float)) else ""
            lines.append(f"قرار AI: ❌ لا تدخل{conf_txt}")
    elif AI_FILTER_ENABLED:
        lines.append("قرار AI: (غير متاح الآن)")

    # Quick context
    if last is not None:
        lines.append(f"السعر الحالي: {last}")
    if rsi_v is not None:
        lines.append(f"RSI14: {rsi_v}")
    if atr_v is not None:
        lines.append(f"ATR14(5m): {atr_v}")
    if sl is not None and tp is not None and entry is not None:
        lines.append(f"Entry: {entry} | SL: {sl} | TP: {tp}")

    if liquidity or spread_risk:
        lq = liquidity or "?"
        sp = spread_risk or "?"
        lines.append(f"السيولة: {lq} | خطر السبريد: {sp}")

    if pat:
        lines.append(f"نموذج شموع: {pat}")

    if reasons:
        # keep it short
        lines.append("السبب: " + ", ".join(reasons[:3]))

    if notes:
        lines.append(f"ملاحظات: {notes}")

    lines.append("⚠️ تنبيه: هذا اقتراح سريع للتطبيق اليدوي فقط. راقب السبريد/السيولة قبل الدخول.")
    return "\n".join(lines)
def _format_pick_d1(c: Candidate, settings: Dict[str, str]) -> str:
    # D1 "Best now" should be live when market is open:
    # use latest trade price as entry reference (manual execution),
    # while keeping daily ATR/score/timeframe filters from the daily scan.
    live_p, live_ts = _get_live_trade_price(c.symbol)
    entry_override = live_p if (live_p is not None and _is_us_market_open()) else None
    plan = _compute_trade_plan(settings, c, entry_override=entry_override)
    if live_p is not None:
        plan["live_price"] = round(float(live_p), 4)
        plan["live_ts"] = live_ts
        plan["ref_close"] = round(float(getattr(c, "last_close", 0.0) or 0.0), 4)
        plan["price_source"] = "LIVE" if entry_override is not None else "LAST_CLOSE"
    return _format_sahm_block("D1", c, plan)


def _get_live_trade_price(symbol: str) -> tuple[float | None, str | None]:
    """Return (price, iso_ts) from Alpaca latest trade (IEX feed).

    Notes:
      - On free plans, quotes/trades may be delayed for some symbols.
      - We still show it to the user as the *reference* price for manual execution.
    """
    try:
        from core.alpaca_client import latest_trade
        data = latest_trade(symbol)
        trade = (data or {}).get("trade") if isinstance(data, dict) else None
        if not isinstance(trade, dict):
            return None, None
        p = trade.get("p")
        ts = trade.get("t")
        if p is None:
            return None, None
        return float(p), (str(ts) if ts is not None else None)
    except Exception:
        return None, None
def _update_cache_d1() -> None:
    s = _settings()
    picks, _ = scan_universe_with_meta()
    if not picks:
        return
    top = picks[:max(5, min(20, len(picks)))]
    items = [{"symbol": c.symbol, "candidate": c, "score": float(c.score)} for c in top]
    with _PICK_LOCK:
        _PICK_CACHE["d1"]["ts"] = time.time()
        _PICK_CACHE["d1"]["items"] = items
def _update_cache_m5() -> None:
    allow_ah = str(os.getenv("ALLOW_AFTER_HOURS", "0")).strip().lower() in ("1","true","yes","y","on")
    if not allow_ah and not _is_us_market_open():
        # Don't compute intraday picks when the market is closed (signals become noisy).
        return

    # Use daily picks as a pre-filter to keep Alpaca calls low.
    picks, _ = scan_universe_with_meta()
    if not picks:
        return
    top_syms = [c.symbol for c in picks[:max(10, min(M5_TOP_K, len(picks)))]]
    items = []
    for sym in top_syms:
        try:
            f = get_symbol_features_m5(sym)
            if f.get("error"):
                continue
            score, direction = _m5_score_from_features(f)
            ai = decide_signal(sym, f, horizon="M5")
            # Basic safety filters for manual scalps
            if f.get("liquidity") == "BAD" or f.get("spread_risk") == "HIGH":
                continue
            # If AI explicitly says SKIP, keep it but de-prioritize
            if isinstance(ai, dict) and str(ai.get("decision","")).upper() == "SKIP":
                score = max(0.0, float(score) - 15.0)
            items.append({
                "symbol": sym,
                "direction": direction,
                "score": float(score),
                "last": f.get("last"),
                "rsi14": f.get("rsi14"),
                "atr14": f.get("atr14"),
                "notes": f.get("notes",""),
                "pattern": f.get("pattern"),
                "liquidity": f.get("liquidity"),
                "spread_risk": f.get("spread_risk"),
                "ai": ai,
            })
        except Exception:
            continue
    items.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
    items = items[:max(3, M5_RETURN_N)]
    with _PICK_LOCK:
        _PICK_CACHE["m5"]["ts"] = time.time()
        _PICK_CACHE["m5"]["items"] = items
def _get_next_pick(tf: str, chat_id: str) -> Optional[Dict[str, Any]]:
    tf = tf.lower()
    with _PICK_LOCK:
        cache = _PICK_CACHE.get(tf) or {}
        items = cache.get("items") or []
        if not items:
            return None
        idx_by_chat = cache.get("idx_by_chat") or {}
        idx = int(idx_by_chat.get(chat_id, 0)) % len(items)
        idx_by_chat[chat_id] = (idx + 1) % len(items)
        cache["idx_by_chat"] = idx_by_chat
        _PICK_CACHE[tf] = cache
        return items[idx]
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
        _tg_send(channel_id, text, reply_markup=reply_markup, silent=silent)
    if send_dm and admin_id:
        _tg_send(admin_id, text, reply_markup=reply_markup, silent=silent)
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
def _extract_json_obj(text: str) -> Optional[dict]:
    """Best-effort JSON extraction from a model response."""
    if not text:
        return None
    t = text.strip()
    # If it's already JSON
    try:
        obj = json.loads(t)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    # Try to find a JSON object inside
    m = re.search(r"\{[\s\S]*\}", t)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        if isinstance(obj, dict):
            return obj
    except Exception:
        return None
    return None
def _ai_direction_for_symbol(symbol: str, settings: Dict[str, str]) -> Optional[dict]:
    """Run Gemini direction prediction for one symbol.
    Returns dict with keys: direction, confidence, reasons, risks, horizon.
    """
    if not _get_bool(settings, "AI_PREDICT_ENABLED", False):
        return None
    frame = _get_str(settings, "PREDICT_FRAME", "D1").upper()
    if frame in ("HYBRID", "M5PLUS"):
        frame = "M5+"
    if frame not in ("D1", "M5", "M5+"):
        frame = "D1"
    # Build features depending on the requested horizon
    features: Dict[str, Any] = {}
    try:
        if frame == "D1":
            features = get_symbol_features(symbol)
        elif frame == "M5":
            features = get_symbol_features_m5(symbol)
        else:  # M5+
            f_d1 = get_symbol_features(symbol)
            f_m5 = get_symbol_features_m5(symbol)
            features = {f"d1_{k}": v for k, v in (f_d1 or {}).items()}
            features.update({f"m5_{k}": v for k, v in (f_m5 or {}).items()})
    except Exception as e:
        return {"direction": "NEUTRAL", "confidence": 0, "horizon": frame, "reasons": [f"feature_error: {e}"], "risks": []}
    # If we don't have enough data, don't waste an AI call
    if isinstance(features, dict) and features.get("error"):
        return {"direction": "NEUTRAL", "confidence": 0, "horizon": frame, "reasons": [str(features.get("error"))], "risks": []}
    raw = gemini_predict_direction(symbol, features, horizon=frame)
    obj = _extract_json_obj(str(raw.get("raw") or ""))
    if not obj:
        return {"direction": "NEUTRAL", "confidence": 0, "horizon": frame, "reasons": ["bad_ai_output"], "risks": []}
    direction = str(obj.get("direction") or "NEUTRAL").upper()
    if direction not in ("UP", "DOWN", "NEUTRAL"):
        direction = "NEUTRAL"
    try:
        confidence = int(round(float(obj.get("confidence") or 0)))
    except Exception:
        confidence = 0
    confidence = max(0, min(100, confidence))
    reasons = obj.get("reasons") if isinstance(obj.get("reasons"), list) else []
    risks = obj.get("risks") if isinstance(obj.get("risks"), list) else []
    return {
        "direction": direction,
        "confidence": confidence,
        "horizon": frame,
        "reasons": [str(x) for x in reasons][:3],
        "risks": [str(x) for x in risks][:3],
    }
def _strength(score: float) -> str:
    if score >= 8.5:
        return "قوي جداً"
    if score >= 7.0:
        return "قوي"
    if score >= 5.0:
        return "متوسط"
    return "ضعيف"

def _dynamic_risk_pct(ai_score: float | None,
                      loss_prob: float | None,
                      settings: Dict[str, str]) -> float:
    """مخاطرة ذكية (% من رأس المال) بناءً على جودة الصفقة واحتمالية الخسارة.
    الهدف: تكبير الربح عبر زيادة الوزن للفرص الأقوى، وتقليل الوزن للفرص الأضعف.
    """
    # Defaults (مناسبة للحسابات الصغيرة)
    min_r = _get_float(settings, "RISK_MIN_PCT", 4.0)
    max_r = _get_float(settings, "RISK_MAX_PCT", 8.0)

    # إذا ما عندنا بيانات كفاية: نستخدم متوسط
    if ai_score is None and loss_prob is None:
        return float((min_r + max_r) / 2.0)

    # إن لم يتوفر أحدهما، نعوّض بقيم وسطية
    sc = float(ai_score) if ai_score is not None else 80.0
    lp = float(loss_prob) if loss_prob is not None else 0.35

    # خريطة بسيطة: كلما زاد Score تقل lp (عادة)، ونعكسها إلى مخاطرة أعلى
    # نطاقات موصى بها للاختبار
    if lp >= float(_get_float(settings, "LOSS_PROB_BLOCK", 0.50)):
        return 0.0  # يعني ممنوع
    if sc >= 90 and lp <= 0.25:
        r = max_r
    elif sc >= 86 and lp <= 0.30:
        r = max(min_r, max_r - 1.0)
    elif sc >= 82 and lp <= 0.35:
        r = max(min_r, max_r - 2.0)
    elif sc >= 75 and lp <= 0.40:
        r = min_r
    else:
        # جودة/احتمال غير مريح: مخاطرة منخفضة جدًا
        r = max(1.0, min_r - 1.5)

    # Clamp
    r = max(0.0, min(float(r), float(max_r)))
    return float(r)

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
        "daily": "يومي D1",
        "scalp": "سكالبينغ M5",
        "swing": "سوينغ",
        "weekly": "أسبوعي",
        "monthly": "شهري",
        "daily_weekly": "يومي + أسبوعي",
        "weekly_monthly": "أسبوعي + شهري",
    }.get(m, m or "يومي D1")
def _entry_type_label(entry_mode: str) -> str:
    em = (entry_mode or "auto").lower()
    return {
        "auto": "تلقائي",
        "market": "سوق",
        "limit": "Limit",
        "breakout": "كسر/تأكيد",
    }.get(em, em or "تلقائي")
def _compute_trade_plan(settings: Dict[str, str], c: Candidate, entry_override: float | None = None) -> Dict[str, Any]:
    """
    خطة يدوية لتطبيق Sahm (ATR):
    - الدخول: سعر الإغلاق الأخير
    - وقف الخسارة: ATR * SL_ATR_MULT (LONG تحت الدخول / SHORT فوق الدخول)
    - جني الربح: (المخاطرة R) * TP_R_MULT (LONG فوق الدخول / SHORT تحت الدخول)
    - الكمية: حسب رأس المال والمخاطرة المتغيرة A+/A/B
    """
    side = (getattr(c, "side", None) or "buy").lower().strip()
    if side not in ("buy", "sell"):
        side = "buy"

    # Default entry reference is the last daily close.
    # For "Best now" use-cases we may override this with a live trade price.
    entry = float(entry_override) if entry_override is not None else float(c.last_close)

    # إعدادات ATR
    sl_atr_mult = _get_float(settings, "SL_ATR_MULT", 2.0)
    tp_r_mult = _get_float(settings, "TP_R_MULT", 2.0)
    atr_val = float(getattr(c, "atr", 0.0) or 0.0)
    if atr_val <= 0:
        atr_val = max(entry * 0.01, 0.5)

    if side == "sell":
        sl = max(0.01, entry + (atr_val * sl_atr_mult))
        risk_per_share = max(sl - entry, 0.01)
        tp = max(0.01, entry - (risk_per_share * tp_r_mult))
    else:
        sl = max(0.01, entry - (atr_val * sl_atr_mult))
        risk_per_share = max(entry - sl, 0.01)
        tp = entry + (risk_per_share * tp_r_mult)


    # === تحسين الخروج لصفقات 1D: Partial TP + Trailing Stop (اقتراحات يدوية) ===
    # افتراضيًا: TP2 هو الهدف النهائي (tp) الموجود سابقًا.
    # TP1 هدف جزئي (مثلاً 1R) + تفعيل Trailing بعده لرفع نسبة الصفقات الرابحة وتقليل الارتداد.
    tp1_r_mult = _get_float(settings, "TP1_R_MULT", 1.0)
    partial_pct = _get_float(settings, "PARTIAL_TP_PCT", 0.5)  # 0..0.95
    trail_atr_mult = _get_float(settings, "TRAIL_ATR_MULT", 1.2)
    trail_after_tp1 = _get_bool(settings, "TRAIL_AFTER_TP1", True)
    move_sl_to_be = _get_bool(settings, "MOVE_SL_TO_BE_AFTER_TP1", True)

    tp1 = None
    try:
        pp = max(0.0, min(0.95, float(partial_pct)))
        if tp1_r_mult and float(tp1_r_mult) > 0 and pp > 0:
            if side == "sell":
                tp1 = max(0.01, entry - (risk_per_share * float(tp1_r_mult)))
            else:
                tp1 = entry + (risk_per_share * float(tp1_r_mult))
        else:
            pp = 0.0
        partial_pct = pp
    except Exception:
        tp1 = None
        partial_pct = 0.0

    # Trailing stop suggestion (manual):
    # بعد TP1 (إذا trail_after_tp1=True) ننقل SL إلى BE (اختياري) ثم نتابع بـ ATR trailing.
    trail_note = ""
    if trail_atr_mult and float(trail_atr_mult) > 0:
        if trail_after_tp1:
            trail_note = f"بعد TP1 فعّل Trailing ≈ ATR×{trail_atr_mult} (و{'حرّك SL لبريك إيفن' if move_sl_to_be else 'بدون تحريك SL'})"
        else:
            trail_note = f"Trailing من البداية ≈ ATR×{trail_atr_mult}"
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

    # === رأس المال + مخاطرة ذكية (مع دعم Fractional Shares) ===
    capital = _get_float(settings, "CAPITAL_USD", 800.0)

    # مخاطر أدنى/أقصى (%)
    risk_min = _get_float(settings, "RISK_MIN_PCT", 0.5)
    risk_max = _get_float(settings, "RISK_MAX_PCT", 2.0)
    risk_pct = max(risk_min, min(risk_max, float(risk_pct)))

    # مبلغ المخاطرة بالدولار
    risk_amount = max(0.10, capital * (risk_pct / 100.0))

    # حجم الصفقة بناءً على المخاطرة (يسمح بالكسور)
    qty_risk = risk_amount / max(risk_per_share, 0.01)

    # حد أقصى لحجم الصفقة (كنسبة من رأس المال)
    pos_pct = _get_float(settings, "POSITION_PCT", 0.20)
    max_notional = max(0.0, capital * pos_pct)
    qty_cap = (max_notional / max(entry, 0.01)) if max_notional > 0 else qty_risk

    qty = max(0.01, min(qty_risk, qty_cap))

    # تقريب مناسب للـ fractional (3 منازل)
    try:
        qty = round(float(qty), 3)
    except Exception:
        qty = float(qty)
    entry_mode = _get_str(settings, "ENTRY_MODE", "auto").lower()

    # تصنيف نوع الفرصة (Breakout / Pullback / Gap / Mixed)
    setup = "MIXED"
    setup_notes: list[str] = []
    try:
        f0 = get_symbol_features(symbol) or {}
        setup, setup_notes = classify_setup(f0, side=side)
    except Exception:
        setup = "MIXED"
        setup_notes = []

    # RR محسوب على أساس المخاطرة R
    rr = (abs(tp - entry)) / max(abs(entry - sl), 0.01)

    return {
        "side": side,
        "setup": setup,
        "setup_notes": setup_notes,
        "entry": round(entry, 2),
        "sl": round(sl, 2),
        "tp": round(tp, 2),
        "tp1": (round(float(tp1), 2) if tp1 is not None else None),
        "tp2": round(tp, 2),
        "partial_pct": float(partial_pct) if partial_pct is not None else 0.0,
        "trail_atr_mult": float(trail_atr_mult) if trail_atr_mult is not None else 0.0,
        "trail_after_tp1": bool(trail_after_tp1),
        "move_sl_to_be_after_tp1": bool(move_sl_to_be),
        "trail_note": trail_note,
        "qty": float(qty),
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


def _build_trade_plan(symbol: str,
                      side: str,
                      entry: float,
                      atr: float = 0.0,
                      settings: Optional[Dict[str, str]] = None,
                      score: Optional[float] = None) -> Dict[str, Any]:
    """
    Build a trade plan for a single symbol (no execution; used for /ai and Top10).
    Auto-select between ATR-based and % based plan.

    Rules:
      - If PLAN_CALC in settings is 'atr' -> force ATR.
      - If PLAN_CALC is 'pct' -> force percent.
      - Else (auto):
          Use ATR when atr is valid and not absurd relative to price, otherwise use percent.

    Score (0-10) is optional; when provided it is used for grade/risk presets and TP% tiering.
    """
    s = settings or _settings()
    side = (side or "buy").lower().strip()
    entry = float(entry or 0.0)
    if entry <= 0:
        entry = 0.01

    # method selection
    mode = _get_str(s, "PLAN_CALC", "auto").lower().strip()  # auto|atr|pct
    atr = float(atr or 0.0)
    atr_ratio = (atr / entry) if entry > 0 else 0.0
    atr_ok = (atr > 0) and (0.002 <= atr_ratio <= 0.30)  # 0.2% .. 30% of price
    use_atr = (mode == "atr") or (mode == "auto" and atr_ok)
    use_pct = (mode == "pct") or (mode == "auto" and not use_atr)

    # grade/risk preset
    grade = ""
    risk_pct = _get_float(s, "RISK_A_PCT", 1.0)
    if score is not None:
        try:
            st = _strength(float(score))
        except Exception:
            st = "متوسط"
        if st == "قوي جداً":
            grade = "A+"
            risk_pct = _get_float(s, "RISK_APLUS_PCT", 1.5)
        elif st == "قوي":
            grade = "A"
            risk_pct = _get_float(s, "RISK_A_PCT", 1.0)
        else:
            grade = "B"
            risk_pct = _get_float(s, "RISK_B_PCT", 0.5)
    else:
        grade = "A"

    capital = _get_float(s, "CAPITAL_USD", 800.0)
    risk_amount = max(1.0, capital * (risk_pct / 100.0))

    # compute SL/TP
    sl_atr_mult = _get_float(s, "SL_ATR_MULT", 1.5)
    tp_r_mult = _get_float(s, "TP_R_MULT", 1.8)
    sl_pct = _get_float(s, "SL_PCT", 3.0) / 100.0

    # TP% tiers (only used for pct-mode)
    tp_pct = _get_float(s, "TP_PCT", 5.0) / 100.0
    if score is not None:
        # map strength tiers to TP% keys
        if _strength(float(score)) == "قوي جداً":
            tp_pct = _get_float(s, "TP_PCT_VSTRONG", 7.5) / 100.0
        elif _strength(float(score)) == "قوي":
            tp_pct = _get_float(s, "TP_PCT_STRONG", 7.0) / 100.0

    if use_atr:
        # fallback ATR when missing
        atr_val = atr if atr > 0 else max(entry * 0.01, 0.5)
        if side == "sell":
            sl = max(0.01, entry + (atr_val * sl_atr_mult))
            risk_per_share = max(sl - entry, 0.01)
            tp = max(0.01, entry - (risk_per_share * tp_r_mult))
        else:
            sl = max(0.01, entry - (atr_val * sl_atr_mult))
            risk_per_share = max(entry - sl, 0.01)
            tp = entry + (risk_per_share * tp_r_mult)
        calc_mode = "ATR"
    else:
        # percent mode
        if side == "sell":
            sl = max(0.01, entry * (1.0 + sl_pct))
            risk_per_share = max(sl - entry, 0.01)
            tp = max(0.01, entry * (1.0 - tp_pct))
        else:
            sl = max(0.01, entry * (1.0 - sl_pct))
            risk_per_share = max(entry - sl, 0.01)
            tp = entry * (1.0 + tp_pct)
        # approximate R/R from pct
        tp_r_mult = (abs(tp - entry) / risk_per_share) if risk_per_share > 0 else 0.0
        calc_mode = "PCT"

    # position sizing with caps
    qty_risk = int(risk_amount / risk_per_share) if risk_per_share > 0 else 1
    if qty_risk < 1:
        qty_risk = 1

    pos_pct = _get_float(s, "POSITION_PCT", 0.20)  # max position size as % of capital
    max_pos_value = max(1.0, capital * pos_pct)
    qty_cap = int(max_pos_value / entry) if entry > 0 else qty_risk
    qty = max(1, min(qty_risk, max(1, qty_cap)))

    rr = (abs(tp - entry) / risk_per_share) if risk_per_share > 0 else 0.0

    return {
        "symbol": symbol,
        "entry": round(entry, 2),
        "sl": round(float(sl), 2),
        "tp": round(float(tp), 2),
        "atr": round(float(atr or 0.0), 2),
        "sl_atr_mult": round(float(sl_atr_mult), 2),
        "tp_r_mult": round(float(tp_r_mult), 2),
        "risk_pct": round(float(risk_pct), 2),
        "risk_amount": round(float(risk_amount), 2),
        "qty": float(qty),
        "risk_per_share": round(float(risk_per_share), 4),
        "rr": round(float(rr), 2),
        "grade": grade,
        "calc_mode": calc_mode,
        "entry_mode": _get_str(s, "ENTRY_MODE", "auto"),
        "ml_prob": None,
        "ev_r": None,
    }



def _format_sahm_block(mode_label: str, c: Candidate, plan: Dict[str, Any], ai_score: int | None = None) -> str:
    strength = _strength(float(c.score))
    entry_type = _entry_type_label(plan["entry_mode"])

    ai_dir = plan.get("ai_dir")
    ai_conf = plan.get("ai_conf")
    ai_h = plan.get("ai_h")
    ai_line = ""
    if ai_dir:
        try:
            ai_line = f"تنبؤ AI ({ai_h or ''}): {ai_dir} ({int(ai_conf)}%)\n"
        except Exception:
            ai_line = f"تنبؤ AI ({ai_h or ''}): {ai_dir}\n"

    side = (plan.get("side") or getattr(c, "side", "buy") or "buy").lower().strip()
    side_lbl = "LONG 🟢" if side == "buy" else "SHORT 🔴"
    op_lbl = "شراء" if side == "buy" else "بيع/شورت"

    # Live price context (for manual execution)
    live_line = ""
    if plan.get("live_price") is not None:
        src = str(plan.get("price_source") or "")
        lp = plan.get("live_price")
        ts = plan.get("live_ts")
        rc = plan.get("ref_close")
        if src == "LIVE":
            live_line = f"سعر مباشر: {lp} ({ts}) | إغلاق D1: {rc}\n"
        else:
            live_line = f"سعر مباشر (مرجعي): {lp} ({ts}) | الدخول محسوب على إغلاق D1: {rc}\n"

    header = (
        f"🚀 سهم: {c.symbol} | {side_lbl} | التصنيف: {plan.get('grade','')} | القوة: {strength} | Score: {c.score:.1f}"
        + (f" | AI: {ai_score}/100" if ai_score is not None else "")
        + (f" | ML: {int(round(float(plan.get('ml_prob') or 0)*100))}%" if plan.get('ml_prob') is not None else "")
        + (f" | EV(R): {float(plan.get('ev_r')):.2f}" if plan.get('ev_r') is not None else "")
    )

    loss_line = ""
    try:
        if plan.get("loss_prob") is not None:
            loss_line = f"احتمال الخسارة: {int(round(float(plan.get('loss_prob'))*100))}%\n"
    except Exception:
        loss_line = ""

    parts: List[str] = []
    parts.append(f"{header}\n")
    parts.append(f"العملية: {op_lbl}\n")
    parts.append(f"النوع: {entry_type}\n")
    parts.append(f"السعر: {plan['entry']}\n")
    if live_line:
        parts.append(live_line)
    parts.append(f"الكمية: {plan['qty']}\n")
    if plan.get("setup"):
        setup_notes = ", ".join((plan.get("setup_notes") or [])[:2])
        parts.append(f"نوع الفرصة: {plan.get('setup','')} | {setup_notes}\n")
    if loss_line:
        parts.append(loss_line)
    parts.append(f"المخاطرة: {plan.get('risk_pct',0)}% (≈ {plan.get('risk_amount',0)}$) | R/R: {plan.get('rr',0)}\n")
    parts.append(f"ATR: {plan.get('atr',0)} | SL×ATR: {plan.get('sl_atr_mult',0)} | TP×R: {plan.get('tp_r_mult',0)}\n")
    parts.append(f"TF: D:{'✅' if getattr(c, 'daily_ok', False) else '❌'} W:{'✅' if getattr(c, 'weekly_ok', False) else '❌'} M:{'✅' if getattr(c, 'monthly_ok', False) else '❌'} | Liquidity(ADV$): {round(float(getattr(c, 'avg_dollar_vol', 0) or 0)/1e6,1)}M\n")
    if ai_line:
        parts.append(ai_line)
    parts.append("الأمر المرفق: جني الربح/وقف الخسارة\n")
    if bool(plan.get("one_day", True)):
        parts.append(f"صلاحية الأمر: يوم | أغلق قبل الإغلاق بـ {int(plan.get('close_exit_minutes') or 15)} دقيقة إذا ما تحقق TP/SL\n")
    if plan.get("tp1") is not None and float(plan.get("partial_pct") or 0) > 0:
        parts.append(f"TP1 (جزئي): {plan.get('tp1')} | نسبة: {int(round(float(plan.get('partial_pct') or 0)*100))}%\n")
    parts.append(f"TP2 (نهائي): {plan.get('tp2', plan.get('tp'))}\n")
    if plan.get("trail_note"):
        parts.append(f"Trailing: {plan.get('trail_note','')}\n")
    parts.append(f"وقف الخسارة: {plan['sl']}\n")
    parts.append(f"الخطة: {mode_label}\n")
    parts.append(f"ملاحظة: {c.notes}\n")
    body = "".join(parts)
    return body
def _select_and_log_new_candidates(picks: List[Candidate], settings: Dict[str, str]) -> Tuple[List[str], List[Dict[str, Any]]]:
    """اختيار أفضل الفرص + تسجيلها + تجهيز رسالة التليجرام.
    - يطبق فلتر AI (Score) + فلتر الأخبار (اختياري) + حماية السحب (Drawdown Guard)
    - يحسب Loss Probability + مخاطرة ذكية + كمية (Fractional) حسب رأس المال
    """
    from core.risk_manager import check_drawdown_and_pause
    from core.probability_model import estimate_loss_probability
    from core.news_filter import check_news_risk

    # --- Capital protection (Drawdown) ---
    paused, dd_meta, dd_reasons = check_drawdown_and_pause()
    if paused:
        msg = "🛑 التداول موقوف لحماية رأس المال.\n"
        if dd_reasons:
            msg += "\n".join(dd_reasons) + "\n"
        try:
            dd = float(dd_meta.get("drawdown_pct") or 0.0)
            hwm = float(dd_meta.get("equity_hwm") or 0.0)
            eq = float(dd_meta.get("equity") or 0.0)
            msg += f"Equity: {eq:.2f}$ | High: {hwm:.2f}$ | DD: {dd:.1f}%\n"
        except Exception:
            pass
        msg += "لإلغاء الإيقاف يدويًا: غيّر Setting TRADING_PAUSED إلى 0."
        return [msg], []

    mode = _get_str(settings, "PLAN_MODE", "daily").lower()
    dedup_hours = _get_int(settings, "DEDUP_HOURS", 6)
    allow_resend_stronger = _get_bool(settings, "ALLOW_RESEND_IF_STRONGER", True)
    max_send = _get_int(settings, "MAX_SEND", 10)
    min_send = _get_int(settings, "MIN_SEND", 7)
    now_utc = datetime.now(timezone.utc)
    cutoff = now_utc - timedelta(hours=dedup_hours)
    mode_label = _mode_label(mode)

    # Optional: require multi-timeframe alignment
    req_daily = _get_bool(settings, "REQUIRE_DAILY_OK", True)
    req_weekly = _get_bool(settings, "REQUIRE_WEEKLY_OK", True)
    req_monthly = _get_bool(settings, "REQUIRE_MONTHLY_OK", False)

    def _tf_ok(c: Candidate) -> bool:
        if req_daily and not bool(getattr(c, "daily_ok", False)):
            return False
        if req_weekly and not bool(getattr(c, "weekly_ok", False)):
            return False
        if req_monthly and not bool(getattr(c, "monthly_ok", False)):
            return False
        return True

    candidates = [c for c in picks if _mode_matches(c, mode) and _tf_ok(c)]
    candidates.sort(key=lambda x: x.score, reverse=True)

    blocks: List[str] = []
    logged: List[Dict[str, Any]] = []

    ai_topn = _get_int(settings, "AI_PREDICT_TOPN", 5)
    ai_cache: Dict[str, Optional[dict]] = {}
    ai_used = 0

    def _recently_sent(symbol: str, strength: str) -> bool:
        last = last_signal(symbol, mode)
        if not last:
            return False
        try:
            last_ts = datetime.fromisoformat(str(last["ts"]).replace("Z", "+00:00"))
        except Exception:
            last_ts = datetime(1970, 1, 1, tzinfo=timezone.utc)
        if last_ts < cutoff:
            return False
        if not allow_resend_stronger:
            return True
        prev_rank = _STRENGTH_RANK.get(str(last.get("strength")), 0)
        cur_rank = _STRENGTH_RANK.get(strength, 0)
        return cur_rank <= prev_rank

    for c in candidates:
        if len(blocks) >= max_send:
            break

        st = _strength(float(c.score))
        if _recently_sent(c.symbol, st):
            continue

        side = (getattr(c, "side", "buy") or "buy")
        ai_score: int | None = None
        _ai_reasons: List[str] = []
        _ai_features: Dict[str, Any] = {}

        # --- Deterministic AI filter (score) ---
        if AI_FILTER_ENABLED:
            ok_ai, ai_score, _ai_reasons, _ai_features = should_alert(c.symbol, side, min_score=AI_FILTER_MIN_SCORE)
            if not ok_ai:
                continue

        # --- News filter (optional) ---
        ok_news, news_reasons, news_meta = check_news_risk(c.symbol)
        if not ok_news:
            # If user wants to see rejects, show one-line reject for transparency
            if _get_bool(settings, "NEWS_FILTER_SEND_REJECTS", False):
                blocks.append(f"📰 تم استبعاد {c.symbol} بسبب الأخبار.\n" + "\n".join(news_reasons[:3]))
            continue

        # --- Trade plan ---
        plan = _compute_trade_plan(settings, c)
        # --- One-day rules: صفقة ثانية فقط إذا الأولى ربحت، وخسارة واحدة فقط في اليوم ---
        from core.storage import get_user_state, set_user_state
        today = datetime.now(timezone.utc).date().isoformat()
        chat_key = "GLOBAL"
        one_day_only = _get_bool(settings, "ONE_DAY_ONLY", True)
        second_only_if_win = _get_bool(settings, "SECOND_TRADE_ONLY_IF_WIN", True)
        if one_day_only:
            trades_count = int(float(get_user_state(chat_key, f"daily_trades_{today}", "0") or 0))
            day_status = (get_user_state(chat_key, f"daily_status_{today}", "") or "").lower().strip()  # open|win|loss|flat
            if day_status == "loss":
                continue
            if day_status == "open":
                continue
            if trades_count >= 1 and second_only_if_win and day_status != "win":
                continue
            if trades_count >= 2:
                continue




            loss_prob, lp_reasons = estimate_loss_probability(_ai_features or plan, score=ai_score if ai_score is not None else c.score)
            plan["loss_prob"] = round(float(loss_prob), 3)
            risk_pct = _dynamic_risk_pct(ai_score=float(ai_score) if ai_score is not None else None,
                                         loss_prob=float(loss_prob),
                                         settings=settings)
            if risk_pct <= 0:
                continue

            try:
                capital = _get_float(settings, "CAPITAL_USD", 800.0)
                risk_amount = max(0.10, capital * (risk_pct / 100.0))
                rps = float(plan.get("risk_per_share") or 0.01)
                qty_risk = risk_amount / max(rps, 0.01)
                pos_pct = _get_float(settings, "POSITION_PCT", 0.20)
                max_notional = max(0.0, capital * pos_pct)
                qty_cap = (max_notional / max(float(plan.get("entry") or 0.01), 0.01)) if max_notional > 0 else qty_risk
                qty = max(0.01, min(qty_risk, qty_cap))
                qty = round(float(qty), 3)
                plan["risk_pct"] = round(float(risk_pct), 2)
                plan["risk_amount"] = round(float(risk_amount), 2)
                plan["qty"] = qty
            except Exception:
                pass

            blocks.append(_format_sahm_block(mode_label, c, plan, ai_score=ai_score))
            logged.append({
                "symbol": c.symbol,
                "side": (getattr(c, "side", "buy") or "buy"),
                "strength": st,
                "score": float(c.score),
                "entry": float(plan["entry"]),
                "sl": float(plan["sl"]),
                "tp": float(plan["tp"]),
                "mode": mode,
                "ai_score": ai_score,
                "ml_prob": plan.get("ml_prob"),
                "reasons": (_ai_reasons if AI_FILTER_ENABLED else None),
                "features": (_ai_features if AI_FILTER_ENABLED else None),
            })

    # persist
    ts = now_utc.isoformat()
    horizon_days = int(_get_int(_settings(), "SIGNAL_EVAL_DAYS", SIGNAL_EVAL_DAYS))
    for d in logged:
        try:
            sig_id = log_signal(
                ts=ts,
                symbol=d["symbol"],
                source="scan",
                side=(d.get("side") or "buy"),
                mode=d["mode"],
                strength=d["strength"],
                score=float(d["score"]),
                entry=float(d["entry"]),
                sl=d.get("sl"),
                tp=(d.get("tp1") if d.get("tp1") is not None else d.get("tp")),
                features_json=json.dumps({"ai_features": (d.get("features") or {}), "plan": {k: d.get(k) for k in ["tp1","tp2","qty","risk_pct","risk_amount","loss_prob","setup","setup_notes","one_day","close_exit_minutes"]}}, ensure_ascii=False),
                reasons_json=json.dumps({"ai_reasons": (d.get("reasons") or []), "news": (d.get("news_meta") or {}), "notes": {"mode": d.get("mode"), "strength": d.get("strength")}}, ensure_ascii=False),
                horizon_days=horizon_days,
                model_prob=(float(d.get("ml_prob")) if d.get("ml_prob") is not None else None),
            )
            try:
                if sig_id:
                    # نضيفها تلقائيًا للمراقبة/السجل (بدون أزرار)
                    due = (datetime.now(timezone.utc) + timedelta(hours=24)).isoformat().replace("+00:00","Z")
                    add_paper_trade("GLOBAL", int(sig_id), due)
            except Exception:
                pass
        except Exception:
            continue

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




_LAST_PAPER_MONITOR_RUN: float = 0.0

def _dt_from_iso(ts: str) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(timezone.utc)
    except Exception:
        return None

def _scan_hit_in_bars(symbol: str, side: str, tp: float, sl: float, start: datetime, end: datetime) -> Optional[Dict[str, Any]]:
    """Return hit dict if TP/SL hit in window, with ordering resolved as best as possible."""
    try:
        data = bars([symbol], start=start, end=end, timeframe="5Min", limit=5000)
        bl = (data.get("bars", {}).get(symbol) or [])
    except Exception:
        bl = []
    if not bl:
        return None

    side = (side or "buy").lower()
    want_buy = side != "sell"

    def bar_dt(b):
        t = b.get("t") or ""
        dt = _dt_from_iso(str(t)) if isinstance(t, str) else None
        if dt is None:
            try:
                # Alpaca can return nanoseconds int in some setups
                dt = datetime.fromtimestamp(float(t)/1e9, tz=timezone.utc)
            except Exception:
                dt = None
        return dt

    for b in bl:
        dt = bar_dt(b)
        if dt is None:
            continue
        h = float(b.get("h") or 0.0)
        l = float(b.get("l") or 0.0)

        if want_buy:
            hit_tp = tp > 0 and h >= tp
            hit_sl = sl > 0 and l <= sl
        else:
            # short: TP is lower target, SL is higher stop
            hit_tp = tp > 0 and l <= tp
            hit_sl = sl > 0 and h >= sl

        if hit_tp and hit_sl:
            # tie-break inside this 5m bar using 1Min bars
            try:
                t0 = dt
                t1 = dt + timedelta(minutes=5)
                data1 = bars([symbol], start=t0, end=t1, timeframe="1Min", limit=1000)
                bl1 = (data1.get("bars", {}).get(symbol) or [])
                for b1 in bl1:
                    dt1 = bar_dt(b1)
                    if dt1 is None:
                        continue
                    h1 = float(b1.get("h") or 0.0)
                    l1 = float(b1.get("l") or 0.0)
                    if want_buy:
                        if tp > 0 and h1 >= tp:
                            return {"kind": "tp", "ts": dt1.isoformat().replace("+00:00","Z"), "price": tp}
                        if sl > 0 and l1 <= sl:
                            return {"kind": "sl", "ts": dt1.isoformat().replace("+00:00","Z"), "price": sl}
                    else:
                        if tp > 0 and l1 <= tp:
                            return {"kind": "tp", "ts": dt1.isoformat().replace("+00:00","Z"), "price": tp}
                        if sl > 0 and h1 >= sl:
                            return {"kind": "sl", "ts": dt1.isoformat().replace("+00:00","Z"), "price": sl}
            except Exception:
                pass
            # conservative fallback
            return {"kind": "sl", "ts": dt.isoformat().replace("+00:00","Z"), "price": sl if sl>0 else 0.0}

        if hit_tp:
            return {"kind": "tp", "ts": dt.isoformat().replace("+00:00","Z"), "price": tp}
        if hit_sl:
            return {"kind": "sl", "ts": dt.isoformat().replace("+00:00","Z"), "price": sl}
    return None
def _scan_runner_window(
    symbol: str,
    side: str,
    entry: float,
    sl: float,
    tp1: float,
    tp2: float,
    tp3: float,
    start: datetime,
    end: datetime,
) -> Optional[Dict[str, Any]]:
    """Scan bars chronologically with a runner state machine.

    Returns dict: {kind, ts, price}
      kind in: 'sl','tp','tp2','tp3','trail','tp1_be'
    """
    if entry <= 0:
        return None
    # fallback: if no SL/TP, nothing to do
    if (tp1 <= 0 and tp2 <= 0 and tp3 <= 0) and sl <= 0:
        return None

    state = 0  # 0 before TP1, 1 runner to TP2, 2 runner to TP3
    trail = 0.0

    try:
        data = bars([symbol], start=start, end=end, timeframe="5Min", limit=3000)
        bl = (data.get("bars", {}).get(symbol) or [])
        if not bl:
            return None

        for b in bl:
            ts = b.get("t")
            dt = _dt_from_iso(str(ts)) or None
            if dt is None:
                continue
            h = float(b.get("h") or 0.0)
            l = float(b.get("l") or 0.0)

            if side == "sell":
                # invert comparisons for short
                # TP hits are lows <= target; SL/trail hits are highs >= stop
                def hit_tp(target: float) -> bool:
                    return target > 0 and l <= target
                def hit_stop(stop: float) -> bool:
                    return stop > 0 and h >= stop
            else:
                def hit_tp(target: float) -> bool:
                    return target > 0 and h >= target
                def hit_stop(stop: float) -> bool:
                    return stop > 0 and l <= stop

            if state == 0:
                if hit_stop(sl):
                    return {"kind": "sl", "ts": dt.isoformat().replace("+00:00","Z"), "price": sl}
                if hit_tp(tp1):
                    state = 1
                    trail = entry  # breakeven
                    continue
            elif state == 1:
                if hit_stop(trail):
                    return {"kind": "trail", "ts": dt.isoformat().replace("+00:00","Z"), "price": trail}
                if hit_tp(tp2):
                    state = 2
                    trail = tp1 if tp1 > 0 else entry
                    # If TP3 already hit in this bar (gap), allow it
                    if hit_tp(tp3):
                        return {"kind": "tp3", "ts": dt.isoformat().replace("+00:00","Z"), "price": tp3}
                    continue
            else:
                if hit_stop(trail):
                    return {"kind": "trail", "ts": dt.isoformat().replace("+00:00","Z"), "price": trail}
                if hit_tp(tp3):
                    return {"kind": "tp3", "ts": dt.isoformat().replace("+00:00","Z"), "price": tp3}

        # End of window: if TP1 happened but nothing else, return breakeven (protected)
        if state >= 1:
            return {"kind": "tp1_be", "ts": end.isoformat().replace("+00:00","Z"), "price": entry}
    except Exception:
        return None

    return None



def _run_open_paper_monitor(ttl_sec: float = 30.0) -> None:
    """Monitor active paper trades for TP1/TP2/TP3/Trail hits and notify immediately.

    Multi-stage logic:
      - status=open: monitor TP1 vs SL
      - on TP1: arm runner (trail_sl = entry, status=runner)
      - status=runner: monitor TP2 vs trail_sl (BE)
      - on TP2: lock profits (trail_sl = TP1, status=tp2)
      - status=tp2: monitor TP3 vs trail_sl (TP1)
      - on TP3: mark tp3 hit (status=tp3)
      - on trail hit: mark as sl with hit_kind=trail
    """
    global _LAST_PAPER_MONITOR_RUN
    now = time.time()
    if (now - float(_LAST_PAPER_MONITOR_RUN or 0.0)) < float(ttl_sec):
        return
    _LAST_PAPER_MONITOR_RUN = now

    try:
        rows = open_paper_trades_for_monitor(limit=500)
    except Exception:
        return
    if not rows:
        return

    now_dt = datetime.now(timezone.utc)

    for r in rows:
        try:
            paper_id = int(r.get("id") or 0)
            chat = str(r.get("chat_id") or "")
            symbol = (r.get("symbol") or "").upper().strip()
            mode = (r.get("mode") or "").upper().strip()
            side = (r.get("side") or "buy").lower().strip()

            entry = float(r.get("entry") or 0.0)
            sl = float(r.get("sl") or 0.0)
            tp1 = float(r.get("tp") or 0.0)
            tp2 = float(r.get("tp2") or 0.0)
            tp3 = float(r.get("tp3") or 0.0)
            trail_sl = float(r.get("trail_sl") or 0.0)
            p_status = (r.get("status") or "open").lower().strip()

            if not paper_id or not chat or not symbol or entry <= 0:
                continue

            sig_dt = _dt_from_iso(str(r.get("signal_ts") or "")) or (now_dt - timedelta(hours=26))
            due_dt = _dt_from_iso(str(r.get("due_ts") or "")) or (sig_dt + timedelta(hours=24))
            end_dt = min(now_dt, due_dt)

            last_check = _dt_from_iso(str(r.get("last_check_ts") or "")) or sig_dt
            start_dt = max(last_check - timedelta(minutes=5), sig_dt)

            if end_dt <= start_dt:
                update_paper_trade_monitor_state(paper_id, last_check_ts=end_dt.isoformat().replace("+00:00","Z"))
                continue

            # Decide which targets to monitor based on status
            tp_target = 0.0
            sl_target = 0.0
            tp_kind = "tp"
            sl_kind = "sl"

            if p_status in ("open", ""):
                tp_target = tp1
                sl_target = sl
                tp_kind = "tp"
                sl_kind = "sl"
            elif p_status == "runner":
                tp_target = tp2 if tp2 > 0 else tp1
                sl_target = trail_sl if trail_sl > 0 else entry
                tp_kind = "tp2"
                sl_kind = "trail"
            elif p_status == "tp2":
                tp_target = tp3 if tp3 > 0 else tp2
                sl_target = trail_sl if trail_sl > 0 else tp1
                tp_kind = "tp3"
                sl_kind = "trail"
            else:
                # not monitorable
                update_paper_trade_monitor_state(paper_id, last_check_ts=end_dt.isoformat().replace("+00:00","Z"))
                continue

            if tp_target <= 0 and sl_target <= 0:
                update_paper_trade_monitor_state(paper_id, last_check_ts=end_dt.isoformat().replace("+00:00","Z"))
                continue

            hit = _scan_hit_in_bars(symbol, side, tp_target if tp_target > 0 else 0.0, sl_target if sl_target > 0 else 0.0, start_dt, end_dt)
            update_paper_trade_monitor_state(paper_id, last_check_ts=end_dt.isoformat().replace("+00:00","Z"))

            if not hit:
                continue

            kind = hit.get("kind")  # 'tp' or 'sl' from scanner
            hit_ts = str(hit.get("ts") or now_dt.isoformat().replace("+00:00","Z"))
            hit_price = float(hit.get("price") or 0.0)

            # Map scanner kind to our stage kind
            mapped_kind = tp_kind if kind == "tp" else sl_kind

            # Stage transitions
            if mapped_kind == "tp":
                # TP1 hit → arm runner (trail to breakeven)
                update_paper_trade_monitor_state(
                    paper_id,
                    status="runner",
                    tp_hit=1,
                    tp1_hit=1,
                    trail_sl=entry,
                    trail_mode="BE",
                    hit_kind="tp",
                    hit_ts=hit_ts,
                    hit_price=hit_price,
                )
                title = "تحقق TP1 — تم تفعيل Runner (Trail إلى الدخول)"
                res_emoji = "✅"
                # تحديث حالة اليوم للسماح بصفقة ثانية
                try:
                    from core.storage import set_user_state
                    today = datetime.now(timezone.utc).date().isoformat()
                    set_user_state("GLOBAL", f"daily_status_{today}", "win")
                except Exception:
                    pass
            elif mapped_kind == "tp2":
                # TP2 hit → lock profits (trail to TP1)
                new_trail = tp1 if tp1 > 0 else entry
                update_paper_trade_monitor_state(
                    paper_id,
                    status="tp2",
                    tp_hit=1,
                    tp2_hit=1,
                    trail_sl=new_trail,
                    trail_mode="TP1",
                    hit_kind="tp2",
                    hit_ts=hit_ts,
                    hit_price=hit_price,
                )
                title = "تحقق TP2 — Runner مستمر (Trail إلى TP1)"
                res_emoji = "🏁"
            elif mapped_kind == "tp3":
                # TP3 hit → mark big win
                update_paper_trade_monitor_state(
                    paper_id,
                    status="tp3",
                    tp_hit=1,
                    tp3_hit=1,
                    hit_kind="tp3",
                    hit_ts=hit_ts,
                    hit_price=hit_price,
                )
                title = "تحقق TP3 — صفقة ذهبية"
                res_emoji = "🏆"
            else:
                # SL / Trail hit
                update_paper_trade_monitor_state(
                    paper_id,
                    status="sl",
                    sl_hit=1,
                    hit_kind=mapped_kind,
                    hit_ts=hit_ts,
                    hit_price=hit_price,
                )
                title = "تحقق وقف الخسارة" if mapped_kind == "sl" else "تحقق Trail Stop"
                res_emoji = "❌"
                # تحديث حالة اليوم: توقف بعد خسارة
                try:
                    from core.storage import set_user_state
                    today = datetime.now(timezone.utc).date().isoformat()
                    set_user_state("GLOBAL", f"daily_status_{today}", "loss")
                except Exception:
                    pass

            # R-multiple estimate (vs SL risk if available)
            r_mult = None
            try:
                risk = abs(entry - sl) if sl > 0 else None
                if risk and risk > 1e-9:
                    if side == "sell":
                        reward = (entry - hit_price)
                    else:
                        reward = (hit_price - entry)
                    r_mult = reward / risk
            except Exception:
                pass

            extra_r = f" | R: {r_mult:+.2f}" if r_mult is not None else ""

            msg = (
                f"🔔 {res_emoji} {symbol} ({mode}) — {title}\n\n"
                f"Entry: {entry:.4g}$\n"
                f"SL: {sl:.4g}$\n"
                f"TP1: {tp1:.4g}$\n"
                f"TP2: {tp2:.4g}$\n"
                f"TP3: {tp3:.4g}$\n"
                f"Trail: {trail_sl:.4g}$\n"
                f"hit@: {hit_ts}\n"
                f"{extra_r}"
            )
            _tg_send(chat, msg, silent=True)

            # log snapshot event
            try:
                signal_id = int(r.get("signal_id") or 0)
                if signal_id > 0:
                    note = json.dumps({
                        "kind": "paper_hit",
                        "hit_kind": mapped_kind,
                        "hit_ts": hit_ts,
                        "hit_price": hit_price,
                        "tp1": tp1,
                        "tp2": tp2,
                        "tp3": tp3,
                        "sl": sl,
                        "entry": entry,
                        "side": side,
                        "status": (r.get("status") or ""),
                    }, ensure_ascii=False)
                    log_signal_review(
                        ts=datetime.now(timezone.utc).isoformat(),
                        signal_id=signal_id,
                        close=float(hit_price),
                        return_pct=0.0,
                        mfe_pct=0.0,
                        mae_pct=0.0,
                        note=note,
                    )
            except Exception:
                pass

        except Exception:
            continue



def _run_eod_close_reminder(ttl_sec: float = 60.0) -> None:
    """Reminder to close 1D one-day positions before market close.

    Since التنفيذ يدوي عبر (سهم)، نرسل تذكير قبل الإغلاق بـ CLOSE_EXIT_MINUTES.
    """
    global _LAST_EOD_REMINDER_RUN
    try:
        _LAST_EOD_REMINDER_RUN
    except Exception:
        _LAST_EOD_REMINDER_RUN = 0.0  # type: ignore

    now = time.time()
    if (now - float(_LAST_EOD_REMINDER_RUN or 0.0)) < float(ttl_sec):
        return
    _LAST_EOD_REMINDER_RUN = now  # type: ignore

    s = get_all_settings()
    close_min = _get_int(s, "CLOSE_EXIT_MINUTES", 15)
    one_day_only = _get_bool(s, "ONE_DAY_ONLY", True)
    if not one_day_only:
        return

    try:
        c = clock() or {}
    except Exception:
        return
    if not c.get("is_open"):
        return

    ts = _dt_from_iso(str(c.get("timestamp") or "")) or datetime.now(timezone.utc)
    next_close = _dt_from_iso(str(c.get("next_close") or "")) or None
    if not next_close:
        return

    minutes_to_close = (next_close - ts).total_seconds() / 60.0
    # trigger once when we enter the window
    if minutes_to_close > float(close_min) or minutes_to_close < max(1.0, float(close_min) - 5.0):
        return

    today = datetime.now(timezone.utc).date().isoformat()
    from core.storage import get_user_state, set_user_state
    status = (get_user_state("GLOBAL", f"daily_status_{today}", "") or "").lower().strip()
    if status != "open":
        return
    if (get_user_state("GLOBAL", f"daily_eod_notified_{today}", "0") or "0") == "1":
        return

    plan_raw = get_user_state("GLOBAL", f"daily_last_plan_{today}", "") or ""
    try:
        plan = json.loads(plan_raw) if plan_raw.strip().startswith("{") else {}
    except Exception:
        plan = {}
    symbol = (plan.get("symbol") or "").upper().strip()
    if not symbol:
        return

    live_p, live_ts = _get_live_trade_price(symbol)
    p_line = f"السعر الحالي: {live_p} ({live_ts})\n" if live_p is not None else ""

    msg = (
        f"⏰ تذكير إغلاق (صفقة يوم واحد)\n"
        f"السهم: {symbol}\n"
        f"{p_line}"
        f"إذا ما تحقق TP/SL: اغلق الصفقة قبل الإغلاق بـ {close_min} دقيقة.\n"
        f"الدخول: {plan.get('entry')} | SL: {plan.get('sl')} | TP1: {plan.get('tp1')} | TP2: {plan.get('tp2')}"
    )
    _notify_simple(msg, s, silent=False)
    set_user_state("GLOBAL", f"daily_eod_notified_{today}", "1")

def _run_due_paper_reviews(ttl_sec: float = 60.0) -> None:
    """Check for due 24h paper trades and send results to their chats (throttled)."""
    global _LAST_PAPER_REVIEW_RUN
    now = time.time()
    if (now - float(_LAST_PAPER_REVIEW_RUN or 0.0)) < float(ttl_sec):
        return
    _LAST_PAPER_REVIEW_RUN = now

    try:
        rows = due_paper_trades(limit=200)
    except Exception:
        return
    if not rows:
        return

    for r in rows:
        paper_id = 0
        try:
            paper_id = int(r.get("id") or 0)
            chat = str(r.get("chat_id") or "")
            symbol = (r.get("symbol") or "").upper().strip()
            mode = (r.get("mode") or "").upper().strip()
            side = (r.get("side") or "buy").lower().strip()
            entry = float(r.get("entry") or 0.0)
            if not chat or not symbol or entry <= 0:
                continue


            tp = float(r.get("tp") or 0.0)
            sl = float(r.get("sl") or 0.0)
            p_status = (r.get("status") or "open").lower().strip()
            hit_price = float(r.get("hit_price") or 0.0)
            hit_ts = str(r.get("hit_ts") or "")

            # If TP/SL was already hit during monitoring, finalize based on that hit.
            if p_status in ("tp", "tp2", "tp3", "sl") and hit_price > 0:
                exit_price = hit_price
                live_p = None
                live_ts = None
                price_src = "TP_HIT" if p_status == "tp" else ("TP2_HIT" if p_status=="tp2" else ("TP3_HIT" if p_status=="tp3" else "SL_HIT"))
            else:
                exit_price = None
                price_src = ""
            
            # If still open at finalize time, scan the full 24h window for TP/SL hit to avoid missing events.
            if exit_price is None and (tp > 0 or sl > 0):
                try:
                    sig_dt = _dt_from_iso(str(r.get("signal_ts") or "")) or datetime.now(timezone.utc) - timedelta(hours=26)
                    due_dt = _dt_from_iso(str(r.get("due_ts") or "")) or (sig_dt + timedelta(hours=24))
                    hit2 = _scan_runner_window(symbol, side, entry, sl, tp, float(r.get('tp2') or 0.0), float(r.get('tp3') or 0.0), sig_dt, due_dt)
                    if hit2:
                        kind2 = hit2.get("kind")
                        hit_ts2 = str(hit2.get("ts") or "")
                        hit_price2 = float(hit2.get("price") or 0.0)
                        if kind2 in ("tp", "tp2", "tp3"):
                            p_status = kind2
                            update_paper_trade_monitor_state(paper_id, status=kind2, tp_hit=1,
                                                           tp1_hit=1 if kind2=="tp" else None,
                                                           tp2_hit=1 if kind2=="tp2" else None,
                                                           tp3_hit=1 if kind2=="tp3" else None,
                                                           hit_kind=kind2, hit_ts=hit_ts2, hit_price=hit_price2)
                            price_src = "TP_HIT" if kind2=="tp" else ("TP2_HIT" if kind2=="tp2" else "TP3_HIT")
                        elif kind2 == "tp1_be":
                            p_status = "final"
                            update_paper_trade_monitor_state(paper_id, status="final", tp_hit=1, tp1_hit=1, hit_kind="tp1_be",
                                                           hit_ts=hit_ts2, hit_price=hit_price2)
                            price_src = "TP1_BE"
                        else:
                            p_status = "sl"
                            hk = "trail" if kind2 == "trail" else "sl"
                            update_paper_trade_monitor_state(paper_id, status="sl", sl_hit=1, hit_kind=hk, hit_ts=hit_ts2, hit_price=hit_price2)
                            price_src = "TRAIL_HIT" if hk=="trail" else "SL_HIT"
                        hit_ts = hit_ts2
                        hit_price = hit_price2
                        exit_price = hit_price2
                except Exception:
                    pass

# Use live price now if possible; fallback to last close for last 1D bar
            if exit_price is None:
                live_p, live_ts = _get_live_trade_price(symbol)
                exit_price = float(live_p) if live_p is not None else None
                price_src = "LIVE" if exit_price is not None else "LAST_CLOSE"

            if exit_price is None:
                try:
                    dt = datetime.now(timezone.utc)
                    data = bars([symbol], start=dt - timedelta(days=3), end=dt, timeframe="1Day", limit=5)
                    bl = (data.get("bars", {}).get(symbol) or [])
                    if bl:
                        exit_price = float(bl[-1].get("c") or entry)
                except Exception:
                    exit_price = entry

            if exit_price is None:
                exit_price = entry

            if side == "sell":
                ret_pct = (entry - exit_price) / entry * 100.0
            else:
                ret_pct = (exit_price - entry) / entry * 100.0

            res = "✅ ربح" if ret_pct > 0 else ("❌ خسارة" if ret_pct < 0 else "➖ تعادل")
            ts_line = f"وقت المراجعة: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
            src_line = f"مصدر سعر الخروج: {price_src}" + (f" ({live_ts})" if live_ts and price_src == "LIVE" else "")
            msg = (
                f"📊 مراجعة بعد 24 ساعة — {symbol} ({mode})\n\n"
                f"الدخول (افتراضي): {entry:.4g}$\n"
                f"سعر الآن: {float(exit_price):.4g}$\n"
                f"النتيجة: {res} ({ret_pct:+.2f}%)\n\n"
                f"{src_line}\n{ts_line}"
            )
            _tg_send(chat, msg, silent=True)
            # Freeze this 24h review as a snapshot so it does NOT change later.
            try:
                signal_id = int(r.get("signal_id") or 0)
                if signal_id > 0:
                    note = json.dumps({
                        "kind": "paper_24h_final",
                        "price_src": price_src,
                        "live_ts": live_ts or "",
                        "review_ts": datetime.now(timezone.utc).isoformat(),
                        "tp": tp,
                        "tp2": float(r.get("tp2") or 0.0),
                        "tp3": float(r.get("tp3") or 0.0),
                        "sl": sl,
                        "entry": entry,
                        "side": side,
                        "hit_kind": str(r.get("hit_kind") or ""),
                        "paper_status": p_status,
                        "hit_ts": hit_ts,
                        "hit_price": hit_price,
                        "tp_hit": 1 if p_status=="tp" else int(r.get("tp_hit") or 0),
                        "sl_hit": 1 if p_status=="sl" else int(r.get("sl_hit") or 0),
                    }, ensure_ascii=False)
                    log_signal_review(
                        ts=datetime.now(timezone.utc).isoformat(),
                        signal_id=signal_id,
                        close=float(exit_price),
                        return_pct=float(ret_pct),
                        mfe_pct=0.0,
                        mae_pct=0.0,
                        note=note,
                    )
            except Exception:
                pass

        except Exception:
            pass
        finally:
            try:
                if paper_id:
                    mark_paper_trade_notified(paper_id)
            except Exception:
                pass
@app.post("/webhook")
def telegram_webhook():
    try:
        if not TELEGRAM_BOT_TOKEN:
            return jsonify({"ok": True})
        data = request.get_json(silent=True) or {}
        message_id: Optional[int] = None  # for UI edits; only set for callback_query messages
        # Handle button clicks
        cb = data.get("callback_query")
        if cb:
            user_id = cb.get("from", {}).get("id")
            chat_id = cb.get("message", {}).get("chat", {}).get("id")
            message_id = cb.get("message", {}).get("message_id")
            action = (cb.get("data") or "").strip()
            callback_id = cb.get("id")

            # Convenience aliases for BotFather-like UX (edit same message on button clicks)
            _chat = str(chat_id) if chat_id is not None else ""
            _mid: Optional[int] = int(message_id) if message_id is not None else None

            def _ui(text: str, reply_markup: Optional[Dict[str, Any]] = None, silent: bool = False) -> None:
                _tg_ui(_chat, _mid, text, reply_markup=reply_markup, silent=silent)

            def _send(text: str, reply_markup: Optional[Dict[str, Any]] = None, silent: bool = False) -> None:
                """Force a new message (use sparingly)."""
                _tg_send(_chat, text, reply_markup=reply_markup, silent=silent)

            # Dedupe / Debounce BEFORE ack text so user gets immediate feedback
            if callback_id and _seen_and_mark(_CB_SEEN, str(callback_id), float(_CB_TTL_SEC)):
                _tg_answer_callback(callback_id, text="⏳ تم تنفيذ هذا الزر للتو", show_alert=False)
                return jsonify({"ok": True})

            if chat_id is not None and action:
                # Don't debounce lightweight UI/navigation actions; otherwise the UI feels "dead"
                _ui_no_debounce = (
                    action == "menu"
                    or action.startswith("show_")
                    or action.startswith("set_")
                    or action.startswith("my_sig_")
                    or action.startswith("wl_")
                    or action in ("show_settings",)
                )
                if (not _ui_no_debounce) and _seen_and_mark(_ACTION_SEEN, f"{chat_id}:{action}", float(_ACTION_DEBOUNCE_SEC)):
                    _tg_answer_callback(callback_id, text="⏳ انتظر لحظة...", show_alert=False)
                    return jsonify({"ok": True})

            # IMPORTANT: acknowledge callback fast to avoid spinner/retries
            _tg_answer_callback(callback_id)
            if not _is_admin(user_id):
                _ui("⛔ هذا البوت للأدمن فقط.", reply_markup=_build_menu(_settings()))
                return jsonify({"ok": True})
            settings = _settings()
            _run_due_paper_reviews()
            if action == "self_check":
                try:
                    rep = _self_check(fix=False)
                    _ui(_self_check_text(rep), reply_markup=_build_menu(_settings()))
                except Exception as e:
                    _ui(f"❌ خطأ في الفحص الذاتي:\n{e}", reply_markup=_build_menu(_settings()))
                return jsonify({"ok": True})

            if action == "paper_log":
                try:
                    from core.storage import get_user_state, set_user_state
                    raw = get_user_state(str(chat_id), "last_pick") or ""
                    if not raw:
                        _ui("⚠️ لا يوجد آخر سهم محفوظ. اضغط D1 أو M5 أولاً.", reply_markup=_build_menu(_settings()))
                        return jsonify({"ok": True})
                    info = json.loads(raw)
                    symbol = (info.get("symbol") or "").upper().strip()
                    mode = (info.get("mode") or "").lower().strip() or "d1"
                    side = (info.get("side") or "buy").lower().strip()
                    entry = float(info.get("entry") or 0.0)
                    sl = info.get("sl")
                    tp = info.get("tp")
                    score = float(info.get("score") or 0.0)
                    strength = (info.get("strength") or "B")
                    if not symbol or entry <= 0:
                        _ui("⚠️ بيانات الإشارة غير مكتملة.", reply_markup=_build_menu(_settings()))
                        return jsonify({"ok": True})

                    ts = datetime.now(timezone.utc).isoformat()
                    sig_id = log_signal(
                        ts=ts,
                        symbol=symbol,
                        source="paper_log",
                        side=side,
                        mode=mode,
                        strength=strength,
                        score=score,
                        entry=entry,
                        sl=float(sl) if sl is not None else None,
                        tp=float(tp) if tp is not None else None,
                        features_json="",
                        reasons_json="",
                        horizon_days=1,
                        model_prob=None,
                    )
                    if sig_id is None:
                        _ui(f"📝 تم تسجيل صفقة وهمية لـ {symbol}. سأراجعها بعد 24 ساعة.", reply_markup=_build_menu(_settings()))
                        return jsonify({"ok": True})

                    due = (datetime.now(timezone.utc) + timedelta(hours=24)).isoformat()
                    add_paper_trade(str(chat_id), int(sig_id), due)
                    set_user_state(str(chat_id), "last_pick_logged", ts)
                    _ui(f"📝 تم تسجيل صفقة وهمية لـ {symbol} بسعر {entry:.4g}$\nسأراجعها بعد 24 ساعة تلقائياً ✅", reply_markup=_build_menu(_settings()), silent=True)
                except Exception as e:
                    _ui(f"❌ خطأ أثناء تسجيل الصفقة الوهمية:\n{e}", reply_markup=_build_menu(_settings()))
                return jsonify({"ok": True})


            # ================= التقرير الأسبوعي =================
            if action == "weekly_report":
                _ui("⏳ جاري إعداد التقرير الأسبوعي...")
                def _job():
                    try:
                        days = int(_get_int(_settings(), "WEEKLY_REPORT_DAYS", 7))
                        msg = _weekly_report(days=days)
                        _tg_ui(_chat, _mid, msg, reply_markup=_build_menu(_settings()))
                    except Exception as e:
                        _tg_ui(_chat, _mid, f"❌ خطأ أثناء إنشاء التقرير الأسبوعي:\n{e}", reply_markup=_build_menu(_settings()))
                _run_async(_job)
                return jsonify({"ok": True})


            if action == "menu":
                _tg_ui(str(chat_id), message_id, "📌 اختر:", reply_markup=_build_menu(settings))
                return jsonify({"ok": True})

            # 📊 إشاراتي (submenu)
            if action == "my_sig_menu":
                _tg_ui(str(chat_id), message_id, "📊 إشاراتي:", reply_markup=_build_my_signals_root_kb())
                return jsonify({"ok": True})

            # Backward compatibility
            if action == "review_signals":
                _tg_ui(str(chat_id), message_id, "📊 إشاراتي:", reply_markup=_build_my_signals_root_kb())
                return jsonify({"ok": True})

            # 📈 مراجعة الأداء (مرتبطة بالشارات المحفوظة فقط)
            if action in ("my_sig_review", "my_sig_review_refresh"):
                msg = _review_my_saved_performance(str(chat_id), lookback_days=2, limit=80)
                _tg_ui(str(chat_id), message_id, msg, reply_markup=_build_my_sig_review_kb(back_action="my_sig_menu"))
                return jsonify({"ok": True})


            
            # 📊 مراجعات 24 ساعة (مقفلة)
            if action in ("my_sig_24h", "my_sig_24h_refresh"):
                try:
                    _run_open_paper_monitor(ttl_sec=0.0)
                    _run_due_paper_reviews(ttl_sec=0.0)
                except Exception:
                    pass
                msg = _my_saved_24h_reviews_message(str(chat_id), lookback_days=30, limit=50)
                _ui(msg, reply_markup=_build_my_sig_24h_kb(back_action="my_sig_menu"))
                return jsonify({"ok": True})


            if action == "my_sig_dash":
                msg = _my_signals_dashboard_message(str(chat_id), lookback_days=30)
                _ui(msg, reply_markup=_ikb([[("⬅️ رجوع", "my_sig_menu")]]))
                return jsonify({"ok": True})

# 📌 شاراتي المحفوظة
            if action in ("my_sig_list", "my_sig_refresh"):
                msg, items = _my_saved_signals_message(str(chat_id), lookback_days=7, limit=80)
                _ui(msg, reply_markup=_build_my_signals_kb(has_items=bool(items), back_action="my_sig_menu"))
                return jsonify({"ok": True})

            # 🗑 حذف صفقة واحدة
            if action == "my_sig_delete":
                msg, items = _my_saved_signals_message(str(chat_id), lookback_days=7, limit=80)
                if not items:
                    _ui(msg, reply_markup=_build_my_signals_kb(has_items=False, back_action="my_sig_menu"))
                    return jsonify({"ok": True})
                _tg_ui(str(chat_id), message_id, "اختر الإشارة التي تريد حذفها:", reply_markup=_build_my_signals_delete_kb(items))
                return jsonify({"ok": True})

            # 🧹 حذف الكل
            if action == "my_sig_delall":
                try:
                    from core.storage import clear_paper_trades_for_chat
                    clear_paper_trades_for_chat(str(chat_id))
                    _tg_ui(str(chat_id), message_id, "✅ تم حذف جميع الشارات من قائمتك.")
                except Exception as e:
                    _tg_ui(str(chat_id), message_id, f"❌ تعذر حذف الكل:\n{e}")
                _tg_ui(str(chat_id), message_id, "📊 إشاراتي:", reply_markup=_build_my_signals_root_kb())
                return jsonify({"ok": True})

            if action.startswith("del_sig:"):
                try:
                    pid = int(action.split(":", 1)[1])
                    delete_paper_trade_for_chat(str(chat_id), pid)
                    _tg_ui(str(chat_id), message_id, "✅ تم حذف الإشارة من قائمتك.")
                except Exception as e:
                    _tg_ui(str(chat_id), message_id, f"❌ تعذر الحذف:\n{e}")
                # show updated list
                msg, items = _my_saved_signals_message(str(chat_id), lookback_days=7, limit=80)
                _ui(msg, reply_markup=_build_my_signals_kb(has_items=bool(items), back_action="my_sig_menu"))
                return jsonify({"ok": True})

            if action in ("ai_top_ev", "ai_top_prob", "ai_top_m5"):
                s = _settings()
                # 3) M5 uses intraday cache (fast)
                if action == "ai_top_m5":
                    try:
                        _update_cache_m5()
                    except Exception:
                        pass
                    with _PICK_LOCK:
                        items = list((_PICK_CACHE.get("m5") or {}).get("items") or [])
                    items = items[:10]
                    out = []
                    for it in items:
                        sym = it.get("symbol")
                        if not sym: 
                            continue
                        sc = float(it.get("score", 0.0))
                        direction = it.get("direction") or ""
                        label = f"{sym} | {direction} | {sc:.0f}"
                        out.append({"symbol": sym, "label": label})
                    if not out:
                        _tg_ui(str(chat_id), message_id, "❌ لا توجد فرص M5 الآن (قد يكون السوق مغلق).", reply_markup=_build_menu(s))
                        return jsonify({"ok": True})
                    _tg_ui(str(chat_id), message_id, "🧠 Top 10 (3- سكالبينغ M5): اختر سهم", reply_markup=_build_top10_kb(out))
                    return jsonify({"ok": True})

                # 1-2) D1 ranking: compute plans + ML probability/EV (best-effort)
                picks, universe_size = scan_universe_with_meta()
                ranked = []
                for c in (picks or [])[:max(30, min(120, len(picks) if picks else 0))]:
                    try:
                        sym = c.symbol
                        side = "buy" if str(getattr(c, "side", "buy")) in ("buy","long") else "sell"
                        passed, ai_score, ai_reasons, ai_features = (True, None, [], {})
                        if AI_FILTER_ENABLED:
                            try:
                                ai_score, ai_reasons, ai_features = score_signal(sym, side=side)
                            except Exception:
                                ai_score, ai_reasons, ai_features = (None, [], {})
                        # Plan based on candidate values
                        entry = float(getattr(c, "last_close", 0.0) or 0.0)
                        atr = float(getattr(c, "atr", 0.0) or 0.0)
                        plan = _build_trade_plan(sym, side=side, entry=entry, atr=atr, settings=s, score=float(getattr(c,'score',0.0) or 0.0))
                        p = None
                        ev = None
                        try:
                            if ML_ENABLED and AI_FILTER_ENABLED and (ai_features is not None):
                                w = parse_weights(s.get("ML_WEIGHTS") or "")
                                x = featurize(ai_features)
                                p = float(predict_prob(w, x))
                                tp_r = float(plan.get("tp_r_mult") or 0.0)
                                ev = (p * tp_r) - ((1.0 - p) * 1.0)
                        except Exception:
                            p, ev = (None, None)
                        ranked.append({
                            "symbol": sym,
                            "ai_score": ai_score,
                            "ml_prob": p,
                            "ev_r": ev,
                        })
                    except Exception:
                        continue

                if not ranked:
                    _tg_ui(str(chat_id), message_id, "❌ لا توجد نتائج الآن.", reply_markup=_build_menu(s))
                    return jsonify({"ok": True})

                if action == "ai_top_prob":
                    ranked.sort(key=lambda x: (x["ml_prob"] is None, -(x["ml_prob"] or 0.0), -(x["ai_score"] or 0)), reverse=False)
                    title = "🧠 Top 10 (2- أعلى احتمال): اختر سهم"
                    out=[]
                    for it in ranked[:10]:
                        sym=it["symbol"]
                        p=it["ml_prob"]
                        sc=it["ai_score"]
                        label=f"{sym} | P {p:.2f}" if p is not None else f"{sym} | P ?"
                        if sc is not None:
                            label += f" | S {sc}"
                        out.append({"symbol": sym, "label": label})
                    _tg_ui(str(chat_id), message_id, title, reply_markup=_build_top10_kb(out))
                    return jsonify({"ok": True})

                # ai_top_ev
                ranked.sort(key=lambda x: (x["ev_r"] is None, -(x["ev_r"] or -999.0), -(x["ml_prob"] or 0.0), -(x["ai_score"] or 0)), reverse=False)
                title = "🧠 Top 10 (1- أفضل EV): اختر سهم"
                out=[]
                for it in ranked[:10]:
                    sym=it["symbol"]
                    ev=it["ev_r"]
                    p=it["ml_prob"]
                    label=f"{sym} | EV {ev:.2f}" if ev is not None else f"{sym} | EV ?"
                    if p is not None:
                        label += f" | P {p:.2f}"
                    out.append({"symbol": sym, "label": label})
                _tg_ui(str(chat_id), message_id, title, reply_markup=_build_top10_kb(out))
                return jsonify({"ok": True})

            if action.startswith("ai_pick:"):
                sym = action.split(":", 1)[1].strip().upper()
                _start_ai_symbol_analysis(str(chat_id), sym)
                return jsonify({"ok": True})
            if action == "ai_symbol_start":
                from core.storage import set_user_state
                set_user_state(str(chat_id), "pending", "ai_symbol")
                _ui("🧠 اكتب رمز السهم الآن (مثال: TSLA)\nأو اكتب /ai TSLA", reply_markup=_build_ai_start_kb())
                return jsonify({"ok": True})
            if action == "ai_cancel":
                from core.storage import clear_user_state
                clear_user_state(str(chat_id), "pending")
                _ui("✅ تم الإلغاء.", reply_markup=_build_menu(_settings()))
                return jsonify({"ok": True})
            if action == "show_modes":
                _tg_ui(str(chat_id), message_id, "📆 اختر الخطة الزمنية:", reply_markup=_build_modes_kb())
                return jsonify({"ok": True})
            if action.startswith("set_mode:"):
                mode = action.split(":", 1)[1]
                set_setting("PLAN_MODE", mode)
                settings = _settings()
                _tg_ui(str(chat_id), message_id, f"✅ تم ضبط الخطة: {_mode_label(mode)}", reply_markup=_build_menu(settings))
                return jsonify({"ok": True})
            if action == "show_entry":
                _tg_ui(str(chat_id), message_id, "🎯 اختر نوع الدخول:", reply_markup=_build_entry_kb())
                return jsonify({"ok": True})
            if action.startswith("set_entry:"):
                entry = action.split(":", 1)[1]
                set_setting("ENTRY_MODE", entry)
                settings = _settings()
                _tg_ui(str(chat_id), message_id, f"✅ نوع الدخول: {_entry_type_label(entry)}", reply_markup=_build_menu(settings))
                return jsonify({"ok": True})
            if action == "toggle_notify":
                cur = _get_bool(settings, "AUTO_NOTIFY", True)
                set_setting("AUTO_NOTIFY", "0" if cur else "1")
                settings = _settings()
                _ui("✅ تم تحديث التنبيهات.", reply_markup=_build_settings_kb(settings))
                return jsonify({"ok": True})
            if action == "toggle_ai_predict":
                cur = _get_bool(settings, "AI_PREDICT_ENABLED", False)
                set_setting("AI_PREDICT_ENABLED", "0" if cur else "1")
                settings = _settings()
                _ui("✅ تم تحديث تنبؤ AI.", reply_markup=_build_settings_kb(settings))
                return jsonify({"ok": True})
            if action == "show_horizon":
                _tg_ui(str(chat_id), message_id, "🤖 اختر إطار التنبؤ (يؤثر على تحليل AI فقط):", reply_markup=_build_horizon_kb(settings))
                return jsonify({"ok": True})
            if action.startswith("set_horizon:"):
                val = action.split(":", 1)[1].strip().upper()
                if val in ("HYBRID", "M5PLUS"):
                    val = "M5+"
                if val not in ("D1", "M5", "M5+"):
                    val = "D1"
                set_setting("PREDICT_FRAME", val)
                s = _settings()
                _tg_ui(str(chat_id), message_id, f"✅ تم ضبط إطار التنبؤ: {val}", reply_markup=_build_settings_kb(s))
                return jsonify({"ok": True})
            if action == "show_notify_route":
                _tg_ui(str(chat_id), message_id, "📨 اختر وجهة التنبيهات:", reply_markup=_build_notify_route_kb())
                return jsonify({"ok": True})
            if action.startswith("set_notify_route:"):
                route = action.split(":", 1)[1].strip().lower()
                if route not in ("dm", "group", "both"):
                    route = "dm"
                set_setting("NOTIFY_ROUTE", route)
                settings = _settings()
                _tg_ui(str(chat_id), message_id, "✅ تم تحديث الوجهة.", reply_markup=_build_menu(settings))
                return jsonify({"ok": True})
            if action == "toggle_silent":
                cur = _get_bool(settings, "NOTIFY_SILENT", True)
                set_setting("NOTIFY_SILENT", "0" if cur else "1")
                settings = _settings()
                _ui("✅ تم تحديث وضع الصامت.", reply_markup=_build_menu(settings))
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
                    f"- إطار التنبؤ (AI): {_get_str(s,'PREDICT_FRAME','D1')} | AI تنبؤ: {'ON' if _get_bool(s,'AI_PREDICT_ENABLED',False) else 'OFF'}\n"
                )
                _tg_ui(str(chat_id), message_id, txt, reply_markup=_build_settings_kb(s))
                return jsonify({"ok": True})
            if action == "show_capital":
                reply = _build_capital_kb() if "_build_capital_kb" in globals() else {"inline_keyboard":[[{"text":"✍️ قيمة مخصصة","callback_data":"set_capital_custom"}],[{"text":"⬅️ رجوع","callback_data":"show_settings"}]]}
                _tg_ui(str(chat_id), message_id, "💰 اختر رأس المال بالدولار:", reply_markup=reply)
                return jsonify({"ok": True})
            if action == "set_capital_custom":
                from core.storage import set_user_state
                set_user_state(str(chat_id), "pending", "capital")
                _ui("✍️ أرسل رقم رأس المال بالدولار (مثال: 5000)")
                return jsonify({"ok": True})
            if action.startswith("set_capital:"):
                val = action.split(":", 1)[1]
                set_setting("CAPITAL_USD", val)
                s = _settings()
                _ui(f"✅ تم ضبط رأس المال: {val}$", reply_markup=_build_settings_kb(s))
                return jsonify({"ok": True})
            if action == "show_position":
                _tg_ui(str(chat_id), message_id, "📦 اختر نسبة حجم الصفقة من رأس المال:", reply_markup=_build_position_kb())
                return jsonify({"ok": True})
            if action.startswith("set_position:"):
                val = action.split(":", 1)[1]
                set_setting("POSITION_PCT", val)
                s = _settings()
                _tg_ui(str(chat_id), message_id, f"✅ تم ضبط حجم الصفقة: {float(val)*100:.0f}%", reply_markup=_build_settings_kb(s))
                return jsonify({"ok": True})
            if action == "show_sl":
                _tg_ui(str(chat_id), message_id, "📉 اختر وقف الخسارة %:", reply_markup=_build_sl_kb())
                return jsonify({"ok": True})
            if action.startswith("set_sl:"):
                val = action.split(":", 1)[1]
                set_setting("SL_PCT", val)
                s = _settings()
                _tg_ui(str(chat_id), message_id, f"✅ تم ضبط وقف الخسارة: {val}%", reply_markup=_build_settings_kb(s))
                return jsonify({"ok": True})
            if action == "show_tp":
                _tg_ui(str(chat_id), message_id, "📈 اختر جني الربح % (لضعيف/متوسط):", reply_markup=_build_tp_kb())
                return jsonify({"ok": True})
            if action.startswith("set_tp:"):
                val = action.split(":", 1)[1]
                set_setting("TP_PCT", val)
                s = _settings()
                _tg_ui(str(chat_id), message_id, f"✅ تم ضبط جني الربح (لضعيف/متوسط): {val}%", reply_markup=_build_settings_kb(s))
                return jsonify({"ok": True})
            if action == "show_send":
                _tg_ui(str(chat_id), message_id, "🎛 اختر عدد الفرص في كل فحص:", reply_markup=_build_send_kb())
                return jsonify({"ok": True})
            if action.startswith("set_send:"):
                parts = action.split(":")
                if len(parts) == 3:
                    set_setting("MIN_SEND", parts[1])
                    set_setting("MAX_SEND", parts[2])
                s = _settings()
                _tg_ui(str(chat_id), message_id, f"✅ تم ضبط عدد الفرص: {s.get('MIN_SEND','7')} إلى {s.get('MAX_SEND','10')}", reply_markup=_build_settings_kb(s))
                return jsonify({"ok": True})
            if action == "toggle_resend":
                cur = _get_bool(settings, "ALLOW_RESEND_IF_STRONGER", True)
                set_setting("ALLOW_RESEND_IF_STRONGER", "0" if cur else "1")
                s = _settings()
                _ui("✅ تم تحديث خيار إعادة الإرسال.", reply_markup=_build_settings_kb(s))
                return jsonify({"ok": True})
            if action == "show_window":
                _tg_ui(str(chat_id), message_id, "🕒 اختر نافذة السوق (بتوقيت الرياض):", reply_markup=_build_window_kb())
                return jsonify({"ok": True})
            if action.startswith("set_window:"):
                parts = action.split(":")
                if len(parts) == 3:
                    set_setting("WINDOW_START", parts[1])
                    set_setting("WINDOW_END", parts[2])
                s = _settings()
                _tg_ui(str(chat_id), message_id, f"✅ تم ضبط النافذة: {s.get('WINDOW_START','17:30')}→{s.get('WINDOW_END','00:00')}", reply_markup=_build_settings_kb(s))
                return jsonify({"ok": True})
            if action == "noop":
                return jsonify({"ok": True})
            if action == "show_risk":
                _tg_ui(str(chat_id), message_id, "⚖️ اختر نسب المخاطرة حسب التصنيف (A+/A/B):", reply_markup=_build_risk_kb(settings))
                return jsonify({"ok": True})
            if action.startswith("set_risk_aplus:"):
                val = action.split(":", 1)[1]
                set_setting("RISK_APLUS_PCT", val)
                s = _settings()
                _tg_ui(str(chat_id), message_id, f"✅ تم ضبط مخاطرة A+: {val}%", reply_markup=_build_settings_kb(s))
                return jsonify({"ok": True})
            if action.startswith("set_risk_a:"):
                val = action.split(":", 1)[1]
                set_setting("RISK_A_PCT", val)
                s = _settings()
                _tg_ui(str(chat_id), message_id, f"✅ تم ضبط مخاطرة A: {val}%", reply_markup=_build_settings_kb(s))
                return jsonify({"ok": True})
            if action.startswith("set_risk_b:"):
                val = action.split(":", 1)[1]
                set_setting("RISK_B_PCT", val)
                s = _settings()
                _tg_ui(str(chat_id), message_id, f"✅ تم ضبط مخاطرة B: {val}%", reply_markup=_build_settings_kb(s))
                return jsonify({"ok": True})
            if action == "show_interval":
                _tg_ui(str(chat_id), message_id, "⏱️ اختر فترة الفحص:", reply_markup=_build_interval_kb(settings))
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
                _tg_ui(str(chat_id), message_id, f"✅ تم ضبط فترة الفحص: {val} دقيقة", reply_markup=_build_settings_kb(s))
                return jsonify({"ok": True})

            if action == "pick_next":
                # Show next cached pick for the last mode (D1/M5) without going back to the main menu
                chat = str(chat_id)
                tf = "d1"
                try:
                    from core.storage import get_user_state
                    raw = get_user_state(chat, "last_pick") or ""
                    info = json.loads(raw) if raw else {}
                    tf = "m5" if str(info.get("mode") or "").lower() == "m5" else "d1"
                except Exception:
                    tf = "d1"

                # Market-hours filter for scalping signals
                if tf == "m5":
                    ms = _market_status_cached()
                    if not ms.get("is_open", True):
                        _tg_ui(chat, message_id, _format_market_status_line(ms) + "\n\n⛔ إشارات M5 تُرسل فقط وقت فتح السوق (لتفادي سيولة ضعيفة).\nجرّب زر D1.", reply_markup=_ikb([[("⬅️ رجوع", "menu")]]))
                        return jsonify({"ok": True})

                pick = _get_next_pick(tf, chat)
                if not pick:
                    _tg_ui(chat, message_id, "⚠️ لا توجد نتائج جاهزة الآن. جرّب تحديث الفحص ثم أعد المحاولة.", reply_markup=_ikb([[("⬅️ رجوع", "menu")]]))
                    return jsonify({"ok": True})

                if tf == "m5":
                    try:
                        from core.storage import set_user_state
                        entry_p = float(pick.get("last") or 0.0)
                        info2 = {"symbol": str(pick.get("symbol") or "").upper(), "mode": "m5", "side": "buy", "entry": entry_p, "score": float(pick.get("score") or 0.0), "strength": "B"}
                        set_user_state(chat, "last_pick", json.dumps(info2, ensure_ascii=False))
                    except Exception:
                        pass
                    _tg_ui(chat, message_id, _format_pick_m5(pick), reply_markup=_build_pick_kb())
                else:
                    c = pick.get("candidate")
                    if isinstance(c, Candidate):
                        try:
                            from core.storage import set_user_state
                            s0 = _settings()
                            live_p0, _ = _get_live_trade_price(c.symbol)
                            entry_override0 = live_p0 if (live_p0 is not None and _is_us_market_open()) else None
                            plan0 = _compute_trade_plan(s0, c, entry_override=entry_override0)
                            info2 = {"symbol": c.symbol, "mode": "d1", "side": plan0.get("side","buy"), "entry": float(plan0.get("entry") or 0.0), "sl": plan0.get("sl"), "tp": plan0.get("tp"), "score": float(getattr(c, "score", 0.0) or 0.0), "strength": str(getattr(c,"grade","B") or "B")}
                            set_user_state(chat, "last_pick", json.dumps(info2, ensure_ascii=False))
                        except Exception:
                            pass
                        _tg_ui(chat, message_id, _format_pick_d1(c, _settings()), reply_markup=_build_pick_kb())
                    else:
                        _tg_ui(chat, message_id, "⚠️ تم العثور على نتيجة لكن غير صالحة.", reply_markup=_ikb([[("⬅️ رجوع", "menu")]]))
                return jsonify({"ok": True})

            if action in ("pick_m5", "pick_d1"):
                tf = "m5" if action == "pick_m5" else "d1"
                chat = str(chat_id)

                # Market-hours filter for scalping signals
                if tf == "m5":
                    ms = _market_status_cached()
                    if not ms.get("is_open", True):
                        _tg_ui(chat, message_id, _format_market_status_line(ms) + "\n\n⛔ إشارات M5 تُرسل فقط وقت فتح السوق (لتفادي سيولة ضعيفة).\nجرّب زر D1.", reply_markup=_ikb([[("⬅️ رجوع", "menu")]]))
                        return jsonify({"ok": True})

                # 1) Try immediate response from cache
                pick = _get_next_pick(tf, chat)
                if pick:
                    if tf == "m5":
                        try:
                            from core.storage import set_user_state
                            entry_p = float(pick.get("last") or 0.0)
                            info = {"symbol": str(pick.get("symbol") or "").upper(), "mode": "m5", "side": "buy", "entry": entry_p, "score": float(pick.get("score") or 0.0), "strength": "B"}
                            set_user_state(chat, "last_pick", json.dumps(info, ensure_ascii=False))
                        except Exception:
                            pass
                        _tg_ui(chat, message_id, _format_pick_m5(pick), reply_markup=_build_pick_kb())
                    else:
                        c = pick.get("candidate")
                        if isinstance(c, Candidate):
                            try:
                                from core.storage import set_user_state
                                s0 = _settings()
                                live_p0, _ = _get_live_trade_price(c.symbol)
                                entry_override0 = live_p0 if (live_p0 is not None and _is_us_market_open()) else None
                                plan0 = _compute_trade_plan(s0, c, entry_override=entry_override0)
                                info = {"symbol": c.symbol, "mode": "d1", "side": plan0.get("side","buy"), "entry": float(plan0.get("entry") or 0.0), "sl": plan0.get("sl"), "tp": plan0.get("tp"), "score": float(getattr(c, "score", 0.0) or 0.0), "strength": str(getattr(c,"grade","B") or "B")}
                                set_user_state(chat, "last_pick", json.dumps(info, ensure_ascii=False))
                            except Exception:
                                pass
                            _tg_ui(chat, message_id, _format_pick_d1(c, _settings()), reply_markup=_build_pick_kb())
                        else:
                            _tg_ui(chat, message_id, "⚠️ لا توجد نتيجة D1 جاهزة الآن، جاري التحديث...", reply_markup=_ikb([[("⬅️ رجوع", "menu")]]))
                    return jsonify({"ok": True})

                # 2) If cache empty/stale: start refresh in background and AUTO-SEND when ready
                key = f"{chat}:{tf}"
                now = time.time()
                started = _PICK_IN_PROGRESS.get(key)
                if started and (now - float(started)) < 180:
                    # already running
                    _tg_ui(chat, message_id, "⏳ لا يزال جاري تجهيز النتائج... سيتم تحديث نفس الرسالة عند الجاهزية.", reply_markup=_ikb([[("⬅️ رجوع", "menu")]]))
                    return jsonify({"ok": True})

                _PICK_IN_PROGRESS[key] = now
                _tg_ui(chat, message_id, "⏳ جاري تجهيز النتائج... سيتم تحديث نفس الرسالة عند الجاهزية.", reply_markup=_ikb([[("⬅️ رجوع", "menu")]]))

                def _refresh_and_send():
                    try:
                        if tf == "m5":
                            _update_cache_m5()
                        else:
                            _update_cache_d1()

                        # After refresh, pull a pick and send it
                        pick2 = _get_next_pick(tf, chat)
                        if not pick2:
                            _tg_ui(chat, message_id, "❌ ما قدرت أجهز فرص حالياً (قد يكون السوق مغلق/بيانات غير كافية). جرّب لاحقاً.", reply_markup=_ikb([[("⬅️ رجوع", "menu")]]))
                            return

                        if tf == "m5":
                            try:
                                from core.storage import set_user_state
                                entry_p = float(pick2.get("last") or 0.0)
                                info = {"symbol": str(pick2.get("symbol") or "").upper(), "mode": "m5", "side": "buy", "entry": entry_p, "score": float(pick2.get("score") or 0.0), "strength": "B"}
                                set_user_state(chat, "last_pick", json.dumps(info, ensure_ascii=False))
                            except Exception:
                                pass
                            _tg_ui(chat, message_id, _format_pick_m5(pick2), reply_markup=_build_pick_kb())
                        else:
                            c2 = pick2.get("candidate")
                            if isinstance(c2, Candidate):
                                try:
                                    from core.storage import set_user_state
                                    s0 = _settings()
                                    live_p0, _ = _get_live_trade_price(c2.symbol)
                                    entry_override0 = live_p0 if (live_p0 is not None and _is_us_market_open()) else None
                                    plan0 = _compute_trade_plan(s0, c2, entry_override=entry_override0)
                                    info = {"symbol": c2.symbol, "mode": "d1", "side": plan0.get("side","buy"), "entry": float(plan0.get("entry") or 0.0), "sl": plan0.get("sl"), "tp": plan0.get("tp"), "score": float(getattr(c2, "score", 0.0) or 0.0), "strength": str(getattr(c2,"grade","B") or "B")}
                                    set_user_state(chat, "last_pick", json.dumps(info, ensure_ascii=False))
                                except Exception:
                                    pass
                                _tg_ui(chat, message_id, _format_pick_d1(c2, _settings()), reply_markup=_build_pick_kb())
                            else:
                                _tg_ui(chat, message_id, "⚠️ تم تحديث D1 لكن النتيجة غير صالحة.", reply_markup=_ikb([[("⬅️ رجوع", "menu")]]))
                    except Exception as e:
                        # IMPORTANT: show error to admin instead of swallowing it
                        _tg_ui(chat, message_id, f"❌ خطأ أثناء تجهيز فرص {tf.upper()}:\n{e}", reply_markup=_ikb([[("⬅️ رجوع", "menu")]]))
                    finally:
                        _PICK_IN_PROGRESS.pop(key, None)

                _run_async(_refresh_and_send)
                return jsonify({"ok": True})
            if action in ("do_analyze", "do_top"):
                settings = _settings()
                # BotFather-like: keep everything in the same message
                _tg_ui(str(chat_id), message_id, "⏳ جاري التحليل...", reply_markup=_ikb([[('⬅️ رجوع', 'menu')]]))
                def _job():
                    try:
                        msg, _ = _run_scan_and_build_message(settings)
                        # Update the same message with results (no extra spam)
                        _tg_ui(str(chat_id), message_id, msg, reply_markup=_ikb([[('⬅️ رجوع', 'menu')], [('🔁 فحص جديد', action)]]))
                    except Exception as e:
                        _tg_ui(str(chat_id), message_id, f"❌ خطأ أثناء الفحص:\n{e}", reply_markup=_ikb([[('⬅️ رجوع', 'menu')]]))
                _run_async(_job)
                return jsonify({"ok": True})
            # Unknown action
            _tg_ui(str(chat_id), message_id, "❓ أمر غير معروف.", reply_markup=_build_menu(settings))
            return jsonify({"ok": True})
        # Handle normal messages
        message = data.get("message") or data.get("channel_post")
        if not message:
            return jsonify({"ok": True})
        chat_id = message["chat"]["id"]
        user_id = message.get("from", {}).get("id")
        text = (message.get("text") or "").strip()
        # إدخال مخصص بعد ضغط زر
        from core.storage import get_user_state, clear_user_state
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
                _tg_ui(str(chat_id), message_id, f"✅ تم تحديث رأس المال إلى {val}$", reply_markup=_build_settings_kb(s))
                return jsonify({"ok": True})
            except Exception:
                _tg_ui(str(chat_id), message_id, "❌ رقم غير صحيح. أرسل رقم مثل: 5000")
                return jsonify({"ok": True})
        
        if pending == "ai_symbol" and text:
            symbol = re.sub(r"[^A-Za-z\.]", "", text.strip().upper())
            if not symbol:
                _tg_ui(str(chat_id), message_id, "❌ اكتب رمز صحيح مثل: TSLA")
                return jsonify({"ok": True})
            from core.storage import clear_user_state
            clear_user_state(str(chat_id), "pending")
            _start_ai_symbol_analysis(str(chat_id), symbol)
            return jsonify({"ok": True})

        if not _is_admin(user_id):
            # Ignore silently for channels, but reply in private
            if str(message.get("chat", {}).get("type")) == "private":
                _tg_ui(str(chat_id), message_id, "⛔ هذا البوت للأدمن فقط.")
            return jsonify({"ok": True})
        settings = _settings()
        if text.startswith("/start"):
            _tg_ui(str(chat_id), message_id, "🤖 البوت شغال.\nاكتب /menu للأزرار.", reply_markup=_build_menu(settings))
            return jsonify({"ok": True})
        if text.startswith("/menu"):
            _tg_ui(str(chat_id), message_id, "📌 اختر:", reply_markup=_build_menu(settings))
            return jsonify({"ok": True})
        if text.startswith("/wl"):
            parts = text.strip().split()
            if len(parts) == 1 or (len(parts) >= 2 and parts[1].lower() in ("list","show")):
                wl = get_watchlist()
                if not wl:
                    _tg_ui(str(chat_id), message_id, "📌 الـ Watchlist فاضي.\nاستخدم: /wl add TSLA")
                    return jsonify({"ok": True})
                _tg_ui(str(chat_id), message_id, "📌 Watchlist:\n" + "\n".join(wl))
                return jsonify({"ok": True})
            if len(parts) >= 3 and parts[1].lower() in ("add","+"):
                sym = parts[2].upper()
                add_watchlist(sym)
                _tg_ui(str(chat_id), message_id, f"✅ تم إضافة {sym} للـ Watchlist.")
                return jsonify({"ok": True})
            if len(parts) >= 3 and parts[1].lower() in ("del","remove","rm","-"):
                sym = parts[2].upper()
                remove_watchlist(sym)
                _tg_ui(str(chat_id), message_id, f"✅ تم حذف {sym} من الـ Watchlist.")
                return jsonify({"ok": True})
            _tg_ui(str(chat_id), message_id, "استخدم: /wl أو /wl add TSLA أو /wl del TSLA")
            return jsonify({"ok": True})
        if text.startswith("/analyze"):
            _tg_ui(str(chat_id), message_id, "⏳ جاري التحليل...", reply_markup=_ikb([[('⬅️ رجوع', 'menu')]]))
            def _job():
                try:
                    msg, _ = _run_scan_and_build_message(settings)
                    _tg_ui(str(chat_id), message_id, msg, reply_markup=_ikb([[('⬅️ رجوع', 'menu')], [('🔁 فحص جديد', 'do_analyze')]]))
                except Exception as e:
                    _tg_ui(str(chat_id), message_id, f"❌ خطأ أثناء الفحص:\n{e}", reply_markup=_ikb([[('⬅️ رجوع', 'menu')]]))
            _run_async(_job)
            return jsonify({"ok": True})
        if text.startswith("/ai"):
            parts = text.split()
            if len(parts) < 2:
                _tg_ui(str(chat_id), message_id, "اكتب: /ai SYMBOL  مثال: /ai TSLA")
                return jsonify({"ok": True})
            symbol = parts[1].upper().strip()
            _start_ai_symbol_analysis(str(chat_id), symbol)
            return jsonify({"ok": True})

        if text.startswith("/settings"):
            _tg_ui(str(chat_id), message_id, "⚙️", reply_markup=_build_menu(settings))
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

@app.get("/stats")
def stats_route():
    """Quick stats for monitoring (no trading).
    Requires RUN_KEY.
    Query params:
      - days: lookback window for latest reviews (default 14)
    """
    if request.args.get("key") != RUN_KEY:
        return jsonify({"ok": False, "error": "unauthorized"}), 401

    try:
        days = int(request.args.get("days") or 14)
    except Exception:
        days = 14
    days = max(1, min(120, days))

    try:
        rows = latest_signal_reviews_since(days=days)
        n = len(rows)
        wins = sum(1 for r in rows if float(r.get("return_pct") or 0) > 0)
        losses = sum(1 for r in rows if float(r.get("return_pct") or 0) < 0)
        flat = n - wins - losses
        winrate = (wins / max(1, wins + losses)) * 100.0

        def _avg(key: str) -> float:
            if not rows:
                return 0.0
            return sum(float(r.get(key) or 0) for r in rows) / len(rows)

        avg_ret = _avg("return_pct")
        avg_mfe = _avg("mfe_pct")
        avg_mae = _avg("mae_pct")

        rows_sorted = sorted(rows, key=lambda x: float(x.get("return_pct") or 0), reverse=True)
        top = [{"symbol": r.get("symbol"), "mode": r.get("mode"), "ret": float(r.get("return_pct") or 0)} for r in rows_sorted[:5]]
        bottom = [{"symbol": r.get("symbol"), "mode": r.get("mode"), "ret": float(r.get("return_pct") or 0)} for r in rows_sorted[-5:]][::-1]

        return jsonify({
            "ok": True,
            "days": days,
            "signals_reviewed": n,
            "wins": wins,
            "losses": losses,
            "flat": flat,
            "winrate_pct": round(winrate, 2),
            "avg_return_pct": round(avg_ret, 3),
            "avg_mfe_pct": round(avg_mfe, 3),
            "avg_mae_pct": round(avg_mae, 3),
            "top5": top,
            "bottom5": bottom,
        })
    except Exception as e:
        # Never crash the service on an empty DB / schema mismatch / missing env vars.
        return jsonify({
            "ok": False,
            "error": "stats_failed",
            "message": str(e),
            "hint": "Check DATABASE_URL/DB_PATH and that init_db() ran. If this is a fresh deploy, run /scan first then /api/review later."
        }), 500

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
    _run_due_paper_reviews()
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


def _my_saved_signals_message(chat_id: str, lookback_days: int = 7, limit: int = 80) -> Tuple[str, List[Dict[str, Any]]]:
    """Return (message, items) for saved paper trades in the last N days."""
    items = []
    try:
        items = list_paper_trades_for_chat(str(chat_id), lookback_days=int(lookback_days), limit=int(limit))
    except Exception:
        items = []
    if not items:
        msg = "📌 إشاراتك المحفوظة\nلا توجد إشارات محفوظة خلال آخر 7 أيام."
        return msg, items

    lines = ["📌 إشاراتك المحفوظة (آخر 7 أيام)", ""]
    # show concise list
    for i, r in enumerate(items[:20], 1):
        sym = (r.get("symbol") or "").upper()
        mode = (r.get("mode") or "").upper() or "D1"
        entry = r.get("entry")
        ts = (r.get("signal_ts") or r.get("ts") or "").replace("T", " ").replace("Z", "")
        try:
            entry_f = float(entry) if entry is not None else 0.0
            entry_s = f"{entry_f:.4g}$" if entry_f > 0 else "-"
        except Exception:
            entry_s = "-"
        lines.append(f"{i}) {sym} ({mode}) | دخول: {entry_s} | وقت: {ts[:16]}")
    if len(items) > 20:
        lines.append(f"… وباقي {len(items)-20} إشارات.")
    lines.append("")
    lines.append("🧹 ملاحظة: يتم تنظيف القائمة تلقائياً بعد 7 أيام (لكن تبقى محفوظة داخلياً للتعلم).")
    return "\n".join(lines), items




def _review_my_saved_performance(chat_id: str, lookback_days: int = 2, limit: int = 50) -> str:
    """Review ONLY the user's saved paper trades.

    - Uses the signal timestamp (signal_ts) from the originating signal (not the paper-trade row).
    - Splits results into: completed (>= due_ts) vs pending (< due_ts).
    - Win/Loss counts are calculated ONLY for completed rows.
    - When market is closed, we may show/use a more recent LIVE last-trade price if available.
    """
    now = datetime.now(timezone.utc)
    try:
        from core.storage import list_paper_trades_for_chat
        rows = list_paper_trades_for_chat(chat_id, lookback_days=max(1, int(lookback_days)), limit=max(20, int(limit)))
    except Exception:
        rows = []
    if not rows:
        return "لا توجد شارات محفوظة لمراجعتها الآن."

    reviewed = 0
    completed = 0
    pending = 0
    winners = 0
    losers = 0
    lines: List[str] = []
    seen: set[tuple] = set()

    for r in rows:
        try:
            ts = r.get("signal_ts") or r.get("ts") or r.get("created_ts") or ""
            if ts:
                sig_dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
                if (now - sig_dt).days > int(lookback_days):
                    continue
            else:
                sig_dt = now

            due_ts = r.get("due_ts") or ""
            try:
                due_dt = datetime.fromisoformat(str(due_ts).replace("Z", "+00:00")) if due_ts else (sig_dt + timedelta(hours=24))
            except Exception:
                due_dt = sig_dt + timedelta(hours=24)

            symbol = (r.get("symbol") or "").upper().strip()
            mode = (r.get("mode") or "D1").upper().strip()
            side = (r.get("side") or "buy").lower().strip()
            entry = float(r.get("entry") or 0.0)
            if not symbol or entry <= 0:
                continue

            k = (symbol, mode, side, round(entry, 4), str(ts)[:16])
            if k in seen:
                continue
            seen.add(k)

            # Pull daily bars around the signal window (enough for MFE/MAE + last close)
            data = bars([symbol], start=sig_dt - timedelta(days=6), end=now + timedelta(days=1), timeframe="1Day", limit=200)
            blist = (data.get("bars", {}).get(symbol) or [])
            if not blist:
                continue

            # last daily close reference
            last_bar = blist[-1]
            last_close = float(last_bar.get("c") or entry)

            # optional live last-trade price (after-hours / pre-market)
            live_p, live_ts = _get_live_trade_price(symbol)
            use_live = False
            live_dt = None
            if live_p is not None and live_ts:
                try:
                    live_dt = datetime.fromisoformat(str(live_ts).replace("Z", "+00:00"))
                    # Use live only if it's reasonably fresh (after-hours / pre-market)
                    use_live = (now - live_dt) <= timedelta(hours=36)
                except Exception:
                    use_live = False

            price = float(live_p) if (use_live and live_p is not None) else last_close
            price_label = "Last" if (use_live and live_p is not None) else "Close"

            highs = [float(b.get("h") or b.get("c") or entry) for b in blist]
            lows = [float(b.get("l") or b.get("c") or entry) for b in blist]
            max_high = max(highs) if highs else entry
            min_low = min(lows) if lows else entry

            if side == "sell":
                ret = (entry - price) / entry * 100.0
                mfe = (entry - min_low) / entry * 100.0
                mae = (entry - max_high) / entry * 100.0
            else:
                ret = (price - entry) / entry * 100.0
                mfe = (max_high - entry) / entry * 100.0
                mae = (min_low - entry) / entry * 100.0

            is_completed = now >= due_dt
            side_label = "شراء" if side != "sell" else "بيع"

            if is_completed:
                completed += 1
                label = "✅" if ret > 0 else ("❌" if ret < 0 else "➖")
                if ret > 0:
                    winners += 1
                elif ret < 0:
                    losers += 1
            else:
                pending += 1
                label = "⏳"

            reviewed += 1

            score = r.get("score")
            score_str = f"{float(score):.1f}" if score is not None else "-"
            t_short = str(ts)[:16]
            lines.append(
                f"{label} {symbol} ({mode.lower()}) | 🎯 {side_label} | Ret: {ret:.2f}% | {price_label}: {price:.2f} | Entry: {entry:.2f} | t: {t_short} | Score: {score_str} | MFE: {mfe:.2f}% | MAE: {mae:.2f}%"
            )
            try:
                if sig_id:
                    # نضيفها تلقائيًا للمراقبة/السجل (بدون أزرار)
                    due = (datetime.now(timezone.utc) + timedelta(hours=24)).isoformat().replace("+00:00","Z")
                    add_paper_trade("GLOBAL", int(sig_id), due)
            except Exception:
                pass
        except Exception:
            continue

    if reviewed == 0:
        return "لا توجد شارات حديثة ضمن فترة المراجعة."

    header = (
        f"📈 مراجعة الإشارات (آخر {lookback_days} يوم):\n"
        f"— تمت مراجعة: {reviewed}\n"
        f"— مكتملة: {completed} (رابحة: {winners} | خاسرة: {losers})\n"
        f"— غير مكتملة: {pending} (قياس لحظي فقط)\n"
        f"ملاحظة: إذا كان السوق مقفل، قد نعرض Last (بعد الإغلاق/قبل الافتتاح) بدل Close.\n"
    )
    body = "\n".join(lines[:25])
    return header + "\n" + body + ("\n\n... (+ المزيد)" if len(lines) > 25 else "")



def _my_signals_dashboard_message(chat_id: str, lookback_days: int = 30) -> str:
    """Premium dashboard summary for the user's paper trades."""
    chat_id = str(chat_id)
    now_dt = datetime.now(timezone.utc)

    paper = list_paper_trades_for_chat(chat_id, lookback_days=lookback_days, limit=500)
    open_trades = [p for p in paper if str(p.get("status") or "open").lower() in ("open","")]
    due_now = []
    for p in open_trades:
        due_dt = _dt_from_iso(str(p.get("due_ts") or "")) or (now_dt + timedelta(days=1))
        if due_dt <= now_dt:
            due_now.append(p)

    finals = list_final_paper_reviews_for_chat(chat_id, lookback_days=lookback_days, limit=500)
    wins = 0
    losses = 0
    tps = 0
    sls = 0
    for r in finals:
        try:
            rp = float(r.get("return_pct") or 0.0)
            if rp > 0:
                wins += 1
            elif rp < 0:
                losses += 1
            note = str(r.get("note") or "")
            if "TP_HIT" in note or '"tp_hit": 1' in note:
                tps += 1
            if "SL_HIT" in note or '"sl_hit": 1' in note:
                sls += 1
        except Exception:
            pass
    total = wins + losses

    wr = (wins / total * 100.0) if total > 0 else 0.0
    msg = (
        f"📊 داشبورد — آخر {int(lookback_days)} يوم\n"
        f"— صفقات محفوظة: {len(paper)}\n"
        f"— مفتوحة: {len(open_trades)}\n"
        f"— مستحقة الآن: {len(due_now)}\n\n"
        f"📈 نتائج 24 ساعة (لقطات ثابتة): {len(finals)}\n"
        f"— Winrate: {wr:.1f}% (رابحة: {wins} | خاسرة: {losses})\n"
        f"— TP Hit: {tps} | SL Hit: {sls}\n\n"
        f"آخر تحديث: {now_dt.strftime('%Y-%m-%d %H:%M UTC')}"
    )
    return msg


def _my_saved_24h_reviews_message(chat_id: str, lookback_days: int = 30, limit: int = 30) -> str:
    """Return frozen 24h review snapshots for this chat."""
    try:
        rows = list_final_paper_reviews_for_chat(str(chat_id), lookback_days=int(lookback_days), limit=int(limit))
    except Exception:
        rows = []
    if not rows:
        return "📊 مراجعات 24 ساعة\n\nلا توجد مراجعات مكتملة محفوظة حتى الآن."
    lines: List[str] = []
    for r in rows[:int(limit)]:
        try:
            symbol = (r.get("symbol") or "").upper().strip()
            mode = (r.get("mode") or "D1").upper().strip()
            side = (r.get("side") or "buy").lower().strip()
            entry = float(r.get("entry") or 0.0)
            exit_price = float(r.get("exit_price") or 0.0)
            ret_pct = float(r.get("return_pct") or 0.0)
            score = r.get("score")
            note_raw = r.get("note") or ""
            price_src = ""
            live_ts = ""
            review_ts = r.get("review_ts") or ""
            try:
                j = json.loads(note_raw) if note_raw.strip().startswith("{") else {}
                price_src = (j.get("price_src") or "")
                live_ts = (j.get("live_ts") or "")
                review_ts = j.get("review_ts") or review_ts
            except Exception:
                pass
            side_label = "شراء" if side != "sell" else "بيع"
            res = "✅" if ret_pct > 0 else ("❌" if ret_pct < 0 else "➖")
            src_line = f" | Src: {price_src}" if price_src else ""
            ts_line = f" | {review_ts[:16].replace('T',' ')}" if review_ts else ""
            sc_line = f" | Score: {float(score):.1f}" if score is not None else ""
            lines.append(
                f"{res} {symbol} ({mode.lower()}) | 🎯 {side_label} | Ret: {ret_pct:+.2f}% | Entry: {entry:.2f} | Exit: {exit_price:.2f}{sc_line}{src_line}{ts_line}"
            )
            try:
                if sig_id:
                    # نضيفها تلقائيًا للمراقبة/السجل (بدون أزرار)
                    due = (datetime.now(timezone.utc) + timedelta(hours=24)).isoformat().replace("+00:00","Z")
                    add_paper_trade("GLOBAL", int(sig_id), due)
            except Exception:
                pass
        except Exception:
            continue
    header = f"📊 مراجعات 24 ساعة (مقفلة)\n— العدد: {len(rows)}\n"
    return header + "\n".join(lines[:25])


def _review_and_saved_message(chat_id: str) -> Tuple[str, List[Dict[str, Any]]]:
    """Combine exploratory performance review (last 2 days) + user's saved paper trades list.

    - Review section uses latest available close (D1 bars) and may include signals beyond paper list.
    - Paper list is what the user can delete from Telegram UI (auto-cleaned after 7 days).
    """
    review = _review_recent_signals(lookback_days=2, limit=60)
    saved_msg, items = _my_saved_signals_message(chat_id, lookback_days=7, limit=80)
    # Combine with a clear separator so the user gets both behaviors.
    combo = review.rstrip() + "\n\n" + "—" * 18 + "\n" + saved_msg
    return combo, items

def _review_recent_signals(lookback_days: int = 2, limit: int = 50) -> str:
    """Build a Telegram message that reviews recent signals using latest available daily close.
    This does NOT place trades. It simply measures how signals performed so far (exploration mode)."
    """
    now = datetime.now(timezone.utc)
    # pull last signals regardless of evaluated status (we want recent performance snapshots)
    rows = last_signals(limit=max(20, int(limit)))
    if not rows:
        return "لا توجد إشارات لمراجعتها الآن."
    reviewed = 0
    winners = 0
    losers = 0
    lines = []
    # Avoid showing duplicates for the same symbol/mode within the same review window.
    seen = set()
    for r in rows:
        try:
            ts = r.get("ts") or ""
            if not ts:
                continue
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            if (now - dt).days > int(lookback_days):
                continue
            symbol = (r.get("symbol") or "").upper().strip()
            side = (r.get("side") or "buy").lower().strip()
            entry = float(r.get("entry") or 0.0)
            if entry <= 0:
                continue
            mode = (r.get("mode") or "").strip() or "D1"
            # key: symbol+mode+side+rounded entry (stable enough to remove spam duplicates)
            k = (symbol, mode.upper(), side, round(entry, 4))
            if k in seen:
                continue
            seen.add(k)
            # get daily bars from signal date to now (few days only)
            data = bars([symbol], start=dt - timedelta(days=2), end=now + timedelta(days=1), timeframe="1Day", limit=50)
            blist = (data.get("bars", {}).get(symbol) or [])
            if not blist:
                continue
            last_close = float(blist[-1].get("c") or entry)
            highs = [float(b.get("h") or b.get("c") or entry) for b in blist]
            lows = [float(b.get("l") or b.get("c") or entry) for b in blist]
            max_high = max(highs) if highs else entry
            min_low = min(lows) if lows else entry
            if side == "sell":
                ret = (entry - last_close) / entry * 100.0
                mfe = (entry - min_low) / entry * 100.0
                mae = (entry - max_high) / entry * 100.0
            else:
                ret = (last_close - entry) / entry * 100.0
                mfe = (max_high - entry) / entry * 100.0
                mae = (min_low - entry) / entry * 100.0
            label = "✅" if ret > 0 else ("❌" if ret < 0 else "➖")
            if ret > 0:
                winners += 1
            elif ret < 0:
                losers += 1
            reviewed += 1
            # store snapshot
            try:
                log_signal_review(
                    ts=now.isoformat(),
                    signal_id=int(r.get("id")),
                    close=float(last_close),
                    return_pct=float(ret),
                    mfe_pct=float(mfe),
                    mae_pct=float(mae),
                    note="daily_review",
                )
            except Exception:
                pass
            score = r.get("score")
            lines.append(
                f"{label} {symbol} ({mode}) | Ret: {ret:.2f}% | Close: {last_close:.2f} | Entry: {entry:.2f} | Score: {float(score):.1f}"
                if score is not None
                else f"{label} {symbol} ({mode}) | Ret: {ret:.2f}% | Close: {last_close:.2f} | Entry: {entry:.2f}"
            )
            try:
                if sig_id:
                    # نضيفها تلقائيًا للمراقبة/السجل (بدون أزرار)
                    due = (datetime.now(timezone.utc) + timedelta(hours=24)).isoformat().replace("+00:00","Z")
                    add_paper_trade("GLOBAL", int(sig_id), due)
            except Exception:
                pass
        except Exception:
            continue

    if reviewed == 0:
        return "لا توجد إشارات حديثة ضمن فترة المراجعة."
    header = f"📈 مراجعة الإشارات (آخر {lookback_days} يوم):\n" \
             f"— تمت مراجعة: {reviewed}\n" \
             f"— رابحة: {winners} | خاسرة: {losers}\n" \
             f"ملاحظة: هذا قياس استكشافي حسب آخر إغلاق/آخر شمعة، وليس تنفيذًا فعليًا.\n"
    body = "\n".join(lines[:25])
    return header + "\n" + body


def _weekly_report(days: int = 7) -> str:
    """Weekly summary based on latest stored reviews (no trading)."""
    try:
        rows = latest_signal_reviews_since(days=int(days))
    except Exception:
        rows = []
    if not rows:
        return f"📅 تقرير أسبوعي (آخر {days} يوم):\nلا توجد بيانات مراجعة كافية. شغّل زر (مراجعة إشاراتي) أو فعّل Cron /api/review."
    n = len(rows)
    wins = sum(1 for r in rows if (r.get("return_pct") or 0) > 0)
    losses = sum(1 for r in rows if (r.get("return_pct") or 0) < 0)
    flat = n - wins - losses
    winrate = (wins / max(1, (wins + losses))) * 100.0
    avg_ret = sum((r.get("return_pct") or 0) for r in rows) / n
    avg_mfe = sum((r.get("mfe_pct") or 0) for r in rows) / n
    avg_mae = sum((r.get("mae_pct") or 0) for r in rows) / n
    tp_hits = sum(1 for r in rows if (r.get("tp_hit") or 0) in (1, True))
    sl_hits = sum(1 for r in rows if (r.get("sl_hit") or 0) in (1, True))

    # Mode comparison (D1 vs M5 etc.)
    mode_map = {}
    for r in rows:
        mode = (r.get("mode") or "").strip() or "?"
        mode_map.setdefault(mode, []).append(float(r.get("return_pct") or 0))
    def _mode_summary(mode: str, vals: list) -> str:
        if not vals:
            return ""
        wins_m = sum(1 for v in vals if v > 0)
        losses_m = sum(1 for v in vals if v < 0)
        winrate_m = (wins_m / max(1, wins_m + losses_m)) * 100.0
        avg_m = sum(vals) / len(vals)
        net_m = sum(vals)
        return f"• {mode}: n={len(vals)} | Winrate {winrate_m:.1f}% | Avg {avg_m:+.2f}% | Net {net_m:+.2f}%"

    # Top/Bottom by return_pct
    rows_sorted = sorted(rows, key=lambda x: float(x.get("return_pct") or 0), reverse=True)
    top5 = rows_sorted[:5]
    bot5 = list(reversed(rows_sorted[-5:]))

    def fmt_row(r):
        sym = (r.get("symbol") or "").upper()
        mode = r.get("mode") or ""
        ret = float(r.get("return_pct") or 0)
        cls = (r.get("tp_gap_class") or "").strip()
        extra = f" | {cls}" if cls else ""
        return f"• {sym} ({mode}) {ret:+.2f}%{extra}"

    header = (
        f"📅 تقرير أسبوعي (آخر {days} يوم)\n"
        f"— إشارات: {n}\n"
        f"— Win/Loss/Flat: {wins}/{losses}/{flat} | Winrate: {winrate:.1f}%\n"
        f"— Avg Ret: {avg_ret:+.2f}% | Avg MFE: {avg_mfe:+.2f}% | Avg MAE: {avg_mae:+.2f}%\n"
        f"— TP Hits: {tp_hits} | SL Hits: {sl_hits}\n"
    )
    compare_lines = []
    # Prefer D1 and M5 first if present
    for k in ["D1", "M5"]:
        if k in mode_map:
            compare_lines.append(_mode_summary(k, mode_map.get(k) or []))
    for k in sorted(mode_map.keys()):
        if k in ("D1","M5"):
            continue
        compare_lines.append(_mode_summary(k, mode_map.get(k) or []))
    compare_block = "📊 مقارنة الأنماط (D1 vs M5)\n" + ("\n".join(compare_lines) if compare_lines else "لا توجد بيانات كافية للمقارنة.")

    body = compare_block + "\n\n" + "🏆 أفضل 5\n" + "\n".join(fmt_row(r) for r in top5) + "\n\n" + "🧊 أسوأ 5\n" + "\n".join(fmt_row(r) for r in bot5)
    footer = "\n\nملاحظة: التقرير يعتمد على آخر مراجعة محفوظة لكل إشارة ضمن الفترة (استكشاف/تعليم)."
    return header + "\n" + body + footer

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
    
    # Precompute fast-pick caches for instant Telegram buttons
    try:
        _scheduler.add_job(
            _update_cache_m5,
            IntervalTrigger(minutes=max(1, int(M5_CACHE_MIN))),
            id="cache_m5_job",
            replace_existing=True,
        )
        _scheduler.add_job(
            _update_cache_d1,
            IntervalTrigger(minutes=max(5, int(D1_CACHE_MIN))),
            id="cache_d1_job",
            replace_existing=True,
        )
    except Exception:
        pass
    _scheduler.add_job(
        _run_scan_and_notify,
        IntervalTrigger(minutes=max(5, interval)),
        kwargs={"force_summary": True},
        id="scan_job",
        replace_existing=True,
    )

    # Premium: monitor TP/SL hits frequently (no broker API needed)
    try:
        _scheduler.add_job(
            _run_open_paper_monitor,
            IntervalTrigger(minutes=int(os.getenv("PAPER_MONITOR_MIN") or 5)),
            kwargs={"ttl_sec": 0.0},
            id="paper_monitor_job",
            replace_existing=True,
        )
        _scheduler.add_job(
            _run_due_paper_reviews,
            IntervalTrigger(minutes=int(os.getenv("PAPER_REVIEW_MIN") or 3)),
            kwargs={"ttl_sec": 0.0},
            id="paper_due_review_job",
            replace_existing=True,
        )
        # تذكير إغلاق صفقات اليوم الواحد قبل الإغلاق (يدوي)
        _scheduler.add_job(
            _run_eod_close_reminder,
            IntervalTrigger(minutes=2),
            kwargs={"ttl_sec": 30.0},
            id="eod_reminder_job",
            replace_existing=True,
        )

    except Exception:
        pass

    _scheduler.start()
    atexit.register(lambda: _scheduler.shutdown(wait=False) if _scheduler else None)
try:
    if os.getenv("ENABLE_SCHEDULER", "1") == "1":
        _start_scheduler()
        # Warm caches in background (so first button press is instant)
        try:
            _run_async(_update_cache_d1)
            _run_async(_update_cache_m5)
        except Exception:
            pass
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

@app.post("/api/outcome")
def api_outcome():
    key = (request.args.get("key") or request.headers.get("X-Run-Key") or "").strip()
    if RUN_KEY and key != RUN_KEY:
        return jsonify({"error":"unauthorized"}), 401
    data = request.get_json(silent=True) or {}
    try:
        signal_id = int(data.get("signal_id") or 0)
    except Exception:
        signal_id = 0
    result = (data.get("result") or "").strip().upper()
    r_mult = data.get("r_mult")
    notes = (data.get("notes") or "").strip()
    if not signal_id:
        return jsonify({"error":"signal_id required"}), 400
    try:
        r_mult_f = float(r_mult) if r_mult is not None and str(r_mult).strip() != "" else None
    except Exception:
        r_mult_f = None
    record_outcome(signal_id, result=result, r_mult=r_mult_f, notes=notes)
    return jsonify({"ok": True, "signal_id": signal_id, "result": result, "r_mult": r_mult_f})

@app.get("/api/manual_stats")
def api_manual_stats():
    key = (request.args.get("key") or "").strip()
    if RUN_KEY and key != RUN_KEY:
        return jsonify({"error":"unauthorized"}), 401
    limit = int(request.args.get("limit") or 200)
    return jsonify(get_recent_stats(limit=limit))

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
