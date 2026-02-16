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
)
from core.scanner import scan_universe_with_meta, Candidate, get_symbol_features, get_symbol_features_m5
app = Flask(__name__)
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
@app.get("/api/review")
def api_review():
    key = (request.args.get("key") or "").strip()
    if RUN_KEY and key != RUN_KEY:
        return jsonify({"ok": False, "error": "unauthorized"}), 403
    lookback = int(request.args.get("days") or 2)
    msg = _review_recent_signals(lookback_days=lookback, limit=80)
    # send to default telegram (admin/channel)
    try:
        send_telegram(msg)
    except Exception:
        pass
    return jsonify({"ok": True, "reviewed_days": lookback})




@app.get("/api/weekly_report")
def api_weekly_report():
    key = (request.args.get("key") or "").strip()
    if RUN_KEY and key != RUN_KEY:
        return jsonify({"ok": False, "error": "unauthorized"}), 403
    days = int(request.args.get("days") or 7)
    msg = _weekly_report(days=days)
    try:
        send_telegram(msg)
    except Exception:
        pass
    return jsonify({"ok": True, "days": days})
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
        [("📈 مراجعة إشاراتي", "review_signals"), ("📅 تقرير أسبوعي", "weekly_report")],
        [("🔁 تحديث القائمة", "menu")],
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
        [("⬅️ رجوع", "menu")],
    ])

def _build_modes_kb() -> Dict[str, Any]:
    return _ikb([
        [("📅 يومي D1", "set_mode:daily"), ("⏱️ سكالبينغ M5", "set_mode:scalp")],
        [("📈 سونق/سوينغ", "set_mode:swing"), ("⬅️ رجوع", "menu")],
    ])

def _build_entry_kb() -> Dict[str, Any]:
    return _ikb([
        [("🧠 تلقائي", "set_entry:auto"), ("✅ كسر/تأكيد", "set_entry:breakout")],
        [("🎯 حد/Limit", "set_entry:limit"), ("⬅️ رجوع", "menu")],
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
        [("🔁 الاثنين", "set_notify_route:both"), ("⬅️ رجوع", "show_settings")],
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
    plan = _compute_trade_plan(settings, c)
    return _format_sahm_block("D1", c, plan)
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
    - وقف الخسارة: ATR * SL_ATR_MULT (LONG تحت الدخول / SHORT فوق الدخول)
    - جني الربح: (المخاطرة R) * TP_R_MULT (LONG فوق الدخول / SHORT تحت الدخول)
    - الكمية: حسب رأس المال والمخاطرة المتغيرة A+/A/B
    """
    side = (getattr(c, "side", None) or "buy").lower().strip()
    if side not in ("buy", "sell"):
        side = "buy"

    entry = float(c.last_close)

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

    # RR محسوب على أساس المخاطرة R
    rr = (abs(tp - entry)) / max(abs(entry - sl), 0.01)

    return {
        "side": side,
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
        "qty": int(qty),
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

    return (
        f"🚀 سهم: {c.symbol} | {side_lbl} | التصنيف: {plan.get('grade','')} | القوة: {strength} | Score: {c.score:.1f}"
        + (f" | AI: {ai_score}/100" if ai_score is not None else "")
        + (f" | ML: {int(round(float(plan.get('ml_prob') or 0)*100))}%" if plan.get('ml_prob') is not None else "")
        + (f" | EV(R): {float(plan.get('ev_r')):.2f}" if plan.get('ev_r') is not None else "")
        + "\n"
        f"العملية: {op_lbl}\n"
        f"النوع: {entry_type}\n"
        f"السعر: {plan['entry']}\n"
        f"الكمية: {plan['qty']}\n"
        f"المخاطرة: {plan.get('risk_pct',0)}% (≈ {plan.get('risk_amount',0)}$) | R/R: {plan.get('rr',0)}\n"
        f"ATR: {plan.get('atr',0)} | SL×ATR: {plan.get('sl_atr_mult',0)} | TP×R: {plan.get('tp_r_mult',0)}\n"
        f"{ai_line}"
        f"الأمر المرفق: جني الربح/وقف الخسارة\n"
        f"جني الربح: {plan['tp']}\n"
        f"وقف الخسارة: {plan['sl']}\n"
        f"الخطة: {mode_label}\n"
        f"ملاحظة: {c.notes}\n"
    )

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
    # Optional: require multi-timeframe alignment (safer entries)
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

    # filter + sort
    candidates = [c for c in picks if _mode_matches(c, mode) and _tf_ok(c)]
    candidates.sort(key=lambda x: x.score, reverse=True)
    blocks: List[str] = []
    logged: List[Dict[str, Any]] = []
    ai_topn = _get_int(settings, "AI_PREDICT_TOPN", 5)
    ai_cache: Dict[str, Optional[dict]] = {}
    ai_used = 0
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
        # Optional AI direction prediction (run only on top N to avoid slowness)
        if _get_bool(settings, "AI_PREDICT_ENABLED", False) and ai_used < max(0, ai_topn):
            if c.symbol not in ai_cache:
                try:
                    ai_cache[c.symbol] = _ai_direction_for_symbol(c.symbol, settings)
                except Exception:
                    ai_cache[c.symbol] = None
            pred = ai_cache.get(c.symbol)
            if pred:
                plan["ai_dir"] = pred.get("direction")
                plan["ai_conf"] = pred.get("confidence")
                plan["ai_h"] = pred.get("horizon")
                ai_used += 1
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
            if _get_bool(settings, "AI_PREDICT_ENABLED", False) and ai_used < max(0, ai_topn):
                if c.symbol not in ai_cache:
                    try:
                        ai_cache[c.symbol] = _ai_direction_for_symbol(c.symbol, settings)
                    except Exception:
                        ai_cache[c.symbol] = None
                pred = ai_cache.get(c.symbol)
                if pred:
                    plan["ai_dir"] = pred.get("direction")
                    plan["ai_conf"] = pred.get("confidence")
                    plan["ai_h"] = pred.get("horizon")
                    ai_used += 1
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
        log_signal(ts=ts, symbol=d["symbol"], source="scan", side=(getattr(c,"side",None) or "buy"), mode=d["mode"], strength=d["strength"], score=float(d["score"]), entry=float(d["entry"]), sl=d.get("sl"), tp=d.get("tp"), features_json=json.dumps(d.get("features") or {}, ensure_ascii=False), reasons_json=json.dumps(d.get("reasons") or [], ensure_ascii=False), horizon_days=int(_get_int(_settings(), "SIGNAL_EVAL_DAYS", SIGNAL_EVAL_DAYS)))
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
                _tg_send(str(chat_id), "⛔ هذا البوت للأدمن فقط.")
                return jsonify({"ok": True})
            settings = _settings()
            if action == "menu":
                _tg_send(str(chat_id), "📌 اختر:", reply_markup=_build_menu(settings))
                return jsonify({"ok": True})

            if action == "review_signals":
                _tg_send(str(chat_id), "⏳ جاري مراجعة الإشارات...")
                def _job():
                    try:
                        msg = _review_recent_signals(lookback_days=int(_get_int(_settings(), "REVIEW_LOOKBACK_DAYS", 2)), limit=80)
                        _tg_send(str(chat_id), msg)
                    except Exception as e:
                        _tg_send(str(chat_id), f"❌ خطأ أثناء المراجعة:\n{e}")
                _run_async(_job)
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
                        _tg_send(str(chat_id), "❌ لا توجد فرص M5 الآن (قد يكون السوق مغلق).", reply_markup=_build_menu(s))
                        return jsonify({"ok": True})
                    _tg_send(str(chat_id), "🧠 Top 10 (3- سكالبينغ M5): اختر سهم", reply_markup=_build_top10_kb(out))
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
                    _tg_send(str(chat_id), "❌ لا توجد نتائج الآن.", reply_markup=_build_menu(s))
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
                    _tg_send(str(chat_id), title, reply_markup=_build_top10_kb(out))
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
                _tg_send(str(chat_id), title, reply_markup=_build_top10_kb(out))
                return jsonify({"ok": True})

            if action.startswith("ai_pick:"):
                sym = action.split(":", 1)[1].strip().upper()
                _start_ai_symbol_analysis(str(chat_id), sym)
                return jsonify({"ok": True})
            if action == "ai_symbol_start":
                from core.storage import set_user_state
                set_user_state(str(chat_id), "pending", "ai_symbol")
                _tg_send(str(chat_id), "🧠 اكتب رمز السهم الآن (مثال: TSLA)\nأو اكتب /ai TSLA", reply_markup=_build_ai_start_kb())
                return jsonify({"ok": True})
            if action == "ai_cancel":
                from core.storage import clear_user_state
                clear_user_state(str(chat_id), "pending")
                _tg_send(str(chat_id), "✅ تم الإلغاء.", reply_markup=_build_menu(_settings()))
                return jsonify({"ok": True})
            if action == "show_modes":
                _tg_send(str(chat_id), "📆 اختر الخطة الزمنية:", reply_markup=_build_modes_kb())
                return jsonify({"ok": True})
            if action.startswith("set_mode:"):
                mode = action.split(":", 1)[1]
                set_setting("PLAN_MODE", mode)
                settings = _settings()
                _tg_send(str(chat_id), f"✅ تم ضبط الخطة: {_mode_label(mode)}", reply_markup=_build_menu(settings))
                return jsonify({"ok": True})
            if action == "show_entry":
                _tg_send(str(chat_id), "🎯 اختر نوع الدخول:", reply_markup=_build_entry_kb())
                return jsonify({"ok": True})
            if action.startswith("set_entry:"):
                entry = action.split(":", 1)[1]
                set_setting("ENTRY_MODE", entry)
                settings = _settings()
                _tg_send(str(chat_id), f"✅ نوع الدخول: {_entry_type_label(entry)}", reply_markup=_build_menu(settings))
                return jsonify({"ok": True})
            if action == "toggle_notify":
                cur = _get_bool(settings, "AUTO_NOTIFY", True)
                set_setting("AUTO_NOTIFY", "0" if cur else "1")
                settings = _settings()
                _tg_send(str(chat_id), "✅ تم تحديث التنبيهات.", reply_markup=_build_settings_kb(settings))
                return jsonify({"ok": True})
            if action == "toggle_ai_predict":
                cur = _get_bool(settings, "AI_PREDICT_ENABLED", False)
                set_setting("AI_PREDICT_ENABLED", "0" if cur else "1")
                settings = _settings()
                _tg_send(str(chat_id), "✅ تم تحديث تنبؤ AI.", reply_markup=_build_settings_kb(settings))
                return jsonify({"ok": True})
            if action == "show_horizon":
                _tg_send(str(chat_id), "🤖 اختر إطار التنبؤ (يؤثر على تحليل AI فقط):", reply_markup=_build_horizon_kb(settings))
                return jsonify({"ok": True})
            if action.startswith("set_horizon:"):
                val = action.split(":", 1)[1].strip().upper()
                if val in ("HYBRID", "M5PLUS"):
                    val = "M5+"
                if val not in ("D1", "M5", "M5+"):
                    val = "D1"
                set_setting("PREDICT_FRAME", val)
                s = _settings()
                _tg_send(str(chat_id), f"✅ تم ضبط إطار التنبؤ: {val}", reply_markup=_build_settings_kb(s))
                return jsonify({"ok": True})
            if action == "show_notify_route":
                _tg_send(str(chat_id), "📨 اختر وجهة التنبيهات:", reply_markup=_build_notify_route_kb())
                return jsonify({"ok": True})
            if action.startswith("set_notify_route:"):
                route = action.split(":", 1)[1].strip().lower()
                if route not in ("dm", "group", "both"):
                    route = "dm"
                set_setting("NOTIFY_ROUTE", route)
                settings = _settings()
                _tg_send(str(chat_id), "✅ تم تحديث الوجهة.", reply_markup=_build_menu(settings))
                return jsonify({"ok": True})
            if action == "toggle_silent":
                cur = _get_bool(settings, "NOTIFY_SILENT", True)
                set_setting("NOTIFY_SILENT", "0" if cur else "1")
                settings = _settings()
                _tg_send(str(chat_id), "✅ تم تحديث وضع الصامت.", reply_markup=_build_menu(settings))
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
                _tg_send(str(chat_id), txt, reply_markup=_build_settings_kb(s))
                return jsonify({"ok": True})
            if action == "show_capital":
                reply = _build_capital_kb() if "_build_capital_kb" in globals() else {"inline_keyboard":[[{"text":"✍️ قيمة مخصصة","callback_data":"set_capital_custom"}],[{"text":"⬅️ رجوع","callback_data":"show_settings"}]]}
                _tg_send(str(chat_id), "💰 اختر رأس المال بالدولار:", reply_markup=reply)
                return jsonify({"ok": True})
            if action == "set_capital_custom":
                from core.storage import set_user_state
                set_user_state(str(chat_id), "pending", "capital")
                _tg_send(str(chat_id), "✍️ أرسل رقم رأس المال بالدولار (مثال: 5000)")
                return jsonify({"ok": True})
            if action.startswith("set_capital:"):
                val = action.split(":", 1)[1]
                set_setting("CAPITAL_USD", val)
                s = _settings()
                _tg_send(str(chat_id), f"✅ تم ضبط رأس المال: {val}$", reply_markup=_build_settings_kb(s))
                return jsonify({"ok": True})
            if action == "show_position":
                _tg_send(str(chat_id), "📦 اختر نسبة حجم الصفقة من رأس المال:", reply_markup=_build_position_kb())
                return jsonify({"ok": True})
            if action.startswith("set_position:"):
                val = action.split(":", 1)[1]
                set_setting("POSITION_PCT", val)
                s = _settings()
                _tg_send(str(chat_id), f"✅ تم ضبط حجم الصفقة: {float(val)*100:.0f}%", reply_markup=_build_settings_kb(s))
                return jsonify({"ok": True})
            if action == "show_sl":
                _tg_send(str(chat_id), "📉 اختر وقف الخسارة %:", reply_markup=_build_sl_kb())
                return jsonify({"ok": True})
            if action.startswith("set_sl:"):
                val = action.split(":", 1)[1]
                set_setting("SL_PCT", val)
                s = _settings()
                _tg_send(str(chat_id), f"✅ تم ضبط وقف الخسارة: {val}%", reply_markup=_build_settings_kb(s))
                return jsonify({"ok": True})
            if action == "show_tp":
                _tg_send(str(chat_id), "📈 اختر جني الربح % (لضعيف/متوسط):", reply_markup=_build_tp_kb())
                return jsonify({"ok": True})
            if action.startswith("set_tp:"):
                val = action.split(":", 1)[1]
                set_setting("TP_PCT", val)
                s = _settings()
                _tg_send(str(chat_id), f"✅ تم ضبط جني الربح (لضعيف/متوسط): {val}%", reply_markup=_build_settings_kb(s))
                return jsonify({"ok": True})
            if action == "show_send":
                _tg_send(str(chat_id), "🎛 اختر عدد الفرص في كل فحص:", reply_markup=_build_send_kb())
                return jsonify({"ok": True})
            if action.startswith("set_send:"):
                parts = action.split(":")
                if len(parts) == 3:
                    set_setting("MIN_SEND", parts[1])
                    set_setting("MAX_SEND", parts[2])
                s = _settings()
                _tg_send(str(chat_id), f"✅ تم ضبط عدد الفرص: {s.get('MIN_SEND','7')} إلى {s.get('MAX_SEND','10')}", reply_markup=_build_settings_kb(s))
                return jsonify({"ok": True})
            if action == "toggle_resend":
                cur = _get_bool(settings, "ALLOW_RESEND_IF_STRONGER", True)
                set_setting("ALLOW_RESEND_IF_STRONGER", "0" if cur else "1")
                s = _settings()
                _tg_send(str(chat_id), "✅ تم تحديث خيار إعادة الإرسال.", reply_markup=_build_settings_kb(s))
                return jsonify({"ok": True})
            if action == "show_window":
                _tg_send(str(chat_id), "🕒 اختر نافذة السوق (بتوقيت الرياض):", reply_markup=_build_window_kb())
                return jsonify({"ok": True})
            if action.startswith("set_window:"):
                parts = action.split(":")
                if len(parts) == 3:
                    set_setting("WINDOW_START", parts[1])
                    set_setting("WINDOW_END", parts[2])
                s = _settings()
                _tg_send(str(chat_id), f"✅ تم ضبط النافذة: {s.get('WINDOW_START','17:30')}→{s.get('WINDOW_END','00:00')}", reply_markup=_build_settings_kb(s))
                return jsonify({"ok": True})
            if action == "noop":
                return jsonify({"ok": True})
            if action == "show_risk":
                _tg_send(str(chat_id), "⚖️ اختر نسب المخاطرة حسب التصنيف (A+/A/B):", reply_markup=_build_risk_kb(settings))
                return jsonify({"ok": True})
            if action.startswith("set_risk_aplus:"):
                val = action.split(":", 1)[1]
                set_setting("RISK_APLUS_PCT", val)
                s = _settings()
                _tg_send(str(chat_id), f"✅ تم ضبط مخاطرة A+: {val}%", reply_markup=_build_settings_kb(s))
                return jsonify({"ok": True})
            if action.startswith("set_risk_a:"):
                val = action.split(":", 1)[1]
                set_setting("RISK_A_PCT", val)
                s = _settings()
                _tg_send(str(chat_id), f"✅ تم ضبط مخاطرة A: {val}%", reply_markup=_build_settings_kb(s))
                return jsonify({"ok": True})
            if action.startswith("set_risk_b:"):
                val = action.split(":", 1)[1]
                set_setting("RISK_B_PCT", val)
                s = _settings()
                _tg_send(str(chat_id), f"✅ تم ضبط مخاطرة B: {val}%", reply_markup=_build_settings_kb(s))
                return jsonify({"ok": True})
            if action == "show_interval":
                _tg_send(str(chat_id), "⏱️ اختر فترة الفحص:", reply_markup=_build_interval_kb(settings))
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
                _tg_send(str(chat_id), f"✅ تم ضبط فترة الفحص: {val} دقيقة", reply_markup=_build_settings_kb(s))
                return jsonify({"ok": True})
            if action in ("pick_m5", "pick_d1"):
                tf = "m5" if action == "pick_m5" else "d1"
                chat = str(chat_id)
                # Market-hours filter for scalping signals
                if tf == "m5":
                    ms = _market_status_cached()
                    if not ms.get("is_open", True):
                        _tg_send(chat, _format_market_status_line(ms) + "\n\n⛔ إشارات M5 تُرسل فقط وقت فتح السوق (لتفادي سيولة ضعيفة).\nجرّب زر D1 للتجهيز.", silent=_get_bool(_settings(), "NOTIFY_SILENT", True))
                        return jsonify({"ok": True})
                # quick response from cache (no Alpaca calls inside webhook)
                pick = _get_next_pick(tf, chat)
                if pick:
                    if tf == "m5":
                        _tg_send(chat, _format_pick_m5(pick), silent=_get_bool(_settings(), "NOTIFY_SILENT", True))
                    else:
                        c = pick.get("candidate")
                        if isinstance(c, Candidate):
                            _tg_send(chat, _format_pick_d1(c, _settings()), silent=_get_bool(_settings(), "NOTIFY_SILENT", True))
                        else:
                            _tg_send(chat, "⚠️ لا توجد نتيجة D1 جاهزة الآن، جاري التحديث...")
                    return jsonify({"ok": True})
                # If cache empty/stale, acknowledge fast then trigger refresh async
                _tg_send(chat, "⏳ جاري تجهيز النتائج... اضغط مرة ثانية بعد ثواني.", silent=True)
                def _refresh():
                    try:
                        if tf == "m5":
                            _update_cache_m5()
                        else:
                            _update_cache_d1()
                    except Exception:
                        pass
                _run_async(_refresh)
                return jsonify({"ok": True})
            if action in ("do_analyze", "do_top"):
                settings = _settings()
                _tg_send(str(chat_id), "⏳ جاري التحليل...")
                def _job():
                    try:
                        msg, _ = _run_scan_and_build_message(settings)
                        send_telegram(msg)
                    except Exception as e:
                        _tg_send(str(chat_id), f"❌ خطأ أثناء الفحص:\n{e}")
                _run_async(_job)
                return jsonify({"ok": True})
            # Unknown action
            _tg_send(str(chat_id), "❓ أمر غير معروف.", reply_markup=_build_menu(settings))
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
                _tg_send(str(chat_id), f"✅ تم تحديث رأس المال إلى {val}$", reply_markup=_build_settings_kb(s))
                return jsonify({"ok": True})
            except Exception:
                _tg_send(str(chat_id), "❌ رقم غير صحيح. أرسل رقم مثل: 5000")
                return jsonify({"ok": True})
        
        if pending == "ai_symbol" and text:
            symbol = re.sub(r"[^A-Za-z\.]", "", text.strip().upper())
            if not symbol:
                _tg_send(str(chat_id), "❌ اكتب رمز صحيح مثل: TSLA")
                return jsonify({"ok": True})
            from core.storage import clear_user_state
            clear_user_state(str(chat_id), "pending")
            _start_ai_symbol_analysis(str(chat_id), symbol)
            return jsonify({"ok": True})

        if not _is_admin(user_id):
            # Ignore silently for channels, but reply in private
            if str(message.get("chat", {}).get("type")) == "private":
                _tg_send(str(chat_id), "⛔ هذا البوت للأدمن فقط.")
            return jsonify({"ok": True})
        settings = _settings()
        if text.startswith("/start"):
            _tg_send(str(chat_id), "🤖 البوت شغال.\nاكتب /menu للأزرار.", reply_markup=_build_menu(settings))
            return jsonify({"ok": True})
        if text.startswith("/menu"):
            _tg_send(str(chat_id), "📌 اختر:", reply_markup=_build_menu(settings))
            return jsonify({"ok": True})
        if text.startswith("/wl"):
            parts = text.strip().split()
            if len(parts) == 1 or (len(parts) >= 2 and parts[1].lower() in ("list","show")):
                wl = get_watchlist()
                if not wl:
                    _tg_send(str(chat_id), "📌 الـ Watchlist فاضي.\nاستخدم: /wl add TSLA")
                    return jsonify({"ok": True})
                _tg_send(str(chat_id), "📌 Watchlist:\n" + "\n".join(wl))
                return jsonify({"ok": True})
            if len(parts) >= 3 and parts[1].lower() in ("add","+"):
                sym = parts[2].upper()
                add_watchlist(sym)
                _tg_send(str(chat_id), f"✅ تم إضافة {sym} للـ Watchlist.")
                return jsonify({"ok": True})
            if len(parts) >= 3 and parts[1].lower() in ("del","remove","rm","-"):
                sym = parts[2].upper()
                remove_watchlist(sym)
                _tg_send(str(chat_id), f"✅ تم حذف {sym} من الـ Watchlist.")
                return jsonify({"ok": True})
            _tg_send(str(chat_id), "استخدم: /wl أو /wl add TSLA أو /wl del TSLA")
            return jsonify({"ok": True})
        if text.startswith("/analyze"):
            _tg_send(str(chat_id), "⏳ جاري التحليل...")
            def _job():
                try:
                    msg, _ = _run_scan_and_build_message(settings)
                    send_telegram(msg)
                except Exception as e:
                    _tg_send(str(chat_id), f"❌ خطأ أثناء الفحص:\n{e}")
            _run_async(_job)
            return jsonify({"ok": True})
        if text.startswith("/ai"):
            parts = text.split()
            if len(parts) < 2:
                _tg_send(str(chat_id), "اكتب: /ai SYMBOL  مثال: /ai TSLA")
                return jsonify({"ok": True})
            symbol = parts[1].upper().strip()
            _start_ai_symbol_analysis(str(chat_id), symbol)
            return jsonify({"ok": True})

        if text.startswith("/settings"):
            _tg_send(str(chat_id), "⚙️", reply_markup=_build_menu(settings))
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
            mode = r.get("mode") or ""
            score = r.get("score")
            lines.append(f"{label} {symbol} ({mode}) | Ret: {ret:.2f}% | Close: {last_close:.2f} | Entry: {entry:.2f} | Score: {float(score):.1f}" if score is not None else f"{label} {symbol} ({mode}) | Ret: {ret:.2f}% | Close: {last_close:.2f} | Entry: {entry:.2f}")
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
    body = "🏆 أفضل 5\n" + "\n".join(fmt_row(r) for r in top5) + "\n\n" + "🧊 أسوأ 5\n" + "\n".join(fmt_row(r) for r in bot5)
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