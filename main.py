from __future__ import annotations
from datetime import datetime, timezone, timedelta
import requests
from flask import Flask, request, jsonify

from config import RUN_KEY, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, TELEGRAM_ADMIN_ID, TELEGRAM_CHANNEL_ID, SEND_DAILY_SUMMARY
from storage import init_db, ensure_default_settings, last_orders, log_scan, last_scans
from scanner import scan_universe_with_meta
from executor import maybe_trade

app = Flask(__name__)
init_db()
ensure_default_settings()


# ================= TELEGRAM SEND =================
def _tg_send(chat_id: str, text: str) -> None:
    if not (TELEGRAM_BOT_TOKEN and chat_id):
        return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        requests.post(url, data={"chat_id": chat_id, "text": text}, timeout=15)
    except Exception:
        pass


def send_telegram(text: str) -> None:
    """Send to channel + admin."""
    if TELEGRAM_CHANNEL_ID:
        _tg_send(TELEGRAM_CHANNEL_ID, text)

    admin_id = TELEGRAM_ADMIN_ID or TELEGRAM_CHAT_ID
    if admin_id:
        _tg_send(admin_id, text)


# ================= TELEGRAM WEBHOOK =================
@app.post("/webhook")
def telegram_webhook():
    if not TELEGRAM_BOT_TOKEN:
        return jsonify({"ok": True})

    data = request.get_json(force=True)

    message = data.get("message") or data.get("channel_post")
    if not message:
        return jsonify({"ok": True})

    chat_id = message["chat"]["id"]
    user_id = message.get("from", {}).get("id")
    text = message.get("text", "")

    # السماح فقط للأدمن
    if TELEGRAM_ADMIN_ID and user_id != TELEGRAM_ADMIN_ID:
    _tg_send(chat_id, "⛔ هذا البوت للأدمن فقط. راجع TELEGRAM_ADMIN_ID في Render.")
    return jsonify({"ok": True})

    # ===== /start =====
    if text.startswith("/start"):
        _tg_send(chat_id, "🤖 البوت شغال.\nاستخدم /analyze لفحص السوق.")
        return jsonify({"ok": True})

    # ===== /analyze =====
    if text.startswith("/analyze"):
        _tg_send(chat_id, "🔎 جاري فحص السوق...")

        try:
            picks, universe_size = scan_universe_with_meta()

            if not picks:
                msg = "❌ لا توجد فرص اليوم."
            else:
                top = picks[0]
                msg = (
                    f"📊 أفضل سهم اليوم:\n"
                    f"{top.symbol}\n"
                    f"السعر: {top.last_close}\n"
                    f"ATR: {top.atr:.2f}\n"
                    f"Score: {top.score}\n"
                    f"ملاحظات: {top.notes}"
                )

            send_telegram(msg)

        except Exception as e:
            _tg_send(chat_id, f"❌ خطأ أثناء الفحص:\n{e}")

        return jsonify({"ok": True})

    return jsonify({"ok": True})


# ================= API ROUTES =================
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


@app.get("/scan")
def scan():
    if request.args.get("key") != RUN_KEY:
        return jsonify({"ok": False, "error": "unauthorized"}), 401

    picks, universe_size = scan_universe_with_meta()

    picks_payload = [
        {"symbol": c.symbol, "score": c.score, "last_close": c.last_close, "atr": c.atr, "notes": c.notes}
        for c in picks
    ]

    ts = datetime.now(timezone.utc).isoformat()
    top_syms = ",".join([c.symbol for c in picks])
    log_scan(ts, universe_size, top_syms, payload="http:/scan")

    trade_logs = maybe_trade([{"symbol": c.symbol, "last_close": c.last_close, "atr": c.atr} for c in picks[:5]])

    if request.args.get("notify") == "1":
        send_telegram(f"Scan done (universe={universe_size}). Top: {top_syms}\n" + "\n".join(trade_logs))

    return jsonify({"ok": True, "universe_size": universe_size, "top": picks_payload, "trade_logs": trade_logs})


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
