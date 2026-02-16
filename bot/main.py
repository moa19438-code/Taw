
import os
import sys
from flask import Flask, request, jsonify

# Ensure core is in path
BASE_DIR = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(BASE_DIR, "core"))

from core.app_main import (
    _weekly_report,
    _tg_send,
    _run_async,
    app  # Flask app imported from core
)

@app.post("/telegram/webhook")
def telegram_webhook():
    data = request.get_json(silent=True) or {}
    cb = data.get("callback_query")

    if cb:
        chat_id = cb.get("message", {}).get("chat", {}).get("id")
        action = cb.get("data")

        # ================= التقرير الأسبوعي =================
        if action == "weekly_report":
            _tg_send(str(chat_id), "⏳ جاري إعداد التقرير الأسبوعي...")

            def _job():
                try:
                    msg = _weekly_report(days=7)
                    _tg_send(str(chat_id), msg)
                except Exception as e:
                    _tg_send(str(chat_id), f"❌ خطأ أثناء إنشاء التقرير:\\n{e}")

            _run_async(_job)
            return jsonify({"ok": True})

        return jsonify({"ok": True})

    return jsonify({"ok": True})
