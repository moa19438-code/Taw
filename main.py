
import os
import logging
from flask import Flask, request, jsonify
from scanner import scan_universe_with_meta
from config import RUN_KEY
from storage import send_telegram

app = Flask(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

def format_scan_summary(meta: dict) -> str:
    return (
        "⏱ Scan finished — No signals\n\n"
        f"Universe: {meta.get('universe', 0)}\n"
        f"Checked: {meta.get('checked', 0)}\n"
        f"Liquidity filtered: {meta.get('filtered_liquidity', 0)}\n"
        f"Strong: {meta.get('strong', 0)}\n"
    )

@app.before_first_request
def startup_message():
    try:
        send_telegram("✅ Taw Scanner started and ready.")
        logging.info("Startup message sent to Telegram")
    except Exception as e:
        logging.error(f"Failed to send startup message: {e}")

@app.route("/scan")
def scan():
    key = request.args.get("key")
    notify = request.args.get("notify") == "1"

    if key != RUN_KEY:
        return jsonify({"ok": False, "error": "unauthorized"}), 401

    top, meta, new_signals = scan_universe_with_meta()

    notify_status = {"sent": False, "reason": "notify disabled"}

    if notify:
        try:
            if new_signals:
                send_telegram("\n".join(new_signals))
                notify_status = {"sent": True, "reason": "sent strong"}
            else:
                summary = format_scan_summary(meta)
                send_telegram(summary)
                notify_status = {"sent": True, "reason": "sent summary"}
        except Exception as e:
            logging.error(f"Telegram send failed: {e}")
            notify_status = {"sent": False, "reason": "telegram error"}

    return jsonify({
        "ok": True,
        "top": top,
        "meta": meta,
        "notify": notify,
        "notify_status": notify_status,
    })

@app.route("/status")
def status():
    return {"ok": True}

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
