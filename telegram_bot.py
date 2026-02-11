from __future__ import annotations
import asyncio
from typing import List, Dict, Any
from datetime import datetime, timezone

from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
from scanner import scan_universe_with_meta
from executor import maybe_trade
from storage import last_orders, log_scan, last_scans

def _allowed_chat(update: Update) -> bool:
    if not TELEGRAM_CHAT_ID:
        return True
    try:
        return str(update.effective_chat.id) == str(TELEGRAM_CHAT_ID)
    except Exception:
        return False

def _format_picks(picks) -> str:
    lines = []
    for i, c in enumerate(picks, start=1):
        lines.append(
            f"{i}) {c.symbol}  score={c.score:.1f}  close={c.last_close:.2f}  ATR={c.atr:.2f}  RSI={c.rsi14:.0f}\n   {c.notes}"
        )
    return "\n".join(lines) if lines else "No picks matched filters."

async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _allowed_chat(update):
        return
    await update.message.reply_text(
        "/analyze - scan the US market and show top picks\n"
        "/summary - show last scan summaries\n"
        "/status - show basic status\n"
        "/orders - show last orders\n"
    )

async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _allowed_chat(update):
        return
    await update.message.reply_text("Bot is running. Use /analyze to scan.")

async def cmd_orders(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _allowed_chat(update):
        return
    orders = last_orders(10)
    if not orders:
        await update.message.reply_text("No orders logged yet.")
        return
    lines = []
    for o in orders:
        lines.append(f"{o['ts']} {o['symbol']} {o['side']} qty={o['qty']} {o['status']} {o['message']}")
    await update.message.reply_text("\n".join(lines))

async def cmd_summary(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _allowed_chat(update):
        return
    scans = last_scans(10)
    if not scans:
        await update.message.reply_text("No scans logged yet.")
        return
    lines = []
    for s in scans:
        lines.append(f"{s['ts']} universe={s['universe_size']} top={s['top_symbols']}")
    await update.message.reply_text("\n".join(lines))

async def cmd_analyze(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _allowed_chat(update):
        return
    await update.message.reply_text("Scanning... this may take a minute.")
    picks, universe_size = scan_universe_with_meta()
    msg = _format_picks(picks)
    await update.message.reply_text(msg)

    # Log scan
    ts = datetime.now(timezone.utc).isoformat()
    top_syms = ",".join([c.symbol for c in picks])
    log_scan(ts, universe_size, top_syms, payload="telegram:/analyze")

    # Optional execution
    picks_payload = [{"symbol": c.symbol, "last_close": c.last_close, "atr": c.atr} for c in picks[:3]]
    logs = maybe_trade(picks_payload)
    await update.message.reply_text("\n".join(logs))

def build_app() -> Application:
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("orders", cmd_orders))
    app.add_handler(CommandHandler("summary", cmd_summary))
    app.add_handler(CommandHandler("analyze", cmd_analyze))
    return app

async def run_polling():
    if not TELEGRAM_BOT_TOKEN:
        return
    app = build_app()
    await app.initialize()
    await app.start()
    await app.updater.start_polling()
