from __future__ import annotations
import asyncio
from typing import List, Dict, Any
from datetime import datetime, timezone

from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, TELEGRAM_ADMIN_ID, TELEGRAM_CHANNEL_ID, EXECUTE_TRADES, ALLOW_LIVE_TRADING
from scanner import scan_universe_with_meta
from executor import maybe_trade
from storage import last_orders, log_scan, last_scans, set_setting, get_all_settings, parse_float, parse_int, parse_bool


# ================= ADMIN CHECK =================
def _is_admin_private(update: Update) -> bool:
    try:
        if update.effective_chat and update.effective_chat.type != 'private':
            return False
        uid = str(update.effective_user.id) if update.effective_user else ''
        admin = str(TELEGRAM_ADMIN_ID or TELEGRAM_CHAT_ID or '')
        return bool(admin) and uid == admin
    except Exception:
        return False


# ================= CHANNEL SEND =================
async def _send_channel(app: Application, text: str) -> None:
    if not TELEGRAM_CHANNEL_ID:
        return
    chunks = [text[i:i+3800] for i in range(0, len(text), 3800)]
    for ch in chunks:
        await app.bot.send_message(chat_id=TELEGRAM_CHANNEL_ID, text=ch)


# ================= FORMAT PICKS =================
def _format_picks(picks) -> str:
    lines = []
    for i, c in enumerate(picks, start=1):
        lines.append(
            f"{i}) {c.symbol} | score={c.score:.1f} | close={c.last_close:.2f} | ATR={c.atr:.2f} | RSI={c.rsi14:.0f}\n{c.notes}"
        )
    return "\n".join(lines) if lines else "No picks matched filters."


# ================= BASIC COMMANDS =================
async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _is_admin_private(update):
        return
    await update.message.reply_text(
        "/analyze - scan market\n"
        "/summary - last scans\n"
        "/orders - last orders\n"
        "/config - show settings\n\n"
        "CONTROL:\n"
        "/capital 1000\n"
        "/size 20\n"
        "/sl 3\n"
        "/tp 5\n"
        "/max 10\n"
        "/plan daily|weekly|monthly|dw|wm\n"
        "/auto on|off\n"
        "/settings"
    )


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _is_admin_private(update):
        return
    await update.message.reply_text("Bot running.")


async def cmd_orders(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _is_admin_private(update):
        return
    orders = last_orders(10)
    if not orders:
        await update.message.reply_text("No orders.")
        return
    await update.message.reply_text(
        "\n".join(f"{o['ts']} {o['symbol']} {o['side']} qty={o['qty']} {o['status']}" for o in orders)
    )


async def cmd_summary(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _is_admin_private(update):
        return
    scans = last_scans(10)
    if not scans:
        await update.message.reply_text("No scans.")
        return
    await update.message.reply_text(
        "\n".join(f"{s['ts']} universe={s['universe_size']} top={s['top_symbols']}" for s in scans)
    )


# ================= ANALYZE =================
async def cmd_analyze(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _is_admin_private(update):
        return

    await update.message.reply_text("Scanning market...")

    picks, universe_size = scan_universe_with_meta()
    msg = _format_picks(picks)

    await update.message.reply_text(msg)
    await _send_channel(context.application, msg)

    ts = datetime.now(timezone.utc).isoformat()
    top_syms = ",".join([c.symbol for c in picks])
    log_scan(ts, universe_size, top_syms, payload="telegram:/analyze")

    picks_payload = [{"symbol": c.symbol, "last_close": c.last_close, "atr": c.atr} for c in picks[:5]]
    logs = maybe_trade(picks_payload)
    await update.message.reply_text("\n".join(logs))


# ================= SETTINGS CONTROL =================

async def cmd_capital(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _is_admin_private(update): return
    if not context.args:
        await update.message.reply_text("Usage: /capital 1000")
        return
    val = parse_float(context.args[0], -1)
    if val <= 0:
        await update.message.reply_text("Invalid capital.")
        return
    set_setting("CAPITAL_USD", str(val))
    await update.message.reply_text(f"Capital set to {val}$")


async def cmd_size(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _is_admin_private(update): return
    if not context.args:
        await update.message.reply_text("Usage: /size 20")
        return
    val = parse_float(context.args[0], -1)
    if val <= 0 or val > 100:
        await update.message.reply_text("Size must be 1-100%.")
        return
    set_setting("POSITION_PCT", str(val))
    await update.message.reply_text(f"Position size {val}%")


async def cmd_sl(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _is_admin_private(update): return
    if not context.args:
        await update.message.reply_text("Usage: /sl 3")
        return
    val = parse_float(context.args[0], -1)
    if val <= 0 or val > 20:
        await update.message.reply_text("SL must be between 0-20%.")
        return
    set_setting("SL_PCT", str(val))
    await update.message.reply_text(f"Stop loss {val}%")


async def cmd_tp(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _is_admin_private(update): return
    if not context.args:
        await update.message.reply_text("Usage: /tp 5")
        return
    val = parse_float(context.args[0], -1)
    if val <= 0 or val > 50:
        await update.message.reply_text("TP must be between 0-50%.")
        return
    set_setting("TP_PCT", str(val))
    await update.message.reply_text(f"Take profit {val}%")


async def cmd_max(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _is_admin_private(update): return
    if not context.args:
        await update.message.reply_text("Usage: /max 10")
        return
    n = parse_int(context.args[0], -1)
    if n <= 0 or n > 50:
        await update.message.reply_text("Invalid number.")
        return
    set_setting("MAX_SEND", str(n))
    await update.message.reply_text(f"Max signals per scan {n}")


async def cmd_plan(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _is_admin_private(update): return
    if not context.args:
        await update.message.reply_text("daily | weekly | monthly | dw | wm")
        return
    val = context.args[0].lower()
    set_setting("PLAN_MODE", val)
    await update.message.reply_text(f"Plan set to {val}")


async def cmd_auto(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _is_admin_private(update): return
    if not context.args:
        await update.message.reply_text("/auto on أو off")
        return
    state = context.args[0].lower() == "on"
    set_setting("AUTO_NOTIFY", "1" if state else "0")
    await update.message.reply_text(f"Auto notify {'ON' if state else 'OFF'}")


async def cmd_settings(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _is_admin_private(update): return
    s = get_all_settings()
    msg = (
        f"⚙️ SETTINGS\n"
        f"Capital: {s.get('CAPITAL_USD')}$\n"
        f"Size: {s.get('POSITION_PCT')}%\n"
        f"SL: {s.get('SL_PCT')}%\n"
        f"TP: {s.get('TP_PCT')}%\n"
        f"Max signals: {s.get('MAX_SEND')}\n"
        f"Plan: {s.get('PLAN_MODE')}\n"
        f"Auto notify: {s.get('AUTO_NOTIFY')}"
    )
    await update.message.reply_text(msg)


# ================= BUILD APP =================
def build_app() -> Application:
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("orders", cmd_orders))
    app.add_handler(CommandHandler("summary", cmd_summary))
    app.add_handler(CommandHandler("analyze", cmd_analyze))

    # new controls
    app.add_handler(CommandHandler("capital", cmd_capital))
    app.add_handler(CommandHandler("size", cmd_size))
    app.add_handler(CommandHandler("sl", cmd_sl))
    app.add_handler(CommandHandler("tp", cmd_tp))
    app.add_handler(CommandHandler("max", cmd_max))
    app.add_handler(CommandHandler("plan", cmd_plan))
    app.add_handler(CommandHandler("auto", cmd_auto))
    app.add_handler(CommandHandler("settings", cmd_settings))

    return app


# ================= RUN =================
async def run_polling():
    if not TELEGRAM_BOT_TOKEN:
        return
    app = build_app()
    await app.initialize()
    await app.start()
    await app.updater.start_polling()
