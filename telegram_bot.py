from __future__ import annotations
import asyncio
from typing import List, Dict, Any
from datetime import datetime, timezone

from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

from config import (
    TELEGRAM_BOT_TOKEN,
    TELEGRAM_CHAT_ID,
    TELEGRAM_ADMIN_ID,
    TELEGRAM_CHANNEL_ID,
    EXECUTE_TRADES,
    ALLOW_LIVE_TRADING,
)
from scanner import scan_universe_with_meta
from executor import maybe_trade
from storage import (
    last_orders,
    log_scan,
    last_scans,
    set_setting,
    get_all_settings,
    parse_float,
    parse_int,
    parse_bool,
)


def _is_admin_private(update: Update) -> bool:
    """Commands allowed only from admin in private chat."""
    try:
        if update.effective_chat and update.effective_chat.type != "private":
            return False
        uid = str(update.effective_user.id) if update.effective_user else ""
        admin = str(TELEGRAM_ADMIN_ID or TELEGRAM_CHAT_ID or "")
        return bool(admin) and uid == admin
    except Exception:
        return False


async def _send_channel(app: Application, text: str) -> None:
    if not TELEGRAM_CHANNEL_ID:
        return
    # Split long messages (Telegram limit ~4096)
    chunks = [text[i : i + 3800] for i in range(0, len(text), 3800)]
    for ch in chunks:
        await app.bot.send_message(chat_id=TELEGRAM_CHANNEL_ID, text=ch)


def _format_picks(picks) -> str:
    lines = []
    for i, c in enumerate(picks, start=1):
        lines.append(
            f"{i}) {c.symbol}  score={c.score:.1f}  close={c.last_close:.2f}  ATR={c.atr:.2f}  RSI={c.rsi14:.0f}\n   {c.notes}"
        )
    return "\n".join(lines) if lines else "No picks matched filters."


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _is_admin_private(update):
        return
    await update.message.reply_text(
        "/analyze - scan the US market and show top picks\n"
        "/summary - show last scan summaries\n"
        "/status - show basic status\n"
        "/orders - show last orders\n"
        "\n"
        "Manual-trading controls (saved in DB):\n"
        "/capital <usd>            مثال: /capital 1200\n"
        "/size <pct>               مثال: /size 20   (يعني 20% من رأس المال)\n"
        "/sl <pct>                 مثال: /sl 3\n"
        "/tp <pct>                 مثال: /tp 5\n"
        "/signals <n>              مثال: /signals 10 (عدد الفرص المرسلة)\n"
        "/plan daily|weekly|monthly|dw|wm\n"
        "/mode breakout|pullback|mixed\n"
        "/auto_notify on|off\n"
        "/auto_status\n"
        "/auto_test\n"
        "/settings\n"
        "\n"
        "Auto-trade (تنفيذ أوامر) موجود: /trade_on /trade_off /risk /max /top\n"
    )


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _is_admin_private(update):
        return
    await update.message.reply_text(
        "Bot is running. Use /analyze to scan. Use /settings to see manual-trading settings."
    )


async def cmd_orders(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _is_admin_private(update):
        return
    orders = last_orders(10)
    if not orders:
        await update.message.reply_text("No orders logged yet.")
        return
    lines = []
    for o in orders:
        lines.append(
            f"{o['ts']} {o['symbol']} {o['side']} qty={o['qty']} {o['status']} {o.get('message','')}"
        )
    await update.message.reply_text("\n".join(lines))


async def cmd_summary(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _is_admin_private(update):
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
    if not _is_admin_private(update):
        return
    await update.message.reply_text("Scanning... this may take a minute.")
    picks, universe_size = scan_universe_with_meta()
    msg = _format_picks(picks)
    await update.message.reply_text(msg)
    # Post same to channel
    await _send_channel(context.application, msg)

    # Log scan
    ts = datetime.now(timezone.utc).isoformat()
    top_syms = ",".join([c.symbol for c in picks])
    log_scan(ts, universe_size, top_syms, payload="telegram:/analyze")

    # Optional execution
    picks_payload = [{"symbol": c.symbol, "last_close": c.last_close, "atr": c.atr} for c in picks[:5]]
    logs = maybe_trade(picks_payload)
    await update.message.reply_text("\n".join(logs))


# ================= Manual trading controls (DB) =================
async def cmd_capital(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _is_admin_private(update):
        return
    if not context.args:
        await update.message.reply_text("Usage: /capital 1200")
        return
    val = parse_float(context.args[0], -1.0)
    if val <= 0:
        await update.message.reply_text("Capital must be > 0")
        return
    set_setting("CAPITAL_USD", str(val))
    await update.message.reply_text(f"✅ CAPITAL_USD = {val}$")


async def cmd_size(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _is_admin_private(update):
        return
    if not context.args:
        await update.message.reply_text("Usage: /size 20   (20% of capital per trade)")
        return
    val = parse_float(context.args[0], -1.0)
    if val <= 0 or val > 100:
        await update.message.reply_text("Size must be between 1 and 100 (percent).")
        return
    # store as percent (0-100)
    set_setting("POSITION_PCT", str(val / 100.0))
    await update.message.reply_text(f"✅ POSITION_PCT = {val}%")


async def cmd_sl(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _is_admin_private(update):
        return
    if not context.args:
        await update.message.reply_text("Usage: /sl 3")
        return
    val = parse_float(context.args[0], -1.0)
    if val <= 0 or val > 20:
        await update.message.reply_text("SL must be between 0 and 20 percent.")
        return
    set_setting("SL_PCT", str(val))
    await update.message.reply_text(f"✅ SL_PCT = {val}%")


async def cmd_tp(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _is_admin_private(update):
        return
    if not context.args:
        await update.message.reply_text("Usage: /tp 5")
        return
    val = parse_float(context.args[0], -1.0)
    if val <= 0 or val > 50:
        await update.message.reply_text("TP must be between 0 and 50 percent.")
        return
    set_setting("TP_PCT", str(val))
    await update.message.reply_text(f"✅ TP_PCT = {val}%")


async def cmd_signals(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """How many opportunities the bot should SEND per scan (manual-trading UI)."""
    if not _is_admin_private(update):
        return
    if not context.args:
        await update.message.reply_text("Usage: /signals 10")
        return
    n = parse_int(context.args[0], -1)
    if n < 1 or n > 30:
        await update.message.reply_text("Signals must be between 1 and 30.")
        return
    set_setting("MAX_SEND", str(n))
    await update.message.reply_text(f"✅ MAX_SEND = {n}")


async def cmd_plan(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _is_admin_private(update):
        return
    if not context.args:
        await update.message.reply_text("Usage: /plan daily|weekly|monthly|dw|wm")
        return
    val = context.args[0].strip().lower()
    if val not in ("daily", "weekly", "monthly", "dw", "wm", "daily_weekly", "weekly_monthly"):
        await update.message.reply_text("Use: daily | weekly | monthly | dw | wm")
        return
    # normalize short forms
    if val == "dw":
        val = "daily_weekly"
    if val == "wm":
        val = "weekly_monthly"
    set_setting("PLAN_MODE", val)
    await update.message.reply_text(f"✅ PLAN_MODE = {val}")


async def cmd_mode(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _is_admin_private(update):
        return
    if not context.args:
        await update.message.reply_text("Usage: /mode breakout|pullback|mixed")
        return
    mode = context.args[0].strip().lower()
    if mode not in ("breakout", "pullback", "mixed"):
        await update.message.reply_text("Use: breakout | pullback | mixed")
        return
    set_setting("SCAN_MODE", mode)
    await update.message.reply_text(f"✅ SCAN_MODE = {mode}")


async def cmd_auto_notify(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Toggle whether /scan?notify=1 broadcasts to Telegram."""
    if not _is_admin_private(update):
        return
    if not context.args:
        await update.message.reply_text("Usage: /auto_notify on|off")
        return
    on = context.args[0].strip().lower() in ("1", "on", "true", "yes")
    set_setting("AUTO_NOTIFY", "1" if on else "0")
    await update.message.reply_text(f"✅ AUTO_NOTIFY = {'ON' if on else 'OFF'}")


async def cmd_auto_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Helps you confirm GitHub schedule is running (based on last scan time)."""
    if not _is_admin_private(update):
        return
    scans = last_scans(1)
    if not scans:
        await update.message.reply_text("No scans logged yet.")
        return
    last_ts = scans[0].get("ts", "")
    msg = f"Last scan logged: {last_ts}\n"
    msg += "If this time updates automatically every ~20 minutes during market hours, then GitHub schedule is working."
    await update.message.reply_text(msg)


async def cmd_auto_test(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Run a manual scan test (doesn't rely on GitHub schedule)."""
    if not _is_admin_private(update):
        return
    await update.message.reply_text("Running scan test now...")
    picks, universe_size = scan_universe_with_meta()
    msg = _format_picks(picks)
    await update.message.reply_text(msg)


async def cmd_settings(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _is_admin_private(update):
        return
    s = get_all_settings()
    lines = [
        "Manual-trading settings (DB):",
        f"- CAPITAL_USD={s.get('CAPITAL_USD')}",
        f"- POSITION_PCT={s.get('POSITION_PCT')}  (fraction of capital)",
        f"- SL_PCT={s.get('SL_PCT')}",
        f"- TP_PCT={s.get('TP_PCT')}",
        f"- MAX_SEND={s.get('MAX_SEND')}",
        f"- PLAN_MODE={s.get('PLAN_MODE')}",
        f"- SCAN_MODE={s.get('SCAN_MODE','mixed')}",
        f"- AUTO_NOTIFY={s.get('AUTO_NOTIFY')}",
        "",
        f"Safety latches (ENV): EXECUTE_TRADES={EXECUTE_TRADES}  ALLOW_LIVE_TRADING={ALLOW_LIVE_TRADING}",
        f"Channel: {TELEGRAM_CHANNEL_ID or '(not set)'}",
    ]
    await update.message.reply_text("\n".join(lines))


# ================= Existing auto-trade commands (kept as-is) =================
async def cmd_trade_on(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _is_admin_private(update):
        return
    set_setting("AUTO_TRADE", "1")
    await update.message.reply_text(
        "AUTO_TRADE enabled (DB). Note: execution still requires EXECUTE_TRADES & ALLOW_LIVE_TRADING env vars."
    )


async def cmd_trade_off(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _is_admin_private(update):
        return
    set_setting("AUTO_TRADE", "0")
    await update.message.reply_text("AUTO_TRADE disabled (DB). Scanner only.")


async def cmd_risk(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _is_admin_private(update):
        return
    if not context.args:
        await update.message.reply_text("Usage: /risk 0.25  (percent of equity risked per trade)")
        return
    val = parse_float(context.args[0], -1.0)
    if val <= 0 or val > 5:
        await update.message.reply_text("Risk must be a small positive percent (example: 0.25).")
        return
    set_setting("RISK_PER_TRADE_PCT", str(val))
    await update.message.reply_text(f"Set RISK_PER_TRADE_PCT={val}")


async def cmd_max(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """MAX_DAILY_TRADES (auto-trade)."""
    if not _is_admin_private(update):
        return
    if not context.args:
        await update.message.reply_text("Usage: /max 2")
        return
    n = parse_int(context.args[0], -1)
    if n < 0 or n > 50:
        await update.message.reply_text("MAX_DAILY_TRADES must be between 0 and 50.")
        return
    set_setting("MAX_DAILY_TRADES", str(n))
    await update.message.reply_text(f"Set MAX_DAILY_TRADES={n}")


async def cmd_top(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _is_admin_private(update):
        return
    if not context.args:
        await update.message.reply_text("Usage: /top 5")
        return
    n = parse_int(context.args[0], -1)
    if n < 1 or n > 30:
        await update.message.reply_text("TOP_N must be between 1 and 30.")
        return
    set_setting("TOP_N", str(n))
    await update.message.reply_text(f"Set TOP_N={n}")


def build_app() -> Application:
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("orders", cmd_orders))
    app.add_handler(CommandHandler("summary", cmd_summary))
    app.add_handler(CommandHandler("analyze", cmd_analyze))

    # manual trading controls
    app.add_handler(CommandHandler("capital", cmd_capital))
    app.add_handler(CommandHandler("size", cmd_size))
    app.add_handler(CommandHandler("sl", cmd_sl))
    app.add_handler(CommandHandler("tp", cmd_tp))
    app.add_handler(CommandHandler("signals", cmd_signals))
    app.add_handler(CommandHandler("plan", cmd_plan))
    app.add_handler(CommandHandler("mode", cmd_mode))
    app.add_handler(CommandHandler("auto_notify", cmd_auto_notify))
    app.add_handler(CommandHandler("auto_status", cmd_auto_status))
    app.add_handler(CommandHandler("auto_test", cmd_auto_test))
    app.add_handler(CommandHandler("settings", cmd_settings))

    # existing auto-trade controls
    app.add_handler(CommandHandler("trade_on", cmd_trade_on))
    app.add_handler(CommandHandler("trade_off", cmd_trade_off))
    app.add_handler(CommandHandler("risk", cmd_risk))
    app.add_handler(CommandHandler("max", cmd_max))
    app.add_handler(CommandHandler("top", cmd_top))

    return app


async def run_polling():
    if not TELEGRAM_BOT_TOKEN:
        return
    app = build_app()
    await app.initialize()
    await app.start()
    await app.updater.start_polling()
