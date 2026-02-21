from __future__ import annotations

import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Tuple

from flask import Blueprint, Response, abort, redirect, render_template, request, url_for

from core.storage import (
    list_paper_trades_for_chat,
    delete_paper_trade_for_chat,
    cleanup_old_paper_trades,
    last_signals,
    get_watchlist,
    add_watchlist,
    remove_watchlist,
    list_final_paper_reviews_for_chat,
)

bp = Blueprint("admin", __name__, url_prefix="/admin")

def _admin_credentials() -> Tuple[str, str]:
    user = (os.getenv("ADMIN_USER") or "admin").strip()
    pw = (os.getenv("ADMIN_PASS") or "change-me").strip()
    return user, pw

def _check_auth() -> bool:
    user, pw = _admin_credentials()
    auth = request.authorization
    return bool(auth and auth.username == user and auth.password == pw)

def _require_auth() -> Response | None:
    if _check_auth():
        return None
    return Response(
        "Authentication required",
        401,
        {"WWW-Authenticate": 'Basic realm="taw-admin"'},
    )

@bp.get("/")
def index():
    need = _require_auth()
    if need:
        return need

    # lightweight stats from paper trades and signals
    lookback_days = int(os.getenv("DASH_LOOKBACK_DAYS") or 7)
    chat_id = (os.getenv("TELEGRAM_CHAT_ID") or "").strip()

    paper: List[Dict[str, Any]] = []
    if chat_id:
        paper = list_paper_trades_for_chat(chat_id, lookback_days=lookback_days, limit=200)

    signals = last_signals(200)

    now = datetime.utcnow()
    cutoff = now - timedelta(days=lookback_days)

    def _in_window(ts: str) -> bool:
        try:
            # ts stored as isoformat
            return datetime.fromisoformat(ts.replace("Z", "+00:00")).replace(tzinfo=None) >= cutoff
        except Exception:
            return False

    signals_window = [s for s in signals if _in_window(str(s.get("ts") or ""))]

    # top symbols in window
    counts: Dict[str, int] = {}
    for s in signals_window:
        sym = str(s.get("symbol") or "").upper()
        if not sym:
            continue
        counts[sym] = counts.get(sym, 0) + 1
    top_symbols = sorted(counts.items(), key=lambda x: (-x[1], x[0]))[:10]

    
    # paper trade KPI
    open_paper = [p for p in paper if str(p.get("status") or "open").lower() in ("open","")]
    due_now = []
    for p in open_paper:
        try:
            dt = datetime.fromisoformat(str(p.get("due_ts") or "").replace("Z","+00:00")).replace(tzinfo=None)
        except Exception:
            continue
        if dt <= datetime.utcnow():
            due_now.append(p)

    finals = []
    try:
        if chat_id:
            finals = list_final_paper_reviews_for_chat(chat_id, lookback_days=lookback_days, limit=500)
    except Exception:
        finals = []
    wins = sum(1 for r in finals if float(r.get("return_pct") or 0.0) > 0)
    losses = sum(1 for r in finals if float(r.get("return_pct") or 0.0) < 0)
    total = wins + losses
    winrate = (wins / total * 100.0) if total else 0.0

    return render_template(
        "admin/index.html",
        lookback_days=lookback_days,
        paper_count=len(paper),
        open_paper_count=len(open_paper),
        due_now_count=len(due_now),
        signals_count=len(signals_window),
        finals_count=len(finals),
        winrate=winrate,
        top_symbols=top_symbols,
        recent_paper=paper[:20],
    )

@bp.get("/paper")
def paper():
    need = _require_auth()
    if need:
        return need

    lookback_days = int(os.getenv("DASH_LOOKBACK_DAYS") or 7)
    chat_id = (os.getenv("TELEGRAM_CHAT_ID") or "").strip()
    if not chat_id:
        return render_template("admin/paper.html", lookback_days=lookback_days, rows=[], chat_missing=True)

    rows = list_paper_trades_for_chat(chat_id, lookback_days=lookback_days, limit=500)
    return render_template("admin/paper.html", lookback_days=lookback_days, rows=rows, chat_missing=False)

@bp.post("/paper/delete")
def paper_delete():
    need = _require_auth()
    if need:
        return need

    chat_id = (os.getenv("TELEGRAM_CHAT_ID") or "").strip()
    if not chat_id:
        abort(400, "TELEGRAM_CHAT_ID not set")

    pid = request.form.get("paper_id", "").strip()
    if not pid.isdigit():
        abort(400, "invalid paper_id")
    delete_paper_trade_for_chat(chat_id, int(pid))
    return redirect(url_for("admin.paper"))

@bp.get("/signals")
def signals():
    need = _require_auth()
    if need:
        return need
    limit = int(os.getenv("DASH_SIGNALS_LIMIT") or 300)
    rows = last_signals(limit)
    return render_template("admin/signals.html", rows=rows, limit=limit)

@bp.get("/wl")
def wl():
    need = _require_auth()
    if need:
        return need
    rows = get_watchlist()
    return render_template("admin/wl.html", rows=rows)

@bp.post("/wl/add")
def wl_add():
    need = _require_auth()
    if need:
        return need
    sym = (request.form.get("symbol") or "").strip().upper()
    if not sym or len(sym) > 12:
        abort(400, "invalid symbol")
    add_watchlist(sym)
    return redirect(url_for("admin.wl"))

@bp.post("/wl/delete")
def wl_delete():
    need = _require_auth()
    if need:
        return need
    sym = (request.form.get("symbol") or "").strip().upper()
    if not sym or len(sym) > 12:
        abort(400, "invalid symbol")
    remove_watchlist(sym)
    return redirect(url_for("admin.wl"))

@bp.post("/maintenance/cleanup")
def cleanup():
    need = _require_auth()
    if need:
        return need
    days = int(request.form.get("days") or 7)
    n = cleanup_old_paper_trades(retention_days=days)
    return redirect(url_for("admin.index"))


@bp.get("/selfcheck")
def selfcheck():
    """Pretty self-check report (same logic as /self_check command in bot)."""
    need = _require_auth()
    if need:
        return need

    fix = (request.args.get("fix") or "0").strip() == "1"
    # Lazy import to avoid circular import at app start
    from core.app_main import _self_check

    report = _self_check(fix=fix)
    return render_template("admin/selfcheck.html", report=report, fix=fix)
