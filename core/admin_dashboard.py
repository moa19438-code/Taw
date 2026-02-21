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
            return datetime.fromisoformat(ts.replace("Z", "+00:00")).replace(tzinfo=None) >= cutoff
        except Exception:
            return False

    signals_window = [s for s in signals if _in_window(str(s.get("ts") or ""))]

    # top symbols in window
    counts: Dict[str, int] = {}
    for s in signals_window:
        sym = (s.get("symbol") or "").upper().strip()
        if sym:
            counts[sym] = counts.get(sym, 0) + 1
    top_symbols = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:10]

    # Paper open / due now
    open_paper = [r for r in paper if str(r.get("status") or "open").lower() in ("open", "runner", "tp2")]
    due_now = [r for r in paper if str(r.get("due_state") or "").lower() == "due"]

    finals: List[Dict[str, Any]] = []
    if chat_id:
        finals = list_final_paper_reviews_for_chat(chat_id, lookback_days=lookback_days, limit=500)

    wins = sum(1 for r in finals if float(r.get("return_pct") or 0.0) > 0)
    losses = sum(1 for r in finals if float(r.get("return_pct") or 0.0) < 0)
    total = wins + losses
    winrate = (wins / total * 100.0) if total else 0.0

    # Advanced metrics (Platinum)
    import json as _json
    start_capital = float(os.getenv("DASH_START_CAPITAL") or 100.0)
    risk_pct = float(os.getenv("DASH_RISK_PCT") or 1.0)

    finals_chrono = list(reversed(finals))  # oldest -> newest
    r_vals: List[float] = []
    equity: List[float] = [start_capital]

    tp1_hits = tp2_hits = tp3_hits = sl_hits = trail_hits = 0

    for fr in finals_chrono:
        note = {}
        try:
            note = _json.loads(fr.get("note") or "{}")
        except Exception:
            note = {}

        hk = str(note.get("hit_kind") or "").lower()
        if hk == "tp":
            tp1_hits += 1
        elif hk == "tp2":
            tp2_hits += 1
        elif hk == "tp3":
            tp3_hits += 1
        elif hk == "trail":
            trail_hits += 1
        elif hk == "sl":
            sl_hits += 1

        entry = float(note.get("entry") or fr.get("entry") or 0.0)
        sl = float(note.get("sl") or 0.0)
        side = str(note.get("side") or fr.get("side") or "buy").lower()
        exit_price = float(fr.get("exit_price") or 0.0)

        if entry <= 0 or sl <= 0 or abs(entry - sl) < 1e-9 or exit_price <= 0:
            continue

        risk = abs(entry - sl)
        reward = (entry - exit_price) if side == "sell" else (exit_price - entry)
        r_mult = reward / risk
        r_vals.append(r_mult)
        equity.append(equity[-1] * (1.0 + (risk_pct / 100.0) * r_mult))

    avg_r = (sum(r_vals) / len(r_vals)) if r_vals else 0.0
    expectancy_r = avg_r

    pos_sum = sum(x for x in r_vals if x > 0)
    neg_sum = sum(x for x in r_vals if x < 0)
    profit_factor = (pos_sum / abs(neg_sum)) if neg_sum != 0 else (float("inf") if pos_sum > 0 else 0.0)
    profit_factor_display = "∞" if profit_factor == float("inf") else f"{profit_factor:.2f}"

    # Max drawdown (equity curve)
    peak = equity[0] if equity else 0.0
    max_dd = 0.0
    for v in equity:
        if v > peak:
            peak = v
        if peak > 0:
            dd = (peak - v) / peak
            if dd > max_dd:
                max_dd = dd

    def _spark_svg(series: List[float], w: int = 520, h: int = 120) -> str:
        if not series or len(series) < 2:
            return ""
        mn = min(series)
        mx = max(series)
        span = (mx - mn) if mx != mn else 1.0
        pts = []
        for i, val in enumerate(series):
            x = (i / (len(series) - 1)) * (w - 10) + 5
            y = h - ((val - mn) / span) * (h - 10) - 5
            pts.append(f"{x:.2f},{y:.2f}")
        poly = " ".join(pts)
        return f'<svg viewBox="0 0 {w} {h}" width="100%" height="{h}" xmlns="http://www.w3.org/2000/svg"><polyline fill="none" stroke="currentColor" stroke-width="2" points="{poly}"/></svg>'

    equity_svg = _spark_svg(equity)

    bins = [(-5, -2), (-2, -1), (-1, 0), (0, 1), (1, 2), (2, 4), (4, 8)]
    r_hist = []
    for lo, hi in bins:
        c = sum(1 for x in r_vals if x >= lo and x < hi)
        r_hist.append((f"[{lo},{hi})", c))

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
        avg_r=avg_r,
        expectancy_r=expectancy_r,
        profit_factor=profit_factor,
        profit_factor_display=profit_factor_display,
        max_dd=max_dd,
        start_capital=start_capital,
        risk_pct=risk_pct,
        equity_last=equity[-1] if equity else start_capital,
        equity_svg=equity_svg,
        r_hist=r_hist,
        tp1_hits=tp1_hits,
        tp2_hits=tp2_hits,
        tp3_hits=tp3_hits,
        sl_hits=sl_hits,
        trail_hits=trail_hits,
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
