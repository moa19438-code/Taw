import os
import sqlite3
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta
import json


def json_dumps(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False)
    except Exception:
        return ""

DB_PATH = "trades.db"
DATABASE_URL = (os.getenv("DATABASE_URL") or "").strip()

IS_POSTGRES = DATABASE_URL.startswith("postgresql://") or DATABASE_URL.startswith("postgres://")

if IS_POSTGRES:
    import psycopg
    from psycopg.rows import dict_row


def _pg_connect():
    # DATABASE_URL لازم يكون فيه ?sslmode=require
    return psycopg.connect(DATABASE_URL)


def init_db() -> None:
    if IS_POSTGRES:
        with _pg_connect() as con:
            with con.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS scans (
                        id BIGSERIAL PRIMARY KEY,
                        ts TEXT NOT NULL,
                        universe_size INTEGER,
                        top_symbols TEXT,
                        payload TEXT
                    );
                """)
                cur.execute("CREATE INDEX IF NOT EXISTS idx_scans_ts ON scans(ts);")

                cur.execute("""
                    CREATE TABLE IF NOT EXISTS orders (
                        id BIGSERIAL PRIMARY KEY,
                        ts TEXT NOT NULL,
                        symbol TEXT NOT NULL,
                        side TEXT NOT NULL,
                        qty DOUBLE PRECISION NOT NULL,
                        order_type TEXT NOT NULL,
                        payload TEXT,
                        broker_order_id TEXT,
                        status TEXT NOT NULL,
                        message TEXT
                    );
                """)
                cur.execute("CREATE INDEX IF NOT EXISTS idx_orders_ts ON orders(ts);")

                cur.execute("""
                    CREATE TABLE IF NOT EXISTS settings (
                        key TEXT PRIMARY KEY,
                        value TEXT NOT NULL
                    );
                """)

                cur.execute("""
                    CREATE TABLE IF NOT EXISTS watchlist (
                        symbol TEXT PRIMARY KEY,
                        added_ts TEXT
                    );
                """)

                cur.execute("""
                    CREATE TABLE IF NOT EXISTS user_state (
                        chat_id TEXT NOT NULL,
                        key TEXT NOT NULL,
                        value TEXT NOT NULL,
                        PRIMARY KEY (chat_id, key)
                    );
                """)

                cur.execute("""
                    CREATE TABLE IF NOT EXISTS signals (
                        id BIGSERIAL PRIMARY KEY,
                        ts TEXT NOT NULL,
                        symbol TEXT NOT NULL,
                        mode TEXT NOT NULL,
                        strength TEXT NOT NULL,
                        score DOUBLE PRECISION,
                        entry DOUBLE PRECISION,
                        sl DOUBLE PRECISION,
                        tp DOUBLE PRECISION
                    );
                """)
                cur.execute("CREATE INDEX IF NOT EXISTS idx_signals_ts ON signals(ts);")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_signals_symbol_mode ON signals(symbol, mode);")
                # --- MIGRATION: ensure signals has extended columns used by insert_signal() ---
                cur.execute("ALTER TABLE signals ADD COLUMN IF NOT EXISTS source TEXT;")
                cur.execute("ALTER TABLE signals ADD COLUMN IF NOT EXISTS side TEXT;")
                cur.execute("ALTER TABLE signals ADD COLUMN IF NOT EXISTS features_json TEXT;")
                cur.execute("ALTER TABLE signals ADD COLUMN IF NOT EXISTS reasons_json TEXT;")
                cur.execute("ALTER TABLE signals ADD COLUMN IF NOT EXISTS horizon_days INTEGER;")
                cur.execute("ALTER TABLE signals ADD COLUMN IF NOT EXISTS evaluated INTEGER DEFAULT 0;")
                cur.execute("ALTER TABLE signals ADD COLUMN IF NOT EXISTS model_prob DOUBLE PRECISION;")


                cur.execute("""
                    CREATE TABLE IF NOT EXISTS signal_outcomes (
                        id BIGSERIAL PRIMARY KEY,
                        ts TEXT NOT NULL,
                        signal_id BIGINT NOT NULL,
                        result TEXT NOT NULL,
                        r_mult DOUBLE PRECISION,
                        notes TEXT
                    );
                """)
                cur.execute("CREATE INDEX IF NOT EXISTS idx_signal_outcomes_signal_id ON signal_outcomes(signal_id);")

                cur.execute("""
                    CREATE TABLE IF NOT EXISTS signal_reviews (
                        id BIGSERIAL PRIMARY KEY,
                        ts TEXT NOT NULL,
                        signal_id BIGINT NOT NULL,
                        close DOUBLE PRECISION,
                        return_pct DOUBLE PRECISION,
                        mfe_pct DOUBLE PRECISION,
                        mae_pct DOUBLE PRECISION,
                        note TEXT
                    );
                """)
                cur.execute("CREATE INDEX IF NOT EXISTS idx_signal_reviews_signal_id ON signal_reviews(signal_id);")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_signal_reviews_ts ON signal_reviews(ts);")

                cur.execute("""
                    CREATE TABLE IF NOT EXISTS paper_trades (
                        id BIGSERIAL PRIMARY KEY,
                        chat_id TEXT NOT NULL,
                        signal_id BIGINT NOT NULL,
                        due_ts TEXT NOT NULL,
                        notified INTEGER DEFAULT 0
                    );
                """)
                cur.execute("CREATE INDEX IF NOT EXISTS idx_paper_trades_due ON paper_trades(due_ts);")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_paper_trades_chat ON paper_trades(chat_id);")


            # ensure backwards-compatible schema additions
            try:
                ensure_signal_schema()
                ensure_signal_reviews_schema()
                ensure_paper_trades_schema()
            except Exception:
                pass

            con.commit()
        return

    # SQLite fallback (زي كودك)
    with sqlite3.connect(DB_PATH) as con:
        con.execute(
            """CREATE TABLE IF NOT EXISTS scans (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                universe_size INTEGER,
                top_symbols TEXT,
                payload TEXT
            )"""
        )
        con.execute("CREATE INDEX IF NOT EXISTS idx_scans_ts ON scans(ts)")

        con.execute(
            """CREATE TABLE IF NOT EXISTS orders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                qty REAL NOT NULL,
                order_type TEXT NOT NULL,
                payload TEXT,
                broker_order_id TEXT,
                status TEXT NOT NULL,
                message TEXT
            )"""
        )
        con.execute("CREATE INDEX IF NOT EXISTS idx_orders_ts ON orders(ts)")

        con.execute(
            """CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )"""
        )

        con.execute(
            """CREATE TABLE IF NOT EXISTS watchlist (
                symbol TEXT PRIMARY KEY,
                added_ts TEXT
            )"""
        )

        con.execute(
            """CREATE TABLE IF NOT EXISTS user_state (
                chat_id TEXT NOT NULL,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                PRIMARY KEY (chat_id, key)
            )"""
        )

        con.execute(
            """CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                symbol TEXT NOT NULL,
                mode TEXT NOT NULL,
                strength TEXT NOT NULL,
                score REAL,
                entry REAL,
                sl REAL,
                tp REAL
            )"""
        )
        con.execute("CREATE INDEX IF NOT EXISTS idx_signals_ts ON signals(ts)")
        con.execute("CREATE INDEX IF NOT EXISTS idx_signals_symbol_mode ON signals(symbol, mode)")
        # --- MIGRATION: ensure signals has extended columns used by insert_signal() ---
        try:
            cols = {row[1] for row in con.execute("PRAGMA table_info(signals)")}
            def _add(col: str, ddl: str):
                if col not in cols:
                    con.execute(f"ALTER TABLE signals ADD COLUMN {ddl}")
            _add("source", "source TEXT")
            _add("side", "side TEXT")
            _add("features_json", "features_json TEXT")
            _add("reasons_json", "reasons_json TEXT")
            _add("horizon_days", "horizon_days INTEGER")
            _add("evaluated", "evaluated INTEGER DEFAULT 0")
            _add("model_prob", "model_prob REAL")
        except Exception:
            pass


        con.execute(
            """CREATE TABLE IF NOT EXISTS signal_outcomes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                signal_id INTEGER NOT NULL,
                result TEXT NOT NULL,
                r_mult REAL,
                notes TEXT
            )"""
        )
        con.execute("CREATE INDEX IF NOT EXISTS idx_signal_outcomes_signal_id ON signal_outcomes(signal_id)")

        con.execute(
            """CREATE TABLE IF NOT EXISTS signal_reviews (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                signal_id INTEGER NOT NULL,
                close REAL,
                return_pct REAL,
                mfe_pct REAL,
                mae_pct REAL,
                note TEXT
            )"""
        )
        con.execute("CREATE INDEX IF NOT EXISTS idx_signal_reviews_signal_id ON signal_reviews(signal_id)")
        con.execute("CREATE INDEX IF NOT EXISTS idx_signal_reviews_ts ON signal_reviews(ts)")

        con.execute(
            """CREATE TABLE IF NOT EXISTS paper_trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id TEXT NOT NULL,
                signal_id INTEGER NOT NULL,
                due_ts TEXT NOT NULL,
                notified INTEGER DEFAULT 0
            )"""
        )
        con.execute("CREATE INDEX IF NOT EXISTS idx_paper_trades_due ON paper_trades(due_ts)")
        con.execute("CREATE INDEX IF NOT EXISTS idx_paper_trades_chat ON paper_trades(chat_id)")

        con.commit()

        # Ensure extended columns
    try:
        ensure_signal_schema()
        ensure_paper_trades_schema()
    except Exception:
        pass

# Ensure signal_reviews has extended columns used by review snapshots
    try:
        ensure_signal_reviews_schema()
    except Exception:
        pass






# --- Signal reviews schema migrations (backwards compatible) ---
_REVIEW_COLS = {
    "high": "REAL",
    "low": "REAL",
    "tp_hit": "INTEGER",
    "sl_hit": "INTEGER",
    "hit": "TEXT",
    "hit_ts": "TEXT",
    "tp_progress": "REAL",
    "tp_gap_pct": "REAL",
    "tp_gap_class": "TEXT",
}

def ensure_signal_reviews_schema() -> None:
    """Add missing columns to signal_reviews table (SQLite/Postgres)."""
    if IS_POSTGRES:
        with _pg_connect() as con:
            with con.cursor() as cur:
                for col, typ in _REVIEW_COLS.items():
                    # Map sqlite-ish types to Postgres
                    ptyp = "DOUBLE PRECISION" if typ == "REAL" else ("INTEGER" if typ == "INTEGER" else "TEXT")
                    cur.execute(f"ALTER TABLE signal_reviews ADD COLUMN IF NOT EXISTS {col} {ptyp};")
            con.commit()
        return

    with sqlite3.connect(DB_PATH) as con:
        cur = con.execute("PRAGMA table_info(signal_reviews)")
        existing = {row[1] for row in cur.fetchall()}
        for col, typ in _REVIEW_COLS.items():
            if col not in existing:
                con.execute(f"ALTER TABLE signal_reviews ADD COLUMN {col} {typ}")
        con.commit()

# --- Signals schema migrations (backwards compatible) ---
_SIGNAL_COLS = {
    # column_name: sqlite_type
    "source": "TEXT",
    "side": "TEXT",
    "features_json": "TEXT",
    "reasons_json": "TEXT",
    "horizon_days": "INTEGER",
    "evaluated": "INTEGER",
    "eval_ts": "TEXT",
    "return_pct": "REAL",
    "mfe_pct": "REAL",
    "mae_pct": "REAL",
    "label": "INTEGER",
    "model_prob": "REAL",
}


# extra columns for paper trades to support premium monitoring (manual execution + TP/SL hit tracking)
_PAPER_TRADE_COLS = {
    # Snapshot fields (freeze signal at save time)
    "symbol": "TEXT",
    "mode": "TEXT",
    "side": "TEXT",            # buy|sell
    "signal_ts": "TEXT",       # UTC ISO
    "entry": "REAL",
    "sl": "REAL",
    "tp": "REAL",              # TP1
    "tp2": "REAL",             # TP2 (runner)
    "tp3": "REAL",             # TP3 (runner)

    # Trade management state
    "status": "TEXT",          # open|runner|tp2|tp3|sl|final
    "exec_price": "REAL",      # user executed price (optional)
    "qty": "REAL",             # executed quantity (optional)

    # Runner / trailing
    "trail_sl": "REAL",        # active trailing stop level
    "trail_mode": "TEXT",      # BE|TP1|ATR (for future)
    "tp1_hit": "INTEGER",
    "tp2_hit": "INTEGER",
    "tp3_hit": "INTEGER",

    # Monitoring / events
    "tp_hit": "INTEGER",       # 1 if any TP touched before finalize
    "sl_hit": "INTEGER",       # 1 if SL/tail stop touched before finalize
    "hit_ts": "TEXT",          # when TP/SL hit (UTC ISO)
    "hit_price": "REAL",       # the level hit (tp/sl/tp2/tp3/trail)
    "hit_kind": "TEXT",        # tp|tp2|tp3|sl|trail
    "last_check_ts": "TEXT",   # last monitoring scan (UTC ISO)
}

def ensure_signal_schema() -> None:
    """Add missing columns to signals table (SQLite/Postgres)."""
    if IS_POSTGRES:
        with _pg_connect() as con:
            with con.cursor() as cur:
                for col, typ in _SIGNAL_COLS.items():
                    cur.execute(f"ALTER TABLE signals ADD COLUMN IF NOT EXISTS {col} {('DOUBLE PRECISION' if typ=='REAL' else 'INTEGER' if typ=='INTEGER' else 'TEXT')};")
            con.commit()
        return

    with sqlite3.connect(DB_PATH) as con:
        cur = con.execute("PRAGMA table_info(signals)")
        existing = {r[1] for r in cur.fetchall()}
        for col, typ in _SIGNAL_COLS.items():
            if col not in existing:
                con.execute(f"ALTER TABLE signals ADD COLUMN {col} {typ}")
        con.commit()



def ensure_paper_trades_schema() -> None:
    """Add missing columns to paper_trades table (SQLite/Postgres)."""
    if IS_POSTGRES:
        with _pg_connect() as con:
            with con.cursor() as cur:
                for col, typ in _PAPER_TRADE_COLS.items():
                    pg_typ = "DOUBLE PRECISION" if typ == "REAL" else "INTEGER" if typ == "INTEGER" else "TEXT"
                    cur.execute(f"ALTER TABLE paper_trades ADD COLUMN IF NOT EXISTS {col} {pg_typ};")
            con.commit()
        return

    with sqlite3.connect(DB_PATH) as con:
        cur = con.execute("PRAGMA table_info(paper_trades)")
        existing = {r[1] for r in cur.fetchall()}
        for col, typ in _PAPER_TRADE_COLS.items():
            if col not in existing:
                con.execute(f"ALTER TABLE paper_trades ADD COLUMN {col} {typ}")
        con.commit()



def log_signal(
    ts: str,
    symbol: str,
    source: str,
    side: str,
    mode: str,
    strength: str,
    score: float,
    entry: float,
    sl: float | None,
    tp: float | None,
    features_json: str = "",
    reasons_json: str = "",
    horizon_days: int = 5,
    model_prob: float | None = None,
) -> int | None:
    """Persist a signal so we can evaluate and learn later.
    Returns inserted signal id when possible.
    """
    if IS_POSTGRES:
        with _pg_connect() as con:
            with con.cursor() as cur:
                cur.execute(
                    """INSERT INTO signals
                    (ts, symbol, mode, strength, score, entry, sl, tp, source, side, features_json, reasons_json, horizon_days, evaluated, model_prob)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,0,%s)
                    RETURNING id""",
                    (ts, symbol, mode, strength, float(score), float(entry),
                     float(sl) if sl is not None else None,
                     float(tp) if tp is not None else None,
                     source, side, features_json, reasons_json, int(horizon_days),
                     float(model_prob) if model_prob is not None else None),
                )
                row = cur.fetchone()
            con.commit()
        try:
            return int(row[0]) if row else None
        except Exception:
            return None

    with sqlite3.connect(DB_PATH) as con:
        cur = con.execute(
            """INSERT INTO signals
            (ts, symbol, mode, strength, score, entry, sl, tp, source, side, features_json, reasons_json, horizon_days, evaluated, model_prob)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,0,?)""",
            (ts, symbol, mode, strength, score, entry, sl, tp, source, side, features_json, reasons_json, horizon_days, model_prob),
        )
        con.commit()
        try:
            return int(cur.lastrowid)
        except Exception:
            return None


def pending_signals_for_eval(limit: int = 200) -> List[Dict[str, Any]]:
    """Signals not evaluated yet."""
    if IS_POSTGRES:
        with _pg_connect() as con:
            with con.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    """SELECT * FROM signals
                    WHERE COALESCE(evaluated,0)=0
                    ORDER BY id ASC
                    LIMIT %s""",
                    (limit,),
                )
                return cur.fetchall()

    with sqlite3.connect(DB_PATH) as con:
        con.row_factory = sqlite3.Row
        rows = con.execute(
            """SELECT * FROM signals WHERE COALESCE(evaluated,0)=0 ORDER BY id ASC LIMIT ?""",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]


def mark_signal_evaluated(
    signal_id: int,
    eval_ts: str,
    return_pct: float,
    mfe_pct: float,
    mae_pct: float,
    label: int,
    model_prob: float | None = None,
) -> None:
    if IS_POSTGRES:
        with _pg_connect() as con:
            with con.cursor() as cur:
                cur.execute(
                    """UPDATE signals SET evaluated=1, eval_ts=%s, return_pct=%s, mfe_pct=%s, mae_pct=%s, label=%s, model_prob=COALESCE(%s, model_prob)
                    WHERE id=%s""",
                    (eval_ts, float(return_pct), float(mfe_pct), float(mae_pct), int(label), float(model_prob) if model_prob is not None else None, int(signal_id)),
                )
            con.commit()
        return

    with sqlite3.connect(DB_PATH) as con:
        con.execute(
            """UPDATE signals SET evaluated=1, eval_ts=?, return_pct=?, mfe_pct=?, mae_pct=?, label=?, model_prob=COALESCE(?, model_prob)
            WHERE id=?""",
            (eval_ts, return_pct, mfe_pct, mae_pct, label, model_prob, signal_id),
        )
        con.commit()


def last_signals(limit: int = 50) -> List[Dict[str, Any]]:
    if IS_POSTGRES:
        with _pg_connect() as con:
            with con.cursor(row_factory=dict_row) as cur:
                cur.execute("SELECT * FROM signals ORDER BY id DESC LIMIT %s", (limit,))
                return cur.fetchall()

    with sqlite3.connect(DB_PATH) as con:
        con.row_factory = sqlite3.Row
        rows = con.execute("SELECT * FROM signals ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]

def log_order(symbol: str, side: str, qty: float, order_type: str, payload: str,
              broker_order_id: Optional[str], status: str, message: str) -> None:
    ts = datetime.utcnow().isoformat()

    if IS_POSTGRES:
        with _pg_connect() as con:
            with con.cursor() as cur:
                cur.execute(
                    "INSERT INTO orders (ts, symbol, side, qty, order_type, payload, broker_order_id, status, message) "
                    "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                    (ts, symbol, side, float(qty), order_type, payload, broker_order_id or "", status, message),
                )
            con.commit()
        return

    with sqlite3.connect(DB_PATH) as con:
        con.execute(
            "INSERT INTO orders (ts, symbol, side, qty, order_type, payload, broker_order_id, status, message) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (ts, symbol, side, qty, order_type, payload, broker_order_id or "", status, message),
        )
        con.commit()


def last_orders(limit: int = 20) -> List[Dict[str, Any]]:
    if IS_POSTGRES:
        with _pg_connect() as con:
            with con.cursor(row_factory=dict_row) as cur:
                cur.execute("SELECT * FROM orders ORDER BY id DESC LIMIT %s", (limit,))
                return cur.fetchall()

    with sqlite3.connect(DB_PATH) as con:
        con.row_factory = sqlite3.Row
        rows = con.execute("SELECT * FROM orders ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]


def log_scan(ts: str, universe_size: int, top_symbols: str, payload: str = "") -> None:
    if IS_POSTGRES:
        with _pg_connect() as con:
            with con.cursor() as cur:
                cur.execute(
                    "INSERT INTO scans (ts, universe_size, top_symbols, payload) VALUES (%s,%s,%s,%s)",
                    (ts, int(universe_size), top_symbols, payload),
                )
            con.commit()
        return

    with sqlite3.connect(DB_PATH) as con:
        con.execute(
            "INSERT INTO scans (ts, universe_size, top_symbols, payload) VALUES (?, ?, ?, ?)",
            (ts, universe_size, top_symbols, payload),
        )
        con.commit()


def last_scans(limit: int = 50) -> List[Dict[str, Any]]:
    if IS_POSTGRES:
        with _pg_connect() as con:
            with con.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    "SELECT ts, universe_size, top_symbols, payload FROM scans ORDER BY id DESC LIMIT %s",
                    (limit,),
                )
                return cur.fetchall()

    with sqlite3.connect(DB_PATH) as con:
        cur = con.execute(
            "SELECT ts, universe_size, top_symbols, payload FROM scans ORDER BY id DESC LIMIT ?",
            (limit,),
        )
        rows = cur.fetchall()

    out: List[Dict[str, Any]] = []
    for ts, usize, syms, payload in rows:
        out.append({"ts": ts, "universe_size": usize, "top_symbols": syms, "payload": payload})
    return out


def _env_defaults() -> Dict[str, str]:
    """Defaults written once (if missing) at startup."""
    try:
        import core.config
        return {
            "TOP_N": str(config.TOP_N),
            "AUTO_TRADE": str(int(config.AUTO_TRADE)),
            "MAX_DAILY_TRADES": str(config.MAX_DAILY_TRADES),
            "RISK_PER_TRADE_PCT": str(config.RISK_PER_TRADE_PCT),
            "TP_R_MULT": str(config.TP_R_MULT),
            "SL_ATR_MULT": str(config.SL_ATR_MULT),
            # Bot UI / manual-trading settings
            "AUTO_NOTIFY": "1",
            "NOTIFY_ROUTE": str(getattr(config, "NOTIFY_ROUTE_DEFAULT", "dm")),
            "NOTIFY_SILENT": str(int(getattr(config, "NOTIFY_SILENT_DEFAULT", True))),
            "SIGNAL_EVAL_DAYS": str(getattr(config, "SIGNAL_EVAL_DAYS", 5)),
            "REVIEW_LOOKBACK_DAYS": "2",
            "WEEKLY_REPORT_DAYS": "7",
            "ML_ENABLED": str(int(getattr(config, "ML_ENABLED", True))),
            "ML_WEIGHTS": "",
            "MAX_SEND": "10",
            "MIN_SEND": "7",
            "DEDUP_HOURS": "6",
            "ALLOW_RESEND_IF_STRONGER": "1",
# Multi-timeframe confirmation
"REQUIRE_DAILY_OK": "1",
"REQUIRE_WEEKLY_OK": "1",
"REQUIRE_MONTHLY_OK": "0",

# Intraday confirmation before execution (VWAP/EMA on 5m)
"INTRADAY_CONFIRM": "1",
            "PLAN_MODE": "daily",  # daily|weekly|monthly|daily_weekly|weekly_monthly
            "ENTRY_MODE": "auto",  # auto|market|limit
            "CAPITAL_USD": "800",
            "RISK_APLUS_PCT": "1.5",
            "RISK_A_PCT": "1.0",
            "RISK_B_PCT": "0.5",
            # مخاطرة ذكية للحسابات الصغيرة (D1)
            "RISK_MIN_PCT": "4",
            "RISK_MAX_PCT": "8",
            # إذا تجاوز احتمال الخسارة هذا الرقم يتم منع الصفقة
            "LOSS_PROB_BLOCK": "0.50",
            # إرسال رسائل رفض بسبب الأخبار؟
            "NEWS_FILTER_SEND_REJECTS": "0",
            "SCAN_INTERVAL_MIN": "20",
            "SCHED_ENABLED": "1",
            "POSITION_PCT": "0.20",
            "SL_PCT": "3",
            "TP_PCT": "5",
            "TP_PCT_STRONG": "7",
            "TP_PCT_VSTRONG": "10",
            "WINDOW_START": "17:30",
            "WINDOW_END": "00:00",

            # AI prediction (direction) settings
            # D1 | M5 | M5+
            "PREDICT_FRAME": "D1",
            # Off by default for speed/cost; enable from Telegram settings
            "AI_PREDICT_ENABLED": "0",
            # Only run AI on top N candidates
            "AI_PREDICT_TOPN": "5",
        }
    except Exception:
        return {}


def ensure_default_settings() -> None:
    defaults = _env_defaults()
    if not defaults:
        return

    if IS_POSTGRES:
        with _pg_connect() as con:
            with con.cursor() as cur:
                for k, v in defaults.items():
                    cur.execute(
                        "INSERT INTO settings(key,value) VALUES(%s,%s) ON CONFLICT(key) DO NOTHING",
                        (k, v),
                    )
            con.commit()
        return

    with sqlite3.connect(DB_PATH) as con:
        for k, v in defaults.items():
            con.execute("INSERT OR IGNORE INTO settings(key,value) VALUES(?,?)", (k, v))
        con.commit()


def get_setting(key: str, default: Optional[str] = None) -> Optional[str]:
    if IS_POSTGRES:
        with _pg_connect() as con:
            with con.cursor() as cur:
                cur.execute("SELECT value FROM settings WHERE key=%s", (key,))
                row = cur.fetchone()
                return row[0] if row else default

    with sqlite3.connect(DB_PATH) as con:
        cur = con.execute("SELECT value FROM settings WHERE key=?", (key,))
        row = cur.fetchone()
        return row[0] if row else default


def set_setting(key: str, value: str) -> None:
    if IS_POSTGRES:
        with _pg_connect() as con:
            with con.cursor() as cur:
                cur.execute(
                    "INSERT INTO settings(key,value) VALUES(%s,%s) ON CONFLICT(key) DO UPDATE SET value=EXCLUDED.value",
                    (key, value),
                )
            con.commit()
        return

    with sqlite3.connect(DB_PATH) as con:
        con.execute(
            "INSERT INTO settings(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
        )
        con.commit()


def get_all_settings() -> Dict[str, str]:
    if IS_POSTGRES:
        with _pg_connect() as con:
            with con.cursor() as cur:
                cur.execute("SELECT key,value FROM settings")
                rows = cur.fetchall()
                return {k: v for (k, v) in rows}

    with sqlite3.connect(DB_PATH) as con:
        cur = con.execute("SELECT key,value FROM settings")
        return {k: v for k, v in cur.fetchall()}


def parse_bool(v: Any, default: bool = False) -> bool:
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    return s in ("1", "true", "yes", "y", "on")


def parse_int(v: Any, default: int = 0) -> int:
    try:
        return int(str(v).strip())
    except Exception:
        return default


def parse_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(str(v).strip())
    except Exception:
        return default


# ===== Signal logging for "send only new" notifications =====

def last_signal(symbol: str, mode: str) -> Optional[Dict[str, Any]]:
    if IS_POSTGRES:
        with _pg_connect() as con:
            with con.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    "SELECT ts, symbol, mode, strength, score, entry, sl, tp FROM signals "
                    "WHERE symbol=%s AND mode=%s ORDER BY id DESC LIMIT 1",
                    (symbol, mode),
                )
                row = cur.fetchone()
                return row if row else None

    with sqlite3.connect(DB_PATH) as con:
        con.row_factory = sqlite3.Row
        row = con.execute(
            "SELECT ts, symbol, mode, strength, score, entry, sl, tp FROM signals WHERE symbol=? AND mode=? ORDER BY id DESC LIMIT 1",
            (symbol, mode),
        ).fetchone()
        return dict(row) if row else None


def signals_since(ts_iso: str, mode: Optional[str] = None) -> List[Dict[str, Any]]:
    if IS_POSTGRES:
        with _pg_connect() as con:
            with con.cursor(row_factory=dict_row) as cur:
                if mode:
                    cur.execute(
                        "SELECT ts, symbol, mode, strength, score, entry, sl, tp FROM signals "
                        "WHERE ts>=%s AND mode=%s ORDER BY id DESC",
                        (ts_iso, mode),
                    )
                else:
                    cur.execute(
                        "SELECT ts, symbol, mode, strength, score, entry, sl, tp FROM signals "
                        "WHERE ts>=%s ORDER BY id DESC",
                        (ts_iso,),
                    )
                return cur.fetchall()

    with sqlite3.connect(DB_PATH) as con:
        con.row_factory = sqlite3.Row
        if mode:
            rows = con.execute(
                "SELECT ts, symbol, mode, strength, score, entry, sl, tp FROM signals WHERE ts>=? AND mode=? ORDER BY id DESC",
                (ts_iso, mode),
            ).fetchall()
        else:
            rows = con.execute(
                "SELECT ts, symbol, mode, strength, score, entry, sl, tp FROM signals WHERE ts>=? ORDER BY id DESC",
                (ts_iso,),
            ).fetchall()
        return [dict(r) for r in rows]


# ===== Per-chat UI state (for button-driven custom input) =====
def set_user_state(chat_id: str, key: str, value: str) -> None:
    if IS_POSTGRES:
        with _pg_connect() as con:
            with con.cursor() as cur:
                cur.execute(
                    "INSERT INTO user_state(chat_id, key, value) VALUES (%s,%s,%s) "
                    "ON CONFLICT(chat_id,key) DO UPDATE SET value=EXCLUDED.value",
                    (str(chat_id), str(key), str(value)),
                )
            con.commit()
        return

    with sqlite3.connect(DB_PATH) as con:
        con.execute(
            "INSERT INTO user_state(chat_id, key, value) VALUES(?,?,?) ON CONFLICT(chat_id,key) DO UPDATE SET value=excluded.value",
            (str(chat_id), str(key), str(value)),
        )
        con.commit()


def get_user_state(chat_id: str, key: str, default: str = "") -> str:
    if IS_POSTGRES:
        with _pg_connect() as con:
            with con.cursor() as cur:
                cur.execute(
                    "SELECT value FROM user_state WHERE chat_id=%s AND key=%s",
                    (str(chat_id), str(key)),
                )
                row = cur.fetchone()
                return str(row[0]) if row else default

    with sqlite3.connect(DB_PATH) as con:
        cur = con.execute("SELECT value FROM user_state WHERE chat_id=? AND key=?", (str(chat_id), str(key)))
        row = cur.fetchone()
        return str(row[0]) if row else default


def clear_user_state(chat_id: str, key: str) -> None:
    if IS_POSTGRES:
        with _pg_connect() as con:
            with con.cursor() as cur:
                cur.execute("DELETE FROM user_state WHERE chat_id=%s AND key=%s", (str(chat_id), str(key)))
            con.commit()
        return

    with sqlite3.connect(DB_PATH) as con:
        con.execute("DELETE FROM user_state WHERE chat_id=? AND key=?", (str(chat_id), str(key)))
        con.commit()


# ===== Watchlist (manual symbols) =====
def get_watchlist() -> List[str]:
    if IS_POSTGRES:
        with _pg_connect() as con:
            with con.cursor() as cur:
                cur.execute("SELECT symbol FROM watchlist ORDER BY symbol")
                return [r[0] for r in cur.fetchall()]
    with sqlite3.connect(DB_PATH) as con:
        cur = con.execute("SELECT symbol FROM watchlist ORDER BY symbol")
        return [r[0] for r in cur.fetchall()]

def add_watchlist(symbol: str) -> None:
    sym = (symbol or '').strip().upper()
    if not sym:
        return
    ts = datetime.utcnow().isoformat()
    if IS_POSTGRES:
        with _pg_connect() as con:
            with con.cursor() as cur:
                cur.execute(
                    "INSERT INTO watchlist(symbol, added_ts) VALUES(%s,%s) ON CONFLICT(symbol) DO NOTHING",
                    (sym, ts),
                )
            con.commit()
        return
    with sqlite3.connect(DB_PATH) as con:
        con.execute("INSERT OR IGNORE INTO watchlist(symbol,added_ts) VALUES(?,?)", (sym, ts))
        con.commit()

def remove_watchlist(symbol: str) -> None:
    sym = (symbol or '').strip().upper()
    if not sym:
        return
    if IS_POSTGRES:
        with _pg_connect() as con:
            with con.cursor() as cur:
                cur.execute("DELETE FROM watchlist WHERE symbol=%s", (sym,))
            con.commit()
        return
    with sqlite3.connect(DB_PATH) as con:
        con.execute("DELETE FROM watchlist WHERE symbol=?", (sym,))
        con.commit()

def record_outcome(signal_id: int, result: str, r_mult: float | None = None, notes: str = "") -> None:
    """Record manual outcome for a signal (WIN/LOSS/SKIP) with optional R multiple."""
    ts = datetime.utcnow().isoformat()
    result_u = (result or "").strip().upper()
    if result_u not in ("WIN", "LOSS", "SKIP"):
        result_u = "SKIP"

    if IS_POSTGRES:
        with _pg_connect() as con:
            with con.cursor() as cur:
                cur.execute(
                    "INSERT INTO signal_outcomes (ts,signal_id,result,r_mult,notes) VALUES (%s,%s,%s,%s,%s);",
                    (ts, int(signal_id), result_u, float(r_mult) if r_mult is not None else None, notes or ""),
                )
            con.commit()
        return

    with sqlite3.connect(DB_PATH) as con:
        con.execute(
            "INSERT INTO signal_outcomes (ts,signal_id,result,r_mult,notes) VALUES (?,?,?,?,?)",
            (ts, int(signal_id), result_u, float(r_mult) if r_mult is not None else None, notes or ""),
        )
        con.commit()


def get_recent_stats(limit: int = 200) -> Dict[str, Any]:
    """Basic stats for manual outcomes."""
    if IS_POSTGRES:
        with _pg_connect() as con:
            with con.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    "SELECT result, r_mult FROM signal_outcomes ORDER BY id DESC LIMIT %s;",
                    (int(limit),),
                )
                rows = cur.fetchall() or []
    else:
        with sqlite3.connect(DB_PATH) as con:
            con.row_factory = sqlite3.Row
            rows = [dict(r) for r in con.execute(
                "SELECT result, r_mult FROM signal_outcomes ORDER BY id DESC LIMIT ?;",
                (int(limit),),
            ).fetchall() or []]

    total = len(rows)
    wins = sum(1 for r in rows if (r.get("result") or "").upper() == "WIN")
    losses = sum(1 for r in rows if (r.get("result") or "").upper() == "LOSS")
    skips = total - wins - losses
    rs = [float(r.get("r_mult")) for r in rows if r.get("r_mult") is not None]
    avg_r = (sum(rs) / len(rs)) if rs else None
    winrate = (wins / (wins + losses)) if (wins + losses) else 0.0
    return {"total": total, "wins": wins, "losses": losses, "skips": skips, "winrate": winrate, "avg_r": avg_r}


def log_signal_review(
    ts: str,
    signal_id: int,
    close: float,
    return_pct: float,
    mfe_pct: float,
    mae_pct: float,
    note: str = "",
    high: float | None = None,
    low: float | None = None,
    tp_hit: bool | None = None,
    sl_hit: bool | None = None,
    hit: str = "",
    hit_ts: str = "",
    tp_progress: float | None = None,
    tp_gap_pct: float | None = None,
    tp_gap_class: str = "",
) -> None:
    """Store a periodic review snapshot for a signal (e.g. daily close performance)."""
    try:
        ensure_signal_reviews_schema()
    except Exception:
        pass
    if IS_POSTGRES:
        with _pg_connect() as con:
            with con.cursor() as cur:
                cur.execute(
                    """INSERT INTO signal_reviews (ts, signal_id, close, return_pct, mfe_pct, mae_pct, note, high, low, tp_hit, sl_hit, hit, hit_ts, tp_progress, tp_gap_pct, tp_gap_class)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                    (ts, int(signal_id), float(close), float(return_pct), float(mfe_pct), float(mae_pct), note or "", float(high) if high is not None else None, float(low) if low is not None else None, int(bool(tp_hit)) if tp_hit is not None else None, int(bool(sl_hit)) if sl_hit is not None else None, hit or "", hit_ts or "", float(tp_progress) if tp_progress is not None else None, float(tp_gap_pct) if tp_gap_pct is not None else None, (tp_gap_class or "")),
                )
            con.commit()
        return

    with sqlite3.connect(DB_PATH) as con:
        con.execute(
            """INSERT INTO signal_reviews (ts, signal_id, close, return_pct, mfe_pct, mae_pct, note, high, low, tp_hit, sl_hit, hit, hit_ts, tp_progress, tp_gap_pct, tp_gap_class)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (ts, int(signal_id), float(close), float(return_pct), float(mfe_pct), float(mae_pct), note or "", float(high) if high is not None else None, float(low) if low is not None else None, int(bool(tp_hit)) if tp_hit is not None else None, int(bool(sl_hit)) if sl_hit is not None else None, hit or "", hit_ts or "", float(tp_progress) if tp_progress is not None else None, float(tp_gap_pct) if tp_gap_pct is not None else None, (tp_gap_class or "")),
        )
        con.commit()


def last_signal_reviews(limit: int = 50) -> List[Dict[str, Any]]:
    if IS_POSTGRES:
        with _pg_connect() as con:
            with con.cursor(row_factory=dict_row) as cur:
                cur.execute("""SELECT * FROM signal_reviews ORDER BY id DESC LIMIT %s""", (limit,))
                return cur.fetchall()
    with sqlite3.connect(DB_PATH) as con:
        con.row_factory = sqlite3.Row
        rows = con.execute("SELECT * FROM signal_reviews ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]


def signal_reviews_since(ts_iso: str) -> List[Dict[str, Any]]:
    """Return signal_reviews rows since ts_iso (inclusive)."""
    if IS_POSTGRES:
        with _pg_connect() as con:
            with con.cursor(row_factory=dict_row) as cur:
                cur.execute("SELECT * FROM signal_reviews WHERE ts >= %s ORDER BY ts ASC", (ts_iso,))
                return cur.fetchall()
    with sqlite3.connect(DB_PATH) as con:
        con.row_factory = sqlite3.Row
        cur = con.execute("SELECT * FROM signal_reviews WHERE ts >= ? ORDER BY ts ASC", (ts_iso,))
        return [dict(r) for r in cur.fetchall()]

def signals_since(ts_iso: str) -> List[Dict[str, Any]]:
    """Return signals rows since ts_iso (inclusive)."""
    if IS_POSTGRES:
        with _pg_connect() as con:
            with con.cursor(row_factory=dict_row) as cur:
                cur.execute("SELECT * FROM signals WHERE ts >= %s ORDER BY ts ASC", (ts_iso,))
                return cur.fetchall()
    with sqlite3.connect(DB_PATH) as con:
        con.row_factory = sqlite3.Row
        cur = con.execute("SELECT * FROM signals WHERE ts >= ? ORDER BY ts ASC", (ts_iso,))
        return [dict(r) for r in cur.fetchall()]



def latest_signal_reviews_since(days: int = 7) -> List[Dict[str, Any]]:
    """Return latest review row per signal within last `days` days.
    Safe on fresh deploys (returns [] if tables are missing).
    """
    try:
        cutoff = (datetime.utcnow() - timedelta(days=int(days))).isoformat()
        if IS_POSTGRES:
            with _pg_connect() as con:
                with con.cursor(row_factory=dict_row) as cur:
                    cur.execute(
                        """
                        SELECT r.*, s.symbol, s.mode, s.score, s.entry, s.tp, s.sl
                        FROM signal_reviews r
                        JOIN (
                            SELECT signal_id, MAX(ts) AS max_ts
                            FROM signal_reviews
                            WHERE ts >= %s
                            GROUP BY signal_id
                        ) x ON x.signal_id = r.signal_id AND x.max_ts = r.ts
                        JOIN signals s ON s.id = r.signal_id
                        ORDER BY r.ts DESC
                        """,
                        (cutoff,),
                    )
                    return cur.fetchall()

        with sqlite3.connect(DB_PATH) as con:
            con.row_factory = sqlite3.Row
            rows = con.execute(
                """
                SELECT r.*, s.symbol, s.mode, s.score, s.entry, s.tp, s.sl
                FROM signal_reviews r
                JOIN (
                    SELECT signal_id, MAX(ts) AS max_ts
                    FROM signal_reviews
                    WHERE ts >= ?
                    GROUP BY signal_id
                ) x ON x.signal_id = r.signal_id AND x.max_ts = r.ts
                JOIN signals s ON s.id = r.signal_id
                ORDER BY r.ts DESC
                """,
                (cutoff,),
            ).fetchall()
            return [dict(r) for r in rows]
    except Exception:
        return []



# --- Paper trades (manual simulation tracking) ---
def add_paper_trade(chat_id: str, signal_id: int, due_ts: str) -> None:
    """Save a signal to a chat for later monitoring/review (24h).

    We freeze (snapshot) key signal fields into paper_trades so later reviews/monitoring
    do not depend on signals table changing.
    """
    ensure_paper_trades_schema()

    # Load signal snapshot
    sig = None
    if IS_POSTGRES:
        with _pg_connect() as con:
            with con.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    """SELECT id, ts, symbol, mode, side, entry, sl, tp, score, strength, features_json, reasons_json
                       FROM signals WHERE id=%s""",
                    (int(signal_id),),
                )
                sig = cur.fetchone()
    else:
        with sqlite3.connect(DB_PATH) as con:
            con.row_factory = sqlite3.Row
            row = con.execute(
                "SELECT id, ts, symbol, mode, side, entry, sl, tp, score, strength FROM signals WHERE id=?",
                (int(signal_id),),
            ).fetchone()
            sig = dict(row) if row else None

    if not sig:
        # Fallback: still insert minimal row (legacy behavior)
        if IS_POSTGRES:
            with _pg_connect() as con:
                with con.cursor() as cur:
                    cur.execute(
                        "INSERT INTO paper_trades (chat_id, signal_id, due_ts, notified) VALUES (%s,%s,%s,0)",
                        (str(chat_id), int(signal_id), due_ts),
                    )
                con.commit()
            return
        with sqlite3.connect(DB_PATH) as con:
            con.execute(
                "INSERT INTO paper_trades (chat_id, signal_id, due_ts, notified) VALUES (?,?,?,0)",
                (str(chat_id), int(signal_id), due_ts),
            )
            con.commit()
        return

    symbol = (sig.get("symbol") or "").upper().strip()
    mode = (sig.get("mode") or "").upper().strip()
    side = (sig.get("side") or "buy").lower().strip()
    signal_ts = str(sig.get("ts") or "")
    entry = float(sig.get("entry") or 0.0)
    sl = float(sig.get("sl") or 0.0)
    tp1 = float(sig.get("tp") or 0.0)

        # إذا كانت خطة الصفقة تحتوي TP2/TP3 مخصصة (من features_json)، استخدمها بدل 4R/8R الافتراضية
    plan_tp2 = 0.0
    plan_tp3 = 0.0
    try:
        fj = str(sig.get("features_json") or "")
        j = json.loads(fj) if fj.strip().startswith("{") else {}
        plan = j.get("plan") if isinstance(j, dict) else None
        if isinstance(plan, dict):
            if plan.get("tp2") is not None:
                plan_tp2 = float(plan.get("tp2") or 0.0)
            if plan.get("tp3") is not None:
                plan_tp3 = float(plan.get("tp3") or 0.0)
    except Exception:
        pass

    # Multi-stage TP: if SL exists, compute TP2/TP3 as 4R/8R runners.
    tp2 = 0.0
    tp3 = 0.0
    try:
        risk = abs(entry - sl) if sl > 0 else 0.0
        if risk > 0:
            if side == "sell":
                tp2 = entry - 4.0 * risk
                tp3 = entry - 8.0 * risk
            else:
                tp2 = entry + 4.0 * risk
                tp3 = entry + 8.0 * risk
    except Exception:
        pass

    # Override with plan-defined targets if available
    try:
        if plan_tp2 and float(plan_tp2) > 0:
            tp2 = float(plan_tp2)
        if plan_tp3 and float(plan_tp3) > 0:
            tp3 = float(plan_tp3)
    except Exception:
        pass

    if IS_POSTGRES:
        with _pg_connect() as con:
            with con.cursor() as cur:
                cur.execute(
                    """INSERT INTO paper_trades
                    (chat_id, signal_id, due_ts, notified, symbol, mode, side, signal_ts, entry, sl, tp, tp2, tp3,
                     status, trail_sl, trail_mode, tp1_hit, tp2_hit, tp3_hit, tp_hit, sl_hit)
                    VALUES (%s,%s,%s,0,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,0,0,0,0,0)""",
                    (str(chat_id), int(signal_id), due_ts,
                     symbol, mode, side, signal_ts,
                     float(entry), float(sl) if sl > 0 else None,
                     float(tp1) if tp1 > 0 else None,
                     float(tp2) if tp2 != 0 else None,
                     float(tp3) if tp3 != 0 else None,
                     "open",
                     None, "BE"),
                )
            con.commit()
        return

    with sqlite3.connect(DB_PATH) as con:
        con.execute(
            """INSERT INTO paper_trades
            (chat_id, signal_id, due_ts, notified, symbol, mode, side, signal_ts, entry, sl, tp, tp2, tp3,
             status, trail_sl, trail_mode, tp1_hit, tp2_hit, tp3_hit, tp_hit, sl_hit)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,0,0,0,0,0)""",
            (str(chat_id), int(signal_id), due_ts, 0,
             symbol, mode, side, signal_ts, float(entry),
             float(sl) if sl > 0 else None,
             float(tp1) if tp1 > 0 else None,
             float(tp2) if tp2 != 0 else None,
             float(tp3) if tp3 != 0 else None,
             "open", None, "BE"),
        )
        con.commit()

def due_paper_trades(limit: int = 200) -> List[Dict[str, Any]]:
    """Paper trades that are due for 24h finalize and not yet notified."""
    if IS_POSTGRES:
        with _pg_connect() as con:
            with con.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    """SELECT
                          p.*,
                          COALESCE(p.signal_ts, s.ts) AS signal_ts,
                          COALESCE(p.symbol, s.symbol) AS symbol,
                          COALESCE(p.mode, s.mode) AS mode,
                          COALESCE(p.side, s.side) AS side,
                          COALESCE(p.entry, s.entry) AS entry,
                          COALESCE(p.sl, s.sl) AS sl,
                          COALESCE(p.tp, s.tp) AS tp,
                          p.tp2, p.tp3, p.trail_sl, p.trail_mode, p.tp1_hit, p.tp2_hit, p.tp3_hit,
                          s.score, s.strength
                       FROM paper_trades p
                       LEFT JOIN signals s ON s.id = p.signal_id
                       WHERE COALESCE(p.notified,0)=0
                         AND (p.due_ts::timestamptz) <= now()
                       ORDER BY p.due_ts ASC
                       LIMIT %s""",
                    (int(limit),),
                )
                return cur.fetchall()

    now_ts = datetime.utcnow().isoformat()
    with sqlite3.connect(DB_PATH) as con:
        con.row_factory = sqlite3.Row
        rows = con.execute(
            """SELECT
                  p.*,
                  COALESCE(p.signal_ts, s.ts) AS signal_ts,
                  COALESCE(p.symbol, s.symbol) AS symbol,
                  COALESCE(p.mode, s.mode) AS mode,
                  COALESCE(p.side, s.side) AS side,
                  COALESCE(p.entry, s.entry) AS entry,
                  COALESCE(p.sl, s.sl) AS sl,
                  COALESCE(p.tp, s.tp) AS tp,
                  p.tp2, p.tp3, p.trail_sl, p.trail_mode, p.tp1_hit, p.tp2_hit, p.tp3_hit,
                  s.score, s.strength
               FROM paper_trades p
               LEFT JOIN signals s ON s.id = p.signal_id
               WHERE COALESCE(p.notified,0)=0
                 AND p.due_ts <= ?
               ORDER BY p.due_ts ASC
               LIMIT ?""",
            (now_ts, int(limit)),
        ).fetchall()
        return [dict(r) for r in rows]

def open_paper_trades_for_monitor(limit: int = 500) -> List[Dict[str, Any]]:
    """Paper trades that are still active (open/runner) and need TP/SL monitoring."""
    if IS_POSTGRES:
        with _pg_connect() as con:
            with con.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    """SELECT
                          p.*,
                          COALESCE(p.signal_ts, s.ts) AS signal_ts,
                          COALESCE(p.symbol, s.symbol) AS symbol,
                          COALESCE(p.mode, s.mode) AS mode,
                          COALESCE(p.side, s.side) AS side,
                          COALESCE(p.entry, s.entry) AS entry,
                          COALESCE(p.sl, s.sl) AS sl,
                          COALESCE(p.tp, s.tp) AS tp,
                          p.tp2, p.tp3, p.trail_sl, p.trail_mode, p.tp1_hit, p.tp2_hit, p.tp3_hit,
                          s.score, s.strength
                       FROM paper_trades p
                       LEFT JOIN signals s ON s.id = p.signal_id
                       WHERE COALESCE(p.notified,0)=0
                         AND COALESCE(p.status,'open') IN ('open','runner','tp2')
                       ORDER BY p.due_ts ASC
                       LIMIT %s""",
                    (int(limit),),
                )
                return cur.fetchall()

    with sqlite3.connect(DB_PATH) as con:
        con.row_factory = sqlite3.Row
        rows = con.execute(
            """SELECT
                  p.*,
                  COALESCE(p.signal_ts, s.ts) AS signal_ts,
                  COALESCE(p.symbol, s.symbol) AS symbol,
                  COALESCE(p.mode, s.mode) AS mode,
                  COALESCE(p.side, s.side) AS side,
                  COALESCE(p.entry, s.entry) AS entry,
                  COALESCE(p.sl, s.sl) AS sl,
                  COALESCE(p.tp, s.tp) AS tp,
                  p.tp2, p.tp3, p.trail_sl, p.trail_mode, p.tp1_hit, p.tp2_hit, p.tp3_hit,
                  s.score, s.strength
               FROM paper_trades p
               LEFT JOIN signals s ON s.id = p.signal_id
               WHERE COALESCE(p.notified,0)=0
                 AND COALESCE(p.status,'open') IN ('open','runner','tp2')
               ORDER BY p.due_ts ASC
               LIMIT ?""",
            (int(limit),),
        ).fetchall()
        return [dict(r) for r in rows]

def update_paper_trade_monitor_state(
    paper_id: int,
    *,
    status: Optional[str] = None,
    tp_hit: Optional[int] = None,
    sl_hit: Optional[int] = None,
    tp1_hit: Optional[int] = None,
    tp2_hit: Optional[int] = None,
    tp3_hit: Optional[int] = None,
    trail_sl: Optional[float] = None,
    trail_mode: Optional[str] = None,
    hit_kind: Optional[str] = None,
    hit_ts: Optional[str] = None,
    hit_price: Optional[float] = None,
    last_check_ts: Optional[str] = None,
) -> None:
    """Update monitoring fields for a paper trade (idempotent)."""
    fields: Dict[str, Any] = {}
    if status is not None:
        fields["status"] = status
    if tp_hit is not None:
        fields["tp_hit"] = int(tp_hit)
    if sl_hit is not None:
        fields["sl_hit"] = int(sl_hit)
    if tp1_hit is not None:
        fields["tp1_hit"] = int(tp1_hit)
    if tp2_hit is not None:
        fields["tp2_hit"] = int(tp2_hit)
    if tp3_hit is not None:
        fields["tp3_hit"] = int(tp3_hit)
    if trail_sl is not None:
        fields["trail_sl"] = float(trail_sl)
    if trail_mode is not None:
        fields["trail_mode"] = str(trail_mode)
    if hit_kind is not None:
        fields["hit_kind"] = str(hit_kind)
    if hit_ts is not None:
        fields["hit_ts"] = hit_ts
    if hit_price is not None:
        fields["hit_price"] = float(hit_price)
    if last_check_ts is not None:
        fields["last_check_ts"] = last_check_ts

    if not fields:
        return

    if IS_POSTGRES:
        cols = ", ".join([f"{k}=%s" for k in fields.keys()])
        vals = list(fields.values()) + [int(paper_id)]
        with _pg_connect() as con:
            with con.cursor() as cur:
                cur.execute(f"UPDATE paper_trades SET {cols} WHERE id=%s", vals)
            con.commit()
        return

    with sqlite3.connect(DB_PATH) as con:
        cols = ", ".join([f"{k}=?" for k in fields.keys()])
        vals = list(fields.values()) + [int(paper_id)]
        con.execute(f"UPDATE paper_trades SET {cols} WHERE id=?", vals)
        con.commit()

def mark_paper_trade_notified(paper_id: int) -> None:
    if IS_POSTGRES:
        with _pg_connect() as con:
            with con.cursor() as cur:
                cur.execute("UPDATE paper_trades SET notified=1 WHERE id=%s", (int(paper_id),))
            con.commit()
        return

    with sqlite3.connect(DB_PATH) as con:
        con.execute("UPDATE paper_trades SET notified=1 WHERE id=?", (int(paper_id),))
        con.commit()


def list_paper_trades_for_chat(chat_id: str, lookback_days: int = 7, limit: int = 80) -> List[Dict[str, Any]]:
    """List saved paper trades for a chat, joined with the originating signal.

    Note: We keep the underlying signal rows (for learning/history) and only manage visibility via paper_trades.
    """
    cutoff = (datetime.utcnow() - timedelta(days=int(lookback_days))).isoformat()
    if IS_POSTGRES:
        with _pg_connect() as con:
            with con.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    """
                    SELECT
                        pt.id AS paper_id,
                        pt.chat_id,
                        pt.signal_id,
                        pt.due_ts,
                        pt.notified,
                        s.ts AS signal_ts,
                        s.symbol,
                        s.mode,
                        s.side,
                        s.strength,
                        s.score,
                        s.entry,
                        s.sl,
                        s.tp
                    FROM paper_trades pt
                    JOIN signals s ON s.id = pt.signal_id
                    WHERE pt.chat_id = %s
                      AND s.ts >= %s
                    ORDER BY s.ts DESC
                    LIMIT %s
                    """,
                    (str(chat_id), cutoff, int(limit)),
                )
                return list(cur.fetchall() or [])
    with sqlite3.connect(DB_PATH) as con:
        con.row_factory = sqlite3.Row
        rows = con.execute(
            """
            SELECT
                pt.id AS paper_id,
                pt.chat_id,
                pt.signal_id,
                pt.due_ts,
                pt.notified,
                s.ts AS signal_ts,
                s.symbol,
                s.mode,
                s.side,
                s.strength,
                s.score,
                s.entry,
                s.sl,
                s.tp
            FROM paper_trades pt
            JOIN signals s ON s.id = pt.signal_id
            WHERE pt.chat_id = ?
              AND s.ts >= ?
            ORDER BY s.ts DESC
            LIMIT ?
            """,
            (str(chat_id), cutoff, int(limit)),
        ).fetchall()
        return [dict(r) for r in (rows or [])]


def delete_paper_trade_for_chat(chat_id: str, paper_id: int) -> None:
    """Delete a single saved paper-trade link for a chat (keeps the signal row intact)."""
    if IS_POSTGRES:
        with _pg_connect() as con:
            with con.cursor() as cur:
                cur.execute(
                    "DELETE FROM paper_trades WHERE id=%s AND chat_id=%s",
                    (int(paper_id), str(chat_id)),
                )
            con.commit()
        return
    with sqlite3.connect(DB_PATH) as con:
        con.execute(
            "DELETE FROM paper_trades WHERE id=? AND chat_id=?",
            (int(paper_id), str(chat_id)),
        )
        con.commit()




def clear_paper_trades_for_chat(chat_id: str) -> int:
    """Delete ALL saved paper trades for a given chat (keeps signals for learning). Returns deleted count."""
    if IS_POSTGRES:
        with _pg_connect() as con:
            with con.cursor() as cur:
                cur.execute("DELETE FROM paper_trades WHERE chat_id=%s", (str(chat_id),))
                deleted = cur.rowcount or 0
            con.commit()
        return int(deleted)
    with sqlite3.connect(DB_PATH) as con:
        cur = con.execute("DELETE FROM paper_trades WHERE chat_id=?", (str(chat_id),))
        deleted = cur.rowcount or 0
        con.commit()
    return int(deleted)


def list_final_paper_reviews_for_chat(chat_id: str, lookback_days: int = 30, limit: int = 50) -> List[Dict[str, Any]]:
    """List frozen 24h paper-review snapshots for a chat (does NOT change over time).

    These rows are written by the 24h paper review runner (kind=paper_24h_final) into signal_reviews.note as JSON.
    """
    cutoff = (datetime.utcnow() - timedelta(days=int(lookback_days))).isoformat()
    like_pat = '%paper_24h_final%'
    if IS_POSTGRES:
        with _pg_connect() as con:
            with con.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    """
                    SELECT
                        pt.id AS paper_id,
                        pt.chat_id,
                        pt.signal_id,
                        pt.due_ts,
                        r.ts AS review_ts,
                        r.close AS exit_price,
                        r.return_pct,
                        r.note,
                        s.ts AS signal_ts,
                        s.symbol,
                        s.mode,
                        s.side,
                        s.score,
                        s.entry
                    FROM paper_trades pt
                    JOIN signals s ON s.id = pt.signal_id
                    JOIN signal_reviews r ON r.signal_id = pt.signal_id
                    WHERE pt.chat_id = %s
                      AND r.ts >= %s
                      AND COALESCE(r.note,'') LIKE %s
                    ORDER BY r.ts DESC
                    LIMIT %s
                    """,
                    (str(chat_id), cutoff, like_pat, int(limit)),
                )
                return cur.fetchall()
    with sqlite3.connect(DB_PATH) as con:
        con.row_factory = sqlite3.Row
        rows = con.execute(
            """
            SELECT
                pt.id AS paper_id,
                pt.chat_id,
                pt.signal_id,
                pt.due_ts,
                r.ts AS review_ts,
                r.close AS exit_price,
                r.return_pct,
                r.note,
                s.ts AS signal_ts,
                s.symbol,
                s.mode,
                s.side,
                s.score,
                s.entry
            FROM paper_trades pt
            JOIN signals s ON s.id = pt.signal_id
            JOIN signal_reviews r ON r.signal_id = pt.signal_id
            WHERE pt.chat_id = ?
              AND r.ts >= ?
              AND COALESCE(r.note,'') LIKE ?
            ORDER BY r.ts DESC
            LIMIT ?
            """,
            (str(chat_id), cutoff, like_pat, int(limit)),
        ).fetchall()
        return [dict(r) for r in rows]

def cleanup_old_paper_trades(retention_days: int = 7) -> int:
    """Auto-clean paper_trades older than retention_days (signals stay for learning). Returns deleted count."""
    cutoff = (datetime.utcnow() - timedelta(days=int(retention_days))).isoformat()
    if IS_POSTGRES:
        with _pg_connect() as con:
            with con.cursor() as cur:
                cur.execute("DELETE FROM paper_trades WHERE due_ts < %s", (cutoff,))
                deleted = cur.rowcount or 0
            con.commit()
        return int(deleted)
    with sqlite3.connect(DB_PATH) as con:
        cur = con.execute("DELETE FROM paper_trades WHERE due_ts < ?", (cutoff,))
        deleted = cur.rowcount or 0
        con.commit()
    return int(deleted)
