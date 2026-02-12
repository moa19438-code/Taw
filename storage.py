import os
import sqlite3
from typing import Any, Dict, List, Optional
from datetime import datetime

DB_PATH = "trades.db"
DATABASE_URL = (os.getenv("DATABASE_URL") or "").strip()

IS_POSTGRES = DATABASE_URL.startswith("postgresql://") or DATABASE_URL.startswith("postgres://")

if IS_POSTGRES:
    import psycopg2
    from psycopg2.extras import RealDictCursor


def _pg_connect():
    # DATABASE_URL لازم يكون فيه ?sslmode=require
    return psycopg2.connect(DATABASE_URL)


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
            con.commit()
        return

    # SQLite fallback (زي كودك تقريبًا)
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
        con.commit()


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
            with con.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("SELECT * FROM orders ORDER BY id DESC LIMIT %s", (limit,))
                return [dict(r) for r in cur.fetchall()]

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
            with con.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    "SELECT ts, universe_size, top_symbols, payload FROM scans ORDER BY id DESC LIMIT %s",
                    (limit,),
                )
                return [dict(r) for r in cur.fetchall()]

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
        import config
        return {
            "TOP_N": str(config.TOP_N),
            "AUTO_TRADE": str(int(config.AUTO_TRADE)),
            "MAX_DAILY_TRADES": str(config.MAX_DAILY_TRADES),
            "RISK_PER_TRADE_PCT": str(config.RISK_PER_TRADE_PCT),
            "TP_R_MULT": str(config.TP_R_MULT),
            "SL_ATR_MULT": str(config.SL_ATR_MULT),
            # Bot UI / manual-trading settings
            "AUTO_NOTIFY": "1",
            "MAX_SEND": "10",
            "MIN_SEND": "7",
            "DEDUP_HOURS": "6",
            "ALLOW_RESEND_IF_STRONGER": "1",
            "PLAN_MODE": "daily",  # daily|weekly|monthly|daily_weekly|weekly_monthly
            "ENTRY_MODE": "auto",  # auto|market|limit
            "CAPITAL_USD": "800",
            "RISK_APLUS_PCT": "1.5",
            "RISK_A_PCT": "1.0",
            "RISK_B_PCT": "0.5",
            "SCAN_INTERVAL_MIN": "20",
            "SCHED_ENABLED": "1",
            "POSITION_PCT": "0.20",
            "SL_PCT": "3",
            "TP_PCT": "5",
            "TP_PCT_STRONG": "7",
            "TP_PCT_VSTRONG": "10",
            "WINDOW_START": "17:30",
            "WINDOW_END": "00:00",
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
                        "INSERT INTO settings(key,value) VALUES(%s,%s) "
                        "ON CONFLICT(key) DO NOTHING",
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
                    "INSERT INTO settings(key,value) VALUES(%s,%s) "
                    "ON CONFLICT(key) DO UPDATE SET value=EXCLUDED.value",
                    (key, value),
                )
            con.commit()
        return

    with sqlite3.connect(DB_PATH) as con:
        con.execute(
            "INSERT INTO settings(key,value) VALUES(?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
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
def log_signal(ts: str, symbol: str, mode: str, strength: str, score: float, entry: float, sl: float, tp: float) -> None:
    if IS_POSTGRES:
        with _pg_connect() as con:
            with con.cursor() as cur:
                cur.execute(
                    "INSERT INTO signals (ts, symbol, mode, strength, score, entry, sl, tp) VALUES (%s,%s,%s,%s,%s,%s,%s,%s)",
                    (ts, symbol, mode, strength, float(score), float(entry), float(sl), float(tp)),
                )
            con.commit()
        return

    with sqlite3.connect(DB_PATH) as con:
        con.execute(
            "INSERT INTO signals (ts, symbol, mode, strength, score, entry, sl, tp) VALUES (?,?,?,?,?,?,?,?)",
            (ts, symbol, mode, strength, float(score), float(entry), float(sl), float(tp)),
        )
        con.commit()


def last_signal(symbol: str, mode: str) -> Optional[Dict[str, Any]]:
    if IS_POSTGRES:
        with _pg_connect() as con:
            with con.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    "SELECT ts, symbol, mode, strength, score, entry, sl, tp FROM signals "
                    "WHERE symbol=%s AND mode=%s ORDER BY id DESC LIMIT 1",
                    (symbol, mode),
                )
                row = cur.fetchone()
                return dict(row) if row else None

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
            with con.cursor(cursor_factory=RealDictCursor) as cur:
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
                return [dict(r) for r in cur.fetchall()]

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
