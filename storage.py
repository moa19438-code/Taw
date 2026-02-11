import sqlite3
from typing import Any, Dict, List, Optional
from datetime import datetime

DB_PATH = "trades.db"

def init_db() -> None:
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
        con.commit()
        con.commit()

def log_order(symbol: str, side: str, qty: float, order_type: str, payload: str, broker_order_id: Optional[str], status: str, message: str) -> None:
    ts = datetime.utcnow().isoformat()
    with sqlite3.connect(DB_PATH) as con:
        con.execute(
            "INSERT INTO orders (ts, symbol, side, qty, order_type, payload, broker_order_id, status, message) VALUES (?,?,?,?,?,?,?,?,?)",
            (ts, symbol, side, qty, order_type, payload, broker_order_id or "", status, message),
        )
        con.commit()

def last_orders(limit: int = 20) -> List[Dict[str, Any]]:
    with sqlite3.connect(DB_PATH) as con:
        con.row_factory = sqlite3.Row
        rows = con.execute("SELECT * FROM orders ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]


def log_scan(ts: str, universe_size: int, top_symbols: str, payload: str = "") -> None:
    with sqlite3.connect(DB_PATH) as con:
        con.execute(
            "INSERT INTO scans (ts, universe_size, top_symbols, payload) VALUES (?, ?, ?, ?)",
            (ts, universe_size, top_symbols, payload),
        )
        con.commit()

def last_scans(limit: int = 50) -> List[Dict[str, Any]]:
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
        }
    except Exception:
        return {}

def ensure_default_settings() -> None:
    defaults = _env_defaults()
    if not defaults:
        return
    with sqlite3.connect(DB_PATH) as con:
        for k, v in defaults.items():
            con.execute("INSERT OR IGNORE INTO settings(key,value) VALUES(?,?)", (k, v))
        con.commit()

def get_setting(key: str, default: Optional[str]=None) -> Optional[str]:
    with sqlite3.connect(DB_PATH) as con:
        cur = con.execute("SELECT value FROM settings WHERE key=?", (key,))
        row = cur.fetchone()
        return row[0] if row else default

def set_setting(key: str, value: str) -> None:
    with sqlite3.connect(DB_PATH) as con:
        con.execute("INSERT INTO settings(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, value))
        con.commit()

def get_all_settings() -> Dict[str, str]:
    with sqlite3.connect(DB_PATH) as con:
        cur = con.execute("SELECT key,value FROM settings")
        return {k: v for k, v in cur.fetchall()}

def parse_bool(v: Any, default: bool=False) -> bool:
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    return s in ("1","true","yes","y","on")

def parse_int(v: Any, default: int=0) -> int:
    try:
        return int(str(v).strip())
    except Exception:
        return default

def parse_float(v: Any, default: float=0.0) -> float:
    try:
        return float(str(v).strip())
    except Exception:
        return default
