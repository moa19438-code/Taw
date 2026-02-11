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
