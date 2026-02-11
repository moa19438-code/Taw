from __future__ import annotations
import requests
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone, timedelta

from config import ALPACA_API_KEY, ALPACA_API_SECRET, ALPACA_BASE_URL, ALPACA_DATA_BASE_URL, REQUEST_TIMEOUT

def _headers() -> Dict[str, str]:
    return {
        "APCA-API-KEY-ID": ALPACA_API_KEY,
        "APCA-API-SECRET-KEY": ALPACA_API_SECRET,
        "Content-Type": "application/json",
    }

def _get(path: str, params: Optional[Dict[str, Any]] = None) -> Any:
    url = ALPACA_BASE_URL.rstrip("/") + path
    r = requests.get(url, headers=_headers(), params=params or {}, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    return r.json()
    
    def _get_data(path: str, params: Optional[Dict[str, Any]] = None) -> Any:
    url = ALPACA_DATA_BASE_URL.rstrip("/") + path
    r = requests.get(url, headers=_headers(), params=params or {}, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    return r.json()

def _post(path: str, payload: Dict[str, Any]) -> Any:
    url = ALPACA_BASE_URL.rstrip("/") + path
    r = requests.post(url, headers=_headers(), json=payload, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    return r.json()

def clock() -> Dict[str, Any]:
    return _get("/v2/clock")

def account() -> Dict[str, Any]:
    return _get("/v2/account")

def list_assets(limit: int = 4000) -> List[Dict[str, Any]]:
    # Alpaca returns a list; we filter client-side.
    assets = _get("/v2/assets", params={"status": "active"})
    # Keep US equities that are tradable and not OTC.
    out = []
    for a in assets:
        if a.get("tradable") and a.get("class") == "us_equity":
            # Some OTC symbols may appear; we'll rely on exchange or fractionable flags.
            out.append(a)
    return out[:limit]

def latest_trade(symbol: str) -> Dict[str, Any]:
    # market data endpoint differs; Alpaca provides data on different host for some plans.
    # Many users use the same base URL for paper/live; if your plan needs a different data host,
    # set ALPACA_DATA_BASE_URL and adjust code.
    # We'll call through base URL for simplicity.
    return _get_data(f"/v2/stocks/{symbol}/trades/latest")

def bars(symbols: List[str], start: datetime, end: datetime, timeframe: str="1Day", limit: int=200) -> Dict[str, Any]:
    # Newer Alpaca API uses /v2/stocks/bars?symbols=...&timeframe=...
    params = {
        "symbols": ",".join(symbols),
        "timeframe": timeframe,
        "start": start.replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z"),
        "end": end.replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z"),
        "limit": limit,
        "adjustment": "raw",
    }
    return return _get_data("/v2/stocks/bars", params=params)

def place_bracket_order(symbol: str, side: str, qty: float, take_profit_price: float, stop_loss_price: float) -> Dict[str, Any]:
    payload = {
        "symbol": symbol,
        "qty": str(qty),
        "side": side,
        "type": "market",
        "time_in_force": "day",
        "order_class": "bracket",
        "take_profit": {"limit_price": f"{take_profit_price:.2f}"},
        "stop_loss": {"stop_price": f"{stop_loss_price:.2f}"},
    }
    return _post("/v2/orders", payload)

def positions() -> Any:
    return _get("/v2/positions")

def open_orders(status: str = "open", limit: int = 500) -> Any:
    return _get("/v2/orders", params={"status": status, "limit": limit, "direction": "desc"})

