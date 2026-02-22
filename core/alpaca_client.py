from __future__ import annotations
import requests
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone

from core.config import (
    ALPACA_API_KEY,
    ALPACA_API_SECRET,
    ALPACA_BASE_URL,
    ALPACA_DATA_BASE_URL,
    REQUEST_TIMEOUT,
)


def _headers() -> Dict[str, str]:
    return {
        "APCA-API-KEY-ID": ALPACA_API_KEY,
        "APCA-API-SECRET-KEY": ALPACA_API_SECRET,
        "Content-Type": "application/json",
    }


# ===== Trading API (paper-api) =====
def _get(path: str, params: Optional[Dict[str, Any]] = None) -> Any:
    url = ALPACA_BASE_URL.rstrip("/") + path
    r = requests.get(url, headers=_headers(), params=params or {}, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    return r.json()


def _post(path: str, payload: Dict[str, Any]) -> Any:
    url = ALPACA_BASE_URL.rstrip("/") + path
    r = requests.post(url, headers=_headers(), json=payload, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    return r.json()


# ===== Market Data API (data.alpaca.markets) =====
def _get_data(path: str, params: Optional[Dict[str, Any]] = None) -> Any:
    url = ALPACA_DATA_BASE_URL.rstrip("/") + path
    r = requests.get(url, headers=_headers(), params=params or {}, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    return r.json()


# ===== Account / Trading =====
def clock() -> Dict[str, Any]:
    return _get("/v2/clock")


def account() -> Dict[str, Any]:
    return _get("/v2/account")


def list_assets(limit: int = 4000) -> List[Dict[str, Any]]:
    assets = _get("/v2/assets", params={"status": "active"})
    out = []
    for a in assets:
        if a.get("tradable") and a.get("class") == "us_equity":
            out.append(a)
    return out[:limit]


def positions() -> Any:
    return _get("/v2/positions")


def open_orders(status: str = "open", limit: int = 500) -> Any:
    return _get("/v2/orders", params={"status": status, "limit": limit, "direction": "desc"})


def place_bracket_order(
    symbol: str,
    side: str,
    qty: float,
    take_profit_price: float,
    stop_loss_price: float,
) -> Dict[str, Any]:
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


# ===== Market Data =====
def latest_trade(symbol: str) -> Dict[str, Any]:
    # add feed=iex to avoid SIP access errors on free plans
    return _get_data(f"/v2/stocks/{symbol}/trades/latest", params={"feed": "iex"})


def bars(
    symbols: List[str],
    start: datetime,
    end: datetime,
    timeframe: str = "1Day",
    limit: int = 200,
) -> Dict[str, Any]:
    params = {
        "symbols": ",".join(symbols),
        "timeframe": timeframe,
        "start": start.replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z"),
        "end": end.replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z"),
        "limit": limit,
        "adjustment": "raw",
        # add feed=iex to avoid SIP access errors on free plans
        "feed": "iex",
    }
    return _get_data("/v2/stocks/bars", params=params)


# ===== News (Market Data) =====
def news(symbols: List[str] | str, limit: int = 20, lookback_hours: int = 48) -> Any:
    """Fetch recent news for given symbol(s) via Alpaca News endpoint (if enabled in account).
    Uses Market Data base URL.
    """
    from datetime import timedelta
    if isinstance(symbols, str):
        sym = symbols
    else:
        sym = ",".join([s for s in symbols if s])
    end = datetime.now(timezone.utc)
    start = end - timedelta(hours=int(max(1, lookback_hours)))
    params = {
        "symbols": sym,
        "start": start.isoformat().replace("+00:00", "Z"),
        "end": end.isoformat().replace("+00:00", "Z"),
        "limit": int(max(1, min(50, limit))),
        "sort": "desc",
    }
    try:
        return _get_data("/v1beta1/news", params=params)
    except Exception:
        # Some accounts may not have the endpoint; return empty.
        return {"news": []}
