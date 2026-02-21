from __future__ import annotations
import os
import json
import requests
from typing import Any, Dict, Optional

# Google Gemini (Generative Language API) integration (optional).
# This module is safe to import even without an API key.
#
# Set:
#   GEMINI_API_KEY=...
# Optional:
#   GEMINI_MODEL=gemini-1.5-pro (default)
#   GEMINI_TIMEOUT_SEC=12

_API_KEY = (os.getenv("GEMINI_API_KEY") or "").strip()
_MODEL = (os.getenv("GEMINI_MODEL") or "gemini-1.5-pro").strip()
_TIMEOUT = float(os.getenv("GEMINI_TIMEOUT_SEC") or "12")

def is_enabled() -> bool:
    return bool(_API_KEY)

def _endpoint() -> str:
    # Using v1beta endpoint style. If you use a different endpoint, update here.
    return f"https://generativelanguage.googleapis.com/v1beta/models/{_MODEL}:generateContent?key={_API_KEY}"

def generate_insight(prompt: str, *, system: Optional[str] = None, max_chars: int = 700) -> Optional[str]:
    """Return a short Arabic insight for a given prompt, or None if disabled/error."""
    if not _API_KEY:
        return None
    sys_txt = system or (
        "أنت مساعد تداول محترف. اكتب ملخصًا قصيرًا جدًا بالعربية، "
        "مباشر، بدون مبالغة، واذكر المخاطر باختصار. لا تعطِ وعود أرباح."
    )
    try:
        payload: Dict[str, Any] = {
            "contents": [
                {"role": "user", "parts": [{"text": sys_txt + "\n\n" + prompt}]}
            ],
            "generationConfig": {
                "temperature": 0.2,
                "topP": 0.9,
                "maxOutputTokens": 220,
            },
        }
        r = requests.post(_endpoint(), json=payload, timeout=_TIMEOUT)
        if r.status_code != 200:
            return None
        data = r.json()
        text = None
        # best-effort parsing
        candidates = data.get("candidates") or []
        if candidates:
            parts = ((candidates[0].get("content") or {}).get("parts") or [])
            if parts:
                text = parts[0].get("text")
        if not text:
            return None
        text = str(text).strip()
        if len(text) > max_chars:
            text = text[:max_chars-1].rstrip() + "…"
        return text
    except Exception:
        return None
