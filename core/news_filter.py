
from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple
from datetime import datetime, timezone

from core.alpaca_client import news as alpaca_news
from core.ai_analyzer import gemini_assess_news


def _split_csv(s: str) -> List[str]:
    out: List[str] = []
    for part in (s or "").split(","):
        p = part.strip()
        if p:
            out.append(p)
    return out


DEFAULT_BLOCK = [
    "earnings", "guidance", "sec", "lawsuit", "investigation", "bankrupt", "bankruptcy",
    "fraud", "downgrade", "recall", "halt", "delist", "offering", "dilution",
]


def check_news_risk(symbol: str) -> Tuple[bool, List[str], Dict[str, Any]]:
    """فلتر أخبار بسيط لتقليل خسائر المفاجآت.

    الفكرة:
    - نجلب آخر الأخبار للسهم خلال lookback
    - إذا وجدنا كلمات حساسة (مثل earnings / lawsuit...) نعتبر المخاطرة عالية ونمنع الإشارة (أو نرفع LossProb)

    يمكن تخصيص الكلمات من ENV:
      NEWS_BLOCK_KEYWORDS="earnings,lawsuit,SEC,..."
      NEWS_LOOKBACK_HOURS="48"
      NEWS_LIMIT="20"

    Returns:
      (ok, reasons, meta)
    """
    enabled = (os.getenv("NEWS_FILTER_ENABLED", "0").strip().lower() in {"1", "true", "yes", "on"})
    if not enabled:
        return True, [], {"enabled": False}

    lookback = int(float(os.getenv("NEWS_LOOKBACK_HOURS", "48") or 48))
    limit = int(float(os.getenv("NEWS_LIMIT", "20") or 20))
    kws = _split_csv(os.getenv("NEWS_BLOCK_KEYWORDS", "")) or list(DEFAULT_BLOCK)

    data = alpaca_news(symbol, limit=limit, lookback_hours=lookback) or {}
    items = []
    # Alpaca sometimes returns {"news":[...]} or list
    if isinstance(data, dict):
        items = data.get("news") or data.get("data") or data.get("items") or []
    elif isinstance(data, list):
        items = data
    reasons: List[str] = []
    hits: List[Dict[str, Any]] = []
    # === (اختياري) تقييم الأخبار بالذكاء الاصطناعي (Gemini) ===
    ai_enabled = (os.getenv("NEWS_AI_ENABLED", "0").strip().lower() in {"1","true","yes","on"})
    ai_block_threshold = (os.getenv("NEWS_AI_BLOCK_RISK", "HIGH") or "HIGH").strip().upper()
    ai_out: Dict[str, Any] = {}
    if ai_enabled and items:
        try:
            ai_out = gemini_assess_news(symbol, items[:8])
            # إذا Gemini قال block=true نمنع الإشارة
            if isinstance(ai_out, dict) and bool(ai_out.get("block")):
                reasons.append("الذكاء الاصطناعي: أخبار عالية المخاطر")
                for rr in (ai_out.get("reasons") or [])[:3]:
                    reasons.append(f"- {rr}")
                return False, reasons, {"enabled": True, "ai": ai_out, "lookback_hours": lookback}
        except Exception as e:
            ai_out = {"error": str(e)}

    for it in (items or [])[:50]:
        try:
            headline = str(it.get("headline") or it.get("title") or "").strip()
            summary = str(it.get("summary") or "").strip()
            ts = str(it.get("created_at") or it.get("updated_at") or it.get("time") or "")
            blob = (headline + " " + summary).lower()
            found = []
            for kw in kws:
                if kw.lower() in blob:
                    found.append(kw)
            if found:
                hits.append({"headline": headline[:180], "ts": ts, "keywords": found[:6]})
        except Exception:
            continue

    if hits:
        # Block by default if any sensitive news
        reasons.append("تم رصد أخبار حساسة قد تسبب تذبذب/قفزات سعرية")
        # include top 2 headlines
        for h in hits[:2]:
            reasons.append(f"- {h.get('headline')} ({','.join(h.get('keywords') or [])})")
        return False, reasons, {"enabled": True, "hits": hits[:5], "lookback_hours": lookback, "ai": ai_out}

    return True, [], {"enabled": True, "hits": [], "lookback_hours": lookback, "ai": ai_out}
