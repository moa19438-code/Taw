import os
from google import genai

# يقرأ المفتاح تلقائيًا من ENV: GEMINI_API_KEY
_client = genai.Client()

DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

def _build_prompt(symbol: str, features: dict) -> str:
    lines = [
        f"حلل السهم الأمريكي التالي للاستكشاف فقط (ليس نصيحة مالية): {symbol}",
        "",
        "بيانات ومؤشرات:",
    ]
    for k, v in features.items():
        lines.append(f"- {k}: {v}")

    lines += [
        "",
        "اكتب الرد بالعربية وبشكل مختصر وواضح:",
        "1) ملخص الحالة (صاعد/متذبذب/ضعيف) + سبب.",
        "2) أهم 3 نقاط قوة.",
        "3) أهم 3 مخاطر.",
        "4) سيناريو اختراق + سيناريو فشل (مع مستويات رقمية إن أمكن).",
        "5) اقتراح إدارة صفقة اختياري (Entry/SL/TP أو Trailing) بدون مبالغة.",
        "6) هل مناسب للاستكشاف الآن؟ (نعم/لا) + سبب واحد.",
    ]
    return "\n".join(lines)

def gemini_analyze(symbol: str, features: dict, model: str | None = None) -> str:
    m = model or DEFAULT_MODEL
    prompt = _build_prompt(symbol, features)
    resp = _client.models.generate_content(model=m, contents=prompt)
    return (resp.text or "").strip()
