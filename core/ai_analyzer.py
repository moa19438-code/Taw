import os
from google import genai

# الأفضل تمرير المفتاح صراحة (أوضح وأضمن)
_api_key = (os.getenv("GEMINI_API_KEY") or "").strip()
_client = genai.Client(api_key=_api_key) if _api_key else genai.Client()

# موديلات آمنة/شائعة (غيّر من Render عبر GEMINI_MODEL إذا تبي)
DEFAULT_MODEL = (os.getenv("GEMINI_MODEL") or "gemini-flash-latest").strip()

_FALLBACK_MODELS = [
    "gemini-flash-latest",
    "gemini-2.5-flash",
    "gemini-2.0-flash",
]

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
        "أبغى:",
        "1) نظرة عامة مختصرة (اتجاه/زخم/تذبذب).",
        "2) أهم 3 نقاط إيجابية وأهم 3 مخاطر.",
        "3) سيناريوهين: صعود/هبوط + مستويات محتملة (تقريبية) بناءً على البيانات.",
        "4) اقتراح وقف خسارة ذكي (فكرة عامة) بدون أرقام مؤكدة إذا البيانات ما تكفي.",
        "5) خلاصة: هل مناسب للاستكشاف الآن؟ ولماذا.",
        "",
        "اكتب بالعربي وبشكل واضح ومختصر."
    ]
    return "\n".join(lines)

def _is_model_not_found(err: Exception) -> bool:
    s = str(err)
    return ("404" in s) or ("NOT_FOUND" in s) or ("is not found for API version" in s)

def gemini_analyze(symbol: str, features: dict, model: str | None = None) -> str:
    m = (model or DEFAULT_MODEL).strip()
    prompt = _build_prompt(symbol, features)

    # نجرب الموديل المحدد أولاً ثم نسقط على بدائل
    tried = []
    candidates = [m] + [x for x in _FALLBACK_MODELS if x != m]

    last_err: Exception | None = None
    for mm in candidates:
        tried.append(mm)
        try:
            resp = _client.models.generate_content(model=mm, contents=prompt)
            return (resp.text or "").strip() or f"تمت الاستجابة لكن بدون نص (model={mm})"
        except Exception as e:
            last_err = e
            # إذا الخطأ مو “موديل غير موجود” لا نكمل Fallback
            if not _is_model_not_found(e):
                raise

    # لو كلهم فشلوا بنفس المشكلة
    raise RuntimeError(f"فشل استدعاء Gemini. الموديلات التي جُربت: {tried}\nآخر خطأ: {last_err}")


def gemini_predict_direction(symbol: str, features: dict, horizon: str = "D1", model: str | None = None) -> dict:
    """Return a small, machine-readable prediction.

    Output schema (best-effort):
      {"direction":"UP|DOWN|NEUTRAL","confidence":0-100,"horizon":"D1|M5|M5+","reasons":[..],"risks":[..]}

    Notes:
      - This is exploratory only, not financial advice.
      - We force JSON-only output to keep the bot stable.
    """
    m = (model or DEFAULT_MODEL).strip()
    h = (horizon or "D1").strip().upper()

    # Keep the prompt concise to reduce latency/cost.
    lines = [
        f"حلل السهم الأمريكي التالي للاستكشاف فقط (ليس نصيحة مالية): {symbol}",
        f"الإطار المطلوب للتوقع: {h} (D1=اليوم/الجلسة، M5=القادم 30-90 دقيقة، M5+=دمج يومي+M5)",
        "", "بيانات ومؤشرات (قيم تقريبية):"
    ]
    for k, v in features.items():
        lines.append(f"- {k}: {v}")
    lines += [
        "",
        "أخرج JSON فقط بدون أي شرح خارج JSON.",
        "المفاتيح المطلوبة:",
        "direction: واحدة من [UP, DOWN, NEUTRAL]",
        "confidence: رقم من 0 إلى 100",
        "horizon: نفس الإطار",
        "reasons: قائمة 1-3 أسباب قصيرة",
        "risks: قائمة 0-3 مخاطر قصيرة",
        "إذا البيانات غير كافية: direction=NEUTRAL و confidence<=55.",
    ]
    prompt = "\n".join(lines)

    tried = []
    candidates = [m] + [x for x in _FALLBACK_MODELS if x != m]
    last_err: Exception | None = None
    for mm in candidates:
        tried.append(mm)
        try:
            resp = _client.models.generate_content(model=mm, contents=prompt)
            txt = (resp.text or "").strip()
            return {"raw": txt, "model": mm, "horizon": h}
        except Exception as e:
            last_err = e
            if not _is_model_not_found(e):
                raise

    raise RuntimeError(f"فشل استدعاء Gemini للتوقع. الموديلات التي جُربت: {tried}\nآخر خطأ: {last_err}")
