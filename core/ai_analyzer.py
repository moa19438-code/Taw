import os
from typing import Any

# Gemini client is optional at runtime.
# If the dependency isn't installed or no API key is provided, we return safe fallbacks.
try:
    from google import genai  # provided by `google-genai`
except Exception:  # ImportError or weird env issues
    genai = None  # type: ignore


_api_key = (os.getenv("GEMINI_API_KEY") or "").strip()

DEFAULT_MODEL = (os.getenv("GEMINI_MODEL") or "gemini-flash-latest").strip()

_FALLBACK_MODELS = [
    "gemini-flash-latest",
    "gemini-2.5-flash",
    "gemini-2.0-flash",
]


def _get_client():
    if genai is None:
        return None
    try:
        # Passing api_key explicitly is clearer; if empty, client may still work in some envs.
        return genai.Client(api_key=_api_key) if _api_key else genai.Client()
    except Exception:
        return None


_CLIENT = _get_client()


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
        "اكتب بالعربي وبشكل واضح ومختصر.",
    ]
    return "\n".join(lines)


def _is_model_not_found(err: Exception) -> bool:
    s = str(err)
    return ("404" in s) or ("NOT_FOUND" in s) or ("is not found for API version" in s)


def gemini_analyze(symbol: str, features: dict, model: str | None = None) -> str:
    """Free-text analysis. If Gemini isn't available, returns a short fallback message."""

    if _CLIENT is None:
        return "Gemini غير متوفر (المكتبة غير مثبتة أو لا يوجد إعدادات API)."

    m = (model or DEFAULT_MODEL).strip()
    prompt = _build_prompt(symbol, features)

    tried: list[str] = []
    candidates = [m] + [x for x in _FALLBACK_MODELS if x != m]
    last_err: Exception | None = None

    for mm in candidates:
        tried.append(mm)
        try:
            resp = _CLIENT.models.generate_content(model=mm, contents=prompt)
            return (resp.text or "").strip() or f"تمت الاستجابة لكن بدون نص (model={mm})"
        except Exception as e:
            last_err = e
            if not _is_model_not_found(e):
                raise

    raise RuntimeError(f"فشل استدعاء Gemini. الموديلات التي جُربت: {tried}\nآخر خطأ: {last_err}")


def gemini_predict_direction(symbol: str, features: dict, horizon: str = "D1", model: str | None = None) -> dict:
    """Return a small, machine-readable prediction.

    Output schema (best-effort):
      {"direction":"UP|DOWN|NEUTRAL","confidence":0-100,"horizon":"D1|M5|M5+","reasons":[..],"risks":[..]}

    We return a safe fallback if Gemini isn't available.
    """

    h = (horizon or "D1").strip().upper()

    if _CLIENT is None:
        return {
            "raw": '{"direction":"NEUTRAL","confidence":50,"horizon":"%s","reasons":["Gemini غير متوفر"],"risks":[]}'
            % h,
            "model": None,
            "horizon": h,
        }

    m = (model or DEFAULT_MODEL).strip()

    lines = [
        f"حلل السهم الأمريكي التالي للاستكشاف فقط (ليس نصيحة مالية): {symbol}",
        f"الإطار المطلوب للتوقع: {h} (D1=اليوم/الجلسة، M5=القادم 30-90 دقيقة، M5+=دمج يومي+M5)",
        "",
        "بيانات ومؤشرات (قيم تقريبية):",
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

    tried: list[str] = []
    candidates = [m] + [x for x in _FALLBACK_MODELS if x != m]
    last_err: Exception | None = None

    for mm in candidates:
        tried.append(mm)
        try:
            resp = _CLIENT.models.generate_content(model=mm, contents=prompt)
            txt = (resp.text or "").strip()
            return {"raw": txt, "model": mm, "horizon": h}
        except Exception as e:
            last_err = e
            if not _is_model_not_found(e):
                raise

    raise RuntimeError(f"فشل استدعاء Gemini للتوقع. الموديلات التي جُربت: {tried}\nآخر خطأ: {last_err}")
