"""Pipeline обработки аудиозвонков Voice-to-CRM (Модуль 2).

Два этапа:
1. **Транскрибация** — аудиофайл (.mp3, .wav и др.) обрабатывается
   faster-whisper (модель ``WHISPER_MODEL``, по умолчанию ``small``);
2. **Структурированная экстракция** — из текста извлекается строго
   типизированная карточка :class:`CRMLeadCard` через Instructor + Pydantic v2.

Если LLM недоступен (``LLM_BACKEND=mock`` или сбой), используется
детерминированное rule-based извлечение, которое гарантированно возвращает
валидную карточку — PoC остаётся рабочим без внешних API.
"""
from __future__ import annotations

import logging
import re
import tempfile
from functools import lru_cache
from pathlib import Path

from ..config import settings
from ..schemas.lead import CRMLeadCard

logger = logging.getLogger(__name__)

SUPPORTED_AUDIO_FORMATS = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".webm", ".mp4"}

_PHONE_RE = re.compile(r"(\+?\d[\d\s\-()]{6,}\d)")
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_SENTENCE_RE = re.compile(r"[^.!?]+[.!?]?", re.UNICODE)


def _extract_spoken_email(text: str) -> str | None:
    """Собирает email, произнесённый словами.

    «Дмитрий точка Петров, собака промстроймаш точка ру» →
    «Дмитрий.Петров@промстроймаш.ру». Устойчив к пунктуации ASR
    («Дмитрий. Точка. Петров. Собака. Промстроймаш. Точка. Ру.»).
    """
    normalized = re.sub(
        r"[.,;:!?]*\s*(?:собака|собачка)\s*[.,;:!?]*", "@", text, flags=re.IGNORECASE
    )
    normalized = re.sub(
        r"[.,;:!?]*\s*(?:точка|тчк)\s*[.,;:!?]*", ".", normalized, flags=re.IGNORECASE
    )
    normalized = re.sub(r"\s+", "", normalized)
    normalized = re.sub(r"\.{2,}", ".", normalized)
    match = re.search(
        r"([a-zа-яё0-9_+-]+\.[a-zа-яё0-9_+-]+)\.?@([a-zа-яё0-9_+-]+)\.([a-zа-яё0-9]+)",
        normalized,
        re.IGNORECASE,
    )
    if match:
        return f"{match.group(1)}@{match.group(2)}.{match.group(3)}"
    return None


# ─── Транскрибация ──────────────────────────────────────────────────────────────


@lru_cache(maxsize=1)
def get_whisper_model():
    """Лениво загружает faster-whisper модель (кэшируется в процессе)."""
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError(
            "faster-whisper не установлен. Установите зависимости: pip install faster-whisper"
        ) from exc

    device = settings.whisper_device
    if device == "auto":
        try:
            import ctranslate2

            device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
        except Exception:  # noqa: BLE001
            device = "cpu"

    logger.info("Загрузка faster-whisper модели %r (device=%s)", settings.whisper_model, device)
    return WhisperModel(
        settings.whisper_model,
        device=device,
        compute_type=settings.whisper_compute_type,
    )


def transcribe_file(path: str | Path) -> str:
    """Транскрибирует аудиофайл в текст (язык определяется автоматически)."""
    model = get_whisper_model()
    segments, info = model.transcribe(str(path), beam_size=5, vad_filter=True)
    logger.info("ASR: %s, язык=%s (вероятность %.2f)", path, info.language, info.language_probability)
    return " ".join(segment.text.strip() for segment in segments).strip()


def transcribe_bytes(data: bytes, suffix: str) -> str:
    """Транскрибирует аудио из байтов (для загрузки через HTTP)."""
    if suffix not in SUPPORTED_AUDIO_FORMATS:
        raise ValueError(f"Неподдерживаемый формат аудио: {suffix}")
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name
    try:
        return transcribe_file(tmp_path)
    finally:
        Path(tmp_path).unlink(missing_ok=True)


# ─── Структурированная экстракция ──────────────────────────────────────────────


def _llm_extract(text: str) -> CRMLeadCard:
    """Извлечение карточки лида через Instructor + LiteLLM (гарантированный JSON)."""
    import litellm

    try:
        import instructor

        client = instructor.from_litellm(litellm.completion)
        kwargs = {
            "model": settings.llm_model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Ты — ассистент менеджера по продажам. Извлеки из транскрипта "
                        "телефонного разговора структурированную карточку для CRM. "
                        "Все поля строго типизированы; заполняй только фактами из текста."
                    ),
                },
                {"role": "user", "content": f"Транскрипт разговора:\n{text}"},
            ],
            "response_model": CRMLeadCard,
            "temperature": 0.0,
            "timeout": settings.llm_timeout,
        }
        if settings.llm_api_base:
            kwargs["base_url"] = settings.llm_api_base
        card = client.chat.completions.create(**kwargs)
        return card
    except Exception as exc:  # noqa: BLE001
        logger.warning("Instructor-экстракция не удалась (%s)", exc)
        raise


def _find_sentence_containing(text: str, keywords: list[str]) -> list[str]:
    result = []
    for sentence in _SENTENCE_RE.findall(text):
        lowered = sentence.lower()
        if any(keyword in lowered for keyword in keywords):
            result.append(_strip_speaker_prefix(sentence))
    return result


def _strip_speaker_prefix(sentence: str) -> str:
    """Убирает префиксы ролей («Менеджер:», «Клиент:») из предложения."""
    return re.sub(r"^\s*(?:менеджер|клиент|оператор|агент)\s*:\s*", "", sentence, flags=re.IGNORECASE)


def _rules_extract(text: str, catalog_names: list[str] | None = None) -> CRMLeadCard:
    """Детерминированное rule-based извлечение (fallback без LLM)."""
    lowered = text.lower()
    catalog_names = catalog_names or []

    phone_match = _PHONE_RE.search(text)
    email_match = _EMAIL_RE.search(text)
    spoken_email = _extract_spoken_email(text)
    if spoken_email:
        contact = spoken_email
    elif email_match:
        contact = email_match.group(0)
    elif phone_match:
        contact = phone_match.group(0)
    else:
        contact = None

    company = None
    client_name = None

    # Приоритет 1: «(это/зовут) <Имя> ... из (компании) <Компания>» — клиент называет СВОЮ компанию
    for pattern in (
        r"(?:меня\s+зовут|это|я)\s+([А-ЯA-Z][а-яёa-z\-]+)"
        r"[^.!?]{0,80}?\s+из\s+(?:компании\s+)?[\"«]?([А-ЯA-Z][\w\-]*(?:[\s\-][\w]+){0,3})",
        r"([А-ЯA-Z][а-яёa-z\-]+)\s+из\s+(?:компании\s+)?[\"«]?([А-ЯA-Z][\w\-]*(?:[\s\-][\w]+){0,3})",
    ):
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            client_name = match.group(1)
            company = match.group(2).strip(" ,.\"»«")
            break

    # Приоритет 2: «меня зовут <Имя> ... компания <Компания>»
    if company is None:
        match = re.search(
            r"(?:меня\s+зовут|это)\s+([А-ЯA-Z][а-яёa-z\-]+)"
            r"[^.!?]{0,80}?\s+компани[яи]\s+[\"«]?([А-ЯA-Z][\w\-]*(?:[\s\-][\w]+){0,3})",
            text,
            re.IGNORECASE,
        )
        if match:
            client_name = client_name or match.group(1)
            company = match.group(2).strip(" ,.\"»«")

    # Приоритет 3: «компания <Компания>» (обычно так представляется менеджер)
    if company is None:
        match = re.search(
            r"(?:компани[яи]\s+|ооо\s+|ао\s+|зао\s+|завод(?:а)?\s+|холдинг(?:а)?\s+|фирм[аы]\s+)"
            r"[\"«]?([А-ЯA-Z][\w\-]*(?:[\s\-][\w]+){0,3})",
            text,
            re.IGNORECASE,
        )
        if match:
            company = match.group(1).strip(" ,.\"»«")

    # Имя контактного лица, если ещё не найдено
    if client_name is None:
        name_match = re.search(
            r"(?:меня\s+зовут|это|здравствуйте,?\s*я)\s+([А-ЯA-Z][а-яёa-z\-]+)",
            text,
            re.IGNORECASE,
        )
        if name_match:
            client_name = name_match.group(1)

    # Срочность
    if re.search(r"срочн|сегодня|завтра|горит|как можно быстрее|на этой неделе", lowered):
        urgency = "критическая"
    elif re.search(r"на неделе|в течение недели|скоро|в ближайшее время", lowered):
        urgency = "средняя"
    else:
        urgency = "низкая"

    # Статус сделки
    if re.search(r"отказ|не нужно|не интересно|отказываемся|не подходит", lowered):
        deal_stage = "отказ"
    elif re.search(r"перезвон|созвон|повторн|свяжусь позже", lowered):
        deal_stage = "повторный звонок"
    else:
        deal_stage = "запрос КП"

    # Запрошенные продукты: сопоставление с каталогом + упоминания типов датчиков
    requested: list[str] = []
    type_keywords = {
        "индуктив": "датчик индуктивный",
        "оптическ": "датчик оптический",
        "давлени": "датчик давления",
        "температур": "датчик температуры",
        "ёмкостн": "датчик ёмкостный",
        "емкостн": "датчик ёмкостный",
    }
    for keyword, label in type_keywords.items():
        if keyword in lowered:
            requested.append(label)
    for name in catalog_names:
        tokens = set(tokenize_text(name)) - {"датчик", "серия"}
        if tokens and all(token in lowered for token in list(tokens)[:2]):
            if name not in requested:
                requested.append(name)
    if not requested:
        requested = ["не определено (уточнить у клиента)"]
    requested = list(dict.fromkeys(requested))[:5]

    # Действия менеджера
    action_keywords = [
        "отправлю", "пришлю", "вышлю", "подготовлю", "сделаю кп", "составлю кп",
        "перезвоню", "уточню", "забронирую", "проверю наличие",
    ]
    action_items = _find_sentence_containing(text, action_keywords)
    if not action_items:
        action_items = ["Связаться с клиентом и уточнить детали"]

    # Краткая суть: первые 2-3 предложения (с очисткой префиксов ролей)
    sentences = [
        _strip_speaker_prefix(sentence).strip()
        for sentence in _SENTENCE_RE.findall(text)
        if sentence.strip()
    ]
    summary = " ".join(sentences[:3])
    if len(summary) > 400:
        summary = summary[:400].rsplit(" ", 1)[0] + "…"

    return CRMLeadCard(
        company_name=company,
        client_name=client_name,
        contact_info=contact,
        requested_products=requested,
        urgency=urgency,
        deal_stage=deal_stage,
        summary=summary,
        action_items=action_items[:5],
    )


def tokenize_text(text: str) -> list[str]:
    return re.findall(r"[a-zа-яё0-9]+", text.lower())


def extract_lead_card(text: str, catalog_names: list[str] | None = None) -> tuple[CRMLeadCard, str]:
    """Извлекает карточку лида; возвращает (карточка, источник: llm|rules)."""
    if settings.llm_backend == "litellm" and (settings.llm_api_key or settings.llm_api_base):
        try:
            return _llm_extract(text), "llm"
        except Exception:  # noqa: BLE001
            pass
    return _rules_extract(text, catalog_names), "rules"
