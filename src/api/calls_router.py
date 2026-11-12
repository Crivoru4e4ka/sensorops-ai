"""REST API модулей Voice-to-CRM (Модуль 2) и скоринга лидов (Модуль 3)."""
from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from ..schemas.lead import (
    CallAnalysisRequest,
    CallAnalysisResponse,
    CRMLeadCard,
    LeadScoreResponse,
)
from ..services import asr_service
from ..state import AppState, get_state

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/calls", tags=["voice-to-crm"])


@router.post("/transcribe", summary="Транскрибация аудиозвонка (faster-whisper)")
async def transcribe_call(
    file: Annotated[UploadFile, File(description="Аудиофайл разговора (.mp3, .wav)")],
) -> dict:
    """Принимает аудиозапись звонка и возвращает текст транскрипта."""
    from pathlib import PurePath

    suffix = PurePath(file.filename or "").suffix.lower()
    if suffix not in asr_service.SUPPORTED_AUDIO_FORMATS:
        raise HTTPException(
            status_code=415,
            detail=f"Неподдерживаемый формат {suffix or '?'}. Разрешены: "
            f"{', '.join(sorted(asr_service.SUPPORTED_AUDIO_FORMATS))}",
        )
    data = await file.read()
    try:
        transcript = asr_service.transcribe_bytes(data, suffix)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {
        "filename": file.filename,
        "asr_model": f"faster-whisper/{asr_service.settings.whisper_model}",
        "transcript": transcript,
    }


@router.post("/analyze", response_model=CallAnalysisResponse, summary="Полный разбор аудиозвонка в карточку CRM")
async def analyze_call(
    file: Annotated[UploadFile, File(description="Аудиофайл разговора (.mp3, .wav)")],
    state: AppState = Depends(get_state),
) -> CallAnalysisResponse:
    """Транскрибация аудиофайла + извлечение CRM-карточки + скоринг лида."""
    from pathlib import PurePath

    suffix = PurePath(file.filename or "").suffix.lower()
    if suffix not in asr_service.SUPPORTED_AUDIO_FORMATS:
        raise HTTPException(status_code=415, detail=f"Неподдерживаемый формат аудио: {suffix}")
    data = await file.read()
    try:
        transcript = asr_service.transcribe_bytes(data, suffix)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return _analyze_transcript(transcript, state, asr_model=f"faster-whisper/{asr_service.settings.whisper_model}")


@router.post("/analyze-text", response_model=CallAnalysisResponse, summary="Разбор текста разговора в карточку CRM")
async def analyze_call_text(
    request: CallAnalysisRequest,
    state: AppState = Depends(get_state),
) -> CallAnalysisResponse:
    """Извлечение CRM-карточки и скоринг по готовому тексту разговора (без ASR)."""
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=422, detail="Передайте текст разговора")
    return _analyze_transcript(request.text.strip(), state)


def _analyze_transcript(transcript: str, state: AppState, asr_model: str | None = None) -> CallAnalysisResponse:
    """Общая логика: экстракция карточки + скоринг."""
    if not transcript.strip():
        raise HTTPException(status_code=422, detail="Транскрипт пуст — не удалось распознать речь")

    catalog_names = [
        row["name"] for row in state.search_service._rows.values()
    ] if state.search_service._rows else None
    lead_card, source = asr_service.extract_lead_card(transcript, catalog_names)
    score = state.scoring_service.score(lead_card, state.catalog)

    return CallAnalysisResponse(
        transcript=transcript,
        lead_card=lead_card,
        score=score,
        extraction_source=source,  # type: ignore[arg-type]
        asr_model=asr_model,
    )


@router.post("/score", response_model=LeadScoreResponse, summary="Скоринг готовой CRM-карточки")
async def score_lead(
    card: CRMLeadCard,
    state: AppState = Depends(get_state),
) -> LeadScoreResponse:
    """Оценивает вероятность конверсии лида и приоритет для отдела продаж."""
    features = state.scoring_service.build_features(card, state.catalog)
    score = state.scoring_service.score(card, state.catalog)
    return LeadScoreResponse(lead_card=card, features=features, score=score)


@router.post("/enrich", summary="Демо Data Enrichment Mock по названию компании")
async def enrich(
    company_name: Annotated[str, Form(description="Название компании клиента")],
) -> dict:
    """Показывает, какие фичи профиля сгенерируются для компании."""
    from ..services.scoring_service import enrich_company

    return enrich_company(company_name or None)
