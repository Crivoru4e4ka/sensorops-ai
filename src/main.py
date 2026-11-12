"""SensorOps AI — FastAPI-приложение.

Запуск:
    uvicorn src.main:app --host 0.0.0.0 --port 8000

Swagger: http://localhost:8000/docs
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api import calls_router, search_router
from .config import settings
from .state import build_state

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper() if hasattr(settings, "log_level") else "INFO"),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("sensorops")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Запуск %s: индексация каталога и загрузка модели скоринга...", settings.app_name)
    state = build_state()
    app.state.sensorops = state
    logger.info(
        "Готово: датчиков=%d, embedding=%s, vector_db=%s, llm=%s",
        state.search_service.catalog_size,
        state.search_service.embedder.name,
        state.search_service.store.name,
        settings.llm_backend,
    )
    yield
    logger.info("Остановка %s", settings.app_name)


app = FastAPI(
    title="SensorOps AI",
    description=(
        "Enterprise-платформа интеллектуального поиска, анализа коммуникаций "
        "и B2B-скоринга для дистрибьютора промышленной электроники (PoC)."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(search_router.router)
app.include_router(calls_router.router)


@app.get("/api/health", tags=["system"], summary="Статус системы")
async def health() -> dict:
    """Сводка о состоянии сервисов (для мониторинга и отладки)."""
    from .state import get_state

    state = get_state()
    search = state.search_service
    return {
        "status": "ok",
        "app": settings.app_name,
        "version": "1.0.0",
        "backends": {
            "embedding": search.embedder.name,
            "vector_db": search.store.name,
            "llm": settings.llm_backend,
            "llm_model": settings.llm_model if settings.llm_backend == "litellm" else None,
            "asr": f"faster-whisper/{settings.whisper_model}",
            "scoring": state.scoring_service.bundle.get("kind") if state.scoring_service.bundle else "not loaded",
        },
        "catalog_size": search.catalog_size,
    }
