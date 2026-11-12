"""REST API модуля гибридного семантического поиска (Модуль 1)."""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException

from ..schemas.search import SearchRequest, SearchResponse
from ..state import AppState, get_state

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/search", tags=["search"])


@router.post("", response_model=SearchResponse, summary="Гибридный поиск датчиков")
async def hybrid_search(
    request: SearchRequest,
    state: AppState = Depends(get_state),
) -> SearchResponse:
    """Семантический поиск по запросу клиента + жёсткие фильтры по атрибутам + RAG-подсказка."""
    try:
        return state.search_service.search(
            query=request.query,
            filters=request.filters,
            top_k=request.top_k,
            with_rag=request.with_rag,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/index/rebuild", summary="Переиндексация каталога")
async def rebuild_index(state: AppState = Depends(get_state)) -> dict:
    """Перечитывает sensors_catalog.csv и переиндексирует векторную БД."""
    import pandas as pd

    catalog = pd.read_csv(state.settings.catalog_path, dtype={"id": str})
    count = state.search_service.index_catalog(catalog)
    return {"indexed": count, "backend": state.search_service.store.name}


@router.get("/stats", summary="Статистика поискового индекса")
async def search_stats(state: AppState = Depends(get_state)) -> dict:
    return {
        "catalog_size": state.search_service.catalog_size,
        "embedding_backend": state.search_service.embedder.name,
        "vector_backend": state.search_service.store.name,
    }
