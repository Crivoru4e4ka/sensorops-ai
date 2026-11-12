"""Глобальное состояние приложения: каталог, сервисы поиска и скоринга.

Состояние создаётся один раз при старте FastAPI (lifespan) и инжектируется
в маршруты через ``Depends(get_state)``.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .config import Settings, settings
from .services.rag_service import HybridSearchService, get_search_service
from .services.scoring_service import ScoringService


@dataclass
class AppState:
    settings: Settings = field(default_factory=lambda: settings)
    catalog: pd.DataFrame = field(default_factory=pd.DataFrame)
    search_service: HybridSearchService | None = None
    scoring_service: ScoringService = field(default_factory=ScoringService)


_STATE: AppState | None = None


def build_state(catalog: pd.DataFrame | None = None) -> AppState:
    """Инициализирует состояние: каталог → индекс → модель скоринга."""
    global _STATE
    if catalog is None:
        catalog = pd.read_csv(settings.catalog_path, dtype={"id": str})
    state = AppState(catalog=catalog)
    state.search_service = get_search_service(catalog)
    state.scoring_service.load_or_train()
    _STATE = state
    return state


def get_state() -> AppState:
    if _STATE is None:
        return build_state()
    return _STATE
