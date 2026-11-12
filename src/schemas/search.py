"""Pydantic-схемы модуля гибридного семантического поиска (Модуль 1)."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class SearchFilters(BaseModel):
    """Жёсткие (sparse/exact) фильтры по техническим атрибутам.

    Все заполненные поля применяются как строгие условия к ``specifications``
    датчика — кандидат, не прошедший фильтр, в выдачу не попадает.
    """

    category: str | None = Field(default=None, description="Тип: индуктивный, оптический, давления, температуры")
    voltage: str | None = Field(default=None, description="Напряжение питания, например '24В'")
    output_type: str | None = Field(default=None, description="Тип выхода: PNP / NPN / аналоговый / RS-485")
    ip_rating: str | None = Field(default=None, description="Класс защиты, например 'IP67'")
    thread: str | None = Field(default=None, description="Резьба корпуса, например 'М18'")
    max_price: float | None = Field(default=None, ge=0, description="Максимальная цена, руб.")
    min_stock: int | None = Field(default=None, ge=0, description="Минимальный остаток на складе")


class SearchRequest(BaseModel):
    query: str = Field(min_length=3, description="Запрос клиента свободным текстом")
    filters: SearchFilters = Field(default_factory=SearchFilters)
    top_k: int = Field(default=5, ge=1, le=20)
    with_rag: bool = Field(default=True, description="Сгенерировать сравнительную подсказку для менеджера")


class SensorHit(BaseModel):
    id: str
    name: str
    category: str
    specifications: dict[str, Any]
    in_stock: int
    price: float
    score: float = Field(description="Гибридный рейтинг релевантности 0..1")
    is_exact_match: bool = Field(description="True, если пройдены все заданные жёсткие фильтры")
    match_reason: str | None = Field(default=None, description="Человекочитаемое объяснение попадания в выдачу")


class SearchResponse(BaseModel):
    query: str
    filters_applied: SearchFilters
    hits: list[SensorHit]
    comparison: str = Field(description="Сравнительная подсказка от RAG/LLM: почему датчик подходит и какие альтернативы")
    generated_by: Literal["llm", "mock"] = "mock"
