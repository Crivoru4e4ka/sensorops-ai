"""Pydantic-схемы модулей Voice-to-CRM (Модуль 2) и скоринга лидов (Модуль 3)."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class CRMLeadCard(BaseModel):
    """Строго типизированная карточка лида для CRM.

    Извлекается из транскрипта звонка с валидацией через Pydantic v2.
    Гарантию чистого JSON (без «мусорного» сопроводительного текста LLM)
    обеспечивает библиотека Instructor.
    """

    company_name: str | None = Field(description="Название компании клиента")
    client_name: str | None = Field(description="Имя контактного лица")
    contact_info: str | None = Field(description="Телефон или email")
    requested_products: list[str] = Field(description="Какие датчики или оборудование искали")
    urgency: str = Field(description="Срочность: низкая, средняя, критическая")
    deal_stage: str = Field(description="Текущий статус: запрос КП, повторный звонок, отказ")
    summary: str = Field(description="Краткая суть разговора в 2-3 предложениях")
    action_items: list[str] = Field(description="Что пообещал сделать менеджер")


class LeadFeatures(BaseModel):
    """Фичи ML-скоринга после обогащения профиля компании (Data Enrichment Mock)."""

    industry: str = Field(description="Сегмент индустрии (мок-обогащение по названию компании)")
    business_scale: str = Field(description="Масштаб бизнеса: малый / средний / крупный")
    urgency: Literal["низкая", "средняя", "критическая"]
    deal_stage: Literal["запрос КП", "повторный звонок", "отказ"]
    stock_availability: float = Field(ge=0.0, le=1.0, description="Доля запрошенных позиций, доступных на складе")
    order_volume: int = Field(ge=1, description="Объём запрашиваемой партии, шт.")
    has_contact: int = Field(ge=0, le=1, description="1 — контакт клиента известен")
    company_known: int = Field(ge=0, le=1, description="1 — компания идентифицирована")


class LeadScore(BaseModel):
    probability: float = Field(ge=0.0, le=1.0, description="Вероятность конверсии лида")
    priority: Literal["High Priority", "Medium", "Low"]
    recommendation: str = Field(description="Рекомендация менеджеру по скидке и скорости ответа")


class LeadScoreResponse(BaseModel):
    lead_card: CRMLeadCard
    features: LeadFeatures
    score: LeadScore


class CallAnalysisRequest(BaseModel):
    text: str | None = Field(default=None, description="Текст разговора (если аудио не загружается)")


class CallAnalysisResponse(BaseModel):
    transcript: str
    lead_card: CRMLeadCard
    score: LeadScore
    extraction_source: Literal["llm", "rules"] = "rules"
    asr_model: str | None = Field(default=None, description="Использованная модель faster-whisper")
