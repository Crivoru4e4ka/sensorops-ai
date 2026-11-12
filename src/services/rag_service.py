"""Гибридный семантический поиск по каталогу датчиков + RAG-генерация (Модуль 1).

Схема гибридного поиска:
1. **Dense Retrieval** — эмбеддинг запроса сравнивается с эмбеддингами датчиков
   (косинусная близость, расширенная выборка кандидатов);
2. **Sparse/Exact Filter** — кандидаты, не прошедшие жёсткие фильтры по
   техническим атрибутам (напряжение, тип выхода, IP, резьба, категория,
   цена, остаток), отбрасываются;
3. **Lexical boost** — штраф/бонус за совпадение токенов запроса с текстом
   датчика (артикулы, аббревиатуры типа М18/PNP/IP67);
4. **RAG** — топ-N датчиков передаётся LLM (LiteLLM) для формирования
   сравнительной подсказки менеджеру; при отсутствии LLM используется
   детерминированный mock-генератор.
"""
from __future__ import annotations

import json
import logging
from typing import Any

import numpy as np
import pandas as pd

from ..config import settings
from ..schemas.search import SearchFilters, SearchResponse, SensorHit
from .vector_db import (
    EmbeddingProvider,
    VectorStore,
    build_embedder,
    build_store,
    normalize,
    normalize_specs,
    sensor_search_text,
    tokenize,
)

logger = logging.getLogger(__name__)

DENSE_WEIGHT = 0.7
LEXICAL_WEIGHT = 0.3
CANDIDATE_MULTIPLIER = 4  # во сколько раз расширяем выборку кандидатов до фильтрации


def _spec_value(specs: dict[str, Any], *keys: str) -> str:
    """Ищет значение атрибута по нескольким возможным ключам."""
    for key in keys:
        for spec_key, value in specs.items():
            if normalize(spec_key) == normalize(key):
                return str(value)
    return ""


class HybridSearchService:
    """Сервис гибридного поиска: индексация каталога + поиск + RAG-ответ."""

    def __init__(self, embedder: EmbeddingProvider, store: VectorStore) -> None:
        self.embedder = embedder
        self.store = store
        self._rows: dict[str, dict[str, Any]] = {}

    # ─── Индексация ────────────────────────────────────────────────────────

    def index_catalog(self, catalog: pd.DataFrame) -> int:
        """Индексирует каталог датчиков в векторной БД."""
        rows = catalog.to_dict(orient="records")
        ids: list[str] = []
        texts: list[str] = []
        payloads: list[dict[str, Any]] = []
        for row in rows:
            sensor_id = str(row["id"])
            ids.append(sensor_id)
            texts.append(sensor_search_text(row))
            payloads.append(
                {
                    "id": sensor_id,
                    "name": str(row["name"]),
                    "category": str(row["category"]),
                    "specifications": normalize_specs(row.get("specifications")),
                    "in_stock": int(row.get("in_stock", 0)),
                    "price": float(row.get("price", 0.0)),
                }
            )
        self.store.reset()
        self.store.upsert(ids, texts, payloads)
        self._rows = {payload["id"]: payload for payload in payloads}
        logger.info("Проиндексировано датчиков: %d (бэкенд: %s)", len(ids), self.store.name)
        return len(ids)

    @property
    def catalog_size(self) -> int:
        return len(self._rows)

    # ─── Фильтры ───────────────────────────────────────────────────────────

    def _filter_matches(self, row: dict[str, Any], filters: SearchFilters) -> tuple[bool, str | None]:
        """Жёсткая проверка технических атрибутов. Возвращает (прошёл, причина)."""
        specs = normalize_specs(row.get("specifications"))
        if filters.category and normalize(row.get("category")) != normalize(filters.category):
            return False, f"категория {row.get('category')} != {filters.category}"
        if filters.voltage and normalize(_spec_value(specs, "voltage", "напряжение")) != normalize(filters.voltage):
            return False, f"напряжение {_spec_value(specs, 'voltage', 'напряжение')} != {filters.voltage}"
        if filters.output_type and normalize(_spec_value(specs, "output_type", "тип выхода")) != normalize(filters.output_type):
            return False, f"выход {_spec_value(specs, 'output_type', 'тип выхода')} != {filters.output_type}"
        if filters.ip_rating and normalize(_spec_value(specs, "ip_rating", "защита")) != normalize(filters.ip_rating):
            return False, f"защита {_spec_value(specs, 'ip_rating', 'защита')} != {filters.ip_rating}"
        if filters.thread and normalize(_spec_value(specs, "thread", "резьба")) != normalize(filters.thread):
            return False, f"резьба {_spec_value(specs, 'thread', 'резьба')} != {filters.thread}"
        if filters.max_price is not None and float(row.get("price", 0.0)) > filters.max_price:
            return False, f"цена {row.get('price')} > {filters.max_price}"
        if filters.min_stock is not None and int(row.get("in_stock", 0)) < filters.min_stock:
            return False, f"остаток {row.get('in_stock')} < {filters.min_stock}"
        return True, None

    @staticmethod
    def _lexical_score(query: str, text: str) -> float:
        """Доля токенов запроса, встретившихся в тексте датчика (0..1)."""
        query_tokens = set(tokenize(query))
        if not query_tokens:
            return 0.0
        text_tokens = set(tokenize(text))
        return len(query_tokens & text_tokens) / len(query_tokens)

    @staticmethod
    def _has_any_filter(filters: SearchFilters) -> bool:
        return any(
            value not in (None, "", 0)
            for value in (
                filters.category,
                filters.voltage,
                filters.output_type,
                filters.ip_rating,
                filters.thread,
                filters.max_price,
                filters.min_stock,
            )
        )

    def _match_reason(self, row: dict[str, Any], filters: SearchFilters) -> str | None:
        if not self._has_any_filter(filters):
            return None
        specs = normalize_specs(row.get("specifications"))
        reasons = []
        if filters.voltage:
            reasons.append(f"питание {_spec_value(specs, 'voltage', 'напряжение')}")
        if filters.output_type:
            reasons.append(f"выход {_spec_value(specs, 'output_type', 'тип выхода')}")
        if filters.ip_rating:
            reasons.append(f"защита {_spec_value(specs, 'ip_rating', 'защита')}")
        if filters.thread:
            reasons.append(f"резьба {_spec_value(specs, 'thread', 'резьба')}")
        return ", ".join(reasons) if reasons else None

    # ─── Поиск ─────────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        filters: SearchFilters | None = None,
        top_k: int = 5,
        with_rag: bool = True,
    ) -> SearchResponse:
        filters = filters or SearchFilters()
        if not self._rows:
            raise RuntimeError("Каталог не проиндексирован")

        query_vector = self.embedder.embed_query(query)
        # Расширенная выборка кандидатов — после неё применяются жёсткие фильтры
        candidates = self.store.search(query_vector, top_k=top_k * CANDIDATE_MULTIPLIER)

        hits: list[SensorHit] = []
        for sensor_id, dense_score in candidates:
            row = self._rows.get(sensor_id)
            if row is None:
                continue
            passed, reason = self._filter_matches(row, filters)
            if not passed:
                continue
            lexical = self._lexical_score(query, sensor_search_text(row))
            hybrid = DENSE_WEIGHT * dense_score + LEXICAL_WEIGHT * lexical
            hits.append(
                SensorHit(
                    id=row["id"],
                    name=row["name"],
                    category=row["category"],
                    specifications=row["specifications"],
                    in_stock=row["in_stock"],
                    price=row["price"],
                    score=round(float(hybrid), 4),
                    is_exact_match=self._has_any_filter(filters),
                    match_reason=self._match_reason(row, filters),
                )
            )

        hits.sort(key=lambda hit: hit.score, reverse=True)
        hits = hits[:top_k]

        if with_rag and hits:
            comparison, source = self._generate_comparison(query, hits)
        else:
            comparison, source = (
                "Ничего не найдено по заданным критериям — попробуйте смягчить фильтры.",
                "mock",
            )

        return SearchResponse(
            query=query,
            filters_applied=filters,
            hits=hits,
            comparison=comparison,
            generated_by=source,  # type: ignore[arg-type]
        )

    # ─── RAG-генерация ─────────────────────────────────────────────────────

    def _hits_context(self, hits: list[SensorHit]) -> str:
        items = [
            {
                "артикул": hit.id,
                "название": hit.name,
                "категория": hit.category,
                "параметры": hit.specifications,
                "остаток": hit.in_stock,
                "цена": hit.price,
                "релевантность": hit.score,
            }
            for hit in hits
        ]
        return json.dumps(items, ensure_ascii=False, indent=2)

    def _generate_comparison(self, query: str, hits: list[SensorHit]) -> tuple[str, str]:
        """Генерирует сравнительную подсказку через LLM; при сбое — mock."""
        if settings.llm_backend == "litellm" and (settings.llm_api_key or settings.llm_api_base):
            try:
                return self._generate_comparison_llm(query, hits), "llm"
            except Exception as exc:  # noqa: BLE001
                logger.warning("LLM-генерация не удалась (%s), использую mock", exc)
        return self._generate_comparison_mock(query, hits), "mock"

    def _generate_comparison_llm(self, query: str, hits: list[SensorHit]) -> str:
        import litellm

        system_prompt = (
            "Ты — инженер-консультант дистрибьютора промышленной электроники. "
            "На основе запроса клиента и найденных датчиков составь краткую "
            "сравнительную подсказку для менеджера по продажам (на русском языке): "
            "почему каждый датчик подходит, чем они отличаются друг от друга, "
            "какие есть альтернативы и что уточнить у клиента. Отвечай строго "
            "по данным из найденных позиций, без выдуманных параметров."
        )
        user_prompt = (
            f"Запрос клиента: {query}\n\n"
            f"Найденные датчики (JSON):\n{self._hits_context(hits)}"
        )
        kwargs: dict[str, Any] = {
            "model": settings.llm_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": settings.llm_temperature,
            "timeout": settings.llm_timeout,
        }
        if settings.llm_api_base:
            kwargs["api_base"] = settings.llm_api_base
        if settings.llm_api_key:
            kwargs["api_key"] = settings.llm_api_key
        response = litellm.completion(**kwargs)
        return str(response.choices[0].message.content).strip()

    def _generate_comparison_mock(self, query: str, hits: list[SensorHit]) -> str:
        """Детерминированный RAG-ответ по найденным датчикам (без LLM)."""
        lines = [f"Запрос клиента: «{query}»", "", "Рекомендуемые позиции:", ""]
        for index, hit in enumerate(hits, start=1):
            specs = ", ".join(f"{key}: {value}" for key, value in hit.specifications.items())
            stock = f"в наличии {hit.in_stock} шт." if hit.in_stock else "под заказ"
            reason = f" — подходит по: {hit.match_reason}" if hit.match_reason else ""
            lines.append(
                f"{index}. **{hit.name}** (арт. {hit.id}, {hit.category}). "
                f"Параметры: {specs}. Цена {hit.price:,.0f} ₽, {stock}{reason}."
            )
        if len(hits) > 1:
            cheapest = min(hits, key=lambda hit: hit.price)
            biggest_stock = max(hits, key=lambda hit: hit.in_stock)
            lines.extend(
                [
                    "",
                    f"**Альтернативы:** если критичен бюджет — {cheapest.name} "
                    f"(арт. {cheapest.id}, {cheapest.price:,.0f} ₽); если нужна немедленная "
                    f"отгрузка — {biggest_stock.name} (арт. {biggest_stock.id}, остаток {biggest_stock.in_stock} шт.).",
                ]
            )
        lines.extend(
            [
                "",
                "Рекомендация менеджеру: подтвердить у клиента условия эксплуатации "
                "(среда, температура, расстояние срабатывания) перед выставлением КП.",
            ]
        )
        return "\n".join(lines)


# ─── Глобальный сервис (инициализируется при старте API) ────────────────────────

_service: HybridSearchService | None = None


def get_search_service(catalog: pd.DataFrame | None = None) -> HybridSearchService:
    """Ленивая инициализация глобального сервиса гибридного поиска.

    При ``EMBEDDING_BACKEND=auto`` и сбое sentence-transformers на этапе
    индексации (например, не скачалась модель) автоматически переключается
    на TF-IDF fallback — PoC остаётся рабочим без сети и GPU.
    """
    global _service
    if _service is not None:
        return _service
    corpus = (
        catalog.apply(lambda row: sensor_search_text(row.to_dict()), axis=1).tolist()
        if catalog is not None
        else None
    )
    embedder, embed_backend = build_embedder(corpus)
    store, store_backend = build_store(embedder)
    logger.info("Эмбеддинги: %s; векторная БД: %s", embed_backend, store_backend)
    _service = HybridSearchService(embedder, store)
    if catalog is not None and not catalog.empty:
        try:
            _service.index_catalog(catalog)
        except Exception as exc:  # noqa: BLE001
            if settings.embedding_backend != "auto":
                raise
            logger.warning(
                "sentence-transformers не смог обработать каталог (%s) — переключаюсь на TF-IDF fallback", exc
            )
            embedder, embed_backend = build_embedder(corpus)
            store, store_backend = build_store(embedder)
            _service = HybridSearchService(embedder, store)
            _service.index_catalog(catalog)
    return _service
