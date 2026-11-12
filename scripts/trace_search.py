"""Трассировка запроса гибридного поиска: во что превращается запрос
и что именно сравнивается с данными в векторной БД.

Запуск:  python scripts/trace_search.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Корень проекта в sys.path, чтобы скрипт можно было запускать напрямую
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.config import settings
from src.schemas.search import SearchFilters
from src.services.rag_service import (
    DENSE_WEIGHT,
    LEXICAL_WEIGHT,
    HybridSearchService,
    get_search_service,
    sensor_search_text,
)
from src.services.vector_db import normalize, tokenize

QUERY = "Нужен оптический датчик приближения с защитой от влаги, резьба М18, выход PNP, питание 24 вольта"
FILTERS = SearchFilters(category="оптический", voltage="24В", output_type="PNP", thread="М18")


def main() -> None:
    print("=" * 78)
    print("ШАГ 1. HTTP-запрос → Pydantic-модель SearchRequest")
    print("=" * 78)
    print(f"  query:   «{QUERY}»")
    print(f"  filters: {FILTERS.model_dump()}")
    print(f"  top_k:   3")

    catalog = pd.read_csv(settings.catalog_path, dtype={"id": str})
    service: HybridSearchService = get_search_service(catalog)

    print()
    print("=" * 78)
    print(f"ШАГ 2. Запрос → dense-вектор (эмбеддинг)")
    print(f"  бэкенд эмбеддингов: {service.embedder.name}")
    print("=" * 78)
    query_vector = service.embedder.embed_query(QUERY)
    print(f"  размерность: {len(query_vector)}")
    print(f"  вектор L2-нормализован: |v| = {float((query_vector ** 2).sum()) ** 0.5:.4f}")
    print(f"  первые значения: {[round(float(x), 4) for x in query_vector[:8]]} ...")

    print()
    print("=" * 78)
    print("ШАГ 3. Dense Retrieval: косинусная близость вектора запроса")
    print("       с векторами ВСЕХ 72 датчиков в БД, берём top_k*4 = 12 кандидатов")
    print("=" * 78)
    candidates = service.store.search(query_vector, top_k=3 * 4)
    print(f"  топ-6 кандидатов до фильтров:")
    for sensor_id, score in candidates[:6]:
        row = service._rows[sensor_id]
        print(f"    {row['id']:9s} dense={score:+.4f}  {row['name']}")

    print()
    print("=" * 78)
    print("ШАГ 4. Sparse/Exact Filter: жёсткое сравнение полей фильтра")
    print("       с полями specifications каждого кандидата (нормализация: lowercase,")
    print("       без пробелов/неразрывных пробелов)")
    print("=" * 78)
    print(f"  фильтры (нормализованные):")
    for key, value in [
        ("category", FILTERS.category),
        ("voltage", FILTERS.voltage),
        ("output_type", FILTERS.output_type),
        ("thread", FILTERS.thread),
    ]:
        print(f"    {key:12s} = {normalize(value)!r}")
    print()
    for sensor_id, _ in candidates[:6]:
        row = service._rows[sensor_id]
        passed, reason = service._filter_matches(row, FILTERS)
        specs = row["specifications"]
        print(
            f"    {row['id']:9s} category={row['category']!r:14s} "
            f"voltage={specs['voltage']!r:9s} output={specs['output_type']!r:10s} "
            f"thread={specs['thread']!r:9s} → {'ПРОШЁЛ' if passed else 'ОТСЕЯН: ' + (reason or '')}"
        )

    print()
    print("=" * 78)
    print("ШАГ 5. Lexical boost: пересечение токенов запроса с текстом датчика")
    print("=" * 78)
    query_tokens = set(tokenize(QUERY))
    print(f"  токены запроса: {sorted(query_tokens)}")
    print()
    for sensor_id, dense in candidates[:4]:
        row = service._rows[sensor_id]
        text = sensor_search_text(row)
        text_tokens = set(tokenize(text))
        overlap = query_tokens & text_tokens
        lexical = service._lexical_score(QUERY, text)
        print(
            f"    {row['id']:9s} пересечение ({len(overlap)} ток.): "
            f"{sorted(overlap)[:12]}"
            f" → lexical = {len(overlap)}/{len(query_tokens)} = {lexical:.3f}"
        )

    print()
    print("=" * 78)
    print("ШАГ 6. Гибридный скор и ранжирование")
    print(f"  score = {DENSE_WEIGHT}*dense + {LEXICAL_WEIGHT}*lexical")
    print("=" * 78)
    response = service.search(QUERY, filters=FILTERS, top_k=3, with_rag=False)
    for hit in response.hits:
        print(f"    {hit.id:9s} score={hit.score:.4f}  {hit.name}  (stock={hit.in_stock}, price={hit.price})")

    print()
    print("=" * 78)
    print("ШАГ 7. Ответ API (без RAG-части)")
    print("=" * 78)
    payload = {
        "query": response.query,
        "filters_applied": response.filters_applied.model_dump(),
        "hits": [
            {
                "id": hit.id,
                "name": hit.name,
                "score": hit.score,
                "is_exact_match": hit.is_exact_match,
                "match_reason": hit.match_reason,
                "in_stock": hit.in_stock,
                "price": hit.price,
            }
            for hit in response.hits
        ],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2)[:1500])


if __name__ == "__main__":
    main()
