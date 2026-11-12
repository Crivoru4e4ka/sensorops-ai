"""Тесты Модуля 1: гибридный семантический поиск."""
from __future__ import annotations

from src.schemas.search import SearchFilters, SearchRequest


def test_plain_semantic_query_returns_hits(search_service) -> None:
    response = search_service.search(
        "Нужен оптический датчик приближения с защитой от влаги, резьба М18, выход PNP, питание 24 вольта",
        top_k=5,
    )
    assert response.hits, "Поиск должен возвращать результаты"
    assert all(0.0 <= hit.score <= 1.0 for hit in response.hits)
    assert response.comparison, "RAG-подсказка должна быть сгенерирована"


def test_exact_filters_are_hard(search_service) -> None:
    filters = SearchFilters(category="индуктивный", voltage="24В", output_type="PNP", thread="М18")
    response = search_service.search("датчик индуктивный", filters=filters, top_k=5)
    assert response.hits, "По обязательным атрибутам должны найтись датчики"
    for hit in response.hits:
        assert hit.category == "индуктивный"
        assert hit.specifications.get("voltage") == "24В"
        assert hit.specifications.get("output_type") == "PNP"
        assert hit.specifications.get("thread") == "М18"
        assert hit.is_exact_match is True


def test_impossible_filter_combination_empty(search_service) -> None:
    filters = SearchFilters(category="давления", thread="М18", voltage="12В", max_price=1000)
    response = search_service.search("датчик давления", filters=filters, top_k=5)
    assert response.hits == [], "Несовместимые фильтры должны давать пустую выдачу"
    assert "не найдено" in response.comparison.lower()


def test_max_price_filter(search_service) -> None:
    response = search_service.search(
        "датчик температуры", filters=SearchFilters(max_price=5000), top_k=10
    )
    assert response.hits
    assert all(hit.price <= 5000 for hit in response.hits)


def test_min_stock_filter(search_service) -> None:
    response = search_service.search(
        "датчик давления", filters=SearchFilters(min_stock=50), top_k=10
    )
    assert all(hit.in_stock >= 50 for hit in response.hits)


def test_results_sorted_by_score(search_service) -> None:
    response = search_service.search("оптический датчик PNP 24В", top_k=5)
    scores = [hit.score for hit in response.hits]
    assert scores == sorted(scores, reverse=True), "Выдача должна быть отсортирована по релевантности"


def test_top_k_limits_results(search_service) -> None:
    for k in (1, 3):
        response = search_service.search("датчик", top_k=k)
        assert len(response.hits) <= k


def test_lexical_boost_for_technical_terms(search_service) -> None:
    """Запрос с артикулоподобными терминами (М18, PNP) должен повышать рейтинг."""
    plain = search_service.search("датчик индуктивный", top_k=5, with_rag=False)
    precise = search_service.search(
        "датчик индуктивный М18 PNP", top_k=5, with_rag=False
    )
    assert plain.hits and precise.hits
    top_plain_id = plain.hits[0].id
    top_precise_id = precise.hits[0].id
    # Как минимум выдачу не должны возглавлять «случайные» позиции:
    # точные термины обязаны встречаться у топ-1 в выдаче precise
    top_hit = precise.hits[0]
    assert "М18" in str(top_hit.specifications.get("thread", "")) or "PNP" in str(
        top_hit.specifications.get("output_type", "")
    )
