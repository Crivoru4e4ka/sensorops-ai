"""Интеграционные тесты REST API (FastAPI + TestClient)."""
from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from src.main import app


@pytest.fixture(scope="module")
def client() -> TestClient:
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client: TestClient) -> None:
    response = client.get("/api/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["catalog_size"] >= 50


def test_search_endpoint(client: TestClient) -> None:
    response = client.post(
        "/api/search",
        json={
            "query": "Нужен оптический датчик с защитой от влаги, резьба М18, выход PNP, питание 24 вольта",
            "top_k": 3,
            "filters": {"category": "оптический", "voltage": "24В"},
        },
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["hits"], "Должны найтись оптические датчики 24В"
    for hit in payload["hits"]:
        assert hit["category"] == "оптический"
        assert hit["specifications"]["voltage"] == "24В"
    assert payload["comparison"]


def test_search_validation_error(client: TestClient) -> None:
    response = client.post("/api/search", json={"query": "ab", "top_k": 3})
    assert response.status_code == 422  # min_length=3


def test_score_endpoint(client: TestClient) -> None:
    card = {
        "company_name": "ПромСтройМаш",
        "client_name": "Дмитрий Петров",
        "contact_info": "d.petrov@promstroymash.ru",
        "requested_products": ["датчик индуктивный"],
        "urgency": "критическая",
        "deal_stage": "запрос КП",
        "summary": "Срочно нужны индуктивные датчики, 200 штук.",
        "action_items": ["Отправить КП"],
    }
    response = client.post("/api/calls/score", json=card)
    assert response.status_code == 200, response.text
    payload = response.json()
    assert 0.0 <= payload["score"]["probability"] <= 1.0
    assert payload["features"]["industry"]
    assert payload["features"]["urgency"] == "критическая"


def test_analyze_call_with_text(client: TestClient) -> None:
    text = (
        "Менеджер: Компания ПромТех, слушаю вас. "
        "Клиент: Здравствуйте, это Дмитрий из ПромСтройМаш, нам срочно нужны индуктивные "
        "датчики М18 PNP 24 вольта, двести штук. "
        "Менеджер: Хорошо, сегодня отправлю вам коммерческое предложение."
    )
    response = client.post("/api/calls/analyze-text", json={"text": text})
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["lead_card"]["company_name"] == "ПромСтройМаш"
    assert payload["lead_card"]["urgency"] == "критическая"
    assert payload["score"]["probability"] >= 0.0
    assert payload["transcript"] == text


def test_analyze_call_without_input_rejected(client: TestClient) -> None:
    response = client.post("/api/calls/analyze-text", json={"text": None})
    assert response.status_code == 422


def test_transcribe_rejects_unsupported_format(client: TestClient) -> None:
    response = client.post(
        "/api/calls/transcribe",
        files={"file": ("notes.txt", b"not audio", "text/plain")},
    )
    assert response.status_code == 415


def test_enrich_endpoint(client: TestClient) -> None:
    response = client.post("/api/calls/enrich", data={"company_name": "ООО НефтеГазСервис"})
    assert response.status_code == 200
    assert response.json()["industry"] == "нефтегаз"


def test_search_stats(client: TestClient) -> None:
    response = client.get("/api/search/stats")
    assert response.status_code == 200
    payload = response.json()
    assert payload["catalog_size"] >= 50
    assert payload["vector_backend"] in {"memory", "chroma", "qdrant"}
