"""Тесты Модуля 3: B2B-профилирование и скоринг лидов."""
from __future__ import annotations

from src.schemas.lead import CRMLeadCard
from src.services.scoring_service import ScoringService, enrich_company, extract_order_volume


def _card(**overrides) -> CRMLeadCard:
    data = dict(
        company_name="ПромСтройМаш",
        client_name="Дмитрий Петров",
        contact_info="d.petrov@promstroymash.ru",
        requested_products=["датчик индуктивный"],
        urgency="критическая",
        deal_stage="запрос КП",
        summary="Клиенту срочно нужны индуктивные датчики, 200 штук на этой неделе.",
        action_items=["Отправить КП"],
    )
    data.update(overrides)
    return CRMLeadCard(**data)


def test_enrichment_is_deterministic() -> None:
    first = enrich_company("ПромСтройМаш")
    second = enrich_company("ПромСтройМаш")
    assert first == second, "Мок-обогащение должно быть детерминированным"
    assert first["industry"] in {
        "машиностроение", "пищепром", "нефтегаз", "ритейл",
        "энергетика", "логистика", "строительство",
    }
    assert first["business_scale"] in {"малый", "средний", "крупный"}


def test_enrichment_keyword_hints() -> None:
    assert enrich_company("ООО НефтеГазСервис")["industry"] == "нефтегаз"
    assert enrich_company("АгроПищеПром")["industry"] == "пищепром"


def test_model_loads_or_trains(scoring_service: ScoringService) -> None:
    assert scoring_service.bundle is not None
    assert scoring_service.bundle["model"] is not None
    assert scoring_service.bundle["kind"] in {"catboost", "sklearn.GradientBoostingClassifier"}


def test_score_probability_in_range(scoring_service: ScoringService, catalog_df) -> None:
    score = scoring_service.score(_card(), catalog_df)
    assert 0.0 <= score.probability <= 1.0
    assert score.priority in {"High Priority", "Medium", "Low"}
    assert score.recommendation


def test_urgent_lead_scores_higher_than_cold(scoring_service: ScoringService, catalog_df) -> None:
    hot = scoring_service.score(_card(urgency="критическая"), catalog_df)
    cold = scoring_service.score(
        _card(urgency="низкая", contact_info=None, company_name=None, requested_products=["экзотика"]),
        catalog_df,
    )
    assert hot.probability > cold.probability, (
        f"Срочный лид с контактом ({hot.probability:.3f}) должен быть выше холодного ({cold.probability:.3f})"
    )


def test_stock_availability_feature(catalog_df) -> None:
    service = ScoringService()
    service.load_or_train()
    card = _card(requested_products=["датчик индуктивный"])
    features = service.build_features(card, catalog_df)
    assert 0.0 <= features.stock_availability <= 1.0
    assert features.has_contact == 1
    assert features.company_known == 1
    assert features.order_volume == 200


def test_priority_mapping(scoring_service: ScoringService, catalog_df) -> None:
    score = scoring_service.score(_card(), catalog_df)
    if score.probability >= 0.75:
        assert score.priority == "High Priority"
    elif score.probability >= 0.4:
        assert score.priority == "Medium"
    else:
        assert score.priority == "Low"


def test_refusal_lead_scores_low(scoring_service: ScoringService, catalog_df) -> None:
    """Явный отказ не должен получать высокий приоритет."""
    refusal = scoring_service.score(_card(deal_stage="отказ"), catalog_df)
    active = scoring_service.score(_card(), catalog_df)
    assert refusal.probability < active.probability, (
        f"Отказ ({refusal.probability:.3f}) должен быть ниже активного лида ({active.probability:.3f})"
    )
    assert refusal.priority == "Low"


def test_feature_order_consistency(scoring_service: ScoringService) -> None:
    """Порядок числовых фич в модели обязан совпадать с сервисом скоринга."""
    from src.services.scoring_service import NUMERIC_FEATURES

    assert scoring_service.bundle["columns"][: len(NUMERIC_FEATURES)] == NUMERIC_FEATURES
