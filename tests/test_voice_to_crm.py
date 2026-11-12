"""Тесты Модуля 2: Voice-to-CRM (rule-based экстракция)."""
from __future__ import annotations

from src.services.asr_service import extract_lead_card
from src.services.scoring_service import extract_order_volume

DIALOG_URGENT = """
Менеджер: Компания ПромТех, менеджер Сергей, слушаю вас.
Клиент: Здравствуйте, это Дмитрий из ПромСтройМаш. Нам срочно нужны индуктивные датчики, резьба М18, выход PNP, питание двадцать четыре вольта.
Менеджер: Какое количество требуется?
Клиент: Двести штук, желательно с запасом на складе. Поставка нужна на этой неделе.
Менеджер: Хорошо. Посмотрю наличие и сегодня отправлю вам коммерческое предложение на почту.
Клиент: Отлично. Мой адрес: Дмитрий точка Петров, собака промстроймаш точка ру.
"""

DIALOG_REFUSAL = """
Менеджер: Компания ПромТех, Олег слушает.
Клиент: Добрый день, Игорь из компании МойкаПро. Мы искали оптические датчики с защитой IP шестьдесят девять.
Менеджер: Сколько штук вам нужно?
Клиент: Ваш конкурент предложил цену ниже. Мы пока отказываемся, не нужно.
"""


def test_company_and_client_extracted() -> None:
    card, source = extract_lead_card(DIALOG_URGENT)
    assert source == "rules"
    assert card.company_name == "ПромСтройМаш", card.company_name
    assert card.client_name == "Дмитрий", card.client_name


def test_client_company_priority_over_manager_company() -> None:
    """Компания клиента («меня зовут X, компания Y») важнее компании менеджера («компания Z, слушаю»)."""
    text = (
        "Менеджер: Добрый день, компания ПромТех, Елена слушает. "
        "Клиент: Здравствуйте, меня зовут Марина, компания АгроПищеПром. "
        "Подбираем датчики температуры и давления."
    )
    card, _ = extract_lead_card(text)
    assert card.company_name == "АгроПищеПром", card.company_name
    assert card.client_name == "Марина", card.client_name


def test_company_from_iz_phrase() -> None:
    text = "Менеджер: Компания ПромТех, Олег слушает. Клиент: Добрый день, Игорь из компании МойкаПро. Искали датчики."
    card, _ = extract_lead_card(text)
    assert card.company_name == "МойкаПро", card.company_name
    assert card.client_name == "Игорь", card.client_name


def test_contact_info_extracted() -> None:
    card, _ = extract_lead_card(DIALOG_URGENT)
    assert card.contact_info and "промстроймаш" in card.contact_info.lower()


def test_urgency_and_stage() -> None:
    card, _ = extract_lead_card(DIALOG_URGENT)
    assert card.urgency == "критическая"
    assert card.deal_stage == "запрос КП"


def test_requested_products_detected() -> None:
    card, _ = extract_lead_card(DIALOG_URGENT)
    joined = " ".join(card.requested_products).lower()
    assert "индуктив" in joined


def test_action_items_captured() -> None:
    card, _ = extract_lead_card(DIALOG_URGENT)
    assert card.action_items, "Должны быть зафиксированы обещания менеджера"
    assert any("отправлю" in item.lower() for item in card.action_items)


def test_refusal_stage_detected() -> None:
    card, _ = extract_lead_card(DIALOG_REFUSAL)
    assert card.deal_stage == "отказ"


def test_schema_validation_always_valid() -> None:
    """Карточка всегда проходит валидацию Pydantic v2."""
    for text in (DIALOG_URGENT, DIALOG_REFUSAL, "", "Короткий разговор без деталей."):
        card, _ = extract_lead_card(text)
        assert card.urgency in {"низкая", "средняя", "критическая"}
        assert card.deal_stage in {"запрос КП", "повторный звонок", "отказ"}
        assert isinstance(card.requested_products, list)
        assert isinstance(card.action_items, list)


def test_order_volume_extraction() -> None:
    assert extract_order_volume("Нужно двести штук датчиков") == 200
    assert extract_order_volume("Заказ на 350 позиций") == 350
    assert extract_order_volume("Партия пятьдесят штук") == 50
    assert extract_order_volume("Просто интересуются") == 1
