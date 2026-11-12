"""B2B-профилирование и скоринг сделки — Lead Scoring (Модуль 3).

Пайплайн:
1. **Data Enrichment Mock** — по названию компании детерминированно
   генерируются фичи: сегмент индустрии и масштаб бизнеса;
2. **ML-скоринг** — модель (CatBoost, если установлен, иначе Scikit-learn)
   оценивает вероятность конверсии лида 0..1 по фичам:
   отрасль, масштаб, срочность, наличие товара на складе (из Модуля 1),
   объём партии, наличие контакта, известность компании;
3. **Вывод** — бейдж High Priority / Medium / Low + рекомендация менеджеру.

Модель обучается скриптом ``scripts/train_scoring.py`` на синтетическом
датасете ``data/leads_training.csv``; артефакт (joblib) сохраняется в
``data/models/scoring_model.joblib``. Если артефакт отсутствует, сервис
обучит модель автоматически при первом запуске.
"""
from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from ..config import settings
from ..schemas.lead import CRMLeadCard, LeadFeatures, LeadScore

logger = logging.getLogger(__name__)

INDUSTRIES = [
    "машиностроение", "пищепром", "нефтегаз", "ритейл",
    "энергетика", "логистика", "строительство",
]
SCALES = ["малый", "средний", "крупный"]
URGENCY_LEVEL = {"низкая": 0, "средняя": 1, "критическая": 2}
DEAL_STAGE_LEVEL = {"отказ": 0, "повторный звонок": 1, "запрос КП": 2}

# Порядок числовых фич: ДОЛЖЕН совпадать в scoring_service и scripts/train_scoring.py
NUMERIC_FEATURES = [
    "deal_stage_level", "urgency_level", "stock_availability",
    "log_order_volume", "has_contact", "company_known",
]

_HIGH_PRIORITY_THRESHOLD = 0.75
_MEDIUM_PRIORITY_THRESHOLD = 0.4

_VOLUME_RE = re.compile(
    r"(\d+)\s*(?:штук\w*|шт\.?|датчик\w*|единиц\w*|позиций\w*)", re.IGNORECASE
)

_INDUSTRY_HINTS = [
    (r"нефте|газ|петро|oil|gas", "нефтегаз"),
    (r"пищ|агро|молок|хлеб|фуд|мяс|food|agro", "пищепром"),
    (r"ритейл|торг|магаз|сеть|retail", "ритейл"),
    (r"энерг|электро|свет|энерго", "энергетика"),
    (r"лог|транс|перевоз|склад|логист", "логистика"),
    (r"стро|монтаж|строй", "строительство"),
    (r"маш|станк|завод|пром|инж|авто|машиностро", "машиностроение"),
]


def _stable_hash(text: str) -> int:
    return int(hashlib.md5(text.encode("utf-8")).hexdigest(), 16)


def enrich_company(company_name: str | None) -> dict[str, str]:
    """Мок-обогащение профиля компании (детерминированное по названию).

    Сначала ищутся ключевые слова в названии, иначе индустрия и масштаб
    выбираются стабильным хешем — повторные вызовы дают одинаковый результат.
    """
    if not company_name:
        key = "unknown"
    else:
        key = company_name.strip().lower()

    industry = "машиностроение"
    for pattern, value in _INDUSTRY_HINTS:
        if re.search(pattern, key):
            industry = value
            break
    if industry == "машиностроение":
        digest = _stable_hash(key)
        industry = INDUSTRIES[digest % len(INDUSTRIES)]

    scale = SCALES[_stable_hash(key + ":scale") % len(SCALES)]
    return {"industry": industry, "business_scale": scale}


def extract_order_volume(text: str) -> int:
    """Извлекает объём партии из текста (например, «двести штук» → 200)."""
    matches = _VOLUME_RE.findall(text or "")
    if matches:
        return max(int(value) for value in matches)
    # Словами: двести, триста, пятьдесят... (по границам слов, чтобы «просто» не ловилось как «сто»)
    numbers_ru = {
        "десять": 10, "двадцать": 20, "тридцать": 30, "сорок": 40, "пятьдесят": 50,
        "шестьдесят": 60, "семьдесят": 70, "восемьдесят": 80, "девяносто": 90,
        "сто": 100, "двести": 200, "триста": 300, "четыреста": 400, "пятьсот": 500,
        "тысяча": 1000, "тысячи": 1000, "тысяч": 1000, "полсотни": 50, "сотня": 100,
    }
    lowered = (text or "").lower()
    found = [
        value
        for word, value in numbers_ru.items()
        if re.search(rf"\b{re.escape(word)}\b", lowered)
    ]
    return max(found, default=1)


def _requested_product_hits(card: CRMLeadCard, catalog: pd.DataFrame) -> list[dict[str, Any]]:
    """Сопоставляет запрошенные продукты со строками каталога (по названию/категории)."""
    if catalog is None or catalog.empty:
        return []
    hits: dict[str, dict[str, Any]] = {}
    for product in card.requested_products:
        lowered = product.lower()
        match = catalog[
            catalog.apply(
                lambda row: (
                    lowered in str(row["name"]).lower()
                    or str(row["category"]).lower() in lowered
                ),
                axis=1,
            )
        ]
        if not match.empty:
            best = match.sort_values("in_stock", ascending=False).iloc[0]
            hits[product] = {
                "found": True,
                "in_stock": bool(best["in_stock"] > 0),
                "price": float(best["price"]),
            }
        else:
            hits[product] = {"found": False, "in_stock": False, "price": 0.0}
    return list(hits.values())


class ScoringService:
    """Сервис ML-скоринга лидов."""

    def __init__(self) -> None:
        self.bundle: dict[str, Any] | None = None

    # ─── Модель ────────────────────────────────────────────────────────────

    def load_or_train(self) -> None:
        model_path = settings.resolve(settings.scoring_model_path)
        if model_path.exists():
            self.bundle = joblib.load(model_path)
            logger.info("Модель скоринга загружена: %s (%s)", model_path, self.bundle.get("kind"))
            return
        logger.warning("Модель скоринга не найдена (%s) — обучаю на синтетических данных", model_path)
        try:
            from scripts.train_scoring import train_model  # отсроченный импорт
        except ImportError:  # scripts/ может быть недоступен — добавляем корень в sys.path
            import sys

            from ..config import PROJECT_ROOT

            if str(PROJECT_ROOT) not in sys.path:
                sys.path.insert(0, str(PROJECT_ROOT))
            from scripts.train_scoring import train_model

        bundle = train_model()
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(bundle, model_path)
        logger.info("Модель скоринга обучена и сохранена: %s", model_path)
        self.bundle = bundle

    # ─── Фичи ──────────────────────────────────────────────────────────────

    def build_features(self, card: CRMLeadCard, catalog: pd.DataFrame | None = None) -> LeadFeatures:
        enrichment = enrich_company(card.company_name)
        hits = _requested_product_hits(card, catalog if catalog is not None else pd.DataFrame())
        available = sum(1 for hit in hits if hit["in_stock"]) if hits else 0
        stock_availability = available / len(hits) if hits else 0.0

        urgency = card.urgency.lower()
        if urgency not in URGENCY_LEVEL:
            urgency = "низкая"

        deal_stage = card.deal_stage.lower()
        if deal_stage not in DEAL_STAGE_LEVEL:
            deal_stage = "запрос КП"

        order_volume = max(extract_order_volume(card.summary), 1)

        return LeadFeatures(
            industry=enrichment["industry"],
            business_scale=enrichment["business_scale"],
            urgency=urgency,  # type: ignore[arg-type]
            deal_stage=deal_stage,  # type: ignore[arg-type]
            stock_availability=round(stock_availability, 3),
            order_volume=order_volume,
            has_contact=1 if card.contact_info else 0,
            company_known=1 if card.company_name else 0,
        )

    # ─── Скоринг ───────────────────────────────────────────────────────────

    def score(self, card: CRMLeadCard, catalog: pd.DataFrame | None = None) -> LeadScore:
        if self.bundle is None:
            self.load_or_train()
        features = self.build_features(card, catalog)

        row = self._features_to_row(features)
        model = self.bundle["model"]
        probability = float(model.predict_proba(row)[0][1])

        if probability >= _HIGH_PRIORITY_THRESHOLD:
            priority = "High Priority"
        elif probability >= _MEDIUM_PRIORITY_THRESHOLD:
            priority = "Medium"
        else:
            priority = "Low"

        return LeadScore(
            probability=round(probability, 4),
            priority=priority,
            recommendation=self._recommendation(priority, features),
        )

    def _features_to_row(self, features: LeadFeatures) -> pd.DataFrame:
        from sklearn.preprocessing import OneHotEncoder, StandardScaler

        encoder: OneHotEncoder = self.bundle["encoder"]
        scaler: StandardScaler = self.bundle["scaler"]

        categorical = encoder.transform(
            pd.DataFrame(
                [[features.industry, features.business_scale]],
                columns=["industry", "business_scale"],
            )
        )
        if hasattr(categorical, "toarray"):
            categorical = categorical.toarray()
        numeric = scaler.transform(
            np.array(
                [
                    [
                        DEAL_STAGE_LEVEL[features.deal_stage],
                        URGENCY_LEVEL[features.urgency],
                        features.stock_availability,
                        np.log1p(features.order_volume),
                        features.has_contact,
                        features.company_known,
                    ]
                ],
                dtype=float,
            )
        )
        return pd.DataFrame(np.hstack([numeric, categorical]), columns=self.bundle["columns"])

    @staticmethod
    def _recommendation(priority: str, features: LeadFeatures) -> str:
        if priority == "High Priority":
            parts = [
                "Связаться с клиентом в течение часа.",
                "Предложить скидку 5–7% при объёме от 100 шт.",
            ]
            if features.stock_availability < 0.5:
                parts.append("Часть позиций под заказ — зафиксировать сроки поставки с клиентом.")
            if features.order_volume >= 100:
                parts.append("Передать лид старшему менеджеру (крупная партия).")
        elif priority == "Medium":
            parts = [
                "Ответить клиенту в течение рабочего дня.",
                "Скидку обсуждать после уточнения объёма партии.",
            ]
        else:
            parts = [
                "Поместить в очередь входящих, ответить в течение 2 рабочих дней.",
                "Скидку не предлагать без встречного интереса клиента.",
            ]
        return " ".join(parts)
