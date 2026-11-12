"""Общие фикстуры для тестов SensorOps AI.

Тесты работают в лёгком оффлайн-режиме: TF-IDF эмбеддинги, in-memory
векторная БД, mock LLM — без torch/Qdrant/внешних API.
"""
from __future__ import annotations

import os

# ВАЖНО: переменные окружения задаются ДО импорта src.*
os.environ.setdefault("VECTOR_BACKEND", "memory")
os.environ.setdefault("EMBEDDING_BACKEND", "tfidf")
os.environ.setdefault("LLM_BACKEND", "mock")
os.environ.setdefault("SCORING_MODEL_PATH", "data/models/test_scoring_model.joblib")
os.environ.setdefault("DATA_DIR", os.path.join(os.path.dirname(__file__), "..", "data"))

import pandas as pd
import pytest

from src.config import settings


@pytest.fixture(scope="session")
def catalog_df() -> pd.DataFrame:
    catalog = pd.read_csv(settings.catalog_path, dtype={"id": str})
    return catalog


@pytest.fixture()
def search_service(catalog_df: pd.DataFrame):
    from src.services.rag_service import get_search_service

    service = get_search_service(catalog_df)
    return service


@pytest.fixture(scope="session")
def scoring_service():
    from src.services.scoring_service import ScoringService

    service = ScoringService()
    service.load_or_train()
    return service
