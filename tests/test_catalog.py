"""Проверка целостности тестовых данных каталога датчиков."""
from __future__ import annotations

import json

import pandas as pd
import pytest

from src.config import settings


def test_catalog_exists_and_has_required_columns(catalog_df: pd.DataFrame) -> None:
    required = {"id", "name", "category", "specifications", "in_stock", "price"}
    assert required.issubset(set(catalog_df.columns)), f"Нет колонок: {required - set(catalog_df.columns)}"


def test_catalog_size_between_50_and_100(catalog_df: pd.DataFrame) -> None:
    assert 50 <= len(catalog_df) <= 100, f"В каталоге {len(catalog_df)} позиций (ожидалось 50–100)"


def test_ids_unique(catalog_df: pd.DataFrame) -> None:
    assert catalog_df["id"].is_unique, "Артикулы (id) должны быть уникальными"


def test_specifications_are_valid_json(catalog_df: pd.DataFrame) -> None:
    for value in catalog_df["specifications"]:
        specs = json.loads(value)
        assert isinstance(specs, dict) and specs, f"Пустая/некорректная спецификация: {value!r}"


def test_categories_allowed(catalog_df: pd.DataFrame) -> None:
    allowed = {"индуктивный", "оптический", "давления", "температуры"}
    assert set(catalog_df["category"].unique()).issubset(allowed)


def test_prices_and_stock_valid(catalog_df: pd.DataFrame) -> None:
    assert (catalog_df["price"] > 0).all(), "Цены должны быть положительными"
    assert (catalog_df["in_stock"] >= 0).all(), "Остатки не могут быть отрицательными"


def test_each_category_present(catalog_df: pd.DataFrame) -> None:
    categories = set(catalog_df["category"].unique())
    assert len(categories) == 4, f"Ожидались все 4 категории, найдены: {categories}"


def test_audio_samples_exist() -> None:
    audio_dir = settings.audio_dir
    wav_files = list(audio_dir.glob("*.wav")) + list(audio_dir.glob("*.mp3"))
    assert len(wav_files) >= 2, f"Нужно 2-3 тестовых аудиофайла, найдено: {len(wav_files)}"
    assert all(f.stat().st_size > 10_000 for f in wav_files), "Аудиофайлы выглядят пустыми"
