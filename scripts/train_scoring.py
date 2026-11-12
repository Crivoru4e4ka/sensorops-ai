"""Обучение модели B2B-скоринга лидов (Модуль 3).

Использует CatBoost, если он установлен, иначе Scikit-learn
(GradientBoostingClassifier). Обучение — на синтетическом датасете
``data/leads_training.csv`` (генерируется здесь же, если файла нет).

Артефакт: ``data/models/scoring_model.joblib`` (бандл с моделью,
OneHotEncoder'ом индустрии/масштаба и StandardScaler'ом числовых фич).

Запуск:
    python scripts/train_scoring.py [--rows 250] [--seed 42]
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INDUSTRIES = [
    "машиностроение", "пищепром", "нефтегаз", "ритейл",
    "энергетика", "логистика", "строительство",
]
SCALES = ["малый", "средний", "крупный"]
URGENCY = ["низкая", "средняя", "критическая"]
URGENCY_LEVEL = {"низкая": 0, "средняя": 1, "критическая": 2}
DEAL_STAGES = ["запрос КП", "повторный звонок", "отказ"]
DEAL_STAGE_LEVEL = {"отказ": 0, "повторный звонок": 1, "запрос КП": 2}

# Порядок числовых фич: ДОЛЖЕН совпадать с src/services/scoring_service.py
NUMERIC_FEATURES = [
    "deal_stage_level", "urgency_level", "stock_availability",
    "log_order_volume", "has_contact", "company_known",
]

FEATURE_COLUMNS = [*NUMERIC_FEATURES, "industry", "business_scale"]


def _conversion_probability(row: pd.Series) -> float:
    """Правдоподобная генеративная функция P(конверсия) для синтетики."""
    if row["deal_stage"] == "отказ":
        return 0.02  # явный отказ почти никогда не конвертируется
    stage_factor = 0.55 if row["deal_stage"] == "повторный звонок" else 1.0
    logit = (
        -1.6
        + 1.1 * row["urgency_level"]
        + 1.5 * row["stock_availability"]
        + 0.35 * row["log_order_volume"]
        + 0.6 * row["has_contact"]
        + 0.4 * row["company_known"]
    )
    return float(stage_factor / (1.0 + np.exp(-logit)))


def generate_training_data(rows: int = 250, seed: int = 42, out_path: Path | None = None) -> pd.DataFrame:
    rng = random.Random(seed)
    data = []
    for _ in range(rows):
        row = {
            "industry": rng.choice(INDUSTRIES),
            "business_scale": rng.choice(SCALES),
            "urgency": rng.choice(URGENCY),
            "urgency_level": 0,
            "deal_stage": rng.choices(DEAL_STAGES, weights=[60, 25, 15])[0],
            "deal_stage_level": 0,
            "stock_availability": round(rng.uniform(0, 1), 2),
            "order_volume": rng.randint(1, 500),
            "has_contact": rng.randint(0, 1),
            "company_known": rng.randint(0, 1),
        }
        row["urgency_level"] = URGENCY_LEVEL[row["urgency"]]
        row["deal_stage_level"] = DEAL_STAGE_LEVEL[row["deal_stage"]]
        row["log_order_volume"] = round(np.log1p(row["order_volume"]), 3)
        row["converted"] = 1 if rng.random() < _conversion_probability(pd.Series(row)) else 0
        data.append(row)
    frame = pd.DataFrame(data)
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(out_path, index=False, encoding="utf-8")
    return frame


def build_pipeline() -> Pipeline:
    """Пайплайн: one-hot (категории) + масштабирование (числа) + градиентный бустинг."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                GradientBoostingClassifier(
                    n_estimators=150,
                    max_depth=3,
                    learning_rate=0.08,
                    subsample=0.9,
                    random_state=42,
                ),
            ),
        ]
    )


def train_model(training_csv: str | Path | None = None, seed: int = 42) -> dict:
    """Обучает модель и возвращает joblib-бандл.

    Пытается использовать CatBoost (если установлен), иначе Scikit-learn.
    """
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    if training_csv is None or not Path(training_csv).exists():
        training_csv = PROJECT_ROOT / "data" / "leads_training.csv"
        if not Path(training_csv).exists():
            generate_training_data(out_path=Path(training_csv), seed=seed)

    frame = pd.read_csv(training_csv)

    # Кодирование категорий
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    categorical = encoder.fit_transform(frame[["industry", "business_scale"]])
    scaler = StandardScaler()
    numeric = scaler.fit_transform(
        frame[
            ["deal_stage_level", "urgency_level", "stock_availability", "log_order_volume", "has_contact", "company_known"]
        ].to_numpy(dtype=float)
    )
    features = np.hstack([numeric, categorical])
    target = frame["converted"].to_numpy()
    columns = [
        *NUMERIC_FEATURES,
        *encoder.get_feature_names_out(["industry", "business_scale"]).tolist(),
    ]
    features_df = pd.DataFrame(features, columns=columns)

    try:
        from catboost import CatBoostClassifier

        model = CatBoostClassifier(
            iterations=200, depth=4, learning_rate=0.08, verbose=False, random_seed=seed
        )
        model.fit(features_df, target)
        kind = "catboost"
    except ImportError:
        model = GradientBoostingClassifier(
            n_estimators=150, max_depth=3, learning_rate=0.08, subsample=0.9, random_state=seed
        )
        model.fit(features_df, target)
        kind = "sklearn.GradientBoostingClassifier"

    return {
        "kind": kind,
        "model": model,
        "encoder": encoder,
        "scaler": scaler,
        "columns": columns,
        "feature_names": FEATURE_COLUMNS,
        "industries": INDUSTRIES,
        "scales": SCALES,
        "metadata": {"trained_on_rows": int(len(frame)), "seed": seed},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Обучение модели скоринга лидов")
    parser.add_argument("--rows", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default=str(PROJECT_ROOT / "data" / "models" / "scoring_model.joblib"))
    parser.add_argument("--training-csv", type=str, default=str(PROJECT_ROOT / "data" / "leads_training.csv"))
    args = parser.parse_args()

    training_csv = Path(args.training_csv)
    if not training_csv.exists():
        generate_training_data(rows=args.rows, seed=args.seed, out_path=training_csv)
        print(f"Синтетический датасет сгенерирован: {training_csv}")

    bundle = train_model(training_csv, seed=args.seed)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, out_path)
    print(
        f"Модель обучена ({bundle['kind']}) на {bundle['metadata']['trained_on_rows']} строках → {out_path}"
    )


if __name__ == "__main__":
    main()
