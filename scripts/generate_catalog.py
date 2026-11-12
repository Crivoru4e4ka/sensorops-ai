"""Генератор синтетического каталога датчиков (data/sensors_catalog.csv).

Создаёт ~70 позиций по 4 категориям (индуктивные, оптические, давления,
температуры) с реалистичными названиями моделей и JSON-спецификациями.
Детерминированный: фиксированный random seed.

Запуск:
    python scripts/generate_catalog.py [--size 70] [--out data/sensors_catalog.csv]
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

BRANDS = ["Autonics", "Sick", "Omron", "Balluff", "IFM", "Pepperl+Fuchs", "ТЕКО", "ОВЕН", "Turck", "Leuze"]

INDUCTIVE_SERIES = ["PR", "PRD", "PRDL", "BES", "IG", "IS", "NJ", "CJ", "PS12", "PS18"]
OPTICAL_SERIES = ["BX", "BGS", "GL", "E3Z", "WTB", "RL", "PRK", "OMT", "OS", "KT5"]
PRESSURE_SERIES = ["PSAN", "PN", "PSE", "PT", "SPAN", "PA", "DMU", "EDS", "PK", "ПД100"]
TEMPERATURE_SERIES = ["ДТС", "ТСП", "TST", "TN", "TR", "PT100", "E2E", "TS", "ДТП", "ТПТ"]

VOLTAGES = ["24В", "10-30В", "12В", "220В", "24В", "10-30В", "24В", "24В", "12-24В", "24В"]
THREADS = ["М8", "М12", "М18", "М30", "М12", "М18", "М12", "М18", "М30", "М18"]
IP_RATINGS = ["IP67", "IP67", "IP68", "IP65", "IP67", "IP69K", "IP67", "IP68", "IP65", "IP67"]
OUTPUTS_INDUCTIVE = ["PNP", "PNP", "NPN", "PNP/NPN", "PNP", "NPN", "PNP", "PNP", "NPN", "PNP/NPN"]
OUTPUTS_OPTICAL = ["PNP", "NPN", "PNP", "PNP/NPN", "PNP", "NPN", "PNP", "PNP/NPN", "PNP", "NPN"]
OUTPUTS_ANALOG = ["аналоговый 4-20мА", "аналоговый 0-10В", "RS-485 Modbus", "аналоговый 4-20мА", "аналоговый 4-20мА"]

DISTANCES_INDUCTIVE = ["2 мм", "4 мм", "8 мм", "15 мм", "5 мм", "8 мм", "10 мм", "4 мм", "8 мм", "22 мм"]
DISTANCES_OPTICAL = ["100 мм", "300 мм", "2 м", "5 м", "150 мм", "1 м", "10 м", "400 мм", "30 м", "800 мм"]
OPTICAL_TYPES = ["диффузный", "ретрорефлекторный", "через луч", "лазерный диффузный", "с подавлением фона"]
PRESSURE_RANGES = ["0-1 МПа", "0-10 бар", "0-16 бар", "0-40 бар", "0-25 бар", "0-6 бар", "0-60 бар", "0-2.5 МПа", "0-100 бар", "0-4 бар"]
TEMP_RANGES = ["-40..+125°C", "0..+150°C", "-50..+100°C", "0..+300°C", "-40..+85°C", "0..+200°C", "-60..+180°C", "0..+500°C", "-30..+120°C", "0..+100°C"]
TEMP_ELEMENTS = ["Pt100", "термопара K", "Pt1000", "NTC 10k", "Pt100", "термопара J", "Pt100", "Pt1000", "NTC 100k", "термопара K"]

CATEGORY_NAMES = {
    "индуктивный": "Датчик индуктивный",
    "оптический": "Датчик оптический",
    "давления": "Датчик давления",
    "температуры": "Датчик температуры",
}


def build_catalog(size: int = 72, seed: int = 777) -> list[dict]:
    rng = random.Random(seed)
    rows: list[dict] = []
    per_category = max(size // 4, 1)

    for category, series in [
        ("индуктивный", INDUCTIVE_SERIES),
        ("оптический", OPTICAL_SERIES),
        ("давления", PRESSURE_SERIES),
        ("температуры", TEMPERATURE_SERIES),
    ]:
        for index in range(per_category):
            brand = BRANDS[rng.randrange(len(BRANDS))]
            series_name = series[index % len(series)]
            suffix = f"{index + 1:02d}"
            name = f"{CATEGORY_NAMES[category]} {brand} {series_name}-{suffix}DN"

            if category == "индуктивный":
                specs = {
                    "voltage": VOLTAGES[rng.randrange(len(VOLTAGES))],
                    "thread": THREADS[rng.randrange(len(THREADS))],
                    "ip_rating": IP_RATINGS[rng.randrange(len(IP_RATINGS))],
                    "output_type": OUTPUTS_INDUCTIVE[rng.randrange(len(OUTPUTS_INDUCTIVE))],
                    "sensing_distance": DISTANCES_INDUCTIVE[rng.randrange(len(DISTANCES_INDUCTIVE))],
                    "housing": rng.choice(["металлический", "пластиковый"]),
                    "temperature_range": "-25..+70°C",
                }
                price = round(rng.uniform(1200, 4200), 2)
            elif category == "оптический":
                specs = {
                    "voltage": VOLTAGES[rng.randrange(len(VOLTAGES))],
                    "thread": rng.choice(["М18", "М12", "корпус 50×50", "корпус 18×50"]),
                    "ip_rating": IP_RATINGS[rng.randrange(len(IP_RATINGS))],
                    "output_type": OUTPUTS_OPTICAL[rng.randrange(len(OUTPUTS_OPTICAL))],
                    "sensing_distance": DISTANCES_OPTICAL[rng.randrange(len(DISTANCES_OPTICAL))],
                    "optical_type": OPTICAL_TYPES[rng.randrange(len(OPTICAL_TYPES))],
                    "temperature_range": "-25..+55°C",
                }
                price = round(rng.uniform(1500, 9500), 2)
            elif category == "давления":
                specs = {
                    "voltage": VOLTAGES[rng.randrange(len(VOLTAGES))],
                    "thread": rng.choice(["G1/4", "G1/2", "М20×1.5", "М12×1"]),
                    "ip_rating": rng.choice(["IP65", "IP67", "IP68"]),
                    "output_type": OUTPUTS_ANALOG[rng.randrange(len(OUTPUTS_ANALOG))],
                    "pressure_range": PRESSURE_RANGES[rng.randrange(len(PRESSURE_RANGES))],
                    "accuracy": rng.choice(["0.5%", "0.25%", "1.0%"]),
                    "media": rng.choice(["жидкости и газы", "пищевые среды", "агрессивные среды"]),
                }
                price = round(rng.uniform(3800, 21000), 2)
            else:  # температуры
                specs = {
                    "voltage": VOLTAGES[rng.randrange(len(VOLTAGES))],
                    "thread": rng.choice(["М12", "М20×1.5", "G1/2", "М8"]),
                    "ip_rating": rng.choice(["IP65", "IP67", "IP68"]),
                    "output_type": OUTPUTS_ANALOG[rng.randrange(len(OUTPUTS_ANALOG))],
                    "temperature_range": TEMP_RANGES[rng.randrange(len(TEMP_RANGES))],
                    "sensing_element": TEMP_ELEMENTS[rng.randrange(len(TEMP_ELEMENTS))],
                    "probe_length": rng.choice(["50 мм", "100 мм", "200 мм", "500 мм"]),
                }
                price = round(rng.uniform(2500, 14500), 2)

            in_stock = rng.choices([0, rng.randint(5, 80), rng.randint(80, 250)], weights=[8, 55, 37])[0]
            rows.append(
                {
                    "id": f"{category[:4].upper()}-{index + 1:03d}",
                    "name": name,
                    "category": category,
                    "specifications": json.dumps(specs, ensure_ascii=False),
                    "in_stock": in_stock,
                    "price": price,
                }
            )

    # Гарантируем наличие «стандартных» конфигураций (примеры из ТЗ должны находиться):
    # у каждой категории есть позиции 24В + PNP + М18 + IP67.
    _apply_standard_configs(rows, seed)
    return rows


def _apply_standard_configs(rows: list[dict], seed: int) -> None:
    """Проставляет стандартные конфигурации первым позициям каждой категории.

    Без этого синтетика может не содержать комбинаций из демо-запросов ТЗ
    (например, «оптический, М18, PNP, 24В»), и жёсткие фильтры дадут пустую выдачу.
    """
    by_category: dict[str, list[dict]] = {}
    for row in rows:
        by_category.setdefault(row["category"], []).append(row)
    for category, items in by_category.items():
        for row in items[:2]:
            specs = json.loads(row["specifications"])
            specs["voltage"] = "24В"
            specs["output_type"] = "PNP"
            specs["ip_rating"] = "IP67"
            if category in {"индуктивный", "оптический"}:
                specs["thread"] = "М18"
            row["specifications"] = json.dumps(specs, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Генерация каталога датчиков")
    parser.add_argument("--size", type=int, default=72, help="Количество позиций (кратно 4)")
    parser.add_argument("--out", type=str, default=str(PROJECT_ROOT / "data" / "sensors_catalog.csv"))
    parser.add_argument("--seed", type=int, default=777)
    args = parser.parse_args()

    rows = build_catalog(size=args.size, seed=args.seed)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["id", "name", "category", "specifications", "in_stock", "price"],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Сгенерировано позиций: {len(rows)} → {out_path}")


if __name__ == "__main__":
    main()
