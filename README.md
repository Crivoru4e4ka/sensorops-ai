# 📡 SensorOps AI

**Enterprise-платформа интеллектуального поиска, анализа коммуникаций и B2B-скоринга
для дистрибьютора промышленной электроники** — комплексный PoC (Proof of Concept),
демонстрирующий компетенции Applied AI / LLM Engineer.

---

## 1. Возможности

| Модуль | Что делает | Технологии |
|---|---|---|
| **1. Гибридный семантический поиск** | Клиент описывает датчик «обычным языком» — система находит точные совпадения и релевантные аналоги, даже если формулировки в каталоге отличаются | Dense Retrieval (эмбеддинги) + Sparse/Exact Filter (жёсткие атрибуты) + RAG-генерация сравнительной подсказки |
| **2. Voice-to-CRM** | Загрузка аудиозаписи звонка (или текста переписки) → транскрипт → строго типизированная карточка лида в JSON | faster-whisper (ASR) + Instructor + Pydantic v2 |
| **3. B2B Profiling & Lead Scoring** | Классификация входящих компаний и скоринг вероятности успешной сделки 0..1 + бейдж приоритета и рекомендация менеджеру | Data Enrichment Mock + CatBoost / Scikit-learn |

## 2. Стек

- **Python 3.10+** · **FastAPI** (REST, async, Swagger `/docs`) · **Streamlit** (мультистраничный UI)
- **LiteLLM** — единый провайдер LLM (OpenAI API / Ollama для локального запуска)
- **Instructor + Pydantic v2** — гарантированный структурированный вывод без «мусора» в JSON
- **Qdrant** (по умолчанию в Docker) **или ChromaDB** (локально, без сервисов)
- **Эмбеддинги**: `sentence-transformers/all-MiniLM-L6-v2` (или `multilingual-e5-base`), TF-IDF fallback для оффлайн-демо
- **faster-whisper** — транскрибация (модель `small`/`medium`, язык определяется автоматически)
- **CatBoost** (если установлен) **или Scikit-learn** — скоринг лидов на табличных данных
- **Docker + Docker Compose**

## 3. Структура репозитория

```
sensorops-ai/
├── data/
│   ├── sensors_catalog.csv       # База датчиков: 72 синтетические позиции (JSON-спецификации)
│   ├── audio_samples/            # 3 тестовых аудиозвонка (.wav + .mp3) и их транскрипты
│   ├── leads_training.csv        # Синтетический датасет для обучения модели скоринга
│   └── models/                   # Обученный артефакт scoring_model.joblib (создаётся автоматически)
├── src/
│   ├── api/
│   │   ├── search_router.py      # POST /api/search, /index/rebuild, /search/stats
│   │   └── calls_router.py       # POST /api/calls/transcribe, /analyze, /score, /enrich
│   ├── services/
│   │   ├── vector_db.py          # Эмбеддинги + Qdrant / ChromaDB / in-memory хранилище
│   │   ├── rag_service.py        # Гибридный поиск + RAG-генерация (LiteLLM / mock)
│   │   ├── asr_service.py        # faster-whisper + экстракция CRMLeadCard (Instructor / rules)
│   │   └── scoring_service.py    # Data Enrichment Mock + ML-скоринг лида
│   ├── schemas/
│   │   ├── lead.py               # CRMLeadCard, LeadFeatures, LeadScore
│   │   └── search.py             # SearchRequest/Response, SearchFilters, SensorHit
│   ├── config.py                 # Настройки окружения (ENV / .env)
│   ├── state.py                  # Глобальное состояние (каталог, индекс, модель)
│   └── main.py                   # FastAPI-приложение (+ /api/health)
├── ui/
│   └── app.py                    # Streamlit Dashboard: Smart Search / Voice-to-CRM / Lead Scoring
├── scripts/
│   ├── generate_catalog.py       # Генерация sensors_catalog.csv (детерминированная)
│   ├── generate_audio.py         # Генерация демо-звонков через системный TTS
│   ├── gen_audio_windows.ps1     # Генерация демо-звонков (Windows SAPI, ru-RU)
│   ├── train_scoring.py          # Обучение модели скоринга (CatBoost/sklearn)
│   ├── smoke_test.py             # Сквозной smoke-тест всех REST-эндпоинтов
│   └── e2e_audio_test.py         # E2E: транскрибация 3 звонков → карточка → скоринг
├── tests/                        # pytest: данные, поиск, Voice-to-CRM, скоринг, API
├── docker-compose.yml            # qdrant + api + ui
├── Dockerfile                    # INSTALL_TRANSFORMERS=true — полный стек эмбеддингов
├── requirements.txt              # основные зависимости (без torch)
├── requirements-embeddings.txt   # sentence-transformers (опционально, тянет torch)
├── .env.example                  # пример конфигурации
└── README.md
```

## 4. Быстрый старт

### Вариант A: локально (Python 3.10+, без Docker)

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate   |  Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt

# (опционально, один раз) пересоздать данные и обучить модель:
python scripts/generate_catalog.py
python scripts/train_scoring.py

# 1. API:
uvicorn src.main:app --host 0.0.0.0 --port 8000
#    → Swagger: http://localhost:8000/docs

# 2. UI (второй терминал):
streamlit run ui/app.py
#    → http://localhost:8501
```

Без внешних сервисов система автоматически переключается в оффлайн-режим:
TF-IDF-эмбеддинги + in-memory/ChromaDB хранилище + детерминированный mock-RAG.
Полноценный стек включается сам, когда доступны sentence-transformers, Qdrant и LLM.

### Вариант B: Docker Compose (полный стек)

```bash
# опционально: .env уже содержит HF_ENDPOINT (зеркало HuggingFace для РФ/СНГ)
docker compose up --build
#   API:  http://localhost:8000/docs
#   UI:   http://localhost:8501
```

В compose поднимаются **Qdrant** (векторная БД), **API** (FastAPI + faster-whisper +
litellm + Instructor) и **UI** (Streamlit). Каталог индексируется автоматически при старте.

Эмбеддинги sentence-transformers (torch CPU) включаются аргументом сборки —
на сетях без доступа к download.pytorch.org образ соберётся и без них,
и система автоматически перейдёт на TF-IDF fallback:

```bash
docker compose build --build-arg INSTALL_TRANSFORMERS=true
# локально без Docker:
pip install torch --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple
pip install -r requirements-embeddings.txt
```

Модели faster-whisper (и sentence-transformers) скачиваются с HuggingFace при
первом вызове ASR. На медленных сетях задайте зеркало: `HF_ENDPOINT=https://hf-mirror.com`.

### Подключение реального LLM

Скопируйте `.env.example` в `.env` и задайте один из вариантов:

```bash
# OpenAI-совместимое API:
LLM_BACKEND=litellm
LLM_MODEL=gpt-4o-mini
LLM_API_KEY=sk-...

# или локальный Ollama:
LLM_BACKEND=litellm
LLM_MODEL=ollama/qwen2.5
LLM_API_BASE=http://localhost:11434/v1
```

После этого RAG-подсказки (Модуль 1) и экстракция карточек (Модуль 2) будут
генерироваться LLM; без ключа работает детерминированный fallback.

## 5. Примеры запросов к API

```bash
# Гибридный поиск с жёсткими фильтрами + RAG-подсказка
curl -X POST http://localhost:8000/api/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Нужен оптический датчик приближения с защитой от влаги, резьба М18, выход PNP, питание 24 вольта",
    "filters": {"category": "оптический", "voltage": "24В", "output_type": "PNP", "thread": "М18"},
    "top_k": 3
  }'

# Разбор аудиозвонка → транскрипт + CRM-карточка + скоринг
curl -X POST http://localhost:8000/api/calls/analyze \
  -F "file=@data/audio_samples/call_01_urgent_inductive.wav"

# Или по тексту (без ASR):
curl -X POST http://localhost:8000/api/calls/analyze-text \
  -H "Content-Type: application/json" \
  -d '{"text": "Менеджер: Компания ПромТех, слушаю. Клиент: Здравствуйте, это Дмитрий из ПромСтройМаш, срочно нужны индуктивные датчики М18 PNP 24В, 200 штук..."}'

# Скоринг готовой карточки лида
curl -X POST http://localhost:8000/api/calls/score \
  -H "Content-Type: application/json" \
  -d '{"company_name": "ПромСтройМаш", "client_name": "Дмитрий", "contact_info": "d@x.ru",
       "requested_products": ["датчик индуктивный"], "urgency": "критическая",
       "deal_stage": "запрос КП", "summary": "200 штук на этой неделе", "action_items": []}'
```

## 6. Тестовые данные

- **`data/sensors_catalog.csv`** — 72 синтетических датчика (индуктивные, оптические,
  давления, температуры) с JSON-спецификациями: напряжение, резьба, IP, тип выхода,
  диапазоны температур/давления, остатки, цены. Перегенерация:
  `python scripts/generate_catalog.py --size 72`.
- **`data/audio_samples/`** — три демо-звонка, озвученные системным TTS:
  | Файл | Сценарий |
  |---|---|
  | `call_01_urgent_inductive.wav/.mp3` | Срочный запрос индуктивных датчиков М18 PNP 24В, 200 шт. |
  | `call_02_medium_temperature_pressure.wav/.mp3` | Подбор датчиков температуры/давления для пищепрома |
  | `call_03_refusal_optical.wav/.mp3` | Отказ клиента (отработка статуса «отказ») |
  Рядом лежат `.txt`-транскрипты. Перегенерация: `python scripts/generate_audio.py`.
- **`data/leads_training.csv`** — 250 синтетических лидов для обучения скоринга.
- **`data/models/scoring_model.joblib`** — обученная модель (создаётся автоматически
  при первом старте, если отсутствует).

## 7. Тесты

```bash
pip install pytest
pytest -v          # 42 теста: данные, поиск, Voice-to-CRM, скоринг, API
```

После запуска API можно прогнать сквозные проверки:

```bash
python scripts/smoke_test.py      # все REST-эндпоинты
python scripts/e2e_audio_test.py  # реальная транскрибация 3 аудиозвонков → CRM-карточка → скоринг
```

Тесты покрывают: целостность каталога и аудиоданных, гибридный поиск (жёсткие
фильтры, сортировка, лимиты, lexical boost), rule-based экстракцию CRM-карточки,
детерминированность обогащения, работу модели скоринга и все REST-эндпоинты.

## 8. Конфигурация (ENV)

| Переменная | По умолчанию | Описание |
|---|---|---|
| `LLM_BACKEND` | `mock` | `mock` или `litellm` |
| `LLM_MODEL` | `gpt-4o-mini` | например `ollama/qwen2.5` |
| `LLM_API_BASE` / `LLM_API_KEY` | — | OpenAI-совместимый endpoint/ключ |
| `VECTOR_BACKEND` | `auto` | `qdrant` / `chroma` / `memory` |
| `QDRANT_URL` | `http://localhost:6333` | адрес Qdrant |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | модель эмбеддингов |
| `WHISPER_MODEL` | `small` | `tiny`/`base`/`small`/`medium` |
| `API_PORT` / `API_URL` | `8000` / `http://localhost:8000` | порт API / URL API для UI |
