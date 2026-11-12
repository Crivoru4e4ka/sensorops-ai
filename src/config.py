"""Конфигурация SensorOps AI.

Все параметры задаются через переменные окружения (или файл `.env` в корне
проекта). Значения по умолчанию подобраны так, чтобы система запускалась
"из коробки" в локальном режиме без внешних сервисов:

- ``LLM_BACKEND=mock``            — детерминированные RAG-ответы без API-ключей;
- ``EMBEDDING_BACKEND=auto``      — sentence-transformers, если установлен, иначе TF-IDF;
- ``VECTOR_BACKEND=auto``         — Qdrant, если доступен, иначе ChromaDB, иначе in-memory;
- ``WHISPER_MODEL=small``         — faster-whisper (модель скачивается при первом вызове).

Чтобы подключить реальный LLM::

    LLM_BACKEND=litellm
    LLM_MODEL=gpt-4o-mini
    LLM_API_KEY=sk-...

или через Ollama::

    LLM_BACKEND=litellm
    LLM_MODEL=ollama/qwen2.5
    LLM_API_BASE=http://localhost:11434/v1
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _env(key: str, default: str = "") -> str:
    return os.getenv(key, default).strip()


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, str(default)))
    except ValueError:
        return default


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, str(default)))
    except ValueError:
        return default


@dataclass
class Settings:
    app_name: str = "SensorOps AI"
    data_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data")

    # ─── Эмбеддинги ────────────────────────────────────────────────────────
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    # auto: sentence-transformers -> TF-IDF fallback; можно принудительно tfidf
    embedding_backend: str = "auto"

    # ─── Векторная БД ──────────────────────────────────────────────────────
    # auto: qdrant (если пингуется) -> chroma (если установлена) -> memory
    vector_backend: str = "auto"
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "sensors"
    chroma_persist_dir: str = "data/chroma"

    # ─── LLM ────────────────────────────────────────────────────────────────
    llm_backend: str = "mock"       # mock | litellm
    llm_model: str = "gpt-4o-mini"  # например ollama/qwen2.5
    llm_api_base: str = ""          # например http://localhost:11434/v1
    llm_api_key: str = ""
    llm_temperature: float = 0.2
    llm_timeout: int = 60

    # ─── ASR (faster-whisper) ───────────────────────────────────────────────
    whisper_model: str = "small"    # tiny | base | small | medium | large-v3
    whisper_device: str = "cpu"
    whisper_compute_type: str = "int8"

    # ─── Скоринг ────────────────────────────────────────────────────────────
    scoring_model_path: str = "data/models/scoring_model.joblib"
    scoring_training_data: str = "data/leads_training.csv"

    # ─── API / UI ───────────────────────────────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_url: str = "http://localhost:8000"   # для Streamlit UI
    log_level: str = "INFO"

    def resolve(self, path: str) -> Path:
        """Превращает относительный путь (от корня проекта) в абсолютный."""
        p = Path(path)
        return p if p.is_absolute() else PROJECT_ROOT / p

    @property
    def catalog_path(self) -> Path:
        return self.data_dir / "sensors_catalog.csv"

    @property
    def audio_dir(self) -> Path:
        return self.data_dir / "audio_samples"


def load_settings() -> Settings:
    """Читает конфигурацию из переменных окружения."""
    return Settings(
        app_name=_env("APP_NAME", "SensorOps AI"),
        data_dir=Path(_env("DATA_DIR", str(PROJECT_ROOT / "data"))),
        embedding_model=_env("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
        embedding_backend=_env("EMBEDDING_BACKEND", "auto").lower(),
        vector_backend=_env("VECTOR_BACKEND", "auto").lower(),
        qdrant_url=_env("QDRANT_URL", "http://localhost:6333"),
        qdrant_collection=_env("QDRANT_COLLECTION", "sensors"),
        chroma_persist_dir=_env("CHROMA_PERSIST_DIR", "data/chroma"),
        llm_backend=_env("LLM_BACKEND", "mock").lower(),
        llm_model=_env("LLM_MODEL", "gpt-4o-mini"),
        llm_api_base=_env("LLM_API_BASE"),
        llm_api_key=_env("LLM_API_KEY", _env("OPENAI_API_KEY")),
        llm_temperature=_env_float("LLM_TEMPERATURE", 0.2),
        llm_timeout=_env_int("LLM_TIMEOUT", 60),
        whisper_model=_env("WHISPER_MODEL", "small"),
        whisper_device=_env("WHISPER_DEVICE", "cpu"),
        whisper_compute_type=_env("WHISPER_COMPUTE_TYPE", "int8"),
        scoring_model_path=_env("SCORING_MODEL_PATH", "data/models/scoring_model.joblib"),
        scoring_training_data=_env("SCORING_TRAINING_DATA", "data/leads_training.csv"),
        api_host=_env("API_HOST", "0.0.0.0"),
        api_port=_env_int("API_PORT", 8000),
        api_url=_env("API_URL", "http://localhost:8000"),
        log_level=_env("LOG_LEVEL", "INFO"),
    )


settings = load_settings()
