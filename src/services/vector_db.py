"""Работа с векторной БД: эмбеддинги + гибридный поиск (Модуль 1).

Поддерживаются три бэкенда хранения (``VECTOR_BACKEND``):

- ``qdrant`` — серверная векторная БД (``docker compose up qdrant``);
- ``chroma`` — локальная персистентная ChromaDB (без внешних сервисов);
- ``memory`` — in-memory brute-force (тесты и оффлайн-демо).

Эмбеддинги (``EMBEDDING_BACKEND``):

- ``sentence-transformers`` — модель ``EMBEDDING_MODEL``
  (по умолчанию ``all-MiniLM-L6-v2``);
- ``tfidf`` — лёгкий язык-независимый fallback на символьных n-граммах
  (не требует torch, используется, когда sentence-transformers недоступен).

Обе реализации возвращают L2-нормализованные плотные векторы, поэтому
косинусная близость = скалярное произведение.
"""
from __future__ import annotations

import json
import logging
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np

from ..config import settings

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[a-zа-яё0-9]+", re.IGNORECASE)


def tokenize(text: str) -> list[str]:
    """Разбивает текст на нормализованные токены (латиница/кириллица/цифры)."""
    return _TOKEN_RE.findall(str(text).lower())


def normalize_specs(specs: Any) -> dict[str, Any]:
    """Приводит specifications к dict (в CSV они хранятся JSON-строкой)."""
    if isinstance(specs, str):
        try:
            return json.loads(specs)
        except json.JSONDecodeError:
            return {}
    return dict(specs) if isinstance(specs, dict) else {}


def sensor_search_text(row: dict[str, Any]) -> str:
    """Текст, по которому строится dense-вектор датчика."""
    parts = [str(row.get("name", "")), str(row.get("category", ""))]
    for key, value in normalize_specs(row.get("specifications")).items():
        parts.append(f"{key} {value}")
    return " ".join(parts)


def normalize(value: Any) -> str:
    """Нормализация значения атрибута для сравнения фильтров."""
    return str(value).strip().lower().replace(" ", "").replace("\u00a0", "")


# ─── Эмбеддинги ────────────────────────────────────────────────────────────────


class EmbeddingProvider(ABC):
    """Абстрактный провайдер эмбеддингов."""

    name: str = "abstract"
    dim: int

    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        """Возвращает L2-нормализованную матрицу (n, dim)."""

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed([text])[0]


class SentenceTransformerEmbedder(EmbeddingProvider):
    """Dense-эмбеддинги через sentence-transformers (модель загружается лениво)."""

    name = "sentence-transformers"

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            logger.info("Загрузка embedding-модели %s ...", self.model_name)
            self._model = SentenceTransformer(self.model_name)
        return self._model

    @property
    def dim(self) -> int:
        return int(self.model.get_sentence_embedding_dimension())

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vectors, dtype=np.float32)


class TfidfEmbedder(EmbeddingProvider):
    """Лёгкий fallback: TF-IDF на символьных n-граммах (3-5), язык-независимый.

    Не даёт полноценной «семантики», но хорошо ловит технические термины
    и артикулы (М18, PNP, IP67), поэтому подходит для оффлайн-демо.
    """

    name = "tfidf"

    def __init__(self, corpus: list[str] | None = None) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer

        self._vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=1,
            sublinear_tf=True,
            norm="l2",
        )
        self._corpus: list[str] = []
        self._matrix: np.ndarray | None = None
        self._dim: int = 0
        if corpus:
            self.fit(corpus)

    def fit(self, corpus: list[str]) -> None:
        self._corpus = list(corpus)
        if not self._corpus:
            self._matrix = np.zeros((0, 128), dtype=np.float32)
            self._dim = 128
            return
        self._matrix = self._vectorizer.fit_transform(self._corpus).toarray().astype(np.float32)
        self._dim = int(self._matrix.shape[1])

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: list[str]) -> np.ndarray:
        if self._matrix is None:
            raise RuntimeError("TfidfEmbedder.fit() должен быть вызван перед embed()")
        if not self._corpus:
            return np.zeros((len(texts), self._dim), dtype=np.float32)
        vectors = self._vectorizer.transform(texts).toarray().astype(np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vectors / norms


def _sentence_transformers_available() -> bool:
    import importlib.util

    return importlib.util.find_spec("sentence_transformers") is not None


def build_embedder(corpus: list[str] | None = None) -> tuple[EmbeddingProvider, str]:
    """Собирает провайдер эмбеддингов согласно конфигурации.

    Возвращает (провайдер, фактический бэкенд).
    """
    backend = settings.embedding_backend
    if backend not in {"auto", "sentence-transformers", "tfidf"}:
        logger.warning("Неизвестный EMBEDDING_BACKEND=%r, использую auto", backend)
        backend = "auto"

    if backend in {"auto", "sentence-transformers"} and _sentence_transformers_available():
        return SentenceTransformerEmbedder(settings.embedding_model), "sentence-transformers"
    if backend == "sentence-transformers":
        raise ImportError(
            "EMBEDDING_BACKEND=sentence-transformers, но пакет не установлен. "
            "Выполните: pip install sentence-transformers"
        )
    logger.info(
        "Эмбеддинги: TF-IDF fallback (sentence-transformers недоступен или отключён)"
    )
    return TfidfEmbedder(corpus), "tfidf"


# ─── Векторные хранилища ───────────────────────────────────────────────────────


class VectorStore(ABC):
    """Абстрактное векторное хранилище."""

    name: str = "abstract"
    embedder: EmbeddingProvider

    def __init__(self, embedder: EmbeddingProvider) -> None:
        self.embedder = embedder

    @abstractmethod
    def reset(self) -> None:
        """Полностью очищает коллекцию."""

    @abstractmethod
    def upsert(self, ids: list[str], texts: list[str], payloads: list[dict[str, Any]]) -> None:
        """Добавляет/обновляет векторы с метаданными."""

    @abstractmethod
    def search(self, query_vector: np.ndarray, top_k: int) -> list[tuple[str, float]]:
        """Возвращает [(id, cosine_score)] по убыванию релевантности."""

    def count(self) -> int:
        raise NotImplementedError


class MemoryVectorStore(VectorStore):
    """In-memory хранилище: brute-force косинусный поиск (numpy)."""

    name = "memory"

    def __init__(self, embedder: EmbeddingProvider) -> None:
        super().__init__(embedder)
        self._ids: list[str] = []
        self._matrix = np.zeros((0, 1), dtype=np.float32)
        self._payloads: dict[str, dict[str, Any]] = {}

    def reset(self) -> None:
        self._ids, self._payloads = [], {}
        self._matrix = np.zeros((0, 1), dtype=np.float32)

    def upsert(self, ids: list[str], texts: list[str], payloads: list[dict[str, Any]]) -> None:
        vectors = self.embedder.embed(texts)
        for idx in ids:
            if idx in self._payloads:  # обновление — удаляем старый вектор
                pos = self._ids.index(idx)
                self._ids.pop(pos)
                self._matrix = np.delete(self._matrix, pos, axis=0)
        self._ids.extend(ids)
        self._matrix = vectors if len(self._ids) == len(vectors) else np.vstack(
            [self._matrix, vectors]
        )
        for idx, payload in zip(ids, payloads):
            self._payloads[idx] = payload

    def search(self, query_vector: np.ndarray, top_k: int) -> list[tuple[str, float]]:
        if len(self._ids) == 0:
            return []
        scores = self._matrix @ query_vector  # косинусная близость (векторы L2-норм.)
        order = np.argsort(-scores)[: min(top_k, len(self._ids))]
        return [(self._ids[int(i)], float(scores[int(i)])) for i in order]

    def count(self) -> int:
        return len(self._ids)


class ChromaVectorStore(VectorStore):
    """Локальная персистентная ChromaDB."""

    name = "chroma"

    def __init__(self, embedder: EmbeddingProvider, persist_dir: str | Path) -> None:
        import chromadb

        super().__init__(embedder)
        Path(persist_dir).mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(persist_dir))
        self._collection = self._client.get_or_create_collection(
            name="sensors", metadata={"hnsw:space": "cosine"}
        )

    def reset(self) -> None:
        self._client.delete_collection("sensors")
        self._collection = self._client.get_or_create_collection(
            name="sensors", metadata={"hnsw:space": "cosine"}
        )

    def upsert(self, ids: list[str], texts: list[str], payloads: list[dict[str, Any]]) -> None:
        vectors = self.embedder.embed(texts)
        self._collection.upsert(
            ids=ids,
            embeddings=vectors.tolist(),
            metadatas=[self._flatten(payload) for payload in payloads],
            documents=texts,
        )

    @staticmethod
    def _flatten(payload: dict[str, Any]) -> dict[str, Any]:
        """ChromaDB хранит только скаляры — сериализуем payload в JSON-строку."""
        return {
            "_payload": json.dumps(payload, ensure_ascii=False),
            "name": str(payload.get("name", "")),
            "category": str(payload.get("category", "")),
        }

    def search(self, query_vector: np.ndarray, top_k: int) -> list[tuple[str, float]]:
        result = self._collection.query(
            query_embeddings=[query_vector.tolist()],
            n_results=min(top_k, max(self.count(), 1)),
            include=["distances"],
        )
        ids = result.get("ids", [[]])[0] or []
        distances = result.get("distances", [[]])[0] or []
        return [(idx, 1.0 - float(dist)) for idx, dist in zip(ids, distances)]

    def count(self) -> int:
        return int(self._collection.count())


class QdrantVectorStore(VectorStore):
    """Серверная векторная БД Qdrant."""

    name = "qdrant"

    def __init__(self, embedder: EmbeddingProvider, url: str, collection: str) -> None:
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams

        super().__init__(embedder)
        self._collection = collection
        self._client = QdrantClient(url=url, timeout=10)
        if not self._client.collection_exists(collection):
            self._client.create_collection(
                collection_name=collection,
                vectors_config=VectorParams(size=embedder.dim, distance=Distance.COSINE),
            )

    def reset(self) -> None:
        from qdrant_client.models import Distance, VectorParams

        self._client.delete_collection(self._collection)
        self._client.create_collection(
            collection_name=self._collection,
            vectors_config=VectorParams(size=self.embedder.dim, distance=Distance.COSINE),
        )

    def upsert(self, ids: list[str], texts: list[str], payloads: list[dict[str, Any]]) -> None:
        import uuid

        from qdrant_client.models import PointStruct

        vectors = self.embedder.embed(texts)
        points = [
            # Qdrant допускает только uint/UUID в качестве point ID →
            # используем детерминированный UUID5 от строкового артикула.
            PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, str(idx))),
                vector=vec.tolist(),
                payload=dict(payload),
            )
            for idx, vec, payload in zip(ids, vectors, payloads)
        ]
        self._client.upsert(collection_name=self._collection, points=points)

    def search(self, query_vector: np.ndarray, top_k: int) -> list[tuple[str, float]]:
        # Совместимость: qdrant-client <1.15 использует search(),
        # новые версии — query_points().
        if hasattr(self._client, "search"):
            result = self._client.search(
                collection_name=self._collection,
                query_vector=query_vector.tolist(),
                limit=top_k,
                with_payload=["id"],
            )
            points = result
        else:
            response = self._client.query_points(
                collection_name=self._collection,
                query=query_vector.tolist(),
                limit=top_k,
                with_payload=["id"],
            )
            points = response.points
        # Возвращаем оригинальный строковый артикул из payload (не UUID точки)
        return [
            (str(hit.payload.get("id") or hit.id), float(hit.score))
            for hit in points
        ]

    def count(self) -> int:
        info = self._client.get_collection(self._collection)
        return int(info.points_count or 0)


# ─── Фабрика хранилищ ──────────────────────────────────────────────────────────


def qdrant_available(url: str) -> bool:
    """Проверяет доступность Qdrant без выбрасывания исключений."""
    try:
        from qdrant_client import QdrantClient

        client = QdrantClient(url=url, timeout=2)
        client.get_collections()
        return True
    except Exception:  # noqa: BLE001
        return False


def build_store(embedder: EmbeddingProvider) -> tuple[VectorStore, str]:
    """Собирает векторное хранилище согласно конфигурации (с fallback-цепочкой).

    Возвращает (хранилище, фактический бэкенд).
    """
    backend = settings.vector_backend
    if backend not in {"auto", "qdrant", "chroma", "memory"}:
        logger.warning("Неизвестный VECTOR_BACKEND=%r, использую auto", backend)
        backend = "auto"

    attempts: list[str] = (
        ["qdrant", "chroma", "memory"] if backend == "auto" else [backend]
    )

    for candidate in attempts:
        try:
            if candidate == "qdrant":
                if not qdrant_available(settings.qdrant_url):
                    if backend == "qdrant":
                        raise ConnectionError(
                            f"Qdrant недоступен по адресу {settings.qdrant_url}"
                        )
                    continue
                store = QdrantVectorStore(
                    embedder, settings.qdrant_url, settings.qdrant_collection
                )
                logger.info("Векторная БД: Qdrant (%s)", settings.qdrant_url)
                return store, "qdrant"
            if candidate == "chroma":
                store = ChromaVectorStore(embedder, settings.resolve(settings.chroma_persist_dir))
                logger.info("Векторная БД: ChromaDB (%s)", settings.chroma_persist_dir)
                return store, "chroma"
            store = MemoryVectorStore(embedder)
            logger.info("Векторная БД: in-memory (без персистентности)")
            return store, "memory"
        except Exception as exc:  # noqa: BLE001
            if backend != "auto":
                raise
            logger.warning("Бэкенд %r недоступен (%s), пробую следующий", candidate, exc)

    raise RuntimeError("Не удалось инициализировать ни одно векторное хранилище")
