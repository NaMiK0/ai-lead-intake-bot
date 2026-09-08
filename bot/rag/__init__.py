# -*- coding: utf-8 -*-
"""
RAG-контур: эмбеддинги заявок и поиск похожих.

Эмбеддинги — fastembed (ONNX через onnxruntime, без torch: легче для
деплоя на бесплатный тариф, см. SPEC.md/Задача 4), модель
paraphrase-multilingual-MiniLM-L12-v2. Векторы хранятся в Qdrant — только
эмбеддинги, Postgres (bot.storage) остаётся источником правды по самим
заявкам. id точки в Qdrant = id заявки в Postgres, что позволяет после
поиска подтянуть полную запись через bot.storage.find_lead().

Эмбеддится original_text (сырой текст клиента) — и при сохранении новой
заявки, и при поиске похожих на входящее сообщение: сравниваем сырой текст
с сырым текстом, а не сырой с уже нормализованной моделью выжимкой.

Публичный API:
    init_rag() / close_rag()           — вызывать из bot/app.py
    embed_text(text)                   — вектор одного текста
    upsert_lead_vector(lead_id, text)  — сохранить вектор заявки
    similar_leads(text, top_k=...)     — похожие прошлые заявки (LeadRecord)

similar_leads() вызывается из bot.llm.analyze() до классификации, чтобы
передать найденные похожие заявки как контекст модели.
"""

from __future__ import annotations

import asyncio
from typing import Optional

from fastembed import TextEmbedding
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from .. import storage
from ..config import (
    QDRANT_API_KEY,
    QDRANT_URL,
    RAG_COLLECTION,
    RAG_MODEL_NAME,
    RAG_SCORE_THRESHOLD,
    RAG_TOP_K,
    RAG_VECTOR_SIZE,
    log,
)
from ..storage.models import LeadRecord

__all__ = [
    "init_rag",
    "close_rag",
    "embed_text",
    "upsert_lead_vector",
    "similar_leads",
]

_model: Optional[TextEmbedding] = None
_client: Optional[AsyncQdrantClient] = None


class RagNotInitialized(RuntimeError):
    """init_rag() ещё не вызывался — обычно значит, что bot.app.main() не
    успел отработать (или тестовый код забыл его вызвать)."""


async def init_rag() -> None:
    """Загружает модель эмбеддингов и поднимает коллекцию в Qdrant.
    Вызывается один раз при старте приложения (bot/app.py). Загрузка модели
    (и, при первом запуске, её скачивание) — блокирующая операция, поэтому
    выполняется в отдельном потоке, чтобы не подвесить event loop."""
    global _model, _client
    if _model is not None and _client is not None:
        return

    try:
        model = await asyncio.to_thread(TextEmbedding, model_name=RAG_MODEL_NAME)
        client = AsyncQdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

        collections = await client.get_collections()
        existing = {c.name for c in collections.collections}
        if RAG_COLLECTION not in existing:
            await client.create_collection(
                collection_name=RAG_COLLECTION,
                vectors_config=VectorParams(size=RAG_VECTOR_SIZE, distance=Distance.COSINE),
            )
            log.info("Коллекция Qdrant '%s' создана.", RAG_COLLECTION)
    except Exception:
        # Ничего не оставляем полуинициализированным — последующие вызовы
        # embed_text/similar_leads должны консистентно падать с понятной
        # RagNotInitialized, а не пытаться работать через недобитый клиент.
        _model = None
        _client = None
        raise

    _model = model
    _client = client
    log.info("RAG-контур инициализирован: модель=%s, Qdrant=%s", RAG_MODEL_NAME, QDRANT_URL)


async def close_rag() -> None:
    global _model, _client
    if _client is not None:
        await _client.close()
    _model = None
    _client = None


def _get_model() -> TextEmbedding:
    if _model is None:
        raise RagNotInitialized(
            "Модель эмбеддингов не загружена — вызови bot.rag.init_rag() при старте приложения."
        )
    return _model


def _get_client() -> AsyncQdrantClient:
    if _client is None:
        raise RagNotInitialized(
            "Клиент Qdrant не инициализирован — вызови bot.rag.init_rag() при старте приложения."
        )
    return _client


async def embed_text(text: str) -> list[float]:
    """Вектор одного текста. fastembed синхронный и CPU-bound — уводим в
    отдельный поток, чтобы не блокировать event loop бота."""
    model = _get_model()

    def _embed() -> list[float]:
        return next(iter(model.embed([text]))).tolist()

    return await asyncio.to_thread(_embed)


async def upsert_lead_vector(lead_id: int, text: str) -> None:
    """Сохранить вектор заявки в Qdrant. id точки = id заявки в Postgres."""
    client = _get_client()
    vector = await embed_text(text)
    await client.upsert(
        collection_name=RAG_COLLECTION,
        points=[PointStruct(id=lead_id, vector=vector, payload={"lead_id": lead_id})],
    )


async def similar_leads(
    text: str,
    *,
    top_k: int = RAG_TOP_K,
    score_threshold: float = RAG_SCORE_THRESHOLD,
) -> list[LeadRecord]:
    """Похожие по смыслу прошлые заявки для текста `text`, отсортированные
    по убыванию похожести. Ниже score_threshold — считаем несвязанным и не
    возвращаем (порог подобран вживую на реальных формулировках, см.
    bot/config.py). Точки без соответствующей записи в Postgres (не должно
    происходить в норме, но данные — не гарантия) молча пропускаются."""
    client = _get_client()
    vector = await embed_text(text)
    hits = await client.query_points(
        collection_name=RAG_COLLECTION,
        query=vector,
        limit=top_k,
        score_threshold=score_threshold,
    )

    results: list[LeadRecord] = []
    for hit in hits.points:
        record = await storage.find_lead(hit.id)
        if record is not None:
            results.append(record)
    return results
