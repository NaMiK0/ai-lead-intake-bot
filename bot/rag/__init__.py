# -*- coding: utf-8 -*-
"""
RAG-контур: эмбеддинги заявок и поиск похожих.

Заглушка. Реализация — Задача 4 из SPEC.md: эмбеддинги через
sentence-transformers (мультиязычная модель, русский текст), хранение и
поиск в Qdrant (локально — docker-compose, в проде — Qdrant Cloud). Зависит
от bot.storage (Задача 3) — векторы привязаны к заявкам, сохранённым в БД.

Планируемый публичный интерфейс этого пакета (сигнатуры могут измениться при
реализации):
    embed_text(text: str) -> list[float]
    upsert_lead_vector(lead_id: int, text: str) -> None
    similar_leads(text: str, top_k: int = 3) -> list[LeadRecord]

similar_leads() будет вызываться из bot.llm.analyze() до классификации, чтобы
передать найденные похожие заявки как контекст модели.
"""

from __future__ import annotations


class RagNotImplemented(NotImplementedError):
    """RAG-контур ещё не реализован — см. SPEC.md, Задача 4."""


def embed_text(*args, **kwargs):
    raise RagNotImplemented("bot.rag.embed_text: см. SPEC.md, Задача 4")


def similar_leads(*args, **kwargs):
    raise RagNotImplemented("bot.rag.similar_leads: см. SPEC.md, Задача 4")
