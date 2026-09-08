# -*- coding: utf-8 -*-
"""
Пул соединений и SQL-запросы. Без ORM — asyncpg и ручной SQL, в стиле
остального проекта (минимум зависимостей, видно, что происходит).

Схема применяется идемпотентно (CREATE TABLE IF NOT EXISTS) при каждом
старте пула — для проекта такого масштаба отдельный инструмент миграций
(Alembic и т.п.) избыточен.
"""

from __future__ import annotations

from typing import Optional

import asyncpg

from ..config import log

SCHEMA = """
CREATE TABLE IF NOT EXISTS leads (
    id BIGSERIAL PRIMARY KEY,
    chat_id BIGINT NOT NULL,
    message_id BIGINT,
    client_name TEXT,
    company TEXT,
    contact TEXT,
    category TEXT,
    request TEXT,
    deadline TEXT,
    budget TEXT,
    urgency TEXT,
    urgency_reason TEXT,
    injection_detected BOOLEAN NOT NULL DEFAULT false,
    original_text TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_leads_contact ON leads (contact);
CREATE INDEX IF NOT EXISTS idx_leads_category ON leads (category);
"""

_pool: Optional[asyncpg.Pool] = None


class StorageNotInitialized(RuntimeError):
    """init_pool() ещё не вызывался — обычно значит, что bot.app.main() не
    успел отработать (или тестовый код забыл его вызвать)."""


async def init_pool(dsn: str) -> None:
    """Создаёт пул соединений и применяет схему. Вызывается один раз при
    старте приложения (bot/app.py)."""
    global _pool
    if _pool is not None:
        return
    _pool = await asyncpg.create_pool(dsn, min_size=1, max_size=5)
    async with _pool.acquire() as conn:
        await conn.execute(SCHEMA)
    log.info("Пул соединений с Postgres создан, схема применена.")


async def close_pool() -> None:
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None


def get_pool() -> asyncpg.Pool:
    if _pool is None:
        raise StorageNotInitialized(
            "Пул соединений с БД не инициализирован — вызови "
            "bot.storage.init_pool(DATABASE_URL) при старте приложения."
        )
    return _pool
