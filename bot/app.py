# -*- coding: utf-8 -*-
"""
Точка сборки приложения: создаёт Bot/LLM-клиент, запускает polling.

TODO(Задача 7, SPEC.md): в проде это заменится на webhook-режим (aiohttp
web-app вместо dp.start_polling) для деплоя на Render. Локальная разработка,
скорее всего, останется на polling — этот файл тогда будет ветвиться по
переменной окружения (например BOT_MODE=polling|webhook).
"""

from __future__ import annotations

from aiogram import Bot
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode

from . import handlers, rag, storage
from .config import DATABASE_URL, TELEGRAM_BOT_TOKEN, OPENROUTER_API_KEY, log
from .handlers import dp
from .llm import build_client


async def main() -> None:
    # Проверяем ключи и, если их нет, ЧЁТКО пишем об этом в консоль.
    missing = []
    if not TELEGRAM_BOT_TOKEN:
        missing.append("TELEGRAM_BOT_TOKEN")
    if not OPENROUTER_API_KEY:
        missing.append("OPENROUTER_API_KEY")
    if not DATABASE_URL:
        missing.append("DATABASE_URL")
    if missing:
        log.error(
            "Не заданы переменные окружения: %s. "
            "Задай их в конфигурации запуска (Environment variables) и запусти снова.",
            ", ".join(missing),
        )
        return

    handlers.llm_client = build_client()
    await storage.init_pool(DATABASE_URL)

    # RAG — необязательный контур (см. bot/rag): его сбои в рантайме не
    # роняют обработку сообщений, поэтому и здесь недоступный Qdrant не
    # должен блокировать старт бота — просто работаем без похожих заявок.
    try:
        await rag.init_rag()
    except Exception as e:  # noqa: BLE001 — RAG необязателен для работы бота
        log.error("RAG-контур не инициализирован (бот продолжит без него): %s", e)

    bot = Bot(
        token=TELEGRAM_BOT_TOKEN,
        default=DefaultBotProperties(parse_mode=ParseMode.HTML),
    )

    log.info("Бот запускается…")
    try:
        # drop_pending_updates=True — при старте не разгребаем накопившееся,
        # чтобы не отвечать на старые сообщения после простоя/перезапуска.
        await dp.start_polling(bot, drop_pending_updates=True)
    finally:
        await handlers.llm_client.close()
        await storage.close_pool()
        await rag.close_rag()
        await bot.session.close()
