# -*- coding: utf-8 -*-
"""
Точка сборки приложения: создаёт Bot/httpx-клиент, запускает polling.

TODO(Задача 7, SPEC.md): в проде это заменится на webhook-режим (aiohttp
web-app вместо dp.start_polling) для деплоя на Render. Локальная разработка,
скорее всего, останется на polling — этот файл тогда будет ветвиться по
переменной окружения (например BOT_MODE=polling|webhook).
"""

from __future__ import annotations

import httpx
from aiogram import Bot
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode

from . import handlers
from .config import TELEGRAM_BOT_TOKEN, OPENROUTER_API_KEY, log
from .handlers import dp


async def main() -> None:
    # Проверяем ключи и, если их нет, ЧЁТКО пишем об этом в консоль.
    missing = []
    if not TELEGRAM_BOT_TOKEN:
        missing.append("TELEGRAM_BOT_TOKEN")
    if not OPENROUTER_API_KEY:
        missing.append("OPENROUTER_API_KEY")
    if missing:
        log.error(
            "Не заданы переменные окружения: %s. "
            "Задай их в конфигурации запуска (Environment variables) и запусти снова.",
            ", ".join(missing),
        )
        return

    handlers.http_client = httpx.AsyncClient()

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
        await handlers.http_client.aclose()
        await bot.session.close()
