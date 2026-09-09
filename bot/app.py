# -*- coding: utf-8 -*-
"""
Точка сборки приложения.

Два режима запуска (BOT_MODE, см. bot/config.py):
    polling — локальная разработка, main() (как раньше, ничего не меняется).
    webhook — прод (Render), run_webhook(): aiohttp-сервер вместо polling.

Инициализация/остановка общих ресурсов (LLM-клиент, пул Postgres, RAG) —
через dp.startup/dp.shutdown хуки aiogram, а не ручной try/finally: они
одинаково срабатывают что при dp.start_polling(), что при setup_application()
в webhook-режиме, так что логика не дублируется между режимами.
"""

from __future__ import annotations

from aiogram import Bot
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.webhook.aiohttp_server import SimpleRequestHandler, setup_application
from aiohttp import web

from . import handlers, rag, storage
from .config import (
    BOT_MODE,
    DATABASE_URL,
    OPENROUTER_API_KEY,
    PORT,
    RAG_ENABLED,
    TELEGRAM_BOT_TOKEN,
    WEBHOOK_PATH,
    WEBHOOK_SECRET,
    WEBHOOK_URL,
    log,
)
from .handlers import dp
from .llm import build_client


def _check_required_env() -> bool:
    """Проверяет обязательные переменные окружения и, если чего-то не хватает,
    ЧЁТКО пишет об этом в консоль. True — можно стартовать."""
    missing = []
    if not TELEGRAM_BOT_TOKEN:
        missing.append("TELEGRAM_BOT_TOKEN")
    if not OPENROUTER_API_KEY:
        missing.append("OPENROUTER_API_KEY")
    if not DATABASE_URL:
        missing.append("DATABASE_URL")
    if BOT_MODE == "webhook":
        if not WEBHOOK_URL:
            missing.append("WEBHOOK_URL")
        if not WEBHOOK_SECRET:
            missing.append("WEBHOOK_SECRET")
    if missing:
        log.error(
            "Не заданы переменные окружения: %s. "
            "Задай их в конфигурации запуска (Environment variables) и запусти снова.",
            ", ".join(missing),
        )
        return False
    return True


@dp.startup.register
async def on_startup(bot: Bot) -> None:
    handlers.llm_client = build_client()
    await storage.init_pool(DATABASE_URL)

    # RAG — необязательный контур (см. bot/rag). RAG_ENABLED=false — модель
    # эмбеддингов даже не пытается загрузиться (на бесплатном тарифе Render
    # это валит процесс OOM ещё до подключения к Qdrant, см. bot/config.py).
    # Иначе — сбой Qdrant в рантайме тоже не должен блокировать старт бота,
    # просто работаем без похожих заявок.
    if not RAG_ENABLED:
        log.info("RAG-контур выключен (RAG_ENABLED=false).")
    else:
        try:
            await rag.init_rag()
        except Exception as e:  # noqa: BLE001 — RAG необязателен для работы бота
            log.error("RAG-контур не инициализирован (бот продолжит без него): %s", e)

    if BOT_MODE == "webhook":
        try:
            await bot.set_webhook(
                url=f"{WEBHOOK_URL}{WEBHOOK_PATH}",
                secret_token=WEBHOOK_SECRET,
                drop_pending_updates=True,
            )
        except Exception as e:  # noqa: BLE001 — превращаем в понятное сообщение
            # Без вебхука бот в этом режиме бесполезен (апдейты не придут
            # никак) — это фатально, но с чёткой причиной в логе, а не
            # сырым трейсбеком TelegramBadRequest.
            log.error(
                "Не удалось установить webhook (%s%s): %s. Проверь WEBHOOK_URL "
                "— это должен быть реальный публичный HTTPS-адрес сервиса.",
                WEBHOOK_URL, WEBHOOK_PATH, e,
            )
            raise
        log.info("Webhook установлен: %s%s", WEBHOOK_URL, WEBHOOK_PATH)

    log.info("Бот запущен (режим=%s).", BOT_MODE)


@dp.shutdown.register
async def on_shutdown(bot: Bot) -> None:
    # Раньше здесь был bot.delete_webhook() в режиме webhook — но Render
    # усыпляет бесплатный веб-сервис при простое тем же graceful SIGTERM,
    # что и при обычном редеплое, а dp.shutdown срабатывает на оба случая
    # одинаково. Удаление вебхука на "усыплении" стирало его у Telegram —
    # а без вебхука Telegram больше не шлёт запросы, которые могли бы
    # разбудить инстанс обратно, так что бот "засыпал" навсегда до
    # ручного вмешательства (баг найден и подтверждён вживую). При
    # реальном редеплое новый инстанс всё равно перевызывает set_webhook
    # в on_startup, так что удалять его при остановке старого — не нужно.
    if handlers.llm_client is not None:
        await handlers.llm_client.close()
    await storage.close_pool()
    await rag.close_rag()


def _build_bot() -> Bot:
    return Bot(
        token=TELEGRAM_BOT_TOKEN,
        default=DefaultBotProperties(parse_mode=ParseMode.HTML),
    )


async def _health(request: web.Request) -> web.Response:
    return web.Response(text="ok")


async def main() -> None:
    """Локальный запуск, long polling."""
    if not _check_required_env():
        return
    bot = _build_bot()
    log.info("Бот запускается (polling)…")
    # handle_signals=True (по умолчанию) — сам ловит SIGINT/SIGTERM и
    # прогоняет dp.shutdown перед выходом; close_bot_session=True закрывает
    # HTTP-сессию бота — ручной try/finally не нужен.
    await dp.start_polling(bot, drop_pending_updates=True)


def run_webhook() -> None:
    """Прод, Render: aiohttp-сервер вместо polling. Синхронная точка входа —
    aiohttp сам поднимает event loop и сам обрабатывает SIGTERM/SIGINT (важно
    для graceful shutdown при деплое новой версии на Render)."""
    if not _check_required_env():
        return
    bot = _build_bot()

    app = web.Application()
    SimpleRequestHandler(dispatcher=dp, bot=bot, secret_token=WEBHOOK_SECRET).register(
        app, path=WEBHOOK_PATH
    )
    setup_application(app, dp, bot=bot)  # вешает dp.startup/dp.shutdown на app
    app.router.add_get("/", _health)  # health-check для Render + ручная проверка в браузере

    log.info("Бот запускается (webhook), порт %s…", PORT)
    web.run_app(app, host="0.0.0.0", port=PORT, print=None)
