# -*- coding: utf-8 -*-
"""Хэндлеры aiogram."""

from __future__ import annotations

from typing import Optional

from aiogram import Dispatcher, F
from aiogram.enums import ChatAction
from aiogram.filters import Command, CommandStart
from aiogram.types import Message
from openai import AsyncOpenAI

from .config import MAX_CARDS, MAX_INPUT_CHARS, log
from .dedup import already_processed, is_duplicate_text
from .llm import AllModelsFailed, analyze
from .rendering import render_lead, render_not_lead

dp = Dispatcher()

WELCOME = (
    "👋 Здравствуйте! Вы можете оставить заявку прямо здесь.\n\n"
    "Пожалуйста, напишите одним сообщением:\n"
    "• что вы хотите заказать или какая задача стоит;\n"
    "• желаемые сроки, если они есть;\n"
    "• телефон или email для связи;\n"
    "• как к вам обращаться.\n\n"
    "Можно писать свободно, как удобно — мы во всём разберёмся и свяжемся с вами."
)

HELP = (
    "ℹ️ Это бот для приёма заявок.\n\n"
    "Просто напишите одним сообщением, что вы хотите заказать, желаемые сроки "
    "и контакт для связи (телефон или email) — и мы свяжемся с вами. "
    "Писать можно свободно, в любом формате.\n\n"
    "Чтобы начать заново, отправьте /start."
)

# Подтверждение клиенту — отправляется ПЕРЕД карточкой(ами), но только если
# в сообщении нашлась хотя бы одна реальная заявка.
CONFIRMATION = "✅ Спасибо, ваша заявка принята! Мы свяжемся с вами в ближайшее время."

# LLM-клиент (openai SDK, направлен на OpenRouter) создаётся один раз при
# старте. Устанавливается из bot/app.py:main() перед стартом polling/webhook.
llm_client: Optional[AsyncOpenAI] = None


@dp.message(CommandStart())
async def on_start(message: Message) -> None:
    await message.answer(WELCOME)


@dp.message(Command("help"))
async def on_help(message: Message) -> None:
    await message.answer(HELP)


@dp.message(F.text)
async def on_text(message: Message) -> None:
    """Основной хэндлер: пришёл текст → карточка(и)."""
    # 1) Дедуп: одно и то же сообщение не обрабатываем дважды.
    if already_processed(message.chat.id, message.message_id):
        log.info("Пропуск дубля апдейта: chat=%s msg=%s",
                 message.chat.id, message.message_id)
        return

    text = (message.text or "").strip()
    if not text:
        await message.answer("Пожалуйста, опишите вашу заявку текстом 🙂")
        return

    # 2) Обрезаем слишком длинный вход (предохранитель по токенам/лимитам).
    if len(text) > MAX_INPUT_CHARS:
        text = text[:MAX_INPUT_CHARS]

    # 3) Проверяем на повтор текста (для флага «возможный дубль»).
    is_dup = is_duplicate_text(text)

    # 4) Показываем «печатает…», пока думает модель.
    try:
        await message.bot.send_chat_action(message.chat.id, ChatAction.TYPING)
    except Exception:  # noqa: BLE001 — не критично
        pass

    # 5) Прогон через каскад моделей.
    assert llm_client is not None
    try:
        result = await analyze(llm_client, text)
    except AllModelsFailed as e:
        log.error("Все модели недоступны: %s", e)
        await message.answer(
            "Извините, сейчас не получилось принять заявку из-за технической "
            "заминки. Пожалуйста, попробуйте отправить сообщение ещё раз через минуту."
        )
        return
    except Exception as e:  # noqa: BLE001 — на всякий случай, чтобы не упасть
        log.exception("Непредвиденная ошибка при анализе: %s", e)
        await message.answer(
            "Извините, при обработке заявки произошла ошибка. "
            "Пожалуйста, попробуйте отправить сообщение ещё раз."
        )
        return

    # 6) Рендерим и отправляем карточки. Каждая — отдельным сообщением,
    #    чтобы их было удобно пересылать по одной.
    items = result.items[:MAX_CARDS]
    leads = [it for it in items if it.kind == "lead"]
    total_leads = len(leads)

    # Подтверждение клиенту — только если есть реальная заявка.
    # (Для не-заявки вроде «спасибо за созвон» подтверждать нечего.)
    if total_leads >= 1:
        await message.answer(CONFIRMATION)

    lead_idx = 0
    for it in items:
        try:
            if it.kind == "not_lead":
                card = render_not_lead(it, text, is_dup)
            else:
                lead_idx += 1
                card = render_lead(it, text, lead_idx, total_leads, is_dup)
            await message.answer(card)
        except Exception as e:  # noqa: BLE001 — один сбой рендера не рушит остальные
            log.exception("Ошибка рендера/отправки карточки: %s", e)
            await message.answer(
                "Извините, часть вашей заявки не удалось обработать. "
                "Пожалуйста, попробуйте отправить сообщение ещё раз."
            )


@dp.message()
async def on_non_text(message: Message) -> None:
    """Любое не-текстовое сообщение (фото, стикер, голосовое и т.п.)."""
    # Дедуп и здесь, чтобы не отвечать дважды.
    if already_processed(message.chat.id, message.message_id):
        return
    # Если у медиа есть подпись — попробуем обработать её как текст.
    if message.caption and message.caption.strip():
        message = message.model_copy(update={"text": message.caption})
        await on_text(message)
        return
    await message.answer(
        "Пожалуйста, опишите вашу заявку текстом — так мы сможем её принять 🙂"
    )


@dp.errors()
async def on_error(event) -> bool:
    """Глобальный перехват: любое необработанное исключение логируем, бот живёт."""
    log.exception("Необработанная ошибка в обработке апдейта: %s", event.exception)
    return True  # помечаем как обработанное, чтобы polling не падал
