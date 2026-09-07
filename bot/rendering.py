# -*- coding: utf-8 -*-
"""Формирование карточки (HTML для Telegram)."""

from __future__ import annotations

import html
from typing import Optional

from .config import CATEGORIES
from .models import Item

URGENCY_EMOJI = {"высокая": "🔴", "средняя": "🟡", "низкая": "🟢"}


def esc(s: Optional[str]) -> str:
    """HTML-экранирование (обязательно для parse_mode=HTML)."""
    return html.escape(s) if s else ""


def render_not_lead(item: Item, original: str, is_dup: bool) -> str:
    reason = esc(item.not_lead_reason) or "сообщение не содержит заявки"
    dup_line = "\n⚠️ Похоже на повтор ранее присланного сообщения." if is_dup else ""
    return (
        f"ℹ️ <b>Не заявка</b>\n"
        f"{reason}.{dup_line}\n\n"
        f"<blockquote expandable>📄 Исходный текст\n{esc(original)}</blockquote>"
    )


def render_lead(item: Item, original: str, idx: int, total: int, is_dup: bool) -> str:
    # Заголовок с нумерацией, если карточек несколько
    header = "📇 <b>Заявка</b>"
    if total > 1:
        header = f"📇 <b>Заявка {idx}/{total}</b>"

    lines = [header]

    # Клиент
    who = " ".join(x for x in [item.client_name, f"«{item.company}»" if item.company else None] if x)
    lines.append(f"Клиент:    {esc(who) if who else '—'}")

    # Контакт (+ флаг отсутствия)
    if item.contact:
        lines.append(f"Контакт:   {esc(item.contact)}")
    else:
        lines.append("Контакт:   — (не указан) ⚠️")

    # Категория (с подстраховкой, если модель выдала что-то вне списка)
    category = item.category if item.category in CATEGORIES else "другое"
    lines.append(f"Категория: {esc(category)}")

    # Запрос (нормализованный)
    lines.append(f"Запрос:    {esc(item.request) if item.request else '—'}")

    # Дедлайн — только если есть
    if item.deadline:
        lines.append(f"Дедлайн:   {esc(item.deadline)}")

    # Бюджет — только если клиент назвал
    if item.budget:
        lines.append(f"Бюджет:    {esc(item.budget)}")

    # Срочность
    urg = item.urgency or "низкая"
    emoji = URGENCY_EMOJI.get(urg, "🟢")
    reason = esc(item.urgency_reason) or "определено по содержанию"
    lines.append(f"{emoji} Срочность: {urg} — {reason}")

    # Флаги-предупреждения
    if total > 1:
        lines.append("⚠️ В сообщении несколько заявок.")
    if is_dup:
        lines.append("⚠️ Похоже на повтор ранее присланного сообщения.")
    if item.injection_detected:
        lines.append(
            "⚠️ В тексте обнаружена инструкция «для системы» — она "
            "проигнорирована, поля определены самостоятельно."
        )

    body = "\n".join(lines)
    original_block = (
        f"\n\n<blockquote expandable>📄 Исходный текст\n{esc(original)}</blockquote>"
    )
    return body + original_block
