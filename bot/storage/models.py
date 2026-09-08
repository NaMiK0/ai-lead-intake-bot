# -*- coding: utf-8 -*-
"""Pydantic-модель строки таблицы leads (то, что реально лежит в БД)."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel


class LeadRecord(BaseModel):
    """Сохранённая заявка. В отличие от bot.models.Item — это уже не сырой
    ответ модели, а строка из БД: есть id, привязка к чату/сообщению,
    исходный текст клиента и время создания. kind не хранится — в таблицу
    попадают только реальные заявки (kind="lead"), not_lead не сохраняется.
    """

    id: int
    chat_id: int
    message_id: Optional[int] = None

    client_name: Optional[str] = None
    company: Optional[str] = None
    contact: Optional[str] = None
    category: Optional[str] = None
    request: Optional[str] = None
    deadline: Optional[str] = None
    budget: Optional[str] = None
    urgency: Optional[str] = None
    urgency_reason: Optional[str] = None
    injection_detected: bool = False

    original_text: str
    created_at: datetime
