# -*- coding: utf-8 -*-
"""
Схема ответа модели — валидируется через pydantic как второй уровень проверки
(первый уровень — сама структура ответа LLM, см. bot/llm.py).
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel


class Item(BaseModel):
    """Одна карточка: либо заявка (lead), либо не-заявка (not_lead)."""
    kind: Literal["lead", "not_lead"]
    not_lead_reason: Optional[str] = None      # почему это не заявка

    client_name: Optional[str] = None
    company: Optional[str] = None
    contact: Optional[str] = None              # телефон / email / @username
    category: Optional[str] = None
    request: Optional[str] = None              # нормализовано: без КАПСА и воды
    deadline: Optional[str] = None
    budget: Optional[str] = None               # только если клиент сам назвал
    urgency: Optional[Literal["высокая", "средняя", "низкая"]] = None
    urgency_reason: Optional[str] = None
    injection_detected: bool = False           # была ли попытка инструкции в тексте


class Extraction(BaseModel):
    items: list[Item]
