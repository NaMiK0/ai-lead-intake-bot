# -*- coding: utf-8 -*-
"""
Персистентность заявок (Postgres).

Сохраняются только реальные заявки (kind="lead") — not_lead (спам/оффтоп/
благодарности) в БД не попадает, там нечего искать и не с чем сравнивать
в будущем RAG-контуре.

Публичный API:
    init_pool(dsn) / close_pool()      — вызывать из bot/app.py
    save_lead(item, chat_id, ...)      — сохранить одну заявку
    find_lead(lead_id)                 — найти по id
    find_leads(contact=..., category=...) — найти по фильтрам
"""

from __future__ import annotations

from typing import Optional

from ..models import Item
from .db import close_pool, get_pool, init_pool
from .models import LeadRecord

__all__ = [
    "init_pool",
    "close_pool",
    "save_lead",
    "find_lead",
    "find_leads",
    "LeadRecord",
]


def _row_to_record(row) -> LeadRecord:
    return LeadRecord(**dict(row))


async def save_lead(
    item: Item,
    *,
    chat_id: int,
    message_id: Optional[int] = None,
    original_text: str,
) -> LeadRecord:
    """Сохранить одну заявку (item.kind должен быть "lead")."""
    assert item.kind == "lead", "save_lead() принимает только kind='lead'"
    pool = get_pool()
    row = await pool.fetchrow(
        """
        INSERT INTO leads (
            chat_id, message_id, client_name, company, contact, category,
            request, deadline, budget, urgency, urgency_reason,
            injection_detected, original_text
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
        RETURNING *
        """,
        chat_id,
        message_id,
        item.client_name,
        item.company,
        item.contact,
        item.category,
        item.request,
        item.deadline,
        item.budget,
        item.urgency,
        item.urgency_reason,
        item.injection_detected,
        original_text,
    )
    return _row_to_record(row)


async def find_lead(lead_id: int) -> Optional[LeadRecord]:
    """Найти заявку по id. None, если не найдена."""
    pool = get_pool()
    row = await pool.fetchrow("SELECT * FROM leads WHERE id = $1", lead_id)
    return _row_to_record(row) if row else None


async def find_leads(
    *,
    contact: Optional[str] = None,
    category: Optional[str] = None,
    limit: int = 20,
) -> list[LeadRecord]:
    """Найти заявки по необязательным фильтрам (contact — частичное
    совпадение без учёта регистра, category — точное). Без фильтров —
    последние `limit` заявок."""
    conditions: list[str] = []
    params: list[object] = []

    if contact:
        params.append(f"%{contact}%")
        conditions.append(f"contact ILIKE ${len(params)}")
    if category:
        params.append(category)
        conditions.append(f"category = ${len(params)}")

    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    params.append(limit)

    pool = get_pool()
    rows = await pool.fetch(
        f"SELECT * FROM leads {where} ORDER BY created_at DESC LIMIT ${len(params)}",
        *params,
    )
    return [_row_to_record(row) for row in rows]
