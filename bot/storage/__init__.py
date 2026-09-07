# -*- coding: utf-8 -*-
"""
Персистентность заявок.

Заглушка. Реализация — Задача 3 из SPEC.md: Postgres как источник правды по
заявкам (структурные поля, поиск по id/контакту/категории). Локально —
docker-compose, в проде — Neon.

Планируемый публичный интерфейс этого пакета (сигнатуры могут измениться при
реализации):
    save_lead(item: Item, chat_id: int) -> LeadRecord
    find_lead(lead_id: int) -> LeadRecord | None
    find_leads(contact: str | None = None, category: str | None = None) -> list[LeadRecord]

Этот модуль сознательно бросает NotImplementedError вместо того, чтобы
молча ничего не делать — чтобы случайный вызов из handlers.py был сразу
заметен, а не терялся.
"""

from __future__ import annotations


class StorageNotImplemented(NotImplementedError):
    """Слой хранения ещё не реализован — см. SPEC.md, Задача 3."""


def save_lead(*args, **kwargs):
    raise StorageNotImplemented("bot.storage.save_lead: см. SPEC.md, Задача 3")


def find_lead(*args, **kwargs):
    raise StorageNotImplemented("bot.storage.find_lead: см. SPEC.md, Задача 3")
