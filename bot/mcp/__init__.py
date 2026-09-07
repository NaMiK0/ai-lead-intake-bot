# -*- coding: utf-8 -*-
"""
MCP-сервер поверх функций бота.

Заглушка. Реализация — Задача 5 из SPEC.md: обёртка bot.storage.find_lead и
bot.rag.similar_leads как MCP tools, с проверкой подключения через локальный
Claude Desktop. Зависит от Задач 3 и 4 — оборачивать пока нечего.

Ожидаемая форма после реализации: отдельный запускаемый модуль
(например `python -m bot.mcp.server`), поднимающий MCP-сервер (mcp SDK,
stdio- или SSE-транспорт) с двумя инструментами:
    find_lead(lead_id: int) -> dict
    similar_leads(query: str, top_k: int = 3) -> list[dict]
"""

from __future__ import annotations


class McpNotImplemented(NotImplementedError):
    """MCP-сервер ещё не реализован — см. SPEC.md, Задача 5."""
