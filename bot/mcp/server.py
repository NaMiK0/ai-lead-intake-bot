# -*- coding: utf-8 -*-
"""
MCP-сервер поверх функций бота — тонкая обёртка, без новой бизнес-логики.

Два инструмента:
    find_lead(lead_id)          -> bot.storage.find_lead
    similar_leads(query, top_k) -> bot.rag.similar_leads

Локальный, stdio-транспорт: MCP-клиент (Claude Desktop) сам запускает этот
модуль как подпроцесс и общается с ним через stdin/stdout — отдельно
поднимать сервер не нужно. Не связан с Telegram-ботом: читает те же
Postgres/Qdrant, но не участвует в обработке сообщений клиентов.

Запуск вручную (для отладки):
    python -m bot.mcp.server

Требует переменные окружения (см. .env.example): DATABASE_URL обязателен —
без него сервер не стартует, find_lead работать не может в принципе.
QDRANT_URL/QDRANT_API_KEY нужны только для similar_leads — как и в
Telegram-боте (bot/app.py), RAG необязателен: если Qdrant недоступен,
find_lead всё равно работает, а similar_leads вернёт понятную ошибку
вместо падения всего сервера.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from mcp.server.mcpserver import MCPServer

from .. import rag, storage
from ..config import DATABASE_URL, log


@asynccontextmanager
async def _lifespan(_: MCPServer) -> AsyncIterator[None]:
    if not DATABASE_URL:
        raise RuntimeError(
            "DATABASE_URL не задан — задай его в конфигурации MCP-сервера "
            "(env в claude_desktop_config.json) или в окружении процесса."
        )
    await storage.init_pool(DATABASE_URL)
    try:
        await rag.init_rag()
    except Exception as e:  # noqa: BLE001 — RAG необязателен, как и в bot/app.py
        log.warning("MCP: RAG-контур не поднялся, similar_leads будет недоступен: %s", e)
    try:
        yield
    finally:
        await storage.close_pool()
        await rag.close_rag()


app = MCPServer(
    name="lead-bot",
    instructions=(
        "Инструменты для чтения заявок IT-студии, принятых Telegram-ботом: "
        "find_lead — заявка по id из Postgres; similar_leads — прошлые "
        "заявки, похожие по смыслу на переданный текст (поиск в Qdrant)."
    ),
    lifespan=_lifespan,
)


@app.tool()
async def find_lead(lead_id: int) -> dict[str, Any]:
    """Найти заявку по id. Если такой заявки нет — {"found": false}."""
    record = await storage.find_lead(lead_id)
    if record is None:
        return {"found": False, "lead_id": lead_id}
    return {"found": True, **record.model_dump(mode="json")}


@app.tool()
async def similar_leads(query: str, top_k: int = 3) -> list[dict[str, Any]]:
    """Похожие по смыслу прошлые заявки для текста query (RAG-поиск в Qdrant,
    см. bot/rag). Пустой список — либо ничего достаточно похожего не нашлось,
    либо RAG-контур недоступен (см. лог сервера при старте)."""
    records = await rag.similar_leads(query, top_k=top_k)
    return [r.model_dump(mode="json") for r in records]


def main() -> None:
    app.run()


if __name__ == "__main__":
    main()
