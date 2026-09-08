# Lead-Intake Bot

Telegram-бот, который разбирает сырые сообщения клиентов в структурированные
заявки (кто клиент, контакт, что хочет, насколько срочно) с помощью LLM.

> Проект в активной разработке. Полное описание архитектуры, стека и инструкция
> по запуску через Docker Compose появятся позже.

## Быстрый старт

```bash
docker compose up -d postgres qdrant   # локальные БД (см. docker-compose.yml)
pip install -r requirements.txt
cp .env.example .env   # и подставь свои токены
export $(cat .env | xargs)
python main.py
```

Нужны (см. `.env.example`):
- `TELEGRAM_BOT_TOKEN` — от [@BotFather](https://t.me/BotFather)
- `OPENROUTER_API_KEY` — с [openrouter.ai/keys](https://openrouter.ai/keys)
- `DATABASE_URL` — строка подключения к Postgres; для локального
  `docker compose up -d postgres` значение по умолчанию уже подходит и
  указано в `.env.example`. Схема БД создаётся автоматически при старте бота.
- `QDRANT_URL` (+ `QDRANT_API_KEY` для облака) — для RAG (поиск похожих
  прошлых заявок); для локального `docker compose up -d qdrant` значение по
  умолчанию уже подходит. Коллекция создаётся автоматически при старте бота.
  При первом запуске модель эмбеддингов (~225 МБ) скачивается один раз и
  кэшируется — старт может занять чуть дольше.

## Структура проекта

```
bot/
├── config.py     # переменные окружения, константы
├── models.py     # pydantic-схема заявки
├── llm.py        # вызов LLM с каскадом фолбэков
├── dedup.py      # дедуп сообщений (in-memory)
├── rendering.py  # карточки для Telegram
├── handlers.py   # aiogram-хэндлеры
├── app.py        # сборка и запуск бота
├── storage/      # персистентность заявок (Postgres, asyncpg, без ORM)
├── rag/          # эмбеддинги (fastembed) и поиск похожих заявок (Qdrant)
└── mcp/          # MCP-сервер поверх storage/rag (find_lead, similar_leads)
main.py             # точка входа
docker-compose.yml  # локальные Postgres + Qdrant для разработки
```

## MCP-сервер

`bot/mcp/server.py` — тонкая обёртка `find_lead`/`similar_leads` как MCP tools.
Не участвует в обработке сообщений Telegram-ботом — отдельный, необязательный
интерфейс для чтения тех же данных из MCP-клиента (например Claude Desktop).
Локальный, stdio-транспорт: клиент сам запускает процесс, поднимать сервер
отдельно не нужно.

Подключение из Claude Desktop — в `claude_desktop_config.json` (Settings →
Developer → Edit Config):

```json
{
  "mcpServers": {
    "lead-bot": {
      "command": "/path/to/lead_bot/.venv/bin/python3",
      "args": ["-m", "bot.mcp.server"],
      "cwd": "/path/to/lead_bot",
      "env": {
        "DATABASE_URL": "postgresql://lead_bot:lead_bot@localhost:5432/lead_bot",
        "QDRANT_URL": "http://localhost:6333"
      }
    }
  }
}
```

Требует запущенных локальных `postgres`/`qdrant` (`docker compose up -d`).
После правки конфига — перезапустить Claude Desktop.
