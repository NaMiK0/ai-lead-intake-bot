# Lead-Intake Bot

Telegram-бот, который разбирает сырые сообщения клиентов в структурированные
заявки (кто клиент, контакт, что хочет, насколько срочно) с помощью LLM.

> Проект в активной разработке. Полное описание архитектуры, стека и инструкция
> по запуску через Docker Compose появятся позже.

## Быстрый старт

```bash
docker compose up -d postgres   # локальная БД (см. docker-compose.yml)
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
├── rag/          # 🔲 эмбеддинги и поиск похожих заявок (Qdrant) — в разработке
└── mcp/          # 🔲 MCP-сервер поверх storage/rag — в разработке
main.py             # точка входа
docker-compose.yml  # локальный Postgres для разработки
```
