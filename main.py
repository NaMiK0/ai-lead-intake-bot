# -*- coding: utf-8 -*-
"""
Точка входа: python main.py

Режим определяется BOT_MODE (см. .env.example): "polling" (по умолчанию,
локальная разработка) или "webhook" (прод, Render) — см. bot/app.py.
"""

from __future__ import annotations

import asyncio

from bot.app import main, run_webhook
from bot.config import BOT_MODE

if __name__ == "__main__":
    if BOT_MODE == "webhook":
        run_webhook()
    else:
        try:
            asyncio.run(main())
        except (KeyboardInterrupt, SystemExit):
            pass
