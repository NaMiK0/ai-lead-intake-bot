# -*- coding: utf-8 -*-
"""Точка входа: python main.py"""

from __future__ import annotations

import asyncio

from bot.app import main

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        pass
