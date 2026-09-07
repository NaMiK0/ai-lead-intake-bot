# -*- coding: utf-8 -*-
"""
Дедуп в памяти (без диска).

Это временное решение до Задачи 3 (SPEC.md): когда появится Postgres, дедуп
по (chat_id, message_id) можно будет оставить в памяти (это чисто защита от
повторной обработки одного апдейта в рамках текущего процесса), а вот
is_duplicate_text по-хорошему тоже переедет на чтение из БД, чтобы работать
между рестартами. Пока — как было.
"""

from __future__ import annotations

import time
from collections import OrderedDict

from .config import DUP_CACHE_MAX, DUP_TTL_S, SEEN_IDS_MAX

# LRU обработанных сообщений: гарантирует «не отвечаем дважды на одно».
_seen_ids: "OrderedDict[tuple[int, int], None]" = OrderedDict()
# Кэш хешей текста для флага «возможный дубль» (одно и то же по нескольку раз).
_recent_hashes: "OrderedDict[int, float]" = OrderedDict()


def already_processed(chat_id: int, message_id: int) -> bool:
    """True, если это сообщение уже обрабатывали (в рамках текущей сессии)."""
    key = (chat_id, message_id)
    if key in _seen_ids:
        return True
    _seen_ids[key] = None
    _seen_ids.move_to_end(key)
    while len(_seen_ids) > SEEN_IDS_MAX:
        _seen_ids.popitem(last=False)  # вытесняем самое старое
    return False


def is_duplicate_text(text: str) -> bool:
    """True, если такой же текст уже прилетал недавно (в пределах DUP_TTL_S)."""
    now = time.time()
    # чистим протухшие записи
    for h, ts in list(_recent_hashes.items()):
        if now - ts > DUP_TTL_S:
            _recent_hashes.pop(h, None)
        else:
            break  # OrderedDict упорядочен по вставке — дальше только свежее
    key = hash(text.strip().lower())
    dup = key in _recent_hashes
    _recent_hashes[key] = now
    _recent_hashes.move_to_end(key)
    while len(_recent_hashes) > DUP_CACHE_MAX:
        _recent_hashes.popitem(last=False)
    return dup
