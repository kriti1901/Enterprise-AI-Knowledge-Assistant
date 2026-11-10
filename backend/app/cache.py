"""Redis answer cache. Optional: every function is a no-op when REDIS_URL is not set or Redis is down.

Cache keys include the caller's departments and a document-index version that is bumped on every
upload or delete, so cached answers are never served across access changes or stale indexes.
"""

from __future__ import annotations

import hashlib
import json
import logging
from functools import lru_cache
from typing import Any

from app.config import get_settings

logger = logging.getLogger("kb")
VERSION_KEY = "kb:index_version"
ANSWER_TTL_SECONDS = 300


@lru_cache
def get_redis():
    url = get_settings().redis_url
    if not url:
        return None
    import redis

    return redis.Redis.from_url(url, socket_timeout=0.5, decode_responses=True)


def _key(departments: list[str], question: str, mode: str) -> str | None:
    client = get_redis()
    if client is None:
        return None
    try:
        version = client.get(VERSION_KEY) or "0"
    except Exception:
        return None
    raw = json.dumps([sorted(departments), question.strip().lower(), mode, version])
    return "kb:answer:" + hashlib.sha256(raw.encode()).hexdigest()


def get_cached_answer(departments: list[str], question: str, mode: str) -> dict[str, Any] | None:
    key = _key(departments, question, mode)
    if key is None:
        return None
    try:
        value = get_redis().get(key)
    except Exception:
        return None
    return json.loads(value) if value else None


def set_cached_answer(departments: list[str], question: str, mode: str, payload: dict[str, Any]) -> None:
    key = _key(departments, question, mode)
    if key is None:
        return
    try:
        get_redis().setex(key, ANSWER_TTL_SECONDS, json.dumps(payload))
    except Exception:
        logger.warning("redis_set_failed")


def bump_index_version() -> None:
    client = get_redis()
    if client is None:
        return
    try:
        client.incr(VERSION_KEY)
    except Exception:
        logger.warning("redis_incr_failed")
