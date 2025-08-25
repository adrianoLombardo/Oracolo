from __future__ import annotations

import json
import logging
import os
from typing import Any

try:
    from redis import Redis  # type: ignore
except Exception:  # pragma: no cover - redis might be missing at runtime
    Redis = None  # type: ignore

logger = logging.getLogger(__name__)

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

_cache: Redis | None = None
if Redis is not None:
    try:
        _cache = Redis.from_url(REDIS_URL, decode_responses=True)
    except Exception:  # pragma: no cover - connection issues
        logger.debug("Redis connection failed", exc_info=True)
        _cache = None


def _safe_call(func, *args, **kwargs):
    if _cache is None:
        return None
    try:
        return func(*args, **kwargs)
    except Exception:  # pragma: no cover - redis unavailable
        logger.debug("Redis operation failed", exc_info=True)
        return None


def cache_get(key: str) -> str | None:
    """Retrieve a raw string value from Redis."""
    return _safe_call(_cache.get, key)  # type: ignore[arg-type]


def cache_set(key: str, value: str, *, ex: int = 3600) -> None:
    """Store a raw string value in Redis with optional expiry."""
    _safe_call(_cache.set, key, value, ex=ex)  # type: ignore[arg-type]


def cache_get_json(key: str) -> Any:
    """Retrieve a JSON-encoded object from Redis."""
    raw = cache_get(key)
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except Exception:
        return None


def cache_set_json(key: str, value: Any, *, ex: int = 3600) -> None:
    """Store a JSON-serialisable object in Redis."""
    try:
        data = json.dumps(value, ensure_ascii=False)
    except TypeError:
        data = json.dumps(str(value))
    cache_set(key, data, ex=ex)
