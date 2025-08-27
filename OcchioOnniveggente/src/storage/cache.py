from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Callable, TypeVar

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


T = TypeVar("T")


def _safe_call(func: Callable[..., T], *args: Any, **kwargs: Any) -> T | None:
    if _cache is None:
        return None
    try:
        return func(*args, **kwargs)
    except Exception:  # pragma: no cover - redis unavailable
        logger.debug("Redis operation failed", exc_info=True)
        return None


def cache_get(key: str) -> str | None:
    """Retrieve a raw string value from Redis."""
    if _cache is None:
        return None
    return _safe_call(_cache.get, key)  # type: ignore[arg-type]


def cache_set(key: str, value: str, *, ex: int = 3600) -> None:
    """Store a raw string value in Redis with optional expiry."""
    if _cache is None:
        return

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


def cache_set_json(key: str, value: Any, *, ttl: int = 3600) -> None:
    """Store a JSON-serialisable object in Redis."""
    try:
        data = json.dumps(value, ensure_ascii=False)
    except TypeError:
        data = json.dumps(str(value))
    cache_set(key, data, ex=ttl)


# ---------------------------------------------------------------------------
#  File-based cache helpers for TTS/STT outputs
# ---------------------------------------------------------------------------

from ..config import Settings  # noqa: E402
from ..utils.container import get_container


def _settings() -> Settings:
    container = get_container()
    return getattr(container, "settings", Settings())


def _cache_root() -> Path:
    return Path(getattr(_settings(), "cache_dir", "data/cache"))


def _ttl() -> int:
    return getattr(_settings(), "cache_ttl", 3600)


def _is_fresh(path: Path, ttl: int) -> bool:
    try:
        return time.time() - path.stat().st_mtime < ttl
    except OSError:
        return False


def _tts_cache_path(text: str, voice: str) -> Path:
    key = hashlib.sha256(f"{voice}:{text}".encode("utf-8")).hexdigest()
    root = _cache_root() / "tts"
    root.mkdir(parents=True, exist_ok=True)
    return root / f"{key}.wav"


def get_tts_cache(text: str, voice: str) -> Path | None:
    """Return cached TTS audio ``Path`` or ``None`` if missing/expired."""
    path = _tts_cache_path(text, voice)
    if _is_fresh(path, _ttl()):
        return path
    if path.exists():
        try:
            path.unlink()
        except OSError:
            pass
    return None


def set_tts_cache(text: str, voice: str, src: Path) -> Path:
    dst = _tts_cache_path(text, voice)
    try:
        dst.write_bytes(src.read_bytes())
    except OSError:
        logger.debug("Failed writing TTS cache", exc_info=True)
    return dst


def _stt_cache_path(audio_hash: str) -> Path:
    root = _cache_root() / "stt"
    root.mkdir(parents=True, exist_ok=True)
    return root / f"{audio_hash}.txt"


def get_stt_cache(audio_hash: str) -> str | None:
    """Return cached transcription for ``audio_hash`` or ``None``."""
    path = _stt_cache_path(audio_hash)
    if _is_fresh(path, _ttl()):
        try:
            return path.read_text(encoding="utf-8")
        except OSError:
            return None
    if path.exists():
        try:
            path.unlink()
        except OSError:
            pass
    return None


def set_stt_cache(audio_hash: str, text: str) -> Path:
    path = _stt_cache_path(audio_hash)
    try:
        path.write_text(text, encoding="utf-8")
    except OSError:
        logger.debug("Failed writing STT cache", exc_info=True)
    return path


def cleanup_cache(ttl: int | None = None) -> int:
    """Remove cache files older than ``ttl`` seconds. Returns count removed."""
    root = _cache_root()
    ttl = ttl or _ttl()
    now = time.time()
    removed = 0
    for p in root.glob("**/*"):
        if p.is_file() and now - p.stat().st_mtime > ttl:
            try:
                p.unlink()
                removed += 1
            except OSError:
                pass
    return removed

