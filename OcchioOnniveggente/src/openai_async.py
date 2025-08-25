"""Lightweight helpers to run async OpenAI calls without a thread pool."""
from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable, TypeVar

T = TypeVar("T")


def run(func: Callable[..., Awaitable[T]], *args: Any, **kwargs: Any) -> T:
    """Run ``func`` and await it if it returns a coroutine."""
    result = func(*args, **kwargs)
    if asyncio.iscoroutine(result):
        return asyncio.run(result)
    return result


async def run_async(func: Callable[..., Awaitable[T]], *args: Any, **kwargs: Any) -> T:
    """Await ``func`` directly."""
    return await func(*args, **kwargs)


def shutdown() -> None:  # pragma: no cover - compatibility stub
    """Previously closed the thread pool; now a no-op for compatibility."""
    return None
