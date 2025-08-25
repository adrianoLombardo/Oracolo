from __future__ import annotations

"""Utilities to offload expensive OpenAI calls to a thread pool."""

from concurrent.futures import Future, ThreadPoolExecutor
from typing import Callable, TypeVar, Any
import asyncio
import os
import atexit

from .config import Settings


T = TypeVar("T")

# Shared executor instance
_executor: ThreadPoolExecutor | None = None


def _get_max_workers() -> int:
    """Return desired worker count from env or settings."""
    env_value = os.getenv("ORACOLO_MAX_WORKERS")
    if env_value:
        try:
            return int(env_value)
        except ValueError:
            pass
    return Settings().openai.max_workers


def _get_executor() -> ThreadPoolExecutor:
    """Lazily create the shared executor."""
    global _executor
    if _executor is None:
        _executor = ThreadPoolExecutor(max_workers=_get_max_workers())
        atexit.register(_executor.shutdown)
    return _executor


def submit(func: Callable[..., T], *args: Any, **kwargs: Any) -> Future:
    """Submit ``func`` to the shared executor and return a Future."""
    return _get_executor().submit(func, *args, **kwargs)


def run(func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    """Run ``func`` synchronously in the executor and wait for the result."""
    return submit(func, *args, **kwargs).result()


def run_async(func: Callable[..., T], *args: Any, **kwargs: Any) -> asyncio.Future:
    """Awaitable version of :func:`run` using ``asyncio`` integration."""
    loop = asyncio.get_running_loop()
    return loop.run_in_executor(_get_executor(), lambda: func(*args, **kwargs))


def shutdown() -> None:
    """Shutdown the shared executor."""
    global _executor
    if _executor is not None:
        _executor.shutdown(wait=True)
        _executor = None


atexit.register(shutdown)
