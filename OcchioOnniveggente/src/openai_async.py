from __future__ import annotations

"""Utilities to offload expensive OpenAI calls to a thread pool."""

from concurrent.futures import Future, ThreadPoolExecutor
from typing import Callable, TypeVar, Any
import asyncio
import os
import atexit

from .service_container import container

T = TypeVar("T")

_executor: ThreadPoolExecutor | None = None


def _get_max_workers() -> int:
    """Return the desired worker count from env or settings."""
    env_value = os.getenv("ORACOLO_MAX_WORKERS")
    if env_value:
        try:
            return int(env_value)
        except ValueError:
            pass
    return container.settings.openai.max_workers


def _executor_instance() -> ThreadPoolExecutor:
    """Lazily create a thread pool executor."""
    global _executor
    if _executor is None:
        _executor = ThreadPoolExecutor(max_workers=_get_max_workers())
        atexit.register(_executor.shutdown)
    return _executor


def submit(func: Callable[..., T], *args: Any, **kwargs: Any) -> Future:
    """Submit ``func`` to the shared executor and return a :class:`Future`."""

    return _executor_instance().submit(func, *args, **kwargs)


def run(func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    """Run ``func`` synchronously in the executor and wait for the result."""

    return submit(func, *args, **kwargs).result()


def run_async(func: Callable[..., T], *args: Any, **kwargs: Any) -> asyncio.Future:
    """Awaitable version of :func:`run` using ``asyncio`` integration."""

    loop = asyncio.get_running_loop()
    return loop.run_in_executor(_executor_instance(), lambda: func(*args, **kwargs))
