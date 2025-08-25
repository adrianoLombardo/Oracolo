from __future__ import annotations

"""Utilities to offload expensive OpenAI calls to a thread pool."""

from concurrent.futures import Future, ThreadPoolExecutor
from typing import Callable, TypeVar, Any
import asyncio
import atexit

from .config import Settings

T = TypeVar("T")

_EXECUTOR: ThreadPoolExecutor | None = None


def _get_executor() -> ThreadPoolExecutor:
    """Lazily create the shared executor."""

    global _EXECUTOR
    if _EXECUTOR is None:
        workers = Settings().openai.max_workers
        _EXECUTOR = ThreadPoolExecutor(max_workers=workers)
    return _EXECUTOR


def submit(func: Callable[..., T], *args: Any, **kwargs: Any) -> Future:
    """Submit ``func`` to the shared executor and return a :class:`Future`."""

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

    global _EXECUTOR
    if _EXECUTOR is not None:
        _EXECUTOR.shutdown(wait=True)
        _EXECUTOR = None


atexit.register(shutdown)
