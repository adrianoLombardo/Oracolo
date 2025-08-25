from __future__ import annotations

"""Utilities to offload expensive OpenAI calls to a thread pool."""

from concurrent.futures import ThreadPoolExecutor, Future
from typing import Callable, TypeVar, Any
import asyncio

T = TypeVar("T")

# A small global executor used across the application.  The size is kept
# conservative so that we do not spawn an excessive amount of threads even if
# many modules import this helper.
_EXECUTOR = ThreadPoolExecutor(max_workers=4)


def submit(func: Callable[..., T], *args: Any, **kwargs: Any) -> Future:
    """Submit ``func`` to the shared executor and return a :class:`Future`."""

    return _EXECUTOR.submit(func, *args, **kwargs)


def run(func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    """Run ``func`` synchronously in the executor and wait for the result."""

    return submit(func, *args, **kwargs).result()


def run_async(func: Callable[..., T], *args: Any, **kwargs: Any) -> asyncio.Future:
    """Awaitable version of :func:`run` using ``asyncio`` integration."""

    loop = asyncio.get_running_loop()
    return loop.run_in_executor(_EXECUTOR, lambda: func(*args, **kwargs))
