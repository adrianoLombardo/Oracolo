from __future__ import annotations

"""Async helpers for potentially blocking OpenAI calls.

This module avoids using a dedicated ``ThreadPoolExecutor``. When a GPU is
available we fall back to a ``ProcessPoolExecutor`` with a single worker to
allow exclusive access to the device. Otherwise ``asyncio.to_thread`` is used
to offload blocking work without creating additional management overhead.
"""

from concurrent.futures import Future, ProcessPoolExecutor
from typing import Callable, TypeVar, Any
import asyncio
import os
import atexit
try:
    import torch
except Exception:  # pragma: no cover
    class _CudaStub:
        @staticmethod
        def is_available() -> bool:
            return False

    class _TorchStub:
        cuda = _CudaStub()

    torch = _TorchStub()  # type: ignore

from .config import Settings


T = TypeVar("T")

_executor: ProcessPoolExecutor | None = None


def _get_max_workers() -> int:
    env_value = os.getenv("ORACOLO_MAX_WORKERS")
    if env_value:
        try:
            return int(env_value)
        except ValueError:
            pass
    return Settings().openai.max_workers


def _get_executor() -> ProcessPoolExecutor | None:
    """Create a process pool when a GPU is present."""
    global _executor
    if _executor is None and torch.cuda.is_available():
        workers = max(1, _get_max_workers())
        _executor = ProcessPoolExecutor(max_workers=workers)
        atexit.register(_executor.shutdown)
    return _executor


def submit(func: Callable[..., T], *args: Any, **kwargs: Any) -> Future:
    exec_ = _get_executor()
    if exec_ is not None:
        return exec_.submit(func, *args, **kwargs)
    loop = asyncio.get_event_loop()
    return loop.run_in_executor(None, lambda: func(*args, **kwargs))


def run(func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    return submit(func, *args, **kwargs).result()


def run_async(func: Callable[..., T], *args: Any, **kwargs: Any) -> asyncio.Future:
    loop = asyncio.get_running_loop()
    exec_ = _get_executor()
    if exec_ is not None:
        return loop.run_in_executor(exec_, lambda: func(*args, **kwargs))
    return asyncio.to_thread(func, *args, **kwargs)


def shutdown() -> None:
    global _executor
    if _executor is not None:
        _executor.shutdown(wait=True)
        _executor = None


atexit.register(shutdown)

