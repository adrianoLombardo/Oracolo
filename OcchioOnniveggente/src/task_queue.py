from __future__ import annotations

"""Simple in-memory task queue for background workers.

The queue groups jobs by name; each job is processed by a dedicated worker
coroutine registered for that name.  It is intentionally lightweight so that
unit tests can run without external services such as Redis or RabbitMQ.
"""

import asyncio
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, Tuple


@dataclass
class Job:
    name: str
    args: Tuple[Any, ...]
    kwargs: Dict[str, Any]


class TaskQueue:
    """Minimal task queue used to decouple work from callers."""

    def __init__(self) -> None:
        self._queues: Dict[str, asyncio.Queue[Job]] = defaultdict(asyncio.Queue)

    def publish(self, name: str, *args: Any, **kwargs: Any) -> None:
        """Enqueue a job ``name`` with the provided arguments."""

        self._queues[name].put_nowait(Job(name, args, kwargs))

    async def worker(
        self, name: str, handler: Callable[..., Awaitable[Any] | Any]
    ) -> None:
        """Continuously process jobs for ``name`` using ``handler``."""

        q = self._queues[name]
        while True:
            job = await q.get()
            try:
                res = handler(*job.args, **job.kwargs)
                if asyncio.iscoroutine(res):
                    await res
            finally:
                q.task_done()

    def size(self, name: str) -> int:
        """Return the approximate size of the queue for ``name``."""

        return self._queues[name].qsize()


task_queue = TaskQueue()

__all__ = ["TaskQueue", "task_queue"]
