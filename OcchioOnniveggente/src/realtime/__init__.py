"""Realtime utilities.

This package exposes the lightweight :class:`TaskQueue` used by the realtime
clients.  Worker helpers live in :mod:`realtime.workers` and are intentionally
not imported here to keep package imports fast and side‑effect free.
"""

from .queue import TaskQueue, task_queue

__all__ = ["TaskQueue", "task_queue"]

