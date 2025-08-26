"""Simple in-memory rate limiting utilities."""
from __future__ import annotations

import time
from collections import defaultdict, deque
from typing import Deque, Dict

from .exceptions import RateLimitExceeded
import os


class RateLimiter:
    """Token-bucket style rate limiter."""

    def __init__(self, limit: int, period: float) -> None:
        self.limit = limit
        self.period = period
        self._hits: Dict[str, Deque[float]] = defaultdict(deque)

    def hit(self, key: str) -> None:
        now = time.monotonic()
        dq = self._hits[key]
        while dq and now - dq[0] > self.period:
            dq.popleft()
        if len(dq) >= self.limit:
            raise RateLimitExceeded(f"rate limit exceeded for {key}")
        dq.append(now)


# Global limiter used by the application. Limits can be customised through
# ``ORACOLO_RATE_LIMIT`` and ``ORACOLO_RATE_PERIOD`` environment variables.
_DEFAULT_LIMIT = int(os.getenv("ORACOLO_RATE_LIMIT", "100"))
_DEFAULT_PERIOD = float(os.getenv("ORACOLO_RATE_PERIOD", "1.0"))
rate_limiter = RateLimiter(limit=_DEFAULT_LIMIT, period=_DEFAULT_PERIOD)
