from __future__ import annotations

import time
from collections import Counter
from typing import Any, Callable, TypeVar

T = TypeVar("T")

# Global counter for telemetry of errors per exception type
ERROR_COUNTS: Counter[str] = Counter()


def retry_with_backoff(
    func: Callable[..., T],
    *args: Any,
    retries: int = 3,
    base_delay: float = 0.5,
    **kwargs: Any,
) -> T:
    """Call ``func`` with retries and exponential backoff.

    Exceptions are counted in :data:`ERROR_COUNTS` keyed by class name.
    The delay grows exponentially: ``base_delay * 2**attempt``.
    """

    for attempt in range(retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:  # noqa: BLE001 - count all exceptions
            ERROR_COUNTS[type(e).__name__] += 1
            if attempt >= retries - 1:
                raise
            time.sleep(base_delay * (2 ** attempt))
    # Unreachable but helps typing
    raise RuntimeError("retry_with_backoff: exhausted retries without raising")
