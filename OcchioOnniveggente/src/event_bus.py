import asyncio
from collections import defaultdict
from typing import Any, Callable, Awaitable, Dict, List


class EventBus:
    """Simple event bus built on top of :class:`asyncio.Queue`.

    Publishers enqueue events via :meth:`publish` while subscribers register
    callbacks with :meth:`subscribe`.  Callbacks may be normal callables or
    coroutines.  If an asyncio event loop is running, coroutine callbacks are
    scheduled on it, otherwise they are executed synchronously via
    :func:`asyncio.run`.
    """

    def __init__(self) -> None:
        self._listeners: Dict[str, List[Callable[..., Any]]] = defaultdict(list)
        self._queue: asyncio.Queue[tuple[str, tuple[Any, ...], dict[str, Any]]] = (
            asyncio.Queue()
        )

    def subscribe(self, event: str, callback: Callable[..., Any]) -> None:
        """Register *callback* to be invoked when *event* is published."""

        self._listeners[event].append(callback)

    def publish(self, event: str, *args: Any, **kwargs: Any) -> None:
        """Publish *event* with the provided arguments."""

        self._queue.put_nowait((event, args, kwargs))
        while not self._queue.empty():
            name, a, kw = self._queue.get_nowait()
            for cb in list(self._listeners.get(name, [])):
                res = cb(*a, **kw)
                if asyncio.iscoroutine(res):
                    try:
                        loop = asyncio.get_running_loop()
                        loop.create_task(res)
                    except RuntimeError:
                        asyncio.run(res)


event_bus = EventBus()

__all__ = ["EventBus", "event_bus"]
