"""Prometheus metrics utilities for the backend."""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Callable, Dict

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
    start_http_server,
)
from starlette.requests import Request
from starlette.responses import Response
from .task_queue import task_queue

try:  # pragma: no cover - optional dependencies
    import torch  # type: ignore
except Exception:  # pragma: no cover
    torch = None  # type: ignore

try:  # pragma: no cover - optional dependencies
    import psutil  # type: ignore
except Exception:  # pragma: no cover
    psutil = None  # type: ignore

logger = logging.getLogger(__name__)

"""Prometheus metrics and simple autoscaling helpers."""

REGISTRY = CollectorRegistry()

# HTTP request metrics
REQUEST_COUNT = Counter(
    "http_requests_total", "Total HTTP requests", ["method", "endpoint"], registry=REGISTRY
)
REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds", "HTTP request latency", ["method", "endpoint"], registry=REGISTRY
)
REQUEST_ERRORS = Counter(
    "http_request_errors_total",
    "Total HTTP requests returning errors",
    ["method", "endpoint", "status"],
    registry=REGISTRY,
)

# Application level errors
LOG_ERRORS = Counter(
    "application_errors_total",
    "Total number of error log records",
    ["logger"],
    registry=REGISTRY,
)

# System metrics
GPU_MEMORY = Gauge("gpu_memory_bytes", "Allocated GPU memory in bytes", registry=REGISTRY)
GPU_UTILIZATION = Gauge("gpu_utilization_percent", "GPU utilisation percentage", registry=REGISTRY)
CPU_PERCENT = Gauge("cpu_usage_percent", "CPU usage percentage", registry=REGISTRY)
QUEUE_LENGTH = Gauge(
    "task_queue_length", "Number of pending tasks", ["job"], registry=REGISTRY
)


async def metrics_middleware(request: Request, call_next):  # type: ignore[no-untyped-def]
    """Middleware capturing request count, errors and latency for each call."""
    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        # Unhandled exception – count as 500 and re-raise
        REQUEST_ERRORS.labels(request.method, request.url.path, "500").inc()
        REQUEST_COUNT.labels(request.method, request.url.path).inc()
        REQUEST_LATENCY.labels(request.method, request.url.path).observe(
            time.perf_counter() - start
        )
        raise

    duration = time.perf_counter() - start
    REQUEST_COUNT.labels(request.method, request.url.path).inc()
    REQUEST_LATENCY.labels(request.method, request.url.path).observe(duration)
    if response.status_code >= 400:
        REQUEST_ERRORS.labels(
            request.method, request.url.path, str(response.status_code)
        ).inc()
    return response


def metrics_endpoint() -> Response:
    """Expose collected metrics in Prometheus text format."""
    return Response(generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)


def start_metrics_server(port: int = 8000) -> None:
    """Start a dedicated Prometheus exporter on ``port``.

    Useful when the hosting application cannot easily expose the
    ``/metrics`` endpoint (e.g. for standalone scripts).
    """
    start_http_server(port, registry=REGISTRY)


def read_system_metrics() -> Dict[str, float]:
    """Return current GPU memory/utilisation and CPU usage."""

    gpu_mem = gpu_util = 0.0
    if torch is not None and getattr(torch, "cuda", None) and torch.cuda.is_available():
        try:
            gpu_mem = float(torch.cuda.memory_allocated())
        except Exception:  # pragma: no cover - defensive
            gpu_mem = 0.0
        util_fn = getattr(torch.cuda, "utilization", None)
        if callable(util_fn):
            try:
                gpu_util = float(util_fn())
            except Exception:  # pragma: no cover - defensive
                gpu_util = 0.0
    cpu = float(psutil.cpu_percent()) if psutil is not None else 0.0
    return {"gpu_memory": gpu_mem, "gpu_util": gpu_util, "cpu": cpu}


def record_system_metrics() -> Dict[str, float]:
    """Update Prometheus gauges with current system metrics."""

    m = read_system_metrics()
    GPU_MEMORY.set(m["gpu_memory"])
    GPU_UTILIZATION.set(m["gpu_util"])
    CPU_PERCENT.set(m["cpu"])
    logger.debug(
        "sys-metrics gpu_mem=%s gpu_util=%s cpu=%s",
        m["gpu_memory"],
        m["gpu_util"],
        m["cpu"],
    )
    return m


def record_queue_metrics() -> None:
    """Update queue length gauges for all registered jobs."""

    for name in ("transcribe", "generate_reply", "synthesize_voice"):
        QUEUE_LENGTH.labels(name).set(task_queue.size(name))


async def metrics_loop(interval: float = 5.0) -> None:
    """Background task periodically collecting system metrics."""

    while True:  # pragma: no cover - simple loop
        record_system_metrics()
        record_queue_metrics()
        await asyncio.sleep(interval)


class Autoscaler:
    """Basic autoscaling helper based on CPU usage.

    ``scale_up`` and ``scale_down`` are callables that perform the
    actual scaling action (e.g. invoking ``kubectl`` or ``docker``).
    The class periodically samples CPU usage and triggers the
    callbacks when thresholds are crossed.
    """

    def __init__(
        self,
        scale_up: Callable[[], None],
        scale_down: Callable[[], None],
        *,
        high: float = 80.0,
        low: float = 20.0,
    ) -> None:
        self.scale_up = scale_up
        self.scale_down = scale_down
        self.high = high
        self.low = low

    async def run(self, interval: float = 30.0) -> None:
        """Continuously check CPU usage and invoke callbacks."""
        while True:  # pragma: no cover - simple loop
            cpu = record_system_metrics()["cpu"]
            if cpu > self.high:
                logger.info("autoscaler: scale up cpu=%s", cpu)
                self.scale_up()
            elif cpu < self.low:
                logger.info("autoscaler: scale down cpu=%s", cpu)
                self.scale_down()
            await asyncio.sleep(interval)


def resolve_device(preferred: str = "auto", *, threshold: float = 90.0) -> str:
    """Return ``"cuda"`` or ``"cpu"`` based on utilisation and preference."""

    if preferred == "cpu":
        return "cpu"
    if torch is None or not getattr(torch, "cuda", None) or not torch.cuda.is_available():
        return "cpu"
    util = read_system_metrics()["gpu_util"]
    if util >= threshold:
        return "cpu"
    return "cuda"

