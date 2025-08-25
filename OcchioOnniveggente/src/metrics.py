"""Prometheus metrics utilities for the backend."""
from __future__ import annotations

import time

from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.requests import Request
from starlette.responses import Response

# HTTP request metrics
REQUEST_COUNT = Counter(
    "http_requests_total", "Total HTTP requests", ["method", "endpoint"],
)
REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds", "HTTP request latency", ["method", "endpoint"],
)


async def metrics_middleware(request: Request, call_next):  # type: ignore[no-untyped-def]
    """Middleware capturing request count and latency for each call."""
    start = time.perf_counter()
    response = await call_next(request)
    duration = time.perf_counter() - start
    REQUEST_COUNT.labels(request.method, request.url.path).inc()
    REQUEST_LATENCY.labels(request.method, request.url.path).observe(duration)
    return response


def metrics_endpoint() -> Response:
    """Expose collected metrics in Prometheus text format."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
