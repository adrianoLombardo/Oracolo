from __future__ import annotations

import json
from contextvars import ContextVar
from typing import Any, Callable, Iterable

from src.profile_utils import save_profile
from src.ui_state import UIState, apply_to_chat

# Per-request UI state container.
_STATE_VAR: ContextVar[UIState] = ContextVar("ui_state")


def _read_json(environ: dict[str, Any]) -> dict[str, Any]:
    """Return request body as JSON dict.

    Raises ``ValueError`` if the body does not contain valid JSON.
    """

    try:
        length = int(environ.get("CONTENT_LENGTH", 0))
    except (TypeError, ValueError):
        length = 0
    body = environ["wsgi.input"].read(length) if length else b""
    if not body:
        return {}
    try:
        return json.loads(body.decode("utf-8"))
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive
        raise ValueError("Invalid JSON") from exc


def app(
    environ: dict[str, Any],
    start_response: Callable[..., Any],
) -> Iterable[bytes]:
    """Minimal WSGI application exposing a couple of endpoints."""
    token = _STATE_VAR.set(UIState())
    try:
        method = environ.get("REQUEST_METHOD", "")
        path = environ.get("PATH_INFO", "")

        if method == "POST" and path == "/api/docs/save":
            try:
                data = _read_json(environ)
            except ValueError:
                start_response(
                    "400 Bad Request", [("Content-Type", "application/json")]
                )
                return [b'{"error": "invalid json"}']
            save_profile(data)
            start_response("200 OK", [("Content-Type", "application/json")])
            return [b'{"status": "saved"}']

        if method == "POST" and path == "/api/docs/apply":
            try:
                data = _read_json(environ)
            except ValueError:
                start_response(
                    "400 Bad Request", [("Content-Type", "application/json")]
                )
                return [b'{"error": "invalid json"}']
            apply_to_chat(_STATE_VAR.get(), data.get("text", ""))
            start_response("200 OK", [("Content-Type", "application/json")])
            return [b'{"status": "applied"}']

        start_response("404 Not Found", [("Content-Type", "text/plain")])
        return [b"Not Found"]
    finally:
        _STATE_VAR.reset(token)


if __name__ == "__main__":  # pragma: no cover - manual invocation helper
    from wsgiref.simple_server import make_server

    srv = make_server("", 8000, app)
    print("Serving on port 8000...")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
