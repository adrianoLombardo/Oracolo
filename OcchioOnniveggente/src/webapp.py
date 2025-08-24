from __future__ import annotations

import json

from src.profile_utils import save_profile
from src.ui_state import UIState, apply_to_chat

# Global UI state used by the application.
_STATE = UIState()


def _read_json(environ) -> dict:
    """Return request body as JSON dict.

    Parameters
    ----------
    environ:
        WSGI environment passed to the application.
    """

    try:
        length = int(environ.get("CONTENT_LENGTH", 0))
    except (TypeError, ValueError):
        length = 0
    body = environ["wsgi.input"].read(length) if length else b""
    try:
        return json.loads(body.decode("utf-8") or "{}")
    except Exception:
        return {}


def app(environ, start_response):
    """Minimal WSGI application exposing a couple of endpoints.

    The implementation intentionally avoids external dependencies so that it
    can run in constrained environments (like the execution sandbox used by
    the tests).  Only the endpoints required by the tests are implemented.
    """

    method = environ.get("REQUEST_METHOD", "")
    path = environ.get("PATH_INFO", "")

    if method == "POST" and path == "/api/docs/save":
        data = _read_json(environ)
        save_profile(data)
        start_response("200 OK", [("Content-Type", "application/json")])
        return [b'{"status": "saved"}']

    if method == "POST" and path == "/api/docs/apply":
        data = _read_json(environ)
        apply_to_chat(_STATE, data.get("text", ""))
        start_response("200 OK", [("Content-Type", "application/json")])
        return [b'{"status": "applied"}']

    start_response("404 Not Found", [("Content-Type", "text/plain")])
    return [b"Not Found"]


if __name__ == "__main__":  # pragma: no cover - manual invocation helper
    from wsgiref.simple_server import make_server

    srv = make_server("", 8000, app)
    print("Serving on port 8000...")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
