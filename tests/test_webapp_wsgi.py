import io
import sys
import threading
from pathlib import Path
from wsgiref.util import setup_testing_defaults

sys.path.append(str(Path(__file__).resolve().parents[1] / "OcchioOnniveggente"))
from src.webapp_wsgi import app


def _make_environ(path: str, body: bytes) -> dict:
    environ: dict[str, object] = {}
    setup_testing_defaults(environ)
    environ["REQUEST_METHOD"] = "POST"
    environ["PATH_INFO"] = path
    environ["CONTENT_LENGTH"] = str(len(body))
    environ["wsgi.input"] = io.BytesIO(body)
    return environ


def test_invalid_json_returns_400():
    environ = _make_environ("/api/docs/save", b"{invalid")
    captured: dict[str, object] = {}

    def start_response(status, headers):
        captured["status"] = status
        captured["headers"] = headers

    body = b"".join(app(environ, start_response))
    assert captured["status"].startswith("400")
    assert b"error" in body


def test_concurrent_state(monkeypatch):
    states = []

    def fake_apply(state, text):
        states.append(state)

    monkeypatch.setattr("src.webapp_wsgi.apply_to_chat", fake_apply)

    env1 = _make_environ("/api/docs/apply", b'{"text": "one"}')
    env2 = _make_environ("/api/docs/apply", b'{"text": "two"}')

    def call_app(env):
        app(env, lambda *_: None)

    t1 = threading.Thread(target=call_app, args=(env1,))
    t2 = threading.Thread(target=call_app, args=(env2,))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert len(states) == 2
    assert states[0] is not states[1]
