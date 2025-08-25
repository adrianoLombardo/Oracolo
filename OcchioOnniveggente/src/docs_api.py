from __future__ import annotations

import json
import os
import sys
import time
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import parse_qs, urlparse

import yaml
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import APIKeyHeader
from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from .service_container import container
from .metrics import metrics_endpoint, metrics_middleware
from .local_llm import stream_generate

try:
    from .ui_state import UIState
except Exception:  # pragma: no cover - fallback for minimal environments
    class UIState:  # type: ignore[no-redef]
        def __init__(self) -> None:  # pragma: no cover - simple stub
            self.settings: dict[str, Any] = {}


STATE = container.ui_state
ROOT = Path(__file__).resolve().parents[1]
SETTINGS_PATH = ROOT / "settings.yaml"

app = FastAPI()
FastAPIInstrumentor.instrument_app(app)
app.middleware("http")(metrics_middleware)
app.get("/metrics")(metrics_endpoint)

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)
RATE_LIMIT = int(os.getenv("API_RATE_LIMIT", "5"))
RATE_WINDOW = int(os.getenv("API_RATE_WINDOW", "60"))
_REQUEST_LOG: dict[str, list[float]] = {}


class DocsOptions(BaseModel):
    """Options toggled from the documentation UI."""

    use_verified: bool | None = None
    allow_citations: bool | None = None
    block_pii: bool | None = None
    min_confidence: float | None = Field(None, ge=0.0, le=1.0)


def _load_settings() -> dict[str, Any]:
    if SETTINGS_PATH.exists():
        try:
            return yaml.safe_load(SETTINGS_PATH.read_text(encoding="utf-8")) or {}
        except Exception:
            return {}
    return {}


def _save_settings(data: dict[str, Any]) -> None:
    SETTINGS_PATH.write_text(
        yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )


def _sanitize(data: dict[str, Any]) -> dict[str, Any]:
    """Return a sanitized copy of ``data`` suitable for logging/storage."""

    clean: dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, str):
            clean[key] = value.replace("\n", " ").strip()
        else:
            clean[key] = value
    return clean


def _rate_limiter(key: str) -> None:
    now = time.time()
    bucket = _REQUEST_LOG.setdefault(key, [])
    bucket = [t for t in bucket if now - t < RATE_WINDOW]
    if len(bucket) >= RATE_LIMIT:
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    bucket.append(now)
    _REQUEST_LOG[key] = bucket


def get_api_key(api_key_header: str | None = Depends(API_KEY_HEADER)) -> str:
    expected = os.getenv("DOCS_API_KEY")
    if not expected or api_key_header != expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
        )
    _rate_limiter(api_key_header)
    return api_key_header


@app.post("/api/docs/options")
def update_docs_options(
    opts: DocsOptions, api_key: str = Depends(get_api_key)
) -> dict[str, Any]:
    """Update documentation-related options in memory and on disk."""

    data = _load_settings()
    payload = _sanitize({k: v for k, v in opts.model_dump(exclude_none=True).items()})
    if payload:
        STATE.settings.update(payload)
        data.update(payload)
        _save_settings(data)
    return {"status": "ok", "settings": STATE.settings}


@app.post("/api/ingest")
def ingest_async(path: str, tasks: BackgroundTasks) -> dict[str, str]:
    from scripts import ingest_docs  # type: ignore
    tasks.add_task(ingest_docs.ingest, path)
    return {"status": "queued"}


@app.post("/api/embeddings")
def embeddings_async(texts: list[str], model: str, tasks: BackgroundTasks) -> dict[str, str]:
    from openai import AsyncOpenAI  # type: ignore
    from .retrieval import _embed_texts
    client = AsyncOpenAI()
    tasks.add_task(_embed_texts, client, model, texts)
    return {"status": "queued"}


@app.post("/api/log")
def log_async(data: dict[str, Any], path: str, tasks: BackgroundTasks) -> dict[str, str]:
    import json
    from pathlib import Path
    def _write(payload: dict[str, Any], dest: str) -> None:
        p = Path(dest)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
    tasks.add_task(_write, data, path)
    return {"status": "queued"}


@app.post("/api/local/stream")
def local_stream(
    messages: list[dict[str, str]],
    model_path: str,
    device: str = "cpu",
    max_new_tokens: int = 256,
    precision: str = "fp32",
) -> StreamingResponse:
    """Stream LLM tokens as Server-Sent Events."""

    def _iter() -> Iterator[str]:
        for tok in stream_generate(
            messages,
            model_path=model_path,
            device=device,
            max_new_tokens=max_new_tokens,
            precision=precision,  # type: ignore[arg-type]
        ):
            yield f"data: {tok}\n\n"

    return StreamingResponse(_iter(), media_type="text/event-stream")

import json
import sys
from http.server import SimpleHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs


# Ensure project root on path to import scripts.ingest_docs
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import ingest_docs  # type: ignore  # noqa: E402

DOCSTORE_PATH = ROOT / "DataBase" / "index.json"


def _get_documents() -> list[dict]:
    """Load document metadata from the docstore.

    Returns a list of dictionaries containing metadata for each document.
    """
    db = ingest_docs._load_db(str(DOCSTORE_PATH))
    if hasattr(db, "get_documents"):
        docs = db.get_documents()
    elif hasattr(db, "_data"):
        docs = db._data.get("documents", [])  # type: ignore[attr-defined]
    else:  # pragma: no cover - unlikely fallback
        docs = []
    out: list[dict] = []
    for d in docs:
        out.append(
            {
                "title": d.get("title") or Path(d.get("id", "")).name,
                "domain": d.get("topic") or d.get("domain", ""),
                "rule": d.get("rule", ""),
                "status": d.get("status", "indexed"),
                "confidence": d.get("confidence", 1.0),
            }
        )
    return out


class DocsHandler(SimpleHTTPRequestHandler):
    """Serve docs metadata and a simple HTML page."""

    def do_GET(self) -> None:  # noqa: N802 (BaseHTTPRequestHandler API)
        parsed = urlparse(self.path)
        if parsed.path == "/api/docs":
            docs = _get_documents()
            params = parse_qs(parsed.query)
            # textual filters
            for key in ("title", "domain", "rule", "status"):
                if key in params:
                    needle = params[key][0].lower()
                    docs = [d for d in docs if needle in str(d.get(key, "")).lower()]
            # numeric filter: minimum confidence
            if "min_confidence" in params:
                try:
                    thr = float(params["min_confidence"][0])
                    docs = [d for d in docs if float(d.get("confidence", 0.0)) >= thr]
                except ValueError:
                    pass
            body = json.dumps(docs, ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        # fall back to default behaviour (serving static files)
        return super().do_GET()


def _make_handler(*args, **kwargs) -> DocsHandler:
    return DocsHandler(*args, directory=str(ROOT), **kwargs)


def run(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Run a minimal HTTP server serving docs.html and /api/docs."""
    HTTPServer((host, port), _make_handler).serve_forever()


if __name__ == "__main__":  # pragma: no cover - manual execution
    try:
        run()
    except KeyboardInterrupt:
        pass
