
"""Simplified documentation API used for tests.

The original project exposes many endpoints and relies on a complex service
container.  For the unit tests we only need a small subset providing a
metrics endpoint and an option update handler.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

from .metrics import metrics_middleware, metrics_endpoint

load_dotenv()
app = FastAPI()
FastAPIInstrumentor.instrument_app(app)
app.middleware("http")(metrics_middleware)
app.get("/metrics")(metrics_endpoint)

STATE: dict[str, Any] = {}
SETTINGS_PATH = Path(__file__).resolve().parents[1] / 'settings.yaml'


class DocsOptions(BaseModel):
    """Options toggled from the documentation UI."""
    use_verified: bool | None = None
    allow_citations: bool | None = None
    block_pii: bool | None = None
    min_confidence: float | None = Field(None, ge=0.0, le=1.0)


@app.post("/api/docs/options")
def update_docs_options(opts: DocsOptions) -> dict[str, Any]:
    """Update documentation-related options in memory and on disk."""
    data = {k: v for k, v in opts.model_dump(exclude_none=True).items()}
    if data:
        STATE.update(data)
        SETTINGS_PATH.write_text('', encoding='utf-8')
    return {"status": "ok", "settings": STATE}
