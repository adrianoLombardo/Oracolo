from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from fastapi import FastAPI
from pydantic import BaseModel, Field

from .ui_state import UIState

STATE = UIState()
ROOT = Path(__file__).resolve().parents[1]
SETTINGS_PATH = ROOT / "settings.yaml"

app = FastAPI()


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


@app.post("/api/docs/options")
def update_docs_options(opts: DocsOptions) -> dict[str, Any]:
    """Update documentation-related options in memory and on disk."""

    data = _load_settings()
    payload = {k: v for k, v in opts.model_dump(exclude_none=True).items()}
    if payload:
        STATE.settings.update(payload)
        data.update(payload)
        _save_settings(data)
    return {"status": "ok", "settings": STATE.settings}
