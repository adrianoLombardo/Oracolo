from __future__ import annotations

import importlib.util
import json
import time
from pathlib import Path


def _load_ingest_module() -> object:
    """Import the ingest_docs script as a module."""

    root = Path(__file__).resolve().parents[1] / "OcchioOnniveggente" / "scripts" / "ingest_docs.py"
    spec = importlib.util.spec_from_file_location("ingest_docs", root)
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    assert spec.loader is not None
    spec.loader.exec_module(module)  # type: ignore[assignment]
    return module


def test_reindex_updates_only_modified_files(tmp_path: Path) -> None:
    ingest = _load_ingest_module()
    docstore = tmp_path / "index.json"

    f = tmp_path / "doc.txt"
    f.write_text("prima versione", encoding="utf-8")

    ingest._add([str(f)], str(docstore))
    data = json.loads(docstore.read_text("utf-8"))
    saved_mtime = data["documents"][0]["mtime"]

    # Reindex without changes → mtime unchanged
    ingest._reindex(str(docstore))
    data = json.loads(docstore.read_text("utf-8"))
    assert data["documents"][0]["mtime"] == saved_mtime
    assert data["documents"][0]["text"] == "prima versione"

    # Modify file and ensure mtime/text updated
    time.sleep(1)
    f.write_text("seconda versione", encoding="utf-8")
    ingest._reindex(str(docstore))
    data = json.loads(docstore.read_text("utf-8"))
    assert data["documents"][0]["text"] == "seconda versione"
    assert data["documents"][0]["mtime"] == f.stat().st_mtime

