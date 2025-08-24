from __future__ import annotations
import json
import sys
from http.server import SimpleHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

# Ensure project root on path to import scripts.ingest_docs
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import ingest_docs  # type: ignore

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


def run(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Run a minimal HTTP server serving docs.html and /api/docs."""
    handler = lambda *args, **kwargs: DocsHandler(*args, directory=str(ROOT), **kwargs)  # type: ignore[arg-type]
    HTTPServer((host, port), handler).serve_forever()


if __name__ == "__main__":  # pragma: no cover - manual execution
    try:
        run()
    except KeyboardInterrupt:
        pass
