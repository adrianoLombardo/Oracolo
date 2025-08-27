from __future__ import annotations

"""Lightweight metadata store for document retrieval.

Provides :class:`MetadataStore`, a tiny wrapper around either SQLite with an
FTS5 index or PostgreSQL with a ``tsvector`` index. Only a subset of CRUD and
search operations is implemented to keep the interface minimal and dependency
free. If PostgreSQL support is requested but the ``psycopg2`` package is not
available a ``RuntimeError`` is raised at runtime.
"""

from pathlib import Path
from typing import Dict, List, Tuple, Any

import sqlite3

try:  # optional PostgreSQL backend
    import psycopg2  # type: ignore
except Exception:  # pragma: no cover - optional
    psycopg2 = None  # type: ignore


class MetadataStore:
    """Simple metadata store backed by SQLite FTS5 or PostgreSQL."""

    def __init__(self, dsn: str):
        self.backend: str
        if dsn.startswith("postgres://") or dsn.startswith("postgresql://"):
            if psycopg2 is None:  # pragma: no cover - optional dependency
                raise RuntimeError("psycopg2 is required for PostgreSQL support")
            self.backend = "postgres"
            self.conn = psycopg2.connect(dsn)  # type: ignore[arg-type]
            cur = self.conn.cursor()
            cur.execute(
                "CREATE TABLE IF NOT EXISTS docs (id TEXT PRIMARY KEY, content TEXT)"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS docs_fts ON docs USING gin(to_tsvector('simple',content))"
            )
            self.conn.commit()
        else:
            self.backend = "sqlite"
            path = dsn
            if dsn.startswith("sqlite://"):
                path = dsn.split("sqlite://", 1)[1]
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            self.conn = sqlite3.connect(p)
            # id stored as UNINDEXED so only ``content`` participates in FTS
            self.conn.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS docs USING fts5(id UNINDEXED, content)"
            )

    # ------------------------------------------------------------------
    # Basic CRUD operations
    # ------------------------------------------------------------------
    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """Insert or replace a batch of documents."""

        if not documents:
            return
        if self.backend == "postgres":  # pragma: no cover - requires psycopg2
            cur = self.conn.cursor()
            for doc in documents:
                cur.execute(
                    "INSERT INTO docs(id, content) VALUES (%s, %s) "
                    "ON CONFLICT(id) DO UPDATE SET content=EXCLUDED.content",
                    (doc.get("id"), doc.get("text", "")),
                )
            self.conn.commit()
        else:
            cur = self.conn.cursor()
            cur.executemany(
                "INSERT INTO docs(id, content) VALUES (?, ?)",
                [(d.get("id"), d.get("text", "")) for d in documents],
            )
            self.conn.commit()

    def delete_documents(self, ids: List[str]) -> None:
        if not ids:
            return
        if self.backend == "postgres":  # pragma: no cover - requires psycopg2
            cur = self.conn.cursor()
            cur.executemany("DELETE FROM docs WHERE id=%s", [(i,) for i in ids])
            self.conn.commit()
        else:
            cur = self.conn.cursor()
            cur.executemany("DELETE FROM docs WHERE id=?", [(i,) for i in ids])
            self.conn.commit()

    def get_documents(self) -> List[Dict[str, str]]:
        cur = self.conn.cursor()
        cur.execute("SELECT id, content FROM docs")
        return [{"id": did, "text": txt} for did, txt in cur.fetchall()]

    def clear(self) -> None:
        cur = self.conn.cursor()
        cur.execute("DELETE FROM docs")
        self.conn.commit()

    # ------------------------------------------------------------------
    # Search utilities
    # ------------------------------------------------------------------
    def search(self, query: str, limit: int = 5) -> List[Tuple[str, str]]:
        """Return ``(id, content)`` tuples matching ``query``."""

        if self.backend == "postgres":  # pragma: no cover - requires psycopg2
            cur = self.conn.cursor()
            cur.execute(
                "SELECT id, content FROM docs WHERE "
                "to_tsvector('simple',content) @@ plainto_tsquery(%s) LIMIT %s",
                (query, limit),
            )
            return cur.fetchall()
        else:
            cur = self.conn.execute(
                "SELECT id, content FROM docs WHERE docs MATCH ? LIMIT ?",
                (query, limit),
            )
            return cur.fetchall()

