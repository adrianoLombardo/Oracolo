from __future__ import annotations

"""Lightweight metadata and vector stores for document retrieval.

This module provides two small helpers used to speed up backend operations:

* :class:`MetadataStore` – wraps either SQLite with an FTS5 index or
  PostgreSQL with a `tsvector` index.  Only a couple of methods are implemented
  to keep the interface extremely small and dependency free.  If PostgreSQL
  support is requested but the ``psycopg2`` package is not available, a
  ``RuntimeError`` is raised at runtime.
* :class:`VectorStore` – optional FAISS-based vector index with a pure Python
  fallback based on cosine similarity.  It is purposely minimal and intended
  for small/medium collections.  Instances can be persisted to disk with
  :meth:`VectorStore.save` and restored through :meth:`VectorStore.load` which
  serialise IDs, vectors and, when available, the FAISS index.

Example
-------

>>> from pathlib import Path
>>> vs = VectorStore(dim=3)
>>> vs.add("doc1", [0.1, 0.2, 0.3])
>>> vs.save(Path("index.bin"))
>>> vs2 = VectorStore.load(Path("index.bin"))

The classes are intentionally tiny so they can be imported without pulling in
heavy dependencies when not used.
"""

from pathlib import Path
from typing import Iterable, List, Tuple, Dict, Any

import sqlite3
import numpy as np
import pickle
from typing import Callable

from .exceptions import ExternalServiceError
from .utils import retry_with_backoff

try:  # pragma: no cover - optional dependency
    from tenacity import retry, stop_after_attempt, wait_exponential
except Exception:  # pragma: no cover - tenacity might be missing
    retry = None  # type: ignore

try:  # optional PostgreSQL backend
    import psycopg2  # type: ignore
except Exception:  # pragma: no cover - optional
    psycopg2 = None  # type: ignore

try:  # optional FAISS backend
    import faiss  # type: ignore
except Exception:  # pragma: no cover - optional
    faiss = None  # type: ignore


class MetadataStore:
    """Simple metadata store backed by SQLite FTS5 or PostgreSQL.

    Parameters
    ----------
    dsn: str
        Database connection string.  Use ``sqlite:///path`` or a plain path for
        SQLite; any string starting with ``postgres`` enables the PostgreSQL
        backend.
    """

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
        """Insert or replace a batch of documents.

        Each document must at least contain ``id`` and ``text`` keys.
        Additional metadata is currently ignored but kept to maintain a stable
        API for future extensions.
        """

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


class VectorStore:
    """Tiny vector store with optional FAISS backend."""

    def __init__(self, dim: int):
        self.dim = dim
        self.ids: List[str] = []
        self.vectors: List[np.ndarray] = []
        if faiss is not None:  # pragma: no cover - optional dependency
            self.index = faiss.IndexFlatL2(dim)
        else:
            self.index = None

    def add(self, doc_id: str, vector: Iterable[float]) -> None:
        vec = np.asarray(list(vector), dtype=np.float32)
        if vec.size != self.dim:
            raise ValueError(f"expected vector of size {self.dim}, got {vec.size}")
        if self.index is not None:  # pragma: no branch - optional
            self.index.add(vec.reshape(1, -1))
        self.ids.append(doc_id)
        self.vectors.append(vec)

    def search(self, vector: Iterable[float], top_k: int = 5) -> List[Tuple[str, float]]:
        """Return ``(id, score)`` pairs ordered by similarity."""

        q = np.asarray(list(vector), dtype=np.float32)
        if q.size != self.dim:
            raise ValueError(f"expected vector of size {self.dim}, got {q.size}")

        if self.index is not None and self.ids:  # pragma: no branch - optional
            def _do_search() -> tuple[np.ndarray, np.ndarray]:
                return self.index.search(q.reshape(1, -1), top_k)

            if retry is not None:
                retryer = retry(
                    reraise=True, stop=stop_after_attempt(3), wait=wait_exponential(0.5, 2)
                )
                try:
                    D, I = retryer(_do_search)()
                except Exception as exc:  # pragma: no cover - defensive
                    raise ExternalServiceError(str(exc)) from exc
            else:  # pragma: no cover - tenacity missing
                try:
                    D, I = retry_with_backoff(_do_search, retries=3, base_delay=0.5)
                except Exception as exc:  # pragma: no cover - defensive
                    raise ExternalServiceError(str(exc)) from exc
            return [
                (self.ids[i], float(D[0][j]))
                for j, i in enumerate(I[0])
                if 0 <= i < len(self.ids)
            ]

        # Fallback cosine similarity
        sims = []
        for did, vec in zip(self.ids, self.vectors):
            denom = float(np.linalg.norm(q) * np.linalg.norm(vec))
            sim = float(np.dot(q, vec) / denom) if denom else 0.0
            sims.append((did, sim))
        sims.sort(key=lambda x: x[1], reverse=True)
        return sims[:top_k]

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------
    def save(self, path: Path) -> None:
        """Serialise the store to ``path``.

        The file contains the dimension, document IDs, vectors and the FAISS
        index when available.  It can be reloaded with
        :meth:`VectorStore.load`.
        """

        data = {
            "dim": self.dim,
            "ids": self.ids,
            "vectors": [v.tolist() for v in self.vectors],
            "index": None,
        }
        if faiss is not None and self.index is not None:  # pragma: no branch - optional
            data["index"] = faiss.serialize_index(self.index)

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, path: Path) -> "VectorStore":
        """Load a previously saved store from ``path``."""

        with Path(path).open("rb") as f:
            data = pickle.load(f)

        store = cls(int(data.get("dim", 0)))
        store.ids = list(data.get("ids", []))
        store.vectors = [np.asarray(v, dtype=np.float32) for v in data.get("vectors", [])]

        if faiss is not None:
            index_bytes = data.get("index")
            if index_bytes is not None:
                store.index = faiss.deserialize_index(index_bytes)
            elif store.vectors:
                store.index.add(np.stack(store.vectors))
        else:
            store.index = None

        return store
