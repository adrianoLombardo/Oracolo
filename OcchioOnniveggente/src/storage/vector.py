from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import pickle

from ..exceptions import ExternalServiceError
from ..utils import retry_with_backoff

try:  # pragma: no cover - optional dependency
    from tenacity import retry, stop_after_attempt, wait_exponential
except Exception:  # pragma: no cover - tenacity might be missing
    retry = None  # type: ignore

try:  # optional FAISS backend
    import faiss  # type: ignore
except Exception:  # pragma: no cover - optional
    faiss = None  # type: ignore


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
        """Serialise the store to ``path``."""

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

