from __future__ import annotations

from pathlib import Path
from difflib import SequenceMatcher
from typing import List


class DocumentDB:
    """Simple document store with naive search.

    Documents are loaded from a directory or single file. Each text file's
    content becomes a document. Search uses a basic similarity metric and
    returns the top matched documents joined as a single context string.
    """

    def __init__(self, docstore_path: str | Path) -> None:
        self.path = Path(docstore_path)
        self.docs: List[str] = []
        if self.path.exists():
            if self.path.is_dir():
                for p in self.path.glob("*.txt"):
                    try:
                        self.docs.append(p.read_text(encoding="utf-8"))
                    except Exception:
                        pass
            else:
                try:
                    self.docs.append(self.path.read_text(encoding="utf-8"))
                except Exception:
                    pass

    def search(self, query: str, top_k: int) -> str:
        if not self.docs or not query:
            return ""
        q = query.lower()
        scored = sorted(
            self.docs,
            key=lambda d: SequenceMatcher(None, d.lower(), q).ratio(),
            reverse=True,
        )
        return "\n\n".join(scored[: max(top_k, 1)])
