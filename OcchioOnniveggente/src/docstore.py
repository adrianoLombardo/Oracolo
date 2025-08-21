from __future__ import annotations

from pathlib import Path
from typing import List

import chromadb
from chromadb.utils import embedding_functions

import PyPDF2
import docx
from PIL import Image
import pytesseract


class _SimpleEmbeddingFunction(embedding_functions.EmbeddingFunction):
    """Very small embedding based on letter frequencies.

    This avoids heavy model downloads while still providing deterministic
    vectors for similarity search."""

    def __call__(self, texts: List[str]) -> List[List[float]]:  # type: ignore[override]
        vecs: List[List[float]] = []
        for t in texts:
            t = t.lower()
            counts = [t.count(chr(ord('a') + i)) for i in range(26)]
            total = sum(counts) or 1
            vecs.append([c / total for c in counts])
        return vecs


class DocumentDB:
    """Simple document store backed by ChromaDB."""

    def __init__(self, persist_directory: Path | None = None) -> None:
        self.persist_directory = (
            persist_directory
            or Path(__file__).resolve().parents[1] / "data" / "docstore"
        )
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(path=str(self.persist_directory))
        self.collection = self.client.get_or_create_collection(
            name="documents", embedding_function=_SimpleEmbeddingFunction()
        )

    # ------------------------------------------------------------------
    # Parsers
    def _parse_pdf(self, path: Path) -> str:
        text = ""
        try:
            with path.open("rb") as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    text += page.extract_text() or ""
        except Exception:
            text = ""
        return text

    def _parse_docx(self, path: Path) -> str:
        try:
            document = docx.Document(path)
            return "\n".join(p.text for p in document.paragraphs)
        except Exception:
            return ""

    def _parse_image(self, path: Path) -> str:
        try:
            image = Image.open(path)
            return pytesseract.image_to_string(image)
        except Exception:
            return ""

    def _parse_text(self, path: Path) -> str:
        try:
            return path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return ""

    def _load_text(self, path: Path) -> str:
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            return self._parse_pdf(path)
        if suffix == ".docx":
            return self._parse_docx(path)
        if suffix in {".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif"}:
            return self._parse_image(path)
        return self._parse_text(path)

    # ------------------------------------------------------------------
    def add_documents(self, paths: List[Path]) -> None:
        documents: List[str] = []
        ids: List[str] = []
        for path in paths:
            text = self._load_text(path)
            if text.strip():
                documents.append(text)
                ids.append(path.as_posix())
        if documents:
            self.collection.add(documents=documents, ids=ids)

    def search(self, query: str, top_k: int) -> List[str]:
        result = self.collection.query(query_texts=[query], n_results=top_k)
        docs = result.get("documents") or []
        return docs[0] if docs else []
