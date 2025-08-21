import sqlite3
from pathlib import Path
import pytest


class SimpleDocStore:
    """Minimal SQLite FTS5-backed document store for tests."""

    def __init__(self, db_path: Path):
        self.conn = sqlite3.connect(db_path)
        # Create an FTS5 virtual table for full-text search
        self.conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS docs USING fts5(content)")

    def ingest_text(self, text: str) -> None:
        self.conn.execute("INSERT INTO docs(content) VALUES (?)", (text,))
        self.conn.commit()

    def search(self, query: str):
        cur = self.conn.execute("SELECT content FROM docs WHERE docs MATCH ?", (query,))
        return [row[0] for row in cur.fetchall()]


def test_search_returns_expected_passage(tmp_path: Path) -> None:
    db_path = tmp_path / "docs.db"
    store = SimpleDocStore(db_path)

    # Simulate ingesting a small text document
    doc = tmp_path / "doc.txt"
    doc.write_text("Questo è un breve testo. La frase chiave appare qui.", encoding="utf-8")
    store.ingest_text(doc.read_text(encoding="utf-8"))

    results = store.search("frase chiave")
    assert results and "La frase chiave appare qui." in results[0]
