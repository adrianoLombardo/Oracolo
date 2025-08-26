import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from OcchioOnniveggente.src.retrieval import retrieve


class DummyStore:
    def get_documents(self):
        return [
            {"id": "1", "text": "hello world", "title": "t", "topic": "g"},
            {"id": "2", "text": "ciao mondo", "title": "t2", "topic": "g"},
        ]


def test_retrieve_from_store():
    store = DummyStore()
    res = retrieve("hello", ".", store=store, top_k=1)
    assert res and res[0]["id"] == "1"
