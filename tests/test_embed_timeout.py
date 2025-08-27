from types import SimpleNamespace
from pathlib import Path
import sys
import time
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.retrieval import _embed_texts  # noqa: E402


class StallEmbClient:
    class Embeddings:
        def create(self, model, input):  # noqa: D401
            time.sleep(1)
            return SimpleNamespace(data=[])

    def __init__(self):
        self.embeddings = self.Embeddings()


def test_embed_timeout(tmp_path: Path):
    client = StallEmbClient()
    with pytest.raises(TimeoutError):
        _embed_texts(client, "m", ["ciao"], timeout=0.1, retries=1)
