from types import SimpleNamespace
from pathlib import Path
import sys
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))
from OcchioOnniveggente.src.retrieval import _embed_texts


class DummyEmbClient:
    def __init__(self):
        self.calls = 0

        class Embeddings:
            def __init__(self, outer):
                self.outer = outer

            def create(self, model, input):  # noqa: D401
                self.outer.calls += 1
                data = []
                for i, _ in enumerate(input):
                    data.append(SimpleNamespace(embedding=[float(i)] * 3))
                return SimpleNamespace(data=data)

        self.embeddings = Embeddings(self)


def test_embed_texts_uses_cache(tmp_path: Path):
    client = DummyEmbClient()
    cache = tmp_path / "c"
    texts = ["uno", "due"]

    vecs1 = _embed_texts(client, "m", texts, cache_dir=cache)
    assert client.calls == 1
    vecs2 = _embed_texts(client, "m", texts, cache_dir=cache)
    assert client.calls == 1  # no extra call thanks to cache
    for v1, v2 in zip(vecs1, vecs2):
        assert np.allclose(v1, v2)
