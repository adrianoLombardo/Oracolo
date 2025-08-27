import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))
from OcchioOnniveggente.src.chat import ChatState  # noqa: E402
from src.retrieval import retrieve  # noqa: E402


class DummyEmbClient:
    def __init__(self, mapping):
        class Embeddings:
            def __init__(self, mapping):
                self.mapping = mapping

            def create(self, model, input):
                data = []
                for text in input:
                    vec = self.mapping[text]
                    data.append(SimpleNamespace(embedding=vec))
                return SimpleNamespace(data=data)
        self.embeddings = Embeddings(mapping)


def test_update_topic_detects_change():
    mapping = {
        "tema1": np.array([1.0, 0.0], dtype=np.float32),
        "tema1 follow": np.array([1.0, 0.1], dtype=np.float32),
        "tema2": np.array([0.0, 1.0], dtype=np.float32),
    }
    client = DummyEmbClient(mapping)
    chat = ChatState()

    chat.push_user("tema1")
    assert chat.update_topic("tema1", client, "m") is False
    chat.push_user("tema1 follow")
    assert chat.update_topic("tema1 follow", client, "m") is False
    chat.push_user("tema2")
    assert chat.update_topic("tema2", client, "m") is True
    assert chat.topic_text == "tema2"
    assert chat.summary  # previous history summarized
    assert len(chat.history) == 1 and chat.history[0]["content"] == "tema2"


def test_retrieve_filters_by_topic(tmp_path: Path):
    index = tmp_path / "index.json"
    index.write_text(
        '{"documents": ['
        '{"id": "a", "text": "common cat", "topic": "t1"},'
        '{"id": "b", "text": "common dog", "topic": "t2"}]}',
        encoding="utf-8",
    )

    res = retrieve("common", index, top_k=5, topic="t1")
    assert res and all(d.get("topic") == "t1" for d in res)
    ids = {d["id"] for d in res}
    assert ids == {"a"}

    res_all = retrieve("common", index, top_k=5)
    ids_all = {d["id"] for d in res_all}
    assert ids_all == {"a", "b"}
