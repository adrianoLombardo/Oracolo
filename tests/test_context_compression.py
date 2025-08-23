import json
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from OcchioOnniveggente.src.retrieval import retrieve, _simple_sentences


def test_retrieve_compresses_passages(tmp_path):
    docs = [
        {
            "id": "1",
            "text": "A apple. B banana. C cherry. D date. E elder. F fig.",
        }
    ]
    p = tmp_path / "idx.json"
    p.write_text(json.dumps(docs), encoding="utf-8")
    res = retrieve("banana", p, top_k=1)
    assert len(_simple_sentences(res[0]["text"])) <= 5
    assert "banana" in res[0]["text"]
