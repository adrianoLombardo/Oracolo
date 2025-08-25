from pathlib import Path
import numpy as np
from src.metadata_store import VectorStore


def test_vector_store_save_load(tmp_path: Path) -> None:
    vs = VectorStore(dim=3)
    vs.add("a", [1.0, 0.0, 0.0])
    vs.add("b", [0.0, 1.0, 0.0])
    save_path = tmp_path / "index.pkl"
    vs.save(save_path)

    loaded = VectorStore.load(save_path)
    assert loaded.ids == ["a", "b"]
    assert np.allclose(loaded.vectors[0], [1.0, 0.0, 0.0])
    top = loaded.search([1.0, 0.0, 0.0], top_k=1)
    assert top and top[0][0] == "a"
