from pathlib import Path
from src.storage import MetadataStore


def test_metadata_store_search(tmp_path: Path) -> None:
    db_path = tmp_path / "docs.db"
    store = MetadataStore(str(db_path))
    store.add_documents([
        {"id": "doc1", "text": "Questo è un test"},
        {"id": "doc2", "text": "La frase chiave è qui"},
    ])
    results = store.search("frase")
    assert results and results[0][0] == "doc2"
    # get_documents should return both entries
    docs = store.get_documents()
    assert any(d["id"] == "doc1" for d in docs)
    store.delete_documents(["doc1"])
    docs = store.get_documents()
    assert all(d["id"] != "doc1" for d in docs)
