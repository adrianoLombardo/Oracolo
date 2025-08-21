import sys
from pathlib import Path

import docx

sys.path.append(str(Path(__file__).resolve().parents[1]))
from OcchioOnniveggente.src.docstore import DocumentDB


def test_add_and_search_text(tmp_path: Path) -> None:
    txt = tmp_path / "sample.txt"
    txt.write_text("hello world", encoding="utf-8")
    db = DocumentDB(persist_directory=tmp_path / "db")
    db.add_documents([txt])
    res = db.search("hello", 1)
    assert any("hello world" in r for r in res)


def test_add_and_search_docx(tmp_path: Path) -> None:
    docx_path = tmp_path / "sample.docx"
    document = docx.Document()
    document.add_paragraph("ciao mondo")
    document.save(docx_path)
    db = DocumentDB(persist_directory=tmp_path / "db")
    db.add_documents([docx_path])
    res = db.search("ciao", 1)
    assert any("ciao mondo" in r for r in res)
