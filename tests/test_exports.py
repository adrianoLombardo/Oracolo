import json
from pathlib import Path

import json
from pathlib import Path

from OcchioOnniveggente.src.chat import ChatState
from OcchioOnniveggente.src.oracle import export_audio_answer, format_citations


class DummySynth:
    def __call__(self, text: str, out_path: Path) -> None:
        out_path.write_bytes(b"audio")


def test_chat_export(tmp_path: Path):
    st = ChatState()
    st.push_user("ciao")
    st.push_assistant("salve")

    txt = tmp_path / "chat.txt"
    st.export_history(txt)
    content = txt.read_text(encoding="utf-8")
    assert "user: ciao" in content

    md = tmp_path / "chat.md"
    st.export_history(md)
    md_content = md.read_text(encoding="utf-8")
    assert "**assistant:** salve" in md_content

    js = tmp_path / "chat.json"
    st.export_history(js)
    data = json.loads(js.read_text(encoding="utf-8"))
    assert data[0]["role"] == "user"


def test_format_citations():
    sources = [{"id": "doc1"}, {"id": "doc2", "score": 0.5}]
    assert format_citations(sources) == "doc1, doc2"


def test_export_audio(tmp_path: Path):
    out = tmp_path / "ans.wav"
    export_audio_answer("hello", out, synth=DummySynth())
    assert out.exists() and out.read_bytes() == b"audio"
