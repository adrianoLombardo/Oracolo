import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from OcchioOnniveggente.src.audio.hotword import strip_hotword_prefix


def test_strip_hotword_prefix_removes_match():
    matched, remainder = strip_hotword_prefix(
        "Ciao Oracolo, come stai?", ["ciao oracolo", "hello oracle"]
    )
    assert matched is True
    assert remainder == "come stai?"


def test_strip_hotword_prefix_no_match():
    text = "Oggi, ciao oracolo, come va?"
    matched, remainder = strip_hotword_prefix(text, ["ciao oracolo"])
    assert matched is False
    assert remainder == text
