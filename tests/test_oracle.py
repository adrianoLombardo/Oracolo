import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from OcchioOnniveggente.src.oracle import is_relevant


def test_is_relevant_case_insensitive():
    topics = ["neuroscience", "contemporary art"]
    assert is_relevant("Tell me about NEUROSCIENCE", topics)
    assert is_relevant("I love Contemporary Art pieces", topics)


def test_is_relevant_no_match():
    topics = ["neuroscience", "contemporary art"]
    assert not is_relevant("Let's talk about cooking", topics)


def test_is_relevant_word_boundaries():
    assert not is_relevant("This is a smart move", ["art"])
