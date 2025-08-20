import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))
from OcchioOnniveggente.src.filters import ProfanityFilter, load_blacklist


@pytest.fixture()
def pf() -> ProfanityFilter:
    base = Path(__file__).parent / "data" / "filters"
    phrases = load_blacklist(base / "phrases.txt")
    return ProfanityFilter(
        base / "italian.txt",
        base / "english.txt",
        extra_it=phrases,
        extra_en=phrases,
    )


def test_recognizes_leet_and_accents(pf: ProfanityFilter) -> None:
    assert pf.contains_profanity("càzz0 che schifo")
    assert pf.contains_profanity("sh1t happens")


def test_mask_preserves_length(pf: ProfanityFilter) -> None:
    text = "This is shit"
    masked = pf.mask(text)
    assert len(masked) == len(text)
    assert masked != text


def test_multiword_phrase_with_punctuation(pf: ProfanityFilter) -> None:
    assert pf.contains_profanity("porco dio")
    assert pf.contains_profanity("Che porco, dio?")
    assert pf.contains_profanity("son of a bitch")
    assert pf.contains_profanity("you son-of a bitch")
