from pathlib import Path
import importlib.util
import pytest

spec = importlib.util.spec_from_file_location(
    "filters", Path(__file__).resolve().parents[1] / "OcchioOnniveggente" / "src" / "filters.py"
)
filters = importlib.util.module_from_spec(spec)
spec.loader.exec_module(filters)
ProfanityFilter = filters.ProfanityFilter


@pytest.fixture()
def pf() -> ProfanityFilter:
    data_dir = Path(__file__).parent / "data" / "filters"
    return ProfanityFilter(data_dir / "it_blacklist.txt", data_dir / "en_blacklist.txt")


def test_recognizes_leet_and_accents(pf: ProfanityFilter) -> None:
    assert pf.contains_profanity("càzz0 che schifo")
    assert pf.contains_profanity("sh1t happens")


def test_mask_preserves_length(pf: ProfanityFilter) -> None:
    text = "This is shit"
    masked = pf.mask(text)
    assert len(masked) == len(text)
    assert masked != text


def test_multiword_phrase_with_punctuation(pf: ProfanityFilter) -> None:
    assert pf.contains_profanity("Che porco, dio?")
    assert pf.contains_profanity("you son-of a bitch")
