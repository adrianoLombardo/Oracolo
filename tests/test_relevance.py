import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))
from OcchioOnniveggente.src.ai import is_relevant


@pytest.fixture()
def allowed_topics() -> list[str]:
    return ["pizza", "musica"]


@pytest.fixture()
def refusal_message() -> str:
    return "Domanda fuori tema"


def test_question_with_topic(allowed_topics, refusal_message) -> None:
    result, message = is_relevant("Parlami della pizza", allowed_topics, refusal_message)
    assert result is True
    assert message == ""


def test_question_without_topic(allowed_topics, refusal_message) -> None:
    result, message = is_relevant("Che tempo fa oggi?", allowed_topics, refusal_message)
    assert result is False
    assert message == refusal_message
