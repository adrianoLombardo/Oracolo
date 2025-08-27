import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.retrieval import Question
from OcchioOnniveggente.src.conversation import QuestionSession
from OcchioOnniveggente.src.oracle import answer_and_log_followup, acknowledge_followup


class DummyResp:
    def __init__(self, text: str):
        self.output_text = text


class DummyResponses:
    def __init__(self):
        self.called_with = None

    def create(self, *, model, instructions, input):
        self.called_with = (model, instructions, input)
        return DummyResp("risposta")


class DummyClient:
    def __init__(self):
        self.responses = DummyResponses()


def test_full_question_followup_cycle(tmp_path: Path):
    client = DummyClient()
    q = Question(domanda="Chi sei?", type="poetica", follow_up="Vuoi continuare?")
    log = tmp_path / "log.jsonl"
    session = QuestionSession(question=q.domanda, follow_up=q.follow_up)

    answer, follow_up = answer_and_log_followup(q, client, "m", log, session_id="s1")
    assert follow_up == "Vuoi continuare?"
    session.record_answer(answer, "Sì")
    assert session.answers == ["risposta"]
    assert session.replies == ["Sì"]

    ack = acknowledge_followup(session.replies[-1])
    assert "Grazie" in ack

    next_q = Question(domanda="Che fai?", type="poetica")
    nxt = acknowledge_followup(session.replies[-1], next_question=next_q)
    assert nxt == "Che fai?"
