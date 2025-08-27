import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))


from OcchioOnniveggente.src.oracle import (
    answer_and_log_followup,
    oracle_answer,
    DEFAULT_FOLLOW_UPS,
)

from OcchioOnniveggente.src.oracle import answer_and_log_followup, oracle_answer
from src.retrieval import Question


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


def test_oracle_answer_returns_response_and_context():
    client = DummyClient()
    context = [{"text": "fonte"}]
    ans, ctx = oracle_answer(
        question="Che cos'è?",
        lang_hint="it",
        client=client,
        llm_model="test-model",
        style_prompt="",
        context=context,
        mode="concise",
    )
    assert ans == "risposta"
    assert ctx == context
    model, instructions, messages = client.responses.called_with
    assert model == "test-model"
    assert "Rispondi in italiano." in instructions
    assert "Rispondi SOLO usando i passaggi" in instructions
    assert messages[-1] == {"role": "user", "content": "Che cos'è?"}


def test_oracle_answer_handles_string_context():
    client = DummyClient()
    context = ["nota di contesto"]
    ans, ctx = oracle_answer(
        question="Che cos'è?",
        lang_hint="it",
        client=client,
        llm_model="test-model",
        context=context,
    )
    assert ans == "risposta"
    assert ctx == context
    model, instructions, messages = client.responses.called_with
    assert model == "test-model"
    # The first system message should include our string context
    assert any("nota di contesto" in m.get("content", "") for m in messages)


def test_answer_and_log_followup(tmp_path: Path):
    client = DummyClient()
    qdata = Question(id="1", domanda="Chi sei?", type="poetica", follow_up="Vuoi continuare?")
    log = tmp_path / "log.jsonl"
    answer, follow_up = answer_and_log_followup(
        qdata, client, "test-model", log, session_id="sess-1"
    )
    assert answer == "risposta"
    assert follow_up == "Vuoi continuare?"
    lines = log.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    first, second = [json.loads(l) for l in lines]
    assert first["question"] == "Chi sei?"
    assert second["question"] == "Vuoi continuare?"
    assert second["answer"] == ""


def test_default_followup_used_when_missing(tmp_path: Path):
    client = DummyClient()
    qdata = {"domanda": "Come stai?", "type": "evocativa"}
    log = tmp_path / "log.jsonl"
    answer, follow_up = answer_and_log_followup(
        qdata, client, "test-model", log, session_id="sess-2"
    )
    assert answer == "risposta"
    assert follow_up == DEFAULT_FOLLOW_UPS["evocativa"]
    lines = log.read_text(encoding="utf-8").strip().splitlines()
    # follow-up should be appended as second log entry
    assert len(lines) == 2
    second = json.loads(lines[1])
    assert second["question"] == DEFAULT_FOLLOW_UPS["evocativa"]
