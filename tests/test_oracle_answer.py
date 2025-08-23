import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from OcchioOnniveggente.src.oracle import oracle_answer


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
