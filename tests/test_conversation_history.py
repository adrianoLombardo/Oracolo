import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from OcchioOnniveggente.src.conversation import ConversationManager
from OcchioOnniveggente.src.oracle import oracle_answer


class DummyResp:
    def __init__(self, text: str):
        self.output_text = text


class DummyResponses:
    def __init__(self):
        self.called_with = None

    def create(self, *, model, instructions, input):
        self.called_with = (model, instructions, input)
        return DummyResp("ok")


class DummyClient:
    def __init__(self):
        self.responses = DummyResponses()


def test_history_summary_is_passed_to_model():
    conv = ConversationManager()
    chat = conv.chat
    chat.max_turns = 1
    conv.push_user("ciao")
    conv.push_assistant("salve")
    conv.push_user("come va?")
    assert conv.messages[0]["role"] == "system"  # summary present
    client = DummyClient()
    oracle_answer(
        question="prossima?",
        lang_hint="it",
        client=client,
        llm_model="m",
        style_prompt="",
        conv=conv,
    )
    _, _, messages = client.responses.called_with
    assert messages[0]["role"] == "system"
    assert messages[-1]["content"] == "prossima?"
