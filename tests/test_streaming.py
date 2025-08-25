import sys
from types import SimpleNamespace
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from OcchioOnniveggente.src.oracle import oracle_answer, oracle_answer_stream


class DummyEvent(SimpleNamespace):
    pass


class DummyStream:
    def __init__(self, text: str):
        self.output_text = text
        self._events = [
            DummyEvent(type="response.output_text.delta", delta="foo"),
            DummyEvent(type="response.output_text.delta", delta="bar"),
        ]

    def __iter__(self):
        return iter(self._events)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class DummyResponses:
    def __init__(self):
        class WithStream:
            def create(self_inner, *, model, instructions, input):
                return DummyStream("foobar")
        self.with_streaming_response = WithStream()

    def create(self, *, model, instructions, input):
        return SimpleNamespace(output_text="foobar")


class DummyClient:
    def __init__(self):
        self.responses = DummyResponses()
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=lambda **k: []))


def test_oracle_answer_stream_option():
    c = DummyClient()
    tokens: list[str] = []
    ans, _ = oracle_answer(
        "q",
        "it",
        c,
        "gpt",
        "",
        stream=True,
        on_token=tokens.append,
    )
    assert ans == "foobar"
    assert tokens == ["foo", "bar"]


import asyncio


def test_oracle_answer_stream_generator():
    c = DummyClient()
    parts: list[tuple[str, bool]] = []

    async def run() -> None:
        async for chunk, done in oracle_answer_stream("q", "it", c, "gpt", ""):
            parts.append((chunk, done))

    asyncio.run(run())
    assert parts == [("foo", False), ("bar", False), ("foobar", True)]
