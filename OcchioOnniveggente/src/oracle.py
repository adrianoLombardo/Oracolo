"""Utility helpers for the Oracle application.

This module provides small, self contained helpers that are used across the
project and in the unit tests.  The original project contains a much more
feature rich implementation, however for the purposes of the tests we only
need a compact subset of the behaviour.
"""

from __future__ import annotations

from datetime import datetime
import asyncio
import csv
import json
from pathlib import Path
from typing import Any, AsyncGenerator, Callable, Iterable, List, Tuple


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def format_citations(sources: Iterable[dict[str, Any]]) -> str:
    """Return a comma separated string of the ``id`` fields in ``sources``.

    Only entries that provide an ``id`` are included.  This mirrors the small
    helper used in the real project and is exercised directly by the tests.
    """

    return ", ".join(str(s["id"]) for s in sources if s.get("id"))


def export_audio_answer(
    text: str, out_path: Path, *, synth: Callable[[str, Path], None] | None = None
) -> None:
    """Create an audio file for ``text`` using ``synth``.

    The tests inject a dummy ``synth`` implementation which simply writes
    bytes to ``out_path``.  The function therefore only needs to make sure the
    destination directory exists and call the callback.
    """

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if synth is None:
        # Fallback: create an empty file so that callers can still read it.
        out_path.write_bytes(b"")
    else:
        synth(text, out_path)


def extract_summary(answer: str) -> str:
    """Extract the summary section from a structured answer.

    Answers in this project may follow a ``1)`` ``2)`` numbered structure
    where the first element contains a short summary introduced either with
    ``Sintesi:`` or directly after ``1)``.  If no such structure is found the
    whole answer is returned.
    """

    for line in answer.splitlines():
        line = line.strip()
        if line.lower().startswith("1)") and ":" in line:
            return line.split(":", 1)[1].strip()
        if line.lower().startswith("sintesi:"):
            return line.split(":", 1)[1].strip()
    return answer.strip()


# ---------------------------------------------------------------------------
# Core answer helpers
# ---------------------------------------------------------------------------


def _build_instructions(
    lang_hint: str, context: List[dict[str, Any]] | None, mode: str
) -> str:
    """Return an instruction string for the LLM."""

    parts: List[str] = []
    if lang_hint == "it":
        parts.append("Rispondi in italiano.")
    elif lang_hint == "en":
        parts.append("Rispondi in inglese.")
    if context:
        parts.append("Rispondi SOLO usando i passaggi forniti.")
    if mode == "concise":
        parts.append("Stile conciso.")
    else:
        parts.append("Stile dettagliato.")
        parts.append("Struttura: 1)")
    return "\n".join(parts)


def _build_messages(
    question: str,
    context: List[dict[str, Any]] | None,
    history: List[dict[str, str]] | None,
) -> List[dict[str, str]]:
    """Create the chat message list passed to the model."""

    messages: List[dict[str, str]] = []
    if history:
        messages.extend(history)
    if context:
        for c in context:
            messages.append({"role": "system", "content": c.get("text", "")})
        sources = "; ".join(
            f"[{i}] {c.get('text', '')}" for i, c in enumerate(context, 1)
        )
        messages.append({"role": "system", "content": f"Fonti: {sources}"})
    messages.append({"role": "user", "content": question})
    return messages


def oracle_answer(
    question: str,
    lang_hint: str,
    client: Any,
    llm_model: str,
    style_prompt: str,
    *,
    context: List[dict[str, Any]] | None = None,
    history: List[dict[str, str]] | None = None,
    policy_prompt: str = "",
    mode: str = "detailed",
    stream: bool = False,
    on_token: Callable[[str], None] | None = None,
) -> Tuple[str, List[dict[str, Any]]]:
    """Return an answer from ``client`` and the context used.

    The ``client`` argument is expected to mimic the OpenAI Python client's
    interface.  In the tests a small dummy object is provided.  When ``stream``
    is ``True`` streaming tokens are forwarded to ``on_token`` and concatenated
    to form the final answer.
    """

    instructions = _build_instructions(lang_hint, context, mode)
    messages = _build_messages(question, context, history)

    if stream:
        response = client.responses.with_streaming_response.create(
            model=llm_model, instructions=instructions, input=messages
        )
        output_text = ""
        on_token = on_token or (lambda _t: None)
        with response as stream_resp:
            for event in stream_resp:
                if getattr(event, "type", "") == "response.output_text.delta":
                    delta = getattr(event, "delta", "")
                    output_text += delta
                    on_token(delta)
        return output_text, context or []

    resp = client.responses.create(
        model=llm_model, instructions=instructions, input=messages
    )
    return resp.output_text, context or []


async def oracle_answer_async(*args, **kwargs):  # pragma: no cover - thin wrapper
    """Asynchronous wrapper around :func:`oracle_answer`."""

    return oracle_answer(*args, **kwargs)


async def oracle_answer_stream(
    question: str,
    lang_hint: str,
    client: Any,
    llm_model: str,
    style_prompt: str,
    *,
    context: List[dict[str, Any]] | None = None,
    history: List[dict[str, str]] | None = None,
    policy_prompt: str = "",
    mode: str = "detailed",
) -> AsyncGenerator[Tuple[str, bool], None]:
    """Stream answer tokens from the model.

    Yields ``(chunk, False)`` for each token and finally ``(full_text, True)``
    with the accumulated output.
    """

    instructions = _build_instructions(lang_hint, context, mode)
    messages = _build_messages(question, context, history)
    response = client.responses.with_streaming_response.create(
        model=llm_model, instructions=instructions, input=messages
    )

    output_text = ""
    with response as stream_resp:
        for event in stream_resp:
            if getattr(event, "type", "") == "response.output_text.delta":
                delta = getattr(event, "delta", "")
                output_text += delta
                yield delta, False
                await asyncio.sleep(0)  # allow cooperative scheduling
    yield output_text, True


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def append_log(
    question: str,
    answer: str,
    path: Path,
    *,
    session_id: str,
    lang: str | None = None,
    topic: str | None = None,
    sources: List[dict[str, Any]] | None = None,
) -> str:
    """Append an interaction to ``path``.

    The log can be either JSON lines (``.jsonl``) or CSV (``.csv``).  Metadata
    such as session id, language and topic are recorded alongside the question
    and answer.  The function returns the ``session_id`` for convenience.
    """

    sources = sources or []
    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "session_id": session_id,
        "lang": lang,
        "topic": topic,
        "question": question,
        "answer": answer,
        "summary": extract_summary(answer),
        "sources": sources,
    }

    path.parent.mkdir(parents=True, exist_ok=True)

    if path.suffix == ".csv":
        file_exists = path.exists()
        with path.open("a", encoding="utf-8", newline="") as f:
            writer = csv.writer(f, quoting=csv.QUOTE_ALL)
            if not file_exists:
                writer.writerow(
                    [
                        "timestamp",
                        "session_id",
                        "lang",
                        "topic",
                        "question",
                        "answer",
                        "sources",
                    ]
                )
            writer.writerow(
                [
                    entry["timestamp"],
                    session_id,
                    lang or "",
                    topic or "",
                    question,
                    answer,
                    json.dumps(sources, ensure_ascii=False),
                ]
            )
    else:  # default to json lines
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    return session_id


# ---------------------------------------------------------------------------
# Text to speech stubs
# ---------------------------------------------------------------------------

def synthesize(
    text: str,
    out_path: Path,
    client: Any | None = None,
    tts_model: str | None = None,
    tts_voice: str | None = None,
) -> None:
    """Minimal TTS helper used by the UI.

    The function simply writes an empty file; the real project would call a TTS
    backend.  It returns ``None`` to mirror the original signature.
    """

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(b"")


async def synthesize_async(*args, **kwargs):  # pragma: no cover - thin wrapper
    """Asynchronous wrapper around :func:`synthesize`."""

    synthesize(*args, **kwargs)



def transcribe(*args, **kwargs) -> str | None:
    """Minimal speech-to-text stub used only for imports in tests."""
    return ""

