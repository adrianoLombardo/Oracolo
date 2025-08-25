from __future__ import annotations

import asyncio
import csv
import json
import uuid
from pathlib import Path
from typing import Any, AsyncGenerator, Callable, Iterable


def format_citations(sources: Iterable[dict[str, Any]]) -> str:
    """Return a comma-separated string of source IDs."""
    return ", ".join(str(s.get("id", "")) for s in sources if s.get("id"))


def export_audio_answer(
    text: str,
    out_path: Path,
    *,
    synth: Callable[[str, Path], None] | None = None,
) -> None:
    """Generate an audio file for ``text`` using ``synth``."""
    synth = synth or (lambda t, p: p.write_bytes(b""))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    synth(text, out_path)


def extract_summary(text: str) -> str:
    """Extract the summary section from a structured answer."""
    import re

    m = re.search(r"1\)\s*[^:]+:\s*(.*?)\n2\)", text, re.S)
    if m:
        return m.group(1).strip()
    return text.strip()


def append_log(
    question: str,
    answer: str,
    path: Path,
    *,
    session_id: str | None = None,
    lang: str = "",
    topic: str = "",
    sources: list[dict[str, Any]] | None = None,
) -> str:
    """Append a QA pair to a log in JSONL or CSV format."""
    session_id = session_id or uuid.uuid4().hex
    sources = sources or []
    entry = {
        "timestamp": asyncio.get_event_loop().time()
        if asyncio.get_event_loop().is_running()
        else 0.0,
        "session_id": session_id,
        "lang": lang,
        "topic": topic,
        "question": question,
        "answer": answer,
        "summary": extract_summary(answer),
        "sources": sources,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".jsonl":
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    else:
        new_file = not path.exists()
        with path.open("a", encoding="utf-8", newline="") as f:
            writer = csv.writer(f, quoting=csv.QUOTE_ALL)
            if new_file:
                writer.writerow([
                    "timestamp",
                    "session_id",
                    "lang",
                    "topic",
                    "question",
                    "answer",
                    "sources",
                ])
            writer.writerow([
                entry["timestamp"],
                session_id,
                lang,
                topic,
                question,
                answer,
                json.dumps(sources),
            ])
    return session_id


def _build_instructions(lang_hint: str, policy: str, mode: str) -> str:
    instr = "Rispondi in italiano." if lang_hint == "it" else "Rispondi in inglese."
    if policy:
        instr += " " + policy
    if mode == "concise":
        instr += " Stile conciso."
    else:
        instr += " Struttura: 1) Sintesi: ... 2) Dettagli: ... 3) Fonti: ..."
    return instr


def _build_messages(question: str, context: Iterable[dict[str, Any]] | None) -> list[dict[str, str]]:
    msgs: list[dict[str, str]] = []
    if context:
        src_lines = "\n".join(f"[{i+1}] {c.get('text', '')}" for i, c in enumerate(context))
        msgs.append({"role": "system", "content": f"Fonti:\n{src_lines}"})
    msgs.append({"role": "user", "content": question})
    return msgs


def oracle_answer(
    question: str,
    lang_hint: str,
    client: Any,
    llm_model: str,
    style_prompt: str,
    *,
    context: list[dict[str, Any]] | None = None,
    policy_prompt: str = "",
    mode: str = "detailed",
    stream: bool = False,
    on_token: Callable[[str], None] | None = None,
) -> tuple[str, list[dict[str, Any]] | None]:
    """Return the answer text and context using the provided ``client``."""
    instructions = _build_instructions(lang_hint, policy_prompt, mode)
    if context:
        instructions += " Rispondi SOLO usando i passaggi forniti."
    messages = _build_messages(question, context)

    if stream and hasattr(client.responses, "with_streaming_response"):
        stream_obj = client.responses.with_streaming_response.create(
            model=llm_model, instructions=instructions, input=messages
        )
        if on_token:
            for event in stream_obj:
                if getattr(event, "type", "") == "response.output_text.delta":
                    on_token(event.delta)
        return stream_obj.output_text, context
    else:
        resp = client.responses.create(
            model=llm_model, instructions=instructions, input=messages
        )
        return resp.output_text, context


async def oracle_answer_stream(
    question: str,
    lang_hint: str,
    client: Any,
    llm_model: str,
    style_prompt: str,
    *,
    context: list[dict[str, Any]] | None = None,
) -> AsyncGenerator[tuple[str, bool], None]:
    """Asynchronously stream answer chunks from the model."""
    instructions = _build_instructions(lang_hint, "", "detailed")
    messages = _build_messages(question, context)
    stream_obj = client.responses.with_streaming_response.create(
        model=llm_model, instructions=instructions, input=messages
    )
    for event in stream_obj:
        if getattr(event, "type", "") == "response.output_text.delta":
            yield event.delta, False
    yield stream_obj.output_text, True


# Placeholder implementations for optional APIs used elsewhere in the project
async def oracle_answer_async(*args, **kwargs):
    return oracle_answer(*args, **kwargs)


def transcribe(*args, **kwargs):
    return ""


def fast_transcribe(*args, **kwargs):
    return ""


def synthesize(*args, **kwargs):
    return b""


async def transcribe_async(*args, **kwargs):
    return "", ""


async def fast_transcribe_async(*args, **kwargs):
    return None
