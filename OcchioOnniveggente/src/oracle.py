"""Semplici helper per le risposte dell'Oracolo.

Questo modulo implementa versioni minimali delle funzioni utilizzate nei test
unitari.  Le funzioni simulano il comportamento dell'API OpenAI rispondendo
tramite l'oggetto ``client`` passato come parametro.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, AsyncGenerator, Callable, Dict, List, Tuple


def _build_instructions(
    lang_hint: str, context: List[Dict[str, Any]] | None, mode: str
) -> str:
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


def oracle_answer(
    question: str,
    lang_hint: str,
    client: Any,
    llm_model: str,
    style_prompt: str,
    *,
    context: List[Dict[str, Any]] | None = None,
    history: List[Dict[str, str]] | None = None,
    topic: str | None = None,
    policy_prompt: str | None = None,
    mode: str = "detailed",
    stream: bool = False,
    on_token: Callable[[str], None] | None = None,
) -> Tuple[str, List[Dict[str, Any]] | None]:
    """Restituisce la risposta del modello e il contesto usato."""

    instructions = _build_instructions(lang_hint, context, mode)
    messages: List[Dict[str, str]] = []
    if context:
        src = "Fonti:\n" + "\n".join(
            f"[{i+1}] {c.get('text', '')}" for i, c in enumerate(context)
        )
        messages.append({"role": "system", "content": src})
    messages.append({"role": "user", "content": question})

    if stream:
        stream_obj = client.responses.with_streaming_response.create(
            model=llm_model, instructions=instructions, input=messages
        )
        tokens: List[str] = []
        for ev in stream_obj:
            if getattr(ev, "type", "") == "response.output_text.delta":
                if on_token:
                    on_token(ev.delta)
                tokens.append(ev.delta)
        text = "".join(tokens) or getattr(stream_obj, "output_text", "")
        return text, context or []

    resp = client.responses.create(
        model=llm_model, instructions=instructions, input=messages
    )
    return getattr(resp, "output_text", ""), context or []


async def oracle_answer_stream(
    question: str,
    lang_hint: str,
    client: Any,
    llm_model: str,
    style_prompt: str,
    *,
    context: List[Dict[str, Any]] | None = None,
) -> AsyncGenerator[Tuple[str, bool], None]:
    """Generatore asincrono che restituisce la risposta a chunk."""

    instructions = _build_instructions(lang_hint, context, "detailed")
    msgs: List[Dict[str, str]] = []
    if context:
        src = "Fonti:\n" + "\n".join(
            f"[{i+1}] {c.get('text', '')}" for i, c in enumerate(context)
        )
        msgs.append({"role": "system", "content": src})
    msgs.append({"role": "user", "content": question})
    stream_obj = client.responses.with_streaming_response.create(
        model=llm_model,
        instructions=instructions,
        input=msgs,
    )
    tokens: List[str] = []
    for ev in stream_obj:
        if getattr(ev, "type", "") == "response.output_text.delta":
            tokens.append(ev.delta)
            yield ev.delta, False
    final = "".join(tokens) or getattr(stream_obj, "output_text", "")
    yield final, True


def format_citations(sources: List[Dict[str, Any]]) -> str:
    """Format source IDs as a comma-separated string."""

    return ", ".join(s.get("id", "") for s in sources if s.get("id"))


def export_audio_answer(
    text: str,
    out_path: Path,
    *,
    synth: Callable[[str, Path], None],
) -> None:
    """Esporta ``text`` in ``out_path`` usando la funzione ``synth``."""

    synth(text, out_path)


def append_log(
    question: str,
    answer: str,
    log_path: Path,
    *,
    session_id: str | None = None,
    lang: str = "",
    topic: str = "",
    sources: List[Dict[str, Any]] | None = None,
) -> str:
    """Aggiunge una voce di log in formato JSONL o CSV."""

    sid = session_id or "session-1"
    if log_path.suffix == ".csv":
        header = (
            "timestamp,session_id,lang,topic,question,answer,sources".split(",")
        )
        if not log_path.exists():
            log_path.write_text(
                ",".join(f'"{h}"' for h in header) + "\n", encoding="utf-8"
            )
        line = ["0", sid, lang, topic, question, answer, json.dumps(sources or [])]
        log_path.write_text(
            log_path.read_text(encoding="utf-8")
            + ",".join(f'"{v}"' for v in line)
            + "\n",
            encoding="utf-8",
        )
        return sid

    entry = {
        "timestamp": 0,
        "session_id": sid,
        "lang": lang,
        "topic": topic,
        "question": question,
        "answer": answer,
        "summary": answer,
        "sources": sources or [],
    }
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")
    return sid


def extract_summary(text: str) -> str:
    """Estrae la sezione di sintesi da un testo strutturato."""

    for line in text.splitlines():
        if line.lower().startswith("1") and ":" in line:
            return line.split(":", 1)[1].strip()
    return text.strip()


def transcribe(
    path_or_bytes: str | Path | bytes,
    client: Any,
    stt_model: str,
    lang_hint: str | None = None,
) -> str | None:
    """Stub di trascrizione locale."""

    return ""


def fast_transcribe(
    path_or_bytes,
    client,
    stt_model: str,
    lang_hint: str | None = None,
) -> str | None:
    """Stub rapido di trascrizione."""

    return ""


def synthesize(text: str, *_, **__) -> bytes:
    """Stub di sintesi vocale."""

    return text.encode("utf-8")

