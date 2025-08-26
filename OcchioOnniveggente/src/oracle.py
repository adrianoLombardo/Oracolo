from __future__ import annotations

"""Utility helpers for the simplified Oracle application used in tests.

This module implements a compact subset of the features of the real project so
that the unit tests can exercise the behaviour of question handling, logging and
basic LLM interaction.  The goal is not feature parity but to provide small
and easy to understand helpers.
"""

import asyncio
import csv
import json
import random
from datetime import datetime
from pathlib import Path
from threading import Event
from typing import Any, AsyncGenerator, Callable, Iterable, List, Dict

from langdetect import LangDetectException, detect  # type: ignore

from .retrieval import Question, load_questions, Context
from .utils.error_handler import handle_error


# ---------------------------------------------------------------------------
# Questions dataset and random sampling
# ---------------------------------------------------------------------------

_QUESTIONS_CACHE: dict[str, List[Question]] | None = None
_QUESTIONS_MTIME: float | None = None


def get_questions() -> dict[str, List[Question]]:
    """Return the questions dataset reloading it when files change."""

    global _QUESTIONS_CACHE, _QUESTIONS_MTIME
    data_dir = Path(__file__).resolve().parent.parent / "data"
    json_files = list(data_dir.glob("*.json"))
    mtime = max((f.stat().st_mtime for f in json_files), default=None)
    if _QUESTIONS_CACHE is None or mtime != _QUESTIONS_MTIME:
        _QUESTIONS_CACHE = load_questions(data_dir)
        _QUESTIONS_MTIME = mtime
    return _QUESTIONS_CACHE


QUESTIONS_BY_TYPE: dict[str, List[Question]] = get_questions()
QUESTIONS_BY_CONTEXT: dict[Context, Dict[str, List[Question]]] = {
    Context.GENERIC: QUESTIONS_BY_TYPE
}

# Track served questions to avoid immediate repeats.
_USED_QUESTIONS: dict[Any, set[int]] = {}


def random_question(category: str, context: Context | None = None) -> Question | None:
    """Return a random question from ``category`` without immediate repeats."""

    ctx = context or Context.GENERIC
    cat = category.lower()
    qs = QUESTIONS_BY_CONTEXT.get(ctx, {}).get(cat)
    if not qs:
        return None
    key: Any = (ctx, cat) if context else cat
    used = _USED_QUESTIONS.setdefault(key, set())
    if len(used) >= len(qs):
        used.clear()
    choices = [i for i in range(len(qs)) if i not in used]
    idx = random.choice(choices)
    used.add(idx)
    return qs[idx]


# ---------------------------------------------------------------------------
# Conversation flow state machine
# ---------------------------------------------------------------------------

class ConversationFlow:
    """Minimal state machine modelling a fixed dialogue flow."""

    DEFAULT_FLOW = ["introduzione", "domanda_principale", "follow_up", "chiusura"]

    def __init__(self, *, context: str | None = None, flows: dict[str, list[str]] | None = None) -> None:
        flows = flows or {}
        self._flow = list(flows.get(context, self.DEFAULT_FLOW))
        if not self._flow:
            raise ValueError("Flow must contain at least one phase")
        self._idx = 0

    @property
    def state(self) -> str:
        return self._flow[self._idx]

    def advance(self) -> str:
        if self._idx < len(self._flow) - 1:
            self._idx += 1
        return self.state

    def is_finished(self) -> bool:
        return self._idx >= len(self._flow) - 1

    def reset(self, *, context: str | None = None, flows: dict[str, list[str]] | None = None) -> None:
        flows = flows or {}
        self._flow = list(flows.get(context, self.DEFAULT_FLOW))
        if not self._flow:
            raise ValueError("Flow must contain at least one phase")
        self._idx = 0


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def format_citations(sources: Iterable[dict[str, Any]]) -> str:
    """Return a comma separated string of ``id`` fields from ``sources``."""

    return ", ".join(str(s["id"]) for s in sources if s.get("id"))


def extract_summary(text: str) -> str:
    """Extract a short summary from structured ``text``."""

    for line in text.splitlines():
        if "Sintesi:" in line:
            return line.split("Sintesi:", 1)[1].strip()
    return text.strip()


def detect_language(text: str) -> str | None:
    """Best effort detection of the language used in ``text``."""

    try:
        return detect(text)
    except LangDetectException:  # pragma: no cover - rare
        return None


# ---------------------------------------------------------------------------
# LLM interaction helpers
# ---------------------------------------------------------------------------


def _build_messages(question: str, context: list[dict[str, Any]] | None, history: list[dict[str, str]] | None) -> List[dict[str, str]]:
    msgs: List[dict[str, str]] = []
    if history:
        msgs.extend(history)
    if context:
        ctx_lines = [f"[{i+1}] {c['text']}" for i, c in enumerate(context)]
        msgs.append({"role": "system", "content": "Fonti:\n" + "\n".join(ctx_lines)})
    msgs.append({"role": "user", "content": question})
    return msgs


def _build_instructions(
    lang_hint: str,
    context: list[dict[str, Any]] | None,
    style_prompt: str,
    mode: str | None,
    policy_prompt: str | None,
) -> str:
    parts = []
    if lang_hint.lower().startswith("it"):
        parts.append("Rispondi in italiano.")
    else:
        parts.append("Rispondi in inglese.")
    if context:
        parts.append("Rispondi SOLO usando i passaggi forniti.")
    if mode == "concise":
        parts.append("Stile conciso")
    elif mode == "detailed":
        parts.append("Struttura: 1) Sintesi; 2) Dettagli; 3) Fonti")
    if style_prompt:
        parts.append(style_prompt)
    if policy_prompt:
        parts.append(policy_prompt)
    return "\n".join(parts)


def oracle_answer(
    question: str,
    lang_hint: str,
    client: Any,
    llm_model: str,
    style_prompt: str,
    *,
    context: list[dict[str, Any]] | None = None,
    history: list[dict[str, str]] | None = None,
    mode: str | None = None,
    policy_prompt: str | None = None,
    stream: bool = False,
    on_token: Callable[[str], None] | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    """Return the model answer and echo back ``context``."""

    msgs = _build_messages(question, context, history)
    instr = _build_instructions(lang_hint, context, style_prompt, mode, policy_prompt)

    if stream and hasattr(client.responses, "with_streaming_response"):
        text = ""
        stream_obj = client.responses.with_streaming_response.create(
            model=llm_model, instructions=instr, input=msgs
        )
        for evt in stream_obj:
            if getattr(evt, "type", "") == "response.output_text.delta":
                delta = getattr(evt, "delta", "")
                text += delta
                if on_token:
                    on_token(delta)
        return text, context or []

    resp = client.responses.create(model=llm_model, instructions=instr, input=msgs)
    return getattr(resp, "output_text", ""), context or []


async def oracle_answer_stream(
    question: str,
    lang_hint: str,
    client: Any,
    llm_model: str,
    style_prompt: str,
    *,
    context: list[dict[str, Any]] | None = None,
    history: list[dict[str, str]] | None = None,
) -> AsyncGenerator[tuple[str, bool], None]:
    """Asynchronous generator yielding streamed answer tokens."""

    msgs = _build_messages(question, context, history)
    instr = _build_instructions(lang_hint, context, style_prompt, None, None)
    stream_obj = client.responses.with_streaming_response.create(
        model=llm_model, instructions=instr, input=msgs
    )
    text = ""
    for evt in stream_obj:
        if getattr(evt, "type", "") == "response.output_text.delta":
            delta = getattr(evt, "delta", "")
            text += delta
            yield delta, False
    yield text, True


def stream_generate(
    question: str,
    lang_hint: str,
    client: Any,
    llm_model: str,
    style_prompt: str,
    *,
    stop_event: Event | None = None,
) -> Iterable[str]:
    """Synchronous generator yielding streamed tokens."""

    stream_obj = client.responses.with_streaming_response.create(
        model=llm_model,
        instructions=_build_instructions(lang_hint, None, style_prompt, None, None),
        input=[{"role": "user", "content": question}],
    )
    for evt in stream_obj:
        if stop_event and stop_event.is_set():
            break
        if getattr(evt, "type", "") == "response.output_text.delta":
            yield getattr(evt, "delta", "")


# ---------------------------------------------------------------------------
# Follow-up handling and logging
# ---------------------------------------------------------------------------

DEFAULT_FOLLOW_UPS: dict[str, str] = {
    "poetica": "Ti va di approfondire questa immagine?",
    "didattica": "Puoi fornire un esempio pratico?",
    "evocativa": "Che altre sensazioni emergono?",
    "orientamento": "Quale sarà il tuo prossimo passo concreto?",
}


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
    """Append an interaction to ``path`` in JSON lines format."""

    ts = datetime.utcnow().isoformat()
    entry = {
        "timestamp": ts,
        "session_id": session_id,
        "lang": lang,
        "topic": topic,
        "question": question,
        "answer": answer,
        "summary": extract_summary(answer),
        "sources": sources or [],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".csv":
        file_exists = path.exists()
        with path.open("a", encoding="utf-8", newline="") as f:
            writer = csv.writer(f, quoting=csv.QUOTE_ALL)
            if not file_exists:
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
                ts,
                session_id,
                lang or "",
                topic or "",
                question,
                answer,
                json.dumps(sources or [], ensure_ascii=False),
            ])
    else:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return session_id


def answer_and_log_followup(
    question_data: Question | dict[str, str],
    client: Any,
    llm_model: str,
    log_path: Path,
    *,
    session_id: str,
    lang_hint: str = "it",
) -> tuple[str, str]:
    """Generate an answer for ``question_data`` and log the interaction."""

    if isinstance(question_data, Question):
        question = question_data.domanda
        qtype = question_data.type
        follow = question_data.follow_up
    else:
        question = question_data.get("domanda", "")
        qtype = question_data.get("type", "")
        follow = question_data.get("follow_up")
    answer, _ = oracle_answer(question, lang_hint, client, llm_model, "")
    follow = follow or DEFAULT_FOLLOW_UPS.get(qtype.lower(), "")
    append_log(question, answer, log_path, session_id=session_id, lang=lang_hint)
    if follow:
        append_log(follow, "", log_path, session_id=session_id, lang=lang_hint)
    return answer, follow


def log_interaction(
    *,
    context: Any | None = None,
    category: str | None = None,
    question: str | None = None,
    follow_up: str | None = None,
    user_response: str | None = None,
    path: Path | None = None,
) -> int:
    """Log a user interaction to ``path`` returning a sequential counter."""

    log_interaction.counter += 1
    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "interaction": log_interaction.counter,
        "context": context,
        "category": category,
        "question": question,
        "follow_up": follow_up,
        "user_response": user_response,
    }
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return log_interaction.counter


log_interaction.counter = 0


def export_audio_answer(text: str, out_path: Path, *, synth: Callable[[str, Path], None] | None = None) -> None:
    """Synthesize ``text`` into ``out_path`` using ``synth`` or a stub."""

    if synth is None:
        synth = synthesize
    synth(text, out_path)


def synthesize(text: str, out_path: Path) -> None:
    """Very small TTS stub used in tests."""

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(text.encode("utf-8"))


async def synthesize_async(*args, **kwargs):  # pragma: no cover - thin wrapper
    synthesize(*args, **kwargs)


async def transcribe(
    audio_path: Path,
    client: Any,
    model: str,
    *,
    lang_hint: str | None = None,
) -> str | None:
    """Transcribe ``audio_path`` using ``client`` handling errors."""

    try:
        if hasattr(client, "transcribe"):
            result = client.transcribe(audio_path, model, lang_hint=lang_hint)
            if asyncio.iscoroutine(result):
                result = await result
            return result
    except Exception as exc:  # noqa: BLE001
        return handle_error(exc, context="transcribe")
    return None


async def fast_transcribe(
    audio_path: Path,
    client: Any,
    model: str,
    *,
    lang_hint: str | None = None,
) -> str | None:
    """Convenience wrapper returning only the transcription text."""

    return await transcribe(audio_path, client, model, lang_hint=lang_hint)


# ---------------------------------------------------------------------------
# Follow-up acknowledgement helper
# ---------------------------------------------------------------------------


def acknowledge_followup(user_reply: str, next_question: Question | None = None) -> str:
    """Return a short acknowledgement or the next question prompt."""

    if next_question is not None:
        return next_question.domanda
    return "Grazie per la tua risposta."


__all__ = [
    "ConversationFlow",
    "DEFAULT_FOLLOW_UPS",
    "QUESTIONS_BY_CONTEXT",
    "QUESTIONS_BY_TYPE",
    "_USED_QUESTIONS",
    "acknowledge_followup",
    "answer_and_log_followup",
    "append_log",
    "export_audio_answer",
    "extract_summary",
    "format_citations",
    "get_questions",
    "log_interaction",
    "oracle_answer",
    "oracle_answer_stream",
    "random_question",
    "stream_generate",
    "synthesize",
    "synthesize_async",
    "transcribe",
]
