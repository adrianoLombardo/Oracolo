"""Utility helpers for the Oracle application.

This module provides small, self contained helpers used across the project and
in the unit tests.  Only a compact subset of the original project's behaviour is
implemented here.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
import asyncio
import csv
import json
import random
import re
from pathlib import Path
from threading import Event
from typing import Any, AsyncGenerator, Callable, Dict, Iterable, Iterator, List

from langdetect import detect, LangDetectException

from .utils.error_handler import handle_error
from .retrieval import Question, load_questions, Context

# ---------------------------------------------------------------------------
# Questions loading and caching
# ---------------------------------------------------------------------------

_QUESTIONS_CACHE: Dict[Context, Dict[str, List[Question]]] | None = None


def get_questions(context: Context | None = None) -> Dict[str, List[Question]]:
    """Return questions for ``context`` loading them once from disk."""

    global _QUESTIONS_CACHE
    if _QUESTIONS_CACHE is None:
        data_dir = Path(__file__).resolve().parent.parent / "data"
        _QUESTIONS_CACHE = load_questions(data_dir)
    ctx = context or Context.GENERIC
    return _QUESTIONS_CACHE.get(ctx, {})


QUESTIONS_BY_CONTEXT = get_questions()

# ---------------------------------------------------------------------------
# Conversation flow helper
# ---------------------------------------------------------------------------


class ConversationFlow:
    """Simple state machine to model multi-phase dialogues."""

    DEFAULT_FLOW = [
        "introduzione",
        "domanda_principale",
        "follow_up",
        "chiusura",
    ]

    def __init__(
        self,
        *,
        context: str | None = None,
        flows: Dict[str, List[str]] | None = None,
    ) -> None:
        flows = flows or {}
        self._flows = flows
        self._context = context
        self._phases = list(flows.get(context, self.DEFAULT_FLOW))
        if not self._phases:
            raise ValueError("Flow must contain at least one phase")
        self._index = 0

    @property
    def state(self) -> str:
        return self._phases[self._index]

    def advance(self) -> str:
        if self._index < len(self._phases) - 1:
            self._index += 1
        return self.state

    def is_finished(self) -> bool:
        return self._index >= len(self._phases) - 1

    def reset(self, *, context: str | None = None) -> None:
        if context is not None:
            self._context = context
            self._phases = list(self._flows.get(context, self.DEFAULT_FLOW))
            if not self._phases:
                raise ValueError("Flow must contain at least one phase")
        self._index = 0


# ---------------------------------------------------------------------------
# Off-topic helpers and small utilities
# ---------------------------------------------------------------------------

OFF_TOPIC_RESPONSES: Dict[str, str] = {
    "poetica": "Preferirei non avventurarmi in slanci poetici.",
    "didattica": "Al momento non posso fornire spiegazioni didattiche.",
    "evocativa": "Queste domande evocative sfuggono al mio scopo.",
    "orientamento": "Non sono in grado di offrire indicazioni stradali.",
    "default": "Mi dispiace, non posso aiutarti con questa richiesta.",
}

OFF_TOPIC_REPLIES = {
    "poetica": "Mi dispiace, ma preferisco non rispondere a richieste poetiche.",
    "didattica": "Questa domanda sembra didattica e non rientra nel mio ambito.",
    "evocativa": "Temo che il suo carattere evocativo mi impedisca di rispondere.",
    "orientamento": "Non posso fornire indicazioni di orientamento in questo contesto.",
}


def off_topic_reply(category: str | None) -> str:
    if not category:
        return "Mi dispiace, ma non posso rispondere a questa domanda."
    return OFF_TOPIC_REPLIES.get(category.lower(), OFF_TOPIC_RESPONSES["default"])


def format_citations(sources: Iterable[dict[str, Any]]) -> str:
    """Return a comma separated string of source ids."""

    return ", ".join(str(s["id"]) for s in sources if s.get("id"))


# Default follow-up messages per question type
DEFAULT_FOLLOW_UPS: Dict[str, str] = {
    "poetica": "Ti va di approfondire questa immagine?",
    "didattica": "Puoi fornire un esempio pratico?",
    "evocativa": "Che altre sensazioni emergono?",
    "orientamento": "Quale sarà il tuo prossimo passo concreto?",
}


def _language_name(code: str) -> str:
    mapping = {"it": "italiano", "en": "inglese"}
    return mapping.get(code.lower(), code)


# ---------------------------------------------------------------------------
# LLM interaction helpers
# ---------------------------------------------------------------------------


def _build_prompt(
    question: str,
    lang_hint: str,
    style_prompt: str,
    context: List[dict[str, Any]] | None,
    policy_prompt: str | None,
    mode: str,
    history: List[dict[str, str]] | None,
) -> tuple[str, List[dict[str, str]]]:
    instructions = f"Rispondi in { _language_name(lang_hint) }."
    if style_prompt:
        instructions += " " + style_prompt
    if policy_prompt:
        instructions += " " + policy_prompt
    if mode == "concise":
        instructions += "\nStile conciso."
    elif mode == "detailed":
        instructions += "\nStruttura: 1) Sintesi; 2) Dettagli; 3) Fonti."
    messages = list(history) if history else []
    if context:
        instructions += "\nRispondi SOLO usando i passaggi forniti."
        lines = "\n".join(f"[{i+1}] {c['text']}" for i, c in enumerate(context))
        messages.append({"role": "system", "content": "Fonti:\n" + lines})
    messages.append({"role": "user", "content": question})
    return instructions, messages


def oracle_answer(
    question: str,
    lang_hint: str,
    client: Any,
    llm_model: str,
    style_prompt: str,
    *,
    context: List[dict[str, Any]] | None = None,
    policy_prompt: str | None = None,
    mode: str = "default",
    stream: bool = False,
    on_token: Callable[[str], None] | None = None,
    history: List[dict[str, str]] | None = None,
) -> tuple[str, List[dict[str, Any]]]:
    """Generate an answer using ``client`` returning the text and ``context``."""

    instructions, messages = _build_prompt(
        question, lang_hint, style_prompt, context, policy_prompt, mode, history
    )
    if stream:
        stream_obj = client.responses.with_streaming_response.create(
            model=llm_model, instructions=instructions, input=messages
        )
        collected = ""
        for ev in stream_obj:
            if ev.type == "response.output_text.delta":
                collected += ev.delta
                if on_token:
                    on_token(ev.delta)
        return collected, context or []
    resp = client.responses.create(
        model=llm_model, instructions=instructions, input=messages
    )
    return getattr(resp, "output_text", str(resp)), context or []


async def oracle_answer_stream(
    question: str,
    lang_hint: str,
    client: Any,
    llm_model: str,
    style_prompt: str,
    *,
    context: List[dict[str, Any]] | None = None,
    policy_prompt: str | None = None,
    history: List[dict[str, str]] | None = None,
) -> AsyncGenerator[tuple[str, bool], None]:
    """Asynchronous generator yielding streamed answer chunks."""

    instructions, messages = _build_prompt(
        question, lang_hint, style_prompt, context, policy_prompt, "default", history
    )
    stream_obj = client.responses.with_streaming_response.create(
        model=llm_model, instructions=instructions, input=messages
    )
    collected = ""
    for ev in stream_obj:
        if ev.type == "response.output_text.delta":
            collected += ev.delta
            yield ev.delta, False
    yield collected, True


def stream_generate(
    question: str,
    lang_hint: str,
    client: Any,
    llm_model: str,
    style_prompt: str,
    *,
    context: List[dict[str, Any]] | None = None,
    policy_prompt: str | None = None,
    stop_event: Event | None = None,
) -> Iterator[str]:
    """Synchronous generator yielding streamed tokens."""

    instructions, messages = _build_prompt(
        question, lang_hint, style_prompt, context, policy_prompt, "default", None
    )
    stream_obj = client.responses.with_streaming_response.create(
        model=llm_model, instructions=instructions, input=messages
    )
    for ev in stream_obj:
        if stop_event and stop_event.is_set():
            break
        if ev.type == "response.output_text.delta":
            yield ev.delta


# ---------------------------------------------------------------------------
# Question session manager
# ---------------------------------------------------------------------------


class QuestionSession:
    """Manage the flow of questions and follow-ups for a conversation."""

    def __init__(self) -> None:
        self._asked: Dict[tuple[Context, str], set[int]] = {}
        self.current_question: Question | None = None
        self._pending_followup: str | None = None

    def next_question(
        self, category: str, *, context: Context = Context.GENERIC
    ) -> Question | None:
        qs = get_questions(context).get(category.lower())
        if not qs:
            self.current_question = None
            self._pending_followup = None
            return None
        key = (context, category.lower())
        used = self._asked.setdefault(key, set())
        if len(used) == len(qs):
            used.clear()
        available = [i for i in range(len(qs)) if i not in used]
        idx = random.choice(available)
        used.add(idx)
        q = qs[idx]
        self.current_question = q
        self._pending_followup = q.follow_up or DEFAULT_FOLLOW_UPS.get(q.type.lower(), "")
        return q

    def record_answer(
        self,
        user_text: str,
        follow_up: str | None = None,
        client: Any | None = None,
        llm_model: str = "",
        *,
        lang_hint: str = "it",
    ) -> str:
        if client is None:
            self._pending_followup = follow_up or ""
            return ""
        answer, _ = oracle_answer(user_text, lang_hint, client, llm_model, "")
        self._pending_followup = follow_up or ""
        return answer

    def next_followup(self) -> str | None:
        follow = self._pending_followup
        self._pending_followup = None
        return follow


SESSION = QuestionSession()


def random_question(
    category: str, context: Context = Context.GENERIC
) -> Question | None:
    """Return a random question using the global session."""

    return SESSION.next_question(category, context=context)


def answer_with_followup(
    question_data: Question | dict[str, str] | str,
    client: Any,
    llm_model: str,
    *,
    lang_hint: str = "it",
) -> tuple[str, str]:
    if isinstance(question_data, Question):
        qtext = question_data.domanda
        follow = question_data.follow_up or DEFAULT_FOLLOW_UPS.get(question_data.type.lower(), "")
    elif isinstance(question_data, dict):
        qtext = question_data.get("domanda", "")
        qtype = question_data.get("type", "").lower()
        follow = question_data.get("follow_up") or DEFAULT_FOLLOW_UPS.get(qtype, "")
    else:
        qtext = str(question_data)
        follow = ""
    ans = SESSION.record_answer(qtext, follow, client, llm_model, lang_hint=lang_hint)
    return ans, SESSION.next_followup() or ""


# ---------------------------------------------------------------------------
# Logging helpers
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
    """Append an interaction entry to ``path`` in JSONL or CSV format."""

    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "session_id": session_id,
        "lang": lang,
        "topic": topic,
        "question": question,
        "answer": answer,
        "summary": answer,
        "sources": sources or [],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".csv":
        new = not path.exists()
        with path.open("a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=[
                    "timestamp",
                    "session_id",
                    "lang",
                    "topic",
                    "question",
                    "answer",
                    "sources",
                ],
                quoting=csv.QUOTE_ALL,
            )
            if new:
                writer.writeheader()
            row = entry.copy()
            row["sources"] = json.dumps(row["sources"], ensure_ascii=False)
            row.pop("summary", None)
            writer.writerow(row)
    else:
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
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
    if isinstance(question_data, Question):
        qtext = question_data.domanda
        qtype = question_data.type.lower()
        follow = question_data.follow_up or DEFAULT_FOLLOW_UPS.get(qtype, "")
    else:
        qtext = question_data.get("domanda", "")
        qtype = question_data.get("type", "").lower()
        follow = question_data.get("follow_up") or DEFAULT_FOLLOW_UPS.get(qtype, "")
    answer = SESSION.record_answer(qtext, follow, client, llm_model, lang_hint=lang_hint)
    follow_up = SESSION.next_followup() or ""
    append_log(qtext, answer, log_path, session_id=session_id)
    if follow_up:
        append_log(follow_up, "", log_path, session_id=session_id)
    return answer, follow_up


# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------


def extract_summary(text: str) -> str:
    """Extract the summary part from a structured answer."""

    m = re.search(r"1\)\s*[^:]*:\s*(.*)", text)
    if m:
        return m.group(1).strip()
    return text.strip()


def export_audio_answer(
    text: str, out_path: Path, *, synth: Callable[[str, Path], None]
) -> Path:
    """Synthesize ``text`` to ``out_path`` using ``synth``."""

    synth(text, out_path)
    return out_path


def synthesize(text: str, out_path: Path, *, synth: Callable[[str, Path], None]) -> Path:
    """Compatibility wrapper delegating to :func:`export_audio_answer`."""

    return export_audio_answer(text, out_path, synth=synth)


def detect_language(text: str) -> str:
    """Detect the language of ``text`` returning ``"unknown"`` on failure."""

    try:
        return detect(text)
    except LangDetectException:
        return "unknown"


# ---------------------------------------------------------------------------
# Transcription helpers
# ---------------------------------------------------------------------------


async def transcribe(
    audio_path: Path,
    client: Any,
    model: str,
    *,
    lang_hint: str | None = None,
) -> str | None:
    """Transcribe audio handling errors with :func:`handle_error`."""

    try:
        result = client.transcribe(audio_path, model, lang_hint=lang_hint)
        if asyncio.iscoroutine(result):
            result = await result
        return result
    except Exception as exc:  # pragma: no cover - delegated to handler
        return handle_error(exc, context="transcribe")


async def fast_transcribe(
    audio_path: Path,
    client: Any,
    model: str,
    *,
    lang_hint: str | None = None,
) -> str | None:
    """Convenience wrapper returning only the transcription text."""

    return await transcribe(audio_path, client, model, lang_hint=lang_hint)

