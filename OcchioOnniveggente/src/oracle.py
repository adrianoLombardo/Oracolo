from __future__ import annotations

"""Simplified oracle helpers used in the unit tests.

The real project contains a very feature rich implementation.  For the
purposes of the exercises we only provide a compact subset that offers the
behaviour exercised by the tests: question handling, logging utilities and a
minimal interface to a fake language model client.
"""

from dataclasses import dataclass
import csv
import json
import logging
import random
from datetime import datetime
from pathlib import Path
from threading import Event
from typing import Any, AsyncGenerator, Callable, ClassVar, Dict, Iterable, Iterator, List, Tuple

from langdetect import LangDetectException, detect  # type: ignore

from .retrieval import Question, load_questions, Context

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Question dataset utilities
# ---------------------------------------------------------------------------

# Load questions once at import time.  ``load_questions`` may return either a
# mapping ``{category: [Question, ...]}`` or ``{Context: {category: [...]}}``.
_loaded = load_questions()
if _loaded and isinstance(next(iter(_loaded.values())), list):  # type: ignore[arg-type]
    _QUESTIONS_BY_CONTEXT: Dict[Context, Dict[str, List[Question]]] = {
        Context.GENERIC: _loaded  # type: ignore[assignment]
    }
else:
    _QUESTIONS_BY_CONTEXT = _loaded  # type: ignore[assignment]


def get_questions(context: Context | str | None = None) -> Dict[str, List[Question]]:
    """Return questions grouped by category for ``context``."""

    if context is None:
        ctx = Context.GENERIC
    elif isinstance(context, Context):
        ctx = context
    else:
        ctx = Context.from_str(context)
    return _QUESTIONS_BY_CONTEXT.get(ctx, {})


# Track which questions have been served per category to avoid immediate
# repetitions.
_USED_QUESTIONS: Dict[str, set[int]] = {}


def random_question(category: str, context: Context | str | None = None) -> Question | None:
    """Return a random question from ``category`` without immediate repeats."""

    questions = get_questions(context).get(category, [])
    if not questions:
        return None

    used = _USED_QUESTIONS.setdefault(category, set())
    available = [i for i in range(len(questions)) if i not in used]
    if not available:
        used.clear()
        available = list(range(len(questions)))
    idx = random.choice(available)
    used.add(idx)
    return questions[idx]


# ---------------------------------------------------------------------------
# Off topic handling
# ---------------------------------------------------------------------------

OFF_TOPIC_RESPONSES: Dict[str, str] = {
    "poetica": "Preferirei non avventurarmi in slanci poetici.",
    "didattica": "Al momento non posso fornire spiegazioni didattiche.",
    "evocativa": "Queste domande evocative sfuggono al mio scopo.",
    "orientamento": "Non sono in grado di offrire indicazioni stradali.",
    "default": "Mi dispiace, non posso aiutarti con questa richiesta.",
}

OFF_TOPIC_REPLIES: Dict[str, str] = {
    "poetica": "Mi dispiace, ma preferisco non rispondere a richieste poetiche.",
    "didattica": "Questa domanda sembra didattica e non rientra nel mio ambito.",
    "evocativa": "Temo che il suo carattere evocativo mi impedisca di rispondere.",
    "orientamento": "Non posso fornire indicazioni di orientamento in questo contesto.",
}


def off_topic_reply(category: str | None) -> str:
    if not category:
        return "Mi dispiace, ma non posso rispondere a questa domanda."
    return OFF_TOPIC_REPLIES.get(
        category.lower(), "Mi dispiace, ma non posso rispondere a questa domanda."
    )


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def format_citations(sources: Iterable[dict[str, Any]]) -> str:
    """Return a comma separated string of the ``id`` fields in ``sources``."""

    return ", ".join(str(s["id"]) for s in sources if s.get("id"))


def extract_summary(text: str) -> str:
    """Extract the summary portion from structured ``text``."""

    for line in text.splitlines():
        if "Sintesi:" in line:
            return line.split("Sintesi:", 1)[1].strip()
    return text.strip()


def detect_language(text: str) -> str | None:
    """Best effort detection of the language used in ``text``."""

    try:
        return detect(text)
    except LangDetectException:
        return None


# ---------------------------------------------------------------------------
# Logging utilities
# ---------------------------------------------------------------------------

def append_log(
    question: str,
    answer: str,
    path: Path,
    *,
    session_id: str | None = None,
    lang: str | None = None,
    topic: str | None = None,
    sources: List[dict[str, Any]] | None = None,
) -> str:
    """Append an interaction entry to ``path`` (JSON lines or CSV)."""

    session_id = session_id or datetime.utcnow().strftime("%Y%m%d%H%M%S%f")
    entry = {
        "timestamp": datetime.utcnow().isoformat(),
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
                    entry["session_id"],
                    entry["lang"],
                    entry["topic"],
                    entry["question"],
                    entry["answer"],
                    json.dumps(entry["sources"], ensure_ascii=False),
                ]
            )
    else:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return session_id


# ---------------------------------------------------------------------------
# Language model interaction helpers
# ---------------------------------------------------------------------------

def _build_messages(
    question: str,
    context: list[dict[str, Any]] | None,
    history: list[dict[str, str]] | None,
) -> List[dict[str, str]]:
    msgs: List[dict[str, str]] = []
    if history:
        msgs.extend(history)
    if context:
        ctx_lines = [f"[{i+1}] {c.get('text', '')}" for i, c in enumerate(context)]
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
    parts: List[str] = []
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
    question_type: str | None = None,
    categoria: str | None = None,
    off_topic_category: str | None = None,
) -> Tuple[str, List[dict[str, Any]]]:
    """Return an answer from ``client`` and the context used."""

    if question_type == "off_topic":
        return off_topic_reply(categoria), []
    if off_topic_category:
        msg = OFF_TOPIC_RESPONSES.get(
            off_topic_category, OFF_TOPIC_RESPONSES["default"]
        )
        return msg, []

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
    mode: str | None = None,
    policy_prompt: str | None = None,
) -> AsyncGenerator[Tuple[str, bool], None]:
    """Async generator yielding response chunks and completion flag."""

    msgs = _build_messages(question, context, None)
    instr = _build_instructions(lang_hint, context, style_prompt, mode, policy_prompt)
    stream = client.responses.with_streaming_response.create(
        model=llm_model, instructions=instr, input=msgs
    )
    text = ""
    for evt in stream:
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
) -> Iterator[str]:
    """Return a generator producing chunks from the streaming response."""

    msgs = _build_messages(question, None, None)
    instr = _build_instructions(lang_hint, None, style_prompt, None, None)

    def _gen() -> Iterator[str]:
        stream = client.responses.with_streaming_response.create(
            model=llm_model, instructions=instr, input=msgs
        )
        for evt in stream:
            if stop_event and stop_event.is_set():
                break
            if getattr(evt, "type", "") == "response.output_text.delta":
                yield getattr(evt, "delta", "")

    return _gen()


# ---------------------------------------------------------------------------
# Follow up helpers
# ---------------------------------------------------------------------------

DEFAULT_FOLLOW_UPS: dict[str, str] = {
    "poetica": "Ti va di approfondire questa immagine?",
    "didattica": "Puoi fornire un esempio pratico?",
    "evocativa": "Che altre sensazioni emergono?",
    "orientamento": "Quale sarà il tuo prossimo passo concreto?",
}


def answer_and_log_followup(
    question_data: Question | dict[str, str],
    client: Any,
    llm_model: str,
    log_path: Path,
    *,
    session_id: str,
    lang_hint: str = "it",
) -> tuple[str, str]:
    """Generate an answer and log the follow-up for the user."""

    if isinstance(question_data, Question):
        qtext = question_data.domanda
        qtype = question_data.type
        follow = question_data.follow_up or DEFAULT_FOLLOW_UPS.get(qtype, "")
    else:
        qtext = question_data.get("domanda", "")
        qtype = question_data.get("type", "")
        follow = question_data.get("follow_up") or DEFAULT_FOLLOW_UPS.get(qtype, "")

    answer, _ = oracle_answer(qtext, lang_hint, client, llm_model, "")
    append_log(qtext, answer, log_path, session_id=session_id, lang=lang_hint)
    if follow:
        append_log(follow, "", log_path, session_id=session_id, lang=lang_hint)
    return answer, follow


def acknowledge_followup(user_reply: str, next_question: Question | None = None) -> str:
    """Return acknowledgement text or ``next_question`` if provided."""

    if next_question is not None:
        return next_question.domanda
    return "Grazie per la tua risposta."


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def export_audio_answer(
    text: str,
    out_path: Path,
    *,
    synth: Callable[[str, Path], None],
) -> Path:
    """Generate an audio file for ``text`` using ``synth`` and save it."""

    out_path.parent.mkdir(parents=True, exist_ok=True)
    synth(text, out_path)
    return out_path


def synthesize(text: str, *_, **__) -> bytes:
    """Placeholder text-to-speech function returning dummy bytes.

    The real project exposes a much richer audio synthesis pipeline.  The
    tests only require the function to exist so that other modules can import
    it.  The returned value is therefore an empty byte string.
    """

    return b""


# ---------------------------------------------------------------------------
# Transcription helpers
# ---------------------------------------------------------------------------

async def transcribe(audio_path: Path, client: Any, model: str) -> str:
    """Best effort transcription with coarse error handling."""

    try:
        result = client.transcribe(audio_path, model=model)
        if isinstance(result, dict) and "text" in result:
            return str(result["text"])
        return str(result)
    except ConnectionError:
        logger.warning(
            "Errore di rete, controlla la connessione (context: transcribe)"
        )
        return "Errore di rete, controlla la connessione"
    except ValueError:
        logger.error("Errore dell'API (context: transcribe)")
        return "Errore dell'API"
    except OSError:
        logger.error("Errore audio (context: transcribe)")
        return "Errore audio"


async def fast_transcribe(audio_path: Path, client: Any, model: str) -> str:
    """Alias of :func:`transcribe` maintained for backwards compatibility."""

    return await transcribe(audio_path, client, model)


# ---------------------------------------------------------------------------
# Conversation flow
# ---------------------------------------------------------------------------

@dataclass
class ConversationFlow:
    """Utility to progress through predefined conversation phases."""

    context: str | None = None
    flows: Dict[str, List[str]] | None = None

    DEFAULT_FLOW: ClassVar[List[str]] = [
        "introduzione",
        "presentazione_opera",
        "domanda_visitatore",
        "follow_up",
        "chiusura",
    ]

    def __post_init__(self) -> None:
        flows = self.flows or {}
        if self.context and self.context in flows:
            self.flow = flows[self.context]
        else:
            self.flow = list(self.DEFAULT_FLOW)
        self.index = 0

    @property
    def state(self) -> str:
        return self.flow[self.index]

    def advance(self) -> str:
        if not self.is_finished():
            self.index += 1
        return self.state

    def is_finished(self) -> bool:
        return self.index >= len(self.flow) - 1


__all__ = [
    "ConversationFlow",
    "DEFAULT_FOLLOW_UPS",
    "OFF_TOPIC_RESPONSES",
    "OFF_TOPIC_REPLIES",
    "_USED_QUESTIONS",
    "acknowledge_followup",
    "answer_and_log_followup",
    "append_log",
    "detect_language",
    "export_audio_answer",
    "synthesize",
    "extract_summary",
    "format_citations",
    "get_questions",
    "off_topic_reply",
    "oracle_answer",
    "oracle_answer_stream",
    "random_question",
    "stream_generate",
    "transcribe",
    "fast_transcribe",
]
