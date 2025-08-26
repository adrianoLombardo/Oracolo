

"""Utility helpers for the Oracle application.

This module provides small, self contained helpers used across the project
and in the unit tests.
"""


from __future__ import annotations

import asyncio
import csv
import inspect
import json
import random
import random
import time

from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from threading import Event

from typing import Any, AsyncGenerator, Callable, Dict, Iterable, Iterator, List, Tuple

import yaml
from langdetect import LangDetectException, detect

from .utils.error_handler import handle_error
from .retrieval import Question, load_questions, Context


# ---------------------------------------------------------------------------
# Question dataset handling
# ---------------------------------------------------------------------------


def _load_category_weights() -> Dict[str, float] | None:
    """Read optional category weights from settings files."""
    root = Path(__file__).resolve().parent.parent
    for name in ("settings.local.yaml", "settings.yaml"):
        cfg = root / name
        if not cfg.exists():
            continue
        try:
            data = yaml.safe_load(cfg.read_text(encoding="utf-8")) or {}
        except Exception:
            continue
        weights = data.get("question_weights")
        if not weights:
            weights = data.get("question_sequence", {}).get("weights")
        if weights:
            try:
                return {str(k).lower(): float(v) for k, v in weights.items()}
            except Exception:
                pass
    return None


CATEGORY_WEIGHTS: Dict[str, float] | None = _load_category_weights()


# Load questions once and expose convenience helpers
_loaded_questions = load_questions()
first_val = next(iter(_loaded_questions.values()))
if isinstance(first_val, list):  # categories mapping
    _QUESTIONS_BY_CONTEXT: Dict[Context, Dict[str, List[Question]]] = {
        Context.GENERIC: _loaded_questions  # type: ignore[assignment]
    }
else:
    _QUESTIONS_BY_CONTEXT = _loaded_questions  # type: ignore[assignment]


def get_questions(context: Context | str | None = None) -> Dict[str, List[Question]]:
    """Return questions grouped by category for ``context``."""
    if context is None:
        ctx = Context.GENERIC
    elif isinstance(context, Context):
        ctx = context
    else:
        ctx = Context.from_str(context)
    return _QUESTIONS_BY_CONTEXT.get(ctx, {})


QUESTIONS_BY_CONTEXT = _QUESTIONS_BY_CONTEXT
QUESTIONS_BY_TYPE: Dict[str, List[Question]] = get_questions()

# Track already served questions per context/category
_USED_QUESTIONS: Dict[str, set[int]] = {}

# Counter for :func:`log_interaction` if ever used
_INTERACTION_COUNTER = 0


from typing import Any, AsyncGenerator, Callable, Iterable, Iterator, List, Tuple, Dict

from langdetect import LangDetectException, detect

from .utils.error_handler import handle_error
from .retrieval import Question, load_questions

# ---------------------------------------------------------------------------
# Question dataset helpers
# ---------------------------------------------------------------------------

_QUESTIONS_CACHE: Dict[str, List[Question]] | None = None
_QUESTIONS_MTIME: float | None = None
_USED_QUESTIONS: Dict[str, set[int]] = {}


def get_questions() -> Dict[str, List[Question]]:
    """Return the questions dataset reloading it when the file changes."""

    global _QUESTIONS_CACHE, _QUESTIONS_MTIME
    data_path = (
        Path(__file__).resolve().parent.parent / "data" / "domande_oracolo.json"
    )
    mtime = data_path.stat().st_mtime if data_path.exists() else None
    if _QUESTIONS_CACHE is None or mtime != _QUESTIONS_MTIME:
        _QUESTIONS_CACHE = load_questions(data_path)
        _QUESTIONS_MTIME = mtime
    return _QUESTIONS_CACHE or {}


def random_question(category: str) -> Question | None:
    """Return a random question from ``category`` without immediate repeats."""

    cat = category.lower()
    qs = get_questions().get(cat)
    if not qs:
        return None
    used = _USED_QUESTIONS.setdefault(cat, set())
    if len(used) == len(qs):
        used.clear()
    available = [i for i in range(len(qs)) if i not in used]
    idx = random.choice(available)
    used.add(idx)
    return qs[idx]

# ---------------------------------------------------------------------------
# Conversation state machine
# ---------------------------------------------------------------------------


class ConversationFlow:
    """Simple state machine to model multi-phase dialogues."""

    DEFAULT_FLOW = ["introduzione", "domanda_principale", "follow_up", "chiusura"]

    def __init__(
        self,
        *,
        context: str | None = None,
        flows: dict[str, list[str]] | None = None,
    ) -> None:
        flows = flows or {}
        self._context = context
        self._flows = flows
        self._phases = list(flows.get(context, self.DEFAULT_FLOW))
        if not self._phases:
            raise ValueError("Flow must contain at least one phase")
        self._index = 0

    @property
    def state(self) -> str:
        return self._phases[self._index]

    def advance(self) -> str:


        """Advance to the next phase and return it."""


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
# Off-topic helpers
# ---------------------------------------------------------------------------

OFF_TOPIC_RESPONSES: dict[str, str] = {
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
    return OFF_TOPIC_REPLIES.get(
        category.lower(), "Mi dispiace, ma non posso rispondere a questa domanda."
    )



# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------



def format_citations(sources: Iterable[dict[str, Any]]) -> str:
    """Return a comma separated string of source ``id`` fields."""
    return ", ".join(str(s["id"]) for s in sources if s.get("id"))



def format_citations(sources: Iterable[dict[str, Any]]) -> str:
    """Return a comma separated string of the ``id`` fields in ``sources``."""


# ---------------------------------------------------------------------------
# Logging helpers
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
                    session_id,
                    lang or "",
                    topic or "",
                    question,
                    answer,
                    json.dumps(sources or [], ensure_ascii=False),
                ]
            )
    else:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return session_id


# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------


def export_audio_answer(
    text: str, out_path: Path, *, synth: Callable[[str, Path], None] | None = None
) -> None:
    """Create an audio file for ``text`` using ``synth``."""


def export_audio_answer(
    text: str,
    out_path: Path,
    *,
    synth: Callable[[str, Path], None] | None = None,
) -> None:
    """Write ``text`` to ``out_path`` using the provided ``synth`` callback."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if synth is None:
        out_path.write_bytes(b"")
    else:
        synth(text, out_path)


def extract_summary(answer: str) -> str:

    """Extract the summary section from ``answer`` when present."""

    """Extract the summary section from a structured answer."""


    for line in answer.splitlines():
        line = line.strip()
        if line.lower().startswith("1)") and ":" in line:
            return line.split(":", 1)[1].strip()
        if line.lower().startswith("sintesi:"):
            return line.split(":", 1)[1].strip()
    return answer.strip()


def detect_language(text: str) -> str | None:
    try:
        return detect(text)
    except LangDetectException:
        return None


# ---------------------------------------------------------------------------
# Core answer helpers
# ---------------------------------------------------------------------------


def _build_instructions(
    lang_hint: str,
    context: List[dict[str, Any]] | None,
    mode: str,
    tone: str,
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
    if tone == "formal":
        parts.append("Tono formale.")
    elif tone == "informal":
        parts.append("Tono informale.")
    return "\n".join(parts)


def _build_messages(
    question: str,
    context: List[dict[str, Any]] | None,
    history: List[dict[str, str]] | None,
) -> List[dict[str, str]]:
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
    tone: str = "informal",
    *,
    context: List[dict[str, Any]] | None = None,
    history: List[dict[str, str]] | None = None,
    policy_prompt: str = "",
    mode: str = "detailed",
    topic: str | None = None,
    stream: bool = False,
    on_token: Callable[[str], None] | None = None,

    question_type: str | None = None,
    categoria: str | None = None,
    off_topic_category: str | None = None,
) -> Tuple[str, List[dict[str, Any]]]:
    """Return an answer from ``client`` and the used context."""
    if question_type == "off_topic":
        return off_topic_reply(categoria), []
    if off_topic_category:
        msg = OFF_TOPIC_RESPONSES.get(
            off_topic_category, OFF_TOPIC_RESPONSES["default"]
        )
        return msg, []


    **_: Any,
) -> Tuple[str, List[dict[str, Any]] | None]:

    instructions = _build_instructions(lang_hint, context, mode, tone)
    messages = _build_messages(question, context, history)
    if stream:

        on_token = on_token or (lambda _t: None)
        response = client.responses.with_streaming_response.create(
            model=llm_model, instructions=instructions, input=messages
        )
        output_text = ""
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

        stream_obj = client.responses.with_streaming_response.create(
            model=llm_model, instructions=instructions, input=messages
        )
        chunks: List[str] = []
        for ev in stream_obj:
            if getattr(ev, "type", "") == "response.output_text.delta":
                delta = getattr(ev, "delta", "")
                if delta:
                    chunks.append(delta)
                    if on_token:
                        on_token(delta)
        return "".join(chunks), context
    resp = client.responses.create(
        model=llm_model, instructions=instructions, input=messages
    )
    return getattr(resp, "output_text", ""), context



async def oracle_answer_stream(
    question: str,
    lang_hint: str,
    client: Any,
    llm_model: str,
    style_prompt: str,

    *,
    context: List[dict[str, Any]] | None = None,
    history: List[dict[str, str]] | None = None,
    mode: str = "detailed",
    tone: str = "informal",
) -> AsyncGenerator[Tuple[str, bool], None]:
    """Asynchronous generator emitting streaming response chunks."""
    instructions = _build_instructions(lang_hint, context, mode, tone)
    messages = _build_messages(question, context, history)
    stream = client.responses.with_streaming_response.create(
        model=llm_model, instructions=instructions, input=messages
    )
    output_text = ""
    for event in stream:
        if getattr(event, "type", "") == "response.output_text.delta":
            delta = getattr(event, "delta", "")
            output_text += delta
            yield delta, False
    yield output_text, True

    **kwargs: Any,
) -> AsyncGenerator[Tuple[str, bool], None]:
    instructions = _build_instructions(
        lang_hint, kwargs.get("context"), kwargs.get("mode", "detailed"), kwargs.get("tone", "informal")
    )
    messages = _build_messages(question, kwargs.get("context"), kwargs.get("history"))
    stream_obj = client.responses.with_streaming_response.create(
        model=llm_model, instructions=instructions, input=messages
    )
    for ev in stream_obj:
        if getattr(ev, "type", "") == "response.output_text.delta":
            yield getattr(ev, "delta", ""), False
    yield getattr(stream_obj, "output_text", ""), True



def stream_generate(
    question: str,
    lang_hint: str,
    client: Any,
    llm_model: str,
    style_prompt: str,
    *,
    stop_event: Event | None = None,
) -> Iterator[str]:

    """Yield tokens from ``client`` as they arrive."""
    stop_event = stop_event or Event()
    instructions = _build_instructions(lang_hint, None, "detailed", "informal")
    messages = _build_messages(question, None, None)
    stream = client.responses.with_streaming_response.create(
        model=llm_model, instructions=instructions, input=messages
    )

    def _gen() -> Iterator[str]:
        for event in stream:
            if stop_event.is_set():
                break
            if getattr(event, "type", "") == "response.output_text.delta":
                yield getattr(event, "delta", "")

    return _gen()

    stream_obj = client.responses.with_streaming_response.create(
        model=llm_model,
        instructions=_build_instructions(lang_hint, None, "detailed", "informal"),
        input=_build_messages(question, None, None),
    )
    for ev in stream_obj:
        if stop_event and stop_event.is_set():
            break
        if getattr(ev, "type", "") == "response.output_text.delta":
            yield getattr(ev, "delta", "")


# Default follow-up messages per question category
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
) -> Tuple[str, str]:
    if isinstance(question_data, Question):
        qtext = question_data.domanda
        follow = question_data.follow_up or DEFAULT_FOLLOW_UPS.get(question_data.type, "")
    else:
        qtext = question_data.get("domanda", "")
        qtype = question_data.get("type", "")
        follow = question_data.get("follow_up") or DEFAULT_FOLLOW_UPS.get(qtype, "")
    answer, _ = oracle_answer(qtext, "it", client, llm_model, "")
    append_log(qtext, answer, log_path, session_id=session_id)
    append_log(follow, "", log_path, session_id=session_id)
    return answer, follow


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
    if path.suffix.lower() == ".csv":
        new = not path.exists()
        with path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, quoting=csv.QUOTE_ALL)
            if new:
                writer.writerow(
                    ["timestamp", "session_id", "lang", "topic", "question", "answer", "sources"]
                )
            writer.writerow(
                [
                    entry["timestamp"],
                    session_id,
                    lang or "",
                    topic or "",
                    question,
                    answer,
                    json.dumps(sources or [], ensure_ascii=False),
                ]
            )
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return session_id



# ---------------------------------------------------------------------------
# Transcription helpers
# ---------------------------------------------------------------------------


def synthesize(text: str, out_path: Path, client: Any | None = None, tts_model: str | None = None, tts_voice: str | None = None) -> None:

    """Synthesize ``text`` into ``out_path`` using ``client``."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import pyttsx3  # type: ignore

        engine = pyttsx3.init()
        if tts_voice:
            try:
                engine.setProperty("voice", tts_voice)
            except Exception:
                pass
        engine.save_to_file(text, out_path.as_posix())
        engine.runAndWait()
        return
    except Exception:
        pass

    try:
        from gtts import gTTS  # type: ignore

        gTTS(text=text, lang="it").save(out_path.as_posix())
        return
    except Exception:
        pass

    out_path.write_bytes(b"")


async def synthesize_async(*args, **kwargs):  # pragma: no cover - thin wrapper

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(b"")


async def synthesize_async(*args, **kwargs) -> None:  # pragma: no cover - thin wrapper

    synthesize(*args, **kwargs)


async def transcribe(
    audio_path: Path,
    client: Any,
    model: str,
    *,
    lang_hint: str | None = None,
) -> str | None:

    """Effettua una trascrizione gestendo gli errori in modo centralizzato."""
=
    try:
        if hasattr(client, "transcribe"):
            result = client.transcribe(audio_path, model, lang_hint=lang_hint)
            if inspect.isawaitable(result):
                result = await result
            return result

        with audio_path.open("rb") as f:
            params: dict[str, Any] = {"model": model, "file": f}
            if lang_hint:
                params["language"] = lang_hint
            create = client.audio.transcriptions.create
            response = create(**params)
            if inspect.isawaitable(response):
                response = await response
        return getattr(response, "text", None)

    except Exception as exc:  # noqa: BLE001 - delegated

    except Exception as exc:  # noqa: BLE001 - delegated to handle_error

        return handle_error(exc, context="transcribe")


async def fast_transcribe(
    audio_path: Path,
    client: Any,
    model: str,
    *,
    lang_hint: str | None = None,
) -> str | None:

    """Wrapper around :func:`transcribe` returning only the text."""
    return await transcribe(audio_path, client, model, lang_hint=lang_hint)


# ---------------------------------------------------------------------------
# Questions utilities
# ---------------------------------------------------------------------------


DEFAULT_FOLLOW_UPS: dict[str, str] = {
    "poetica": "Ti va di approfondire questa immagine?",
    "didattica": "Puoi fornire un esempio pratico?",
    "evocativa": "Che altre sensazioni emergono?",
    "orientamento": "Quale sarà il tuo prossimo passo concreto?",
}


def random_question(category: str, context: Context | str | None = None) -> Question | None:
    """Return a random question from ``category`` without immediate repeats."""
    cat = category.lower()
    qs = get_questions(context).get(cat)
    if not qs:
        return None
    if context is None:
        key = cat
    else:
        if isinstance(context, Context):
            ctx = context
        else:
            ctx = Context.from_str(context)
        key = (ctx, cat)
    used = _USED_QUESTIONS.setdefault(key, set())
    if len(used) == len(qs):
        used.clear()
    available = [i for i in range(len(qs)) if i not in used]
    idx = random.choice(available)
    used.add(idx)
    return qs[idx]


class QuestionSession:
    """Manage sequential question selection."""

    def __init__(
        self,
        *,
        context: Context = Context.GENERIC,
        weights: Dict[str, float] | None = None,
    ) -> None:
        self.context = context
        self.weights = {k.lower(): float(v) for k, v in (weights or CATEGORY_WEIGHTS or {}).items()}
        self._categories = list(get_questions(context).keys())
        if not self._categories:
            raise ValueError("No question categories available")
        self._rr_index = 0

    def next_question(self, category: str | None = None) -> Question | None:
        """Return the next question optionally forcing ``category``."""
        if category is None:
            cats = self._categories
            if self.weights:
                weights = [self.weights.get(c, 0.0) for c in cats]
                if any(weights) and sum(weights) > 0:
                    category = random.choices(cats, weights=weights, k=1)[0]
                else:
                    category = cats[self._rr_index % len(cats)]
                    self._rr_index += 1
            else:
                category = cats[self._rr_index % len(cats)]
                self._rr_index += 1
        return random_question(category, context=self.context)


# ---------------------------------------------------------------------------
# High level helpers combining answer and logging
# ---------------------------------------------------------------------------


def answer_and_log_followup(
    question_data: Question | dict[str, str],
    client: Any,
    llm_model: str,
    log_path: Path,
    *,
    session_id: str | None = None,
    lang: str = "it",
) -> Tuple[str, str]:
    """Ask ``question_data`` and log both question and follow-up."""
    if isinstance(question_data, Question):
        domanda = question_data.domanda
        qtype = question_data.type
        follow_up = question_data.follow_up or DEFAULT_FOLLOW_UPS.get(qtype.lower(), "")
    else:
        domanda = question_data.get("domanda", "")
        qtype = question_data.get("type", "")
        follow_up = question_data.get("follow_up") or DEFAULT_FOLLOW_UPS.get(qtype.lower(), "")
    answer, _ = oracle_answer(domanda, lang, client, llm_model, "")
    append_log(domanda, answer, log_path, session_id=session_id, lang=lang)
    append_log(follow_up, "", log_path, session_id=session_id, lang=lang)
    return answer, follow_up


    return await transcribe(audio_path, client, model, lang_hint=lang_hint)

