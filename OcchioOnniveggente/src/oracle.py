from __future__ import annotations

"""Simplified oracle helpers used in the unit tests.

The real project contains a very feature rich implementation.  For the
purposes of the exercises we only provide a compact subset that offers the
behaviour exercised by the tests: question handling, logging utilities and a
minimal interface to a fake language model client.
"""

from dataclasses import dataclass
import asyncio
import csv
import json
import logging
import os
import random
import hashlib
from datetime import datetime
from pathlib import Path
from threading import Event
from typing import Any, AsyncGenerator, Callable, ClassVar, Dict, Iterable, Iterator, List, Tuple
import requests

from langdetect import LangDetectException, detect  # type: ignore

from .conversation import ConversationManager, ChatState
from .retrieval import Question, load_questions, Context
from .event_bus import event_bus
from .utils.container import get_container
from .task_queue import task_queue
from .cache import cache_get_json, cache_set_json
from .rate_limiter import rate_limiter
from .exceptions import RateLimitExceeded, ExternalServiceError
from .utils import retry_with_backoff
from .metrics import pipeline_timer

try:  # pragma: no cover - optional dependency
    from tenacity import retry, stop_after_attempt, wait_exponential
except Exception:  # pragma: no cover - tenacity not available
    retry = None  # type: ignore


logger = logging.getLogger(__name__)
API_URL = os.getenv("ORACOLO_API_URL")
container = get_container()

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


@pipeline_timer("oracle_answer")
def oracle_answer(
    question: str,
    lang_hint: str,
    client: Any | None = None,
    llm_model: str | None = None,
    style_prompt: str = "",
    *,
    context: list[dict[str, Any]] | None = None,
    conv: ConversationManager | None = None,
    mode: str | None = None,
    policy_prompt: str | None = None,
    stream: bool = False,
    on_token: Callable[[str], None] | None = None,
    question_type: str | None = None,
    categoria: str | None = None,
    off_topic_category: str | None = None,
) -> Tuple[str, List[dict[str, Any]]]:
    """Return an answer from ``client`` and the context used."""
    # Check cache first
    cache_key = "oracle:" + hashlib.sha256(question.encode("utf-8")).hexdigest()
    cached = cache_get_json(cache_key)
    if cached:
        return cached.get("answer", ""), cached.get("context", [])

    # Enforce rate limiting
    rate_limiter.hit("oracle_answer")

    client = client or container.llm_client()
    llm_model = llm_model or container.settings.openai.llm_model

    if question_type == "off_topic":
        ans = off_topic_reply(categoria)
        event_bus.publish("response_ready", ans)
        cache_set_json(cache_key, {"answer": ans, "context": []})
        return ans, []
    if off_topic_category:
        msg = OFF_TOPIC_RESPONSES.get(
            off_topic_category, OFF_TOPIC_RESPONSES["default"]
        )
        event_bus.publish("response_ready", msg)
        cache_set_json(cache_key, {"answer": msg, "context": []})
        return msg, []

    if API_URL:
        resp = requests.post(
            f"{API_URL}/chat",
            json={"message": question},
            timeout=30,
        )
        resp.raise_for_status()
        ans = resp.json().get("response", "")
        event_bus.publish("response_ready", ans)
        if conv:
            conv.push_user(question)
            conv.push_assistant(ans)
        cache_set_json(cache_key, {"answer": ans, "context": context or []})
        return ans, context or []

    history = conv.messages_for_llm() if conv else None
    msgs = _build_messages(question, context, history)
    if conv:
        conv.push_user(question)
    instr = _build_instructions(lang_hint, context, style_prompt, mode, policy_prompt)
    chat_state = conv.chat if conv else None

    def _call_openai(**kwargs: Any) -> Any:
        return client.responses.create(**kwargs)

    def _call_stream(**kwargs: Any) -> Any:
        return client.responses.with_streaming_response.create(**kwargs)

    # Apply retry strategy
    if retry is not None:
        retryer = retry(reraise=True, stop=stop_after_attempt(3), wait=wait_exponential(1, 3))
        openai_call = retryer(_call_openai)
        stream_call = retryer(_call_stream)
    else:  # pragma: no cover - fallback
        openai_call = lambda **kw: retry_with_backoff(_call_openai, retries=3, base_delay=1, **kw)
        stream_call = lambda **kw: retry_with_backoff(_call_stream, retries=3, base_delay=1, **kw)

    if stream and hasattr(client.responses, "with_streaming_response"):
        text = ""
        stream_obj = stream_call(model=llm_model, instructions=instr, input=msgs)
        for evt in stream_obj:
            if getattr(evt, "type", "") == "response.output_text.delta":
                delta = getattr(evt, "delta", "")
                text += delta
                if on_token:
                    on_token(delta)

        if chat_state:
            chat_state.push_assistant(text)

        event_bus.publish("response_ready", text)

        if conv:
            conv.push_assistant(text)

        cache_set_json(cache_key, {"answer": text, "context": context or []})
        return text, context or []

    try:
        resp = openai_call(model=llm_model, instructions=instr, input=msgs)
    except Exception as exc:  # pragma: no cover - defensive
        raise ExternalServiceError(str(exc)) from exc
    ans = getattr(resp, "output_text", "")

    if chat_state:
        chat_state.push_assistant(ans)

    event_bus.publish("response_ready", ans)

    if conv:
        conv.push_assistant(ans)

    cache_set_json(cache_key, {"answer": ans, "context": context or []})
    return ans, context or []


@pipeline_timer("oracle_answer_async")
async def oracle_answer_async(
    question: str,
    lang_hint: str,
    client: Any | None = None,
    llm_model: str | None = None,
    style_prompt: str = "",
    *,
    context: list[dict[str, Any]] | None = None,
    conv: ConversationManager | None = None,
    mode: str | None = None,
    policy_prompt: str | None = None,
    stream: bool = False,
    on_token: Callable[[str], None] | None = None,
    question_type: str | None = None,
    categoria: str | None = None,
    off_topic_category: str | None = None,
) -> Tuple[str, List[dict[str, Any]]]:
    """Async wrapper around :func:`oracle_answer` using ``asyncio.to_thread``."""
    client = client or container.llm_client()
    llm_model = llm_model or container.settings.openai.llm_model

    return await asyncio.to_thread(
        oracle_answer,
        question,
        lang_hint,
        client,
        llm_model,
        style_prompt,
        context=context,
        conv=conv,
        mode=mode,
        policy_prompt=policy_prompt,
        stream=stream,
        on_token=on_token,
        question_type=question_type,
        categoria=categoria,
        off_topic_category=off_topic_category,
    )


@pipeline_timer("oracle_answer_stream")
async def oracle_answer_stream(
    question: str,
    lang_hint: str,
    client: Any | None = None,
    llm_model: str | None = None,
    style_prompt: str = "",
    *,
    context: list[dict[str, Any]] | None = None,
    mode: str | None = None,
    policy_prompt: str | None = None,
    conv: ConversationManager | None = None,
) -> AsyncGenerator[Tuple[str, bool], None]:
    """Async generator yielding response chunks and completion flag."""
    if API_URL:
        resp = requests.post(
            f"{API_URL}/chat",
            json={"message": question},
            timeout=30,
        )
        resp.raise_for_status()
        text = resp.json().get("response", "")
        event_bus.publish("response_ready", text)
        if conv:
            conv.push_user(question)
            conv.push_assistant(text)
        yield text, True
        return

    client = client or container.llm_client()
    llm_model = llm_model or container.settings.openai.llm_model

    history = conv.messages_for_llm() if conv else None
    msgs = _build_messages(question, context, history)
    if conv:
        conv.push_user(question)
    instr = _build_instructions(lang_hint, context, style_prompt, mode, policy_prompt)
    stream = client.responses.with_streaming_response.create(
        model=llm_model, instructions=instr, input=msgs
    )
    chat_state = conv.chat if conv else None
    text = ""
    for evt in stream:
        if getattr(evt, "type", "") == "response.output_text.delta":
            delta = getattr(evt, "delta", "")
            text += delta
            yield delta, False


    if chat_state:
        chat_state.push_assistant(text)

    event_bus.publish("response_ready", text)

    if conv:
        conv.push_assistant(text)

    yield text, True


def stream_generate(
    question: str,
    lang_hint: str,
    client: Any | None = None,
    llm_model: str | None = None,
    style_prompt: str = "",
    *,
    stop_event: Event | None = None,
    conv: ConversationManager | None = None,
) -> Iterator[str]:
    """Return a generator producing chunks from the streaming response."""
    client = client or container.llm_client()
    llm_model = llm_model or container.settings.openai.llm_model

    history = conv.messages_for_llm() if conv else None
    msgs = _build_messages(question, None, history)
    if conv:
        conv.push_user(question)
    instr = _build_instructions(lang_hint, None, style_prompt, None, None)

    def _gen() -> Iterator[str]:
        stream = client.responses.with_streaming_response.create(
            model=llm_model, instructions=instr, input=msgs
        )
        text = ""
        for evt in stream:
            if stop_event and stop_event.is_set():
                break
            if getattr(evt, "type", "") == "response.output_text.delta":
                delta = getattr(evt, "delta", "")
                text += delta
                yield delta


        chat_state = conv.chat if conv else None
        if chat_state:
            chat_state.push_assistant(text)

        event_bus.publish("response_ready", text)

        if conv:
            conv.push_assistant(text)


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
    """Synthesize ``text`` using the configured TTS provider."""

    tts = container.text_to_speech()
    return tts.synthesize(text)


async def synthesize_async(
    text: str,
    path: Path,
    client: Any | None = None,
    model: str = "",
    voice: str = "",
    **kwargs: Any,
) -> bytes:
    """Async wrapper around :func:`synthesize` using ``asyncio.to_thread``."""

    tts = container.text_to_speech()
    return await asyncio.to_thread(tts.synthesize, text)


# ---------------------------------------------------------------------------
# Transcription helpers
# ---------------------------------------------------------------------------

async def transcribe(
    audio_path: Path,
    client: Any | None = None,
    model: str | None = None,
    *,

    chat: ChatState | None = None,

    conv: ConversationManager | None = None,
) -> str:
    """Best effort transcription with coarse error handling."""

    if client is None:
        stt = container.speech_to_text()
        text = await asyncio.to_thread(stt.transcribe, audio_path)

        if chat:
            chat.push_user(text)

        if conv:
            conv.push_user(text)
        return text


    # ``AsyncOpenAI`` removed the ``client.transcribe`` shortcut in favour of
    # the ``audio.transcriptions.create`` endpoint. The helper now supports both

    # ``AsyncOpenAI`` removed the ``client.transcribe`` shortcut in favour of the
    # ``audio.transcriptions.create`` endpoint.  The helper now supports both

    # calling conventions for compatibility with older clients used in the tests.
    # When ``conv`` is provided the transcribed text is appended to it as a user
    # message.


    try:
        result: Any
        audio_api = getattr(client, "audio", None)
        if audio_api and hasattr(audio_api, "transcriptions"):
            create = getattr(audio_api.transcriptions, "create", None)
            if create is not None:
                with audio_path.open("rb") as fp:
                    result = await create(model=model, file=fp)
            else:  # pragma: no cover - unexpected API shape
                raise AttributeError("transcriptions.create missing")
        else:
            call = getattr(client, "transcribe")
            result = call(audio_path, model=model)
            if asyncio.iscoroutine(result):
                result = await result

        if isinstance(result, dict) and "text" in result:
            text = str(result["text"])
        elif hasattr(result, "text"):
            text = str(getattr(result, "text"))
        else:
            text = str(result)
        if conv:
            conv.push_user(text)
        return text
    except ConnectionError:
        msg = "Errore di rete, controlla la connessione"
        logger.warning("Errore di rete, controlla la connessione (context: transcribe)")
        if conv:
            conv.push_user(msg)
        return msg
    except ValueError:
        msg = "Errore dell'API"
        logger.error("Errore dell'API (context: transcribe)")
        if conv:
            conv.push_user(msg)
        return msg
    except OSError:
        msg = "Errore audio"
        logger.error("Errore audio (context: transcribe)")
        if conv:
            conv.push_user(msg)
        return msg


async def fast_transcribe(
    audio_path: Path,
    client: Any | None = None,
    model: str | None = None,
    *,
    conv: ConversationManager | None = None,
) -> str:
    """Alias of :func:`transcribe` maintained for backwards compatibility."""

    return await transcribe(audio_path, client, model, conv=conv)


# ---------------------------------------------------------------------------
# Task queue helpers
# ---------------------------------------------------------------------------

def enqueue_transcription(
    audio_path: Path,
    *,
    client: Any | None = None,
    model: str | None = None,
    conv: ConversationManager | None = None,
) -> None:
    """Publish a transcription job to the background queue."""

    task_queue.publish(
        "transcribe",
        audio_path=audio_path,
        client=client,
        model=model,
        conv=conv,
    )


def enqueue_generate_reply(
    question: str,
    lang_hint: str | None,
    client: Any | None,
    model: str,
    *,
    conv: ConversationManager | None = None,
) -> None:
    """Publish a reply generation job to the queue."""

    task_queue.publish(
        "generate_reply",
        question=question,
        lang_hint=lang_hint,
        client=client,
        model=model,
        conv=conv,
    )


def enqueue_synthesize_voice(text: str) -> None:
    """Publish a voice synthesis job to the queue."""

    task_queue.publish("synthesize_voice", text=text)


async def transcribe_worker() -> None:
    """Background worker processing transcription jobs."""

    async def _handle(
        *,
        audio_path: Path,
        client: Any | None = None,
        model: str | None = None,
        conv: ConversationManager | None = None,
    ) -> None:
        text = await transcribe(audio_path, client, model, conv=conv)
        event_bus.publish("transcript_ready", text)

    await task_queue.worker("transcribe", _handle)


async def generate_reply_worker() -> None:
    """Background worker generating replies from queued jobs."""

    async def _handle(
        *,
        question: str,
        lang_hint: str | None,
        client: Any | None = None,
        model: str | None = None,
        conv: ConversationManager | None = None,
    ) -> None:
        answer, _ = await oracle_answer_async(
            question, lang_hint, client, model, conv=conv
        )
        event_bus.publish("response_ready", answer)

    await task_queue.worker("generate_reply", _handle)


async def synthesize_voice_worker() -> None:
    """Background worker turning text into audio."""

    async def _handle(*, text: str) -> None:
        audio = await synthesize_async(text)
        event_bus.publish("tts_ready", audio)

    await task_queue.worker("synthesize_voice", _handle)


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



@dataclass
class QuestionSession:
    """Serve questions cycling through categories.

    When ``weights`` is provided, categories with a weight of ``0`` are ignored
    and the remaining ones are sampled according to the specified weights.  When
    weights are not supplied, categories are served in round-robin order.
    """

    weights: Dict[str, float] | None = None
    rng: random.Random | None = None

    def __post_init__(self) -> None:
        self._categories = list(get_questions().keys())
        self._index = 0
        self._rng = self.rng or random

    def next_question(self) -> Question:
        if self.weights:
            cats = [c for c in self._categories if self.weights.get(c, 0) > 0]
            weights = [self.weights.get(c, 0) for c in cats]
            if not cats:
                return Question(domanda="", type="")
            cat = self._rng.choices(cats, weights=weights, k=1)[0]
        else:
            cat = self._categories[self._index]
            self._index = (self._index + 1) % len(self._categories)
        q = random_question(cat)
        return q if q is not None else Question(domanda="", type=cat)



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
    "synthesize_async",
    "extract_summary",
    "format_citations",
    "get_questions",
    "off_topic_reply",
    "oracle_answer",
    "oracle_answer_async",
    "oracle_answer_stream",
    "random_question",
    "stream_generate",
    "transcribe",
    "fast_transcribe",
    "enqueue_transcription",
    "enqueue_generate_reply",
    "enqueue_synthesize_voice",
    "transcribe_worker",
    "generate_reply_worker",
    "synthesize_voice_worker",
]
