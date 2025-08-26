"""Utility helpers for the Oracle application.

This module provides small, self contained helpers that are used across the
project and in the unit tests.  The real project contains a much richer
implementation but the simplified version below focuses only on the features
exercised by the tests.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
import asyncio
import csv
import inspect
import json
import random
import time
from pathlib import Path
from threading import Event
from typing import Any, AsyncGenerator, Callable, Dict, Iterable, Iterator, List, Tuple

from langdetect import LangDetectException, detect

from .utils.error_handler import handle_error
from .retrieval import Context, Question, load_questions


# ---------------------------------------------------------------------------
# Questions handling
# ---------------------------------------------------------------------------

_QUESTIONS_CACHE: Dict[Context, Dict[str, List[Question]]] | None = None
_QUESTIONS_MTIME: float | None = None


def _load_all_questions() -> Dict[Context, Dict[str, List[Question]]]:
    """Load question datasets, reloading them when files change."""

    global _QUESTIONS_CACHE, _QUESTIONS_MTIME
    data_dir = Path(__file__).resolve().parent.parent / "data"
    json_files = list(data_dir.glob("*.json"))
    mtime = max((f.stat().st_mtime for f in json_files), default=None)
    if _QUESTIONS_CACHE is None or mtime != _QUESTIONS_MTIME:
        data = load_questions(data_dir)
        if isinstance(data, dict) and all(isinstance(v, list) for v in data.values()):
            _QUESTIONS_CACHE = {Context.GENERIC: data}  # type: ignore[arg-type]
        else:
            _QUESTIONS_CACHE = data  # type: ignore[assignment]
        _QUESTIONS_MTIME = mtime
    return _QUESTIONS_CACHE


def get_questions(context: Context | str | None = None) -> Dict[str, List[Question]]:
    """Return questions for ``context`` (defaults to generic)."""

    data = _load_all_questions()
    if context is None:
        ctx = Context.GENERIC
    elif isinstance(context, Context):
        ctx = context
    else:
        ctx = Context.from_str(context)
    return data.get(ctx, {})


QUESTIONS_BY_CONTEXT: Dict[Context, Dict[str, List[Question]]] = _load_all_questions()
QUESTIONS_BY_TYPE: Dict[str, List[Question]] = get_questions()

# Track questions already served to avoid immediate repetitions.  Keys are
# either a category name or the pair ``(context, category)`` when a specific
# context is used.
_USED_QUESTIONS: Dict[Any, set[int]] = {}

# Counter for user interactions logged via :func:`log_interaction`.
_INTERACTION_COUNTER = 0


# ---------------------------------------------------------------------------
# Conversation state machine
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
        flows: dict[str, list[str]] | None = None,
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
        """Return the name of the current phase."""

        return self._phases[self._index]

    def advance(self) -> str:
        """Advance to the next phase and return it.

        If already at the last phase, the state remains unchanged.
        """

        if self._index < len(self._phases) - 1:
            self._index += 1
        return self.state

    def is_finished(self) -> bool:
        """Return ``True`` when the flow reached its final phase."""

        return self._index >= len(self._phases) - 1

    def reset(self, *, context: str | None = None) -> None:
        """Reset to the first phase optionally switching ``context``."""

        if context is not None:
            self._context = context
            self._phases = list(self._flows.get(context, self.DEFAULT_FLOW))
            if not self._phases:
                raise ValueError("Flow must contain at least one phase")
        self._index = 0


# ---------------------------------------------------------------------------
# Questions utilities
# ---------------------------------------------------------------------------

# Default follow-up messages per question category.  Used when a question does
# not define its own ``follow_up`` field.
DEFAULT_FOLLOW_UPS: Dict[str, str] = {
    "poetica": "Ti va di approfondire questa immagine?",
    "didattica": "Puoi fornire un esempio pratico?",
    "evocativa": "Che altre sensazioni emergono?",
    "orientamento": "Quale sarà il tuo prossimo passo concreto?",
}


def random_question(category: str, context: Context | str | None = None) -> Question | None:
    """Return a random question from ``category`` avoiding repeats."""

    cat = category.lower()
    qs_map = get_questions(context)
    qs = qs_map.get(cat)
    if not qs:
        return None

    if context is None:
        key: Any = cat
    else:
        ctx = context if isinstance(context, Context) else Context.from_str(context)
        key = (ctx, cat)

    used = _USED_QUESTIONS.setdefault(key, set())
    if len(used) >= len(qs):
        used.clear()
    available = [i for i in range(len(qs)) if i not in used]
    idx = random.choice(available)
    used.add(idx)
    return qs[idx]


# Off-topic handling ---------------------------------------------------------

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
    """Return a polite refusal message for the given ``category``."""

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


def export_audio_answer(
    text: str, out_path: Path, *, synth: Callable[[str, Path], None] | None = None
) -> None:
    """Create an audio file for ``text`` using ``synth``."""

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if synth is None:
        out_path.write_bytes(b"")
    else:
        synth(text, out_path)


def extract_summary(answer: str) -> str:
    """Extract the summary section from a structured answer."""

    for line in answer.splitlines():
        line = line.strip()
        if line.lower().startswith("1)") and ":" in line:
            return line.split(":", 1)[1].strip()
        if line.lower().startswith("sintesi:"):
            return line.split(":", 1)[1].strip()
    return answer.strip()


def detect_language(text: str) -> str | None:
    """Detect the language of ``text`` using ``langdetect``."""

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
    """Return an answer from ``client`` and the context used."""

    if question_type == "off_topic":
        return off_topic_reply(categoria), []
    if off_topic_category:
        msg = OFF_TOPIC_RESPONSES.get(
            off_topic_category, OFF_TOPIC_RESPONSES["default"]
        )
        return msg, []

    instructions = _build_instructions(lang_hint, context, mode, tone)
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


async def oracle_answer_async(
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
    """Async variant of :func:`oracle_answer` supporting ``AsyncOpenAI``."""

    if question_type == "off_topic":
        return off_topic_reply(categoria), []
    if off_topic_category:
        msg = OFF_TOPIC_RESPONSES.get(
            off_topic_category, OFF_TOPIC_RESPONSES["default"]
        )
        return msg, []

    instructions = _build_instructions(lang_hint, context, mode, tone)
    messages = _build_messages(question, context, history)

    if stream:
        create_fn = client.responses.with_streaming_response.create
        if inspect.iscoroutinefunction(create_fn):
            response = await create_fn(
                model=llm_model, instructions=instructions, input=messages
            )
            output_text = ""
            on_token = on_token or (lambda _t: None)
            async with response as stream_resp:
                async for event in stream_resp:
                    if getattr(event, "type", "") == "response.output_text.delta":
                        delta = getattr(event, "delta", "")
                        output_text += delta
                        on_token(delta)
            return output_text, context or []
        return oracle_answer(
            question,
            lang_hint,
            client,
            llm_model,
            style_prompt,
            context=context,
            history=history,
            policy_prompt=policy_prompt,
            mode=mode,
            topic=topic,
            stream=True,
            on_token=on_token,
            question_type=question_type,
            categoria=categoria,
        )

    create_fn = client.responses.create
    if inspect.iscoroutinefunction(create_fn):
        resp = await create_fn(
            model=llm_model, instructions=instructions, input=messages
        )
        return resp.output_text, context or []
    return oracle_answer(
        question,
        lang_hint,
        client,
        llm_model,
        style_prompt,
        context=context,
        history=history,
        policy_prompt=policy_prompt,
        mode=mode,
        topic=topic,
        question_type=question_type,
        categoria=categoria,
    )


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
    topic: str | None = None,
    tone: str = "informal",
    question_type: str | None = None,
    categoria: str | None = None,
) -> AsyncGenerator[Tuple[str, bool], None]:
    """Stream answer tokens from the model."""

    if question_type == "off_topic":
        yield off_topic_reply(categoria), True
        return

    instructions = _build_instructions(lang_hint, context, mode, tone)
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
                await asyncio.sleep(0)
    yield output_text, True


def stream_generate(
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
    timeout: float | None = None,
    stop_event: Event | None = None,
    question_type: str | None = None,
    categoria: str | None = None,
) -> Iterator[str]:
    """Yield answer tokens from the model synchronously."""

    if question_type == "off_topic":
        yield off_topic_reply(categoria)
        return

    instructions = _build_instructions(lang_hint, context, mode, tone)
    messages = _build_messages(question, context, history)
    response = client.responses.with_streaming_response.create(
        model=llm_model, instructions=instructions, input=messages
    )

    start = time.monotonic()
    with response as stream_resp:
        for event in stream_resp:
            if stop_event is not None and stop_event.is_set():
                break
            if timeout is not None and (time.monotonic() - start) > timeout:
                break
            if getattr(event, "type", "") == "response.output_text.delta":
                delta = getattr(event, "delta", "")
                if delta:
                    yield delta


def answer_with_followup(
    question_data: Question | dict[str, Any] | str,
    client: Any,
    llm_model: str,
    *,
    lang_hint: str = "it",
) -> tuple[str, str]:
    """Generate an answer for ``question_data`` and return its follow-up."""

    if isinstance(question_data, Question):
        question = question_data.domanda
        follow_up = question_data.follow_up or DEFAULT_FOLLOW_UPS.get(
            question_data.type.lower(), ""
        )
    elif isinstance(question_data, dict):
        question = question_data.get("domanda", "")
        qtype = question_data.get("type", "").lower()
        follow_up = question_data.get("follow_up") or DEFAULT_FOLLOW_UPS.get(
            qtype, ""
        )
    else:
        question = str(question_data)
        follow_up = ""

    answer, _ = oracle_answer(question, lang_hint, client, llm_model, "")
    return answer, follow_up


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

    answer, follow_up = answer_with_followup(
        question_data, client, llm_model, lang_hint=lang_hint
    )

    if isinstance(question_data, Question):
        question_text = question_data.domanda
    else:
        question_text = question_data.get("domanda", "")

    append_log(
        question_text,
        answer,
        log_path,
        session_id=session_id,
        lang=lang_hint,
    )
    if follow_up:
        append_log(
            follow_up,
            "",
            log_path,
            session_id=session_id,
            lang=lang_hint,
        )
    return answer, follow_up


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
    """Append an interaction to ``path``."""

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
    else:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    return session_id


def log_interaction(
    *,
    context: Any | None = None,
    category: str | None = None,
    question: str | None = None,
    follow_up: str | None = None,
    user_response: str | None = None,
    path: Path | None = None,
    endpoint: str | None = None,
) -> int:
    """Log a user interaction and return its sequential counter."""

    global _INTERACTION_COUNTER
    _INTERACTION_COUNTER += 1
    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "interaction": _INTERACTION_COUNTER,
        "context": context,
        "category": category,
        "question": question,
        "follow_up": follow_up,
        "user_response": user_response,
    }

    data = json.dumps(entry, ensure_ascii=False)

    if endpoint:
        try:
            import urllib.request

            req = urllib.request.Request(
                endpoint,
                data=data.encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            urllib.request.urlopen(req, timeout=5)
        except Exception:
            pass

    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(data + "\n")

    return _INTERACTION_COUNTER


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
    """Synthesize ``text`` into ``out_path`` using a local TTS engine."""

    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:  # pragma: no cover - optional dependency
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

    try:  # pragma: no cover - optional dependency
        from gtts import gTTS  # type: ignore

        gTTS(text=text, lang="it").save(out_path.as_posix())
        return
    except Exception:
        pass

    out_path.write_bytes(b"")


async def synthesize_async(*args, **kwargs):  # pragma: no cover - thin wrapper
    """Asynchronous wrapper around :func:`synthesize`."""

    synthesize(*args, **kwargs)


async def transcribe(
    audio_path: Path,
    client: Any,
    model: str,
    *,
    lang_hint: str | None = None,
) -> str | None:
    """Effettua una trascrizione gestendo gli errori in modo centralizzato."""

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
            if inspect.iscoroutinefunction(create):
                response = await create(**params)
            else:
                response = create(**params)
            if inspect.isawaitable(response):
                response = await response
        return getattr(response, "text", None)
    except Exception as exc:  # noqa: BLE001 - delegated to handler
        return handle_error(exc, context="transcribe")


async def fast_transcribe(
    audio_path: Path,
    client: Any,
    model: str,
    *,
    lang_hint: str | None = None,
) -> str | None:
    """Wrapper around :func:`transcribe` returning only the transcription."""

    return await transcribe(audio_path, client, model, lang_hint=lang_hint)


