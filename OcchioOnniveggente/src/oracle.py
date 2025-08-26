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
import inspect
import time
import random
from threading import Event
from pathlib import Path
from typing import Any, AsyncGenerator, Callable, Iterable, Iterator, List, Tuple

from langdetect import LangDetectException, detect

from .utils.error_handler import handle_error
from .retrieval import Question, load_questions


_QUESTIONS_CACHE: dict[str, List[dict[str, Any]]] | None = None
_QUESTIONS_MTIME: float | None = None


def get_questions() -> dict[str, List[dict[str, Any]]]:
    """Return the questions dataset reloading it when the file changes."""

    global _QUESTIONS_CACHE, _QUESTIONS_MTIME
    path = Path(__file__).resolve().parent.parent / "data" / "domande_oracolo.json"
    mtime = path.stat().st_mtime if path.exists() else None

    if _QUESTIONS_CACHE is None or mtime != _QUESTIONS_MTIME:
        _QUESTIONS_CACHE = load_questions(path)
        _QUESTIONS_MTIME = mtime

    return _QUESTIONS_CACHE or {}

QUESTIONS_BY_TYPE: dict[str, List[Question]] = load_questions()

# Track questions already asked for each category during the current session.
# Keys are category names (lowercase) and values are the indexes of questions
# that have been served.  Once all questions in a category have been used the
# set is cleared to start a new cycle.
_USED_QUESTIONS: dict[str, set[int]] = {}



# Risposte predefinite per domande fuori tema
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
    """Return an answer from ``client`` and the context used.

    The ``client`` argument is expected to mimic the OpenAI Python client's
    interface.  In the tests a small dummy object is provided.  When ``stream``
    is ``True`` streaming tokens are forwarded to ``on_token`` and concatenated
    to form the final answer.
    """

    # The ``topic`` argument is accepted for API compatibility with the real
    # project.  In this lightweight implementation it is currently unused but
    # allowing it avoids unexpected ``TypeError`` exceptions when higher level
    # components pass the parameter.

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
        # Fallback to synchronous implementation
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
    # Fallback to synchronous implementation
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
    """Stream answer tokens from the model.

    Yields ``(chunk, False)`` for each token and finally ``(full_text, True)``
    with the accumulated output.
    """

    # ``topic`` is accepted for interface compatibility.  It is not used by the
    # simplified streaming helper but allows callers to pass the argument
    # unconditionally.
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
                await asyncio.sleep(0)  # allow cooperative scheduling
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
    stop_event: "Event" | None = None,
    question_type: str | None = None,
    categoria: str | None = None,
) -> Iterator[str]:
    """Yield answer tokens from the model synchronously.

    The generator mirrors :func:`oracle_answer_stream` but operates in a
    synchronous context.  It forwards each token as soon as it is produced and
    can be interrupted either by setting ``stop_event`` or after ``timeout``
    seconds have elapsed.
    """

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


# ---------------------------------------------------------------------------
# Questions handling
# ---------------------------------------------------------------------------



def random_question(category: str) -> Question | None:
    """Return a random question object from the desired ``category``."""


    qs = get_questions().get(category.lower())

def random_question(category: str) -> dict[str, str] | None:
    """Return a random question from ``category`` without immediate repeats.

    Questions already returned are tracked per category.  Once all questions in
    a category have been used the tracking set is cleared, allowing the cycle to
    restart.
    """


    cat = category.lower()
    qs = QUESTIONS_BY_TYPE.get(cat)

    if not qs:
        return None

    used = _USED_QUESTIONS.setdefault(cat, set())
    if len(used) == len(qs):
        used.clear()

    available = [i for i in range(len(qs)) if i not in used]
    idx = random.choice(available)
    used.add(idx)
    return qs[idx]


def answer_with_followup(
    question_data: Question | dict[str, Any],
    client: Any,
    llm_model: str,
    *,
    lang_hint: str = "it",
) -> tuple[str, str]:
    """Generate an answer for ``question_data`` and return its follow-up."""
    if isinstance(question_data, dict):
        question = question_data.get("domanda", "")
        follow_up = question_data.get("follow_up") or ""
    else:
        question = question_data.domanda
        follow_up = question_data.follow_up or ""
    answer, _ = oracle_answer(question, lang_hint, client, llm_model, "")
    return answer, follow_up


def answer_and_log_followup(
    question_data: dict[str, str],
    client: Any,
    llm_model: str,
    log_path: Path,
    *,
    session_id: str,
    lang_hint: str = "it",
) -> tuple[str, str]:
    """Generate an answer and log the follow-up for the user.

    The question and its answer are written to ``log_path`` via
    :func:`append_log`.  If a ``follow_up`` field is present in
    ``question_data`` it is appended to the same log so that the caller can
    immediately propose it to the user.  The function returns both the
    ``answer`` and ``follow_up``.
    """

    answer, follow_up = answer_with_followup(
        question_data, client, llm_model, lang_hint=lang_hint
    )
    append_log(
        question_data.get("domanda", ""),
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
    """Synthesize ``text`` into ``out_path`` using a local TTS engine.

    The implementation favours ``pyttsx3`` for offline speech synthesis and
    falls back to ``gTTS`` when available. If no backend can generate audio an
    empty placeholder file is created so callers do not fail.
    """

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
    """Asynchronous wrapper around :func:`synthesize`."""

    synthesize(*args, **kwargs)




async def transcribe(
    audio_path: Path,
    client: Any,
    model: str,
    *,
    lang_hint: str | None = None,
) -> str | None:
    """Effettua una trascrizione gestendo gli errori in modo centralizzato.

    Il client passato deve esporre un metodo ``transcribe`` compatibile con le
    firme utilizzate nei test. Per ``AsyncOpenAI`` – che non implementa tale
    metodo – effettuiamo la chiamata diretta all'endpoint di trascrizione.
    In caso di eccezioni la funzione utilizza :func:`handle_error` per
    classificare l'errore e restituisce il messaggio destinato all'utente.
    """

    try:
        if hasattr(client, "transcribe"):
            result = client.transcribe(audio_path, model, lang_hint=lang_hint)
            if inspect.isawaitable(result):
                result = await result
            return result

        # Fallback per ``AsyncOpenAI`` che espone l'API moderna
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
    except Exception as exc:  # noqa: BLE001 - delegato a handle_error
        return handle_error(exc, context="transcribe")


async def fast_transcribe(
    audio_path: Path,
    client: Any,
    model: str,
    *,
    lang_hint: str | None = None,
) -> str | None:
    """Wrapper around :func:`transcribe` returning only the transcription text."""

    return await transcribe(audio_path, client, model, lang_hint=lang_hint)

