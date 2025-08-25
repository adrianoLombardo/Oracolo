from __future__ import annotations

import asyncio
import csv
import json

from datetime import datetime
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Tuple

from .openai_async import run, run_async


def format_citations(sources: List[Dict[str, Any]]) -> str:
    ids = [s.get("id", "") for s in sources if s.get("id")]
    return ", ".join(ids)


def export_audio_answer(text: str, out_path: Path, *, synth) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    synth(text, out_path)


def extract_summary(answer: str) -> str:
    for line in answer.splitlines():
        line = line.strip()
        if line.lower().startswith("1)") and ":" in line:
            return line.split(":", 1)[1].strip()
        if line.lower().startswith("sintesi:"):
            return line.split(":", 1)[1].strip()
    return answer.strip()

import uuid
from pathlib import Path
from typing import Any, AsyncGenerator, Callable, Iterable
from typing import Any, Dict, List, Tuple, Callable, AsyncGenerator

import asyncio
import logging
import io
from langdetect import detect
from pydub import AudioSegment
import tempfile
import base64
import json as _json
import openai
import websockets

from .openai_async import run_async, run
from .local_audio import tts_local, stt_local, stt_local_faster
from .local_llm import llm_local
from .utils import retry_with_backoff
from .cache import cache_get_json, cache_set_json
from .service_container import container
from .chat import ChatState



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

    device: str = "cpu",
) -> str | None:
    """Perform a single transcription call with optional language hint."""



    if container.settings.stt_backend != "openai":
        pass


    if stt_model == "local":
        container.load_stt_model()
        p = Path(path_or_bytes) if isinstance(path_or_bytes, (str, Path)) else Path("temp.wav")
        if not isinstance(path_or_bytes, (str, Path)):
            p.write_bytes(path_or_bytes)
        return stt_local(p, lang_hint or "it")
    kwargs: Dict[str, Any] = {}
    if lang_hint in ("it", "en"):
        kwargs["language"] = lang_hint

    tmp_path: Path | None = None

    try:
        if stt_model == "local":
            container.load_stt_model()
            if isinstance(path_or_bytes, (str, Path)):
                p = Path(path_or_bytes)
            else:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                    tmp.write(path_or_bytes)
                    tmp_path = Path(tmp.name)
                p = tmp_path

            return stt_local(
                p,
                lang=lang_hint or "it",
                device=container.settings.compute.stt.device,
            )

            return stt_local_faster(p, lang_hint or "it", device=device)


        kwargs: Dict[str, Any] = {}
        if lang_hint in ("it", "en"):
            kwargs["language"] = lang_hint

        if isinstance(path_or_bytes, (str, Path)):
            with open(path_or_bytes, "rb") as f:
                tx = await _maybe_await(
                    client.audio.transcriptions.create(
                        model=stt_model, file=f, **kwargs
                    )
                )
        else:

            tx = await _maybe_await(
                client.audio.transcriptions.create(
                    model=stt_model, file=path_or_bytes, **kwargs
                )
            )


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

    log_path: Path,

    path: Path,

    *,
    session_id: str | None = None,
    lang: str = "",
    topic: str = "",

    sources: List[Dict[str, Any]] | None = None,
) -> str:
    sid = session_id or str(int(datetime.utcnow().timestamp()))
    entry = {
        "timestamp": int(datetime.utcnow().timestamp()),
        "session_id": sid,

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

        "sources": sources or [],
    }
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if log_path.suffix == ".csv":
        write_header = not log_path.exists()
        fields = ["timestamp", "session_id", "lang", "topic", "question", "answer", "sources"]
        row = {k: entry[k] for k in fields}
        with log_path.open("a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields, quoting=csv.QUOTE_ALL)
            if write_header:
                writer.writeheader()
            writer.writerow(row)
    else:
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return sid


def transcribe(
    path_or_bytes, client, stt_model: str, *, debug: bool = False, lang_hint: str | None = None
) -> Tuple[str | None, str]:
    return "", ""


def fast_transcribe(
    path_or_bytes, client, stt_model: str, lang_hint: str | None = None
) -> str | None:
    return ""


def synthesize(
    text: str,
    out_path: Path,
    client: Any | None = None,
    tts_model: str | None = None,
    tts_voice: str | None = None,
) -> None:
    pass


def _build_messages(
    question: str,
    context: List[Dict[str, Any]] | None,
    history: List[Dict[str, str]] | None,
) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = []
    if history:
        messages.extend(history)
    if context:
        sources: List[str] = []
        for idx, c in enumerate(context, 1):
            txt = c.get("text", "")
            messages.append({"role": "system", "content": txt})
            sources.append(f"[{idx}] {txt}")
        if sources:
            messages.append({"role": "system", "content": "Fonti: " + ", ".join(sources)})
    messages.append({"role": "user", "content": question})
    return messages


async def oracle_answer_async(

        "sources": sources,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".jsonl":
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    """Detect conversation language from audio or text.

    Returns ``"it"`` for Italian, ``"en"`` for English or ``""`` if unknown.
    When ``state`` is provided the detected language is stored in
    ``state.language`` for future reuse.
    """

    snippet = ""
    if text:
        snippet = text
    elif path_or_bytes is not None and client is not None and stt_model:
        try:
            if isinstance(path_or_bytes, (str, Path)):
                audio = AudioSegment.from_file(path_or_bytes)
            else:
                audio = AudioSegment.from_file(io.BytesIO(path_or_bytes))
            audio = audio[:2000]
            buf = io.BytesIO()
            audio.export(buf, format="wav")
            buf.seek(0)
            snippet = fast_transcribe(buf, client, stt_model) or ""
        except Exception as e:
            logger.error("Errore rilevamento lingua: %s", e, exc_info=True)

    lang = ""
    if snippet:
        try:
            det = detect(snippet)
            if det.startswith("it"):
                lang = "it"
            elif det.startswith("en"):
                lang = "en"
        except Exception:
            lang = ""

    if state is not None and lang in ("it", "en"):
        state.language = lang
    return lang

def fast_transcribe(
    path_or_bytes,
    client,
    stt_model: str,
    lang_hint: str | None = None,
) -> str | None:
    return run(
        fast_transcribe_async,
        path_or_bytes,
        client,
        stt_model,
        lang_hint=lang_hint,
    )


async def transcribe_async(

    path_or_bytes: str | Path | bytes,
    client,
    stt_model: str,
    *,
    debug: bool = False,
    lang_hint: str | None = None,

    device: str = "cpu",


    backend: str | None = None,

    state: ChatState | None = None,


) -> Tuple[str | None, str]:

    """Trascrive un percorso o dei ``bytes`` e restituisce testo e lingua.

    ``lang_hint`` forza la lingua ("it" o "en") migliorando l'accuratezza
    della trascrizione quando la lingua di conversazione è nota.
    """


    backend = backend or container.settings.stt_backend
    if backend != "openai":

    if lang_hint not in ("it", "en") and state and state.language in ("it", "en"):
        lang_hint = state.language
    if lang_hint not in ("it", "en"):
        lang_hint = detect_language(
            path_or_bytes,
            client=client,
            stt_model=stt_model,
            state=state,
        )


    """Trascrive un percorso o dei ``bytes`` e restituisce testo e lingua."""

    if stt_model == "local":


        p = Path(path_or_bytes) if isinstance(path_or_bytes, (str, Path)) else Path("temp.wav")
        if not isinstance(path_or_bytes, (str, Path)):
            p.write_bytes(path_or_bytes)
        return stt_local(p, lang_hint or "it"), lang_hint or ""

        tmp_path: Path | None = None
        try:
            if isinstance(path_or_bytes, (str, Path)):
                p = Path(path_or_bytes)
            else:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                    tmp.write(path_or_bytes)
                    tmp_path = Path(tmp.name)
                p = tmp_path

            return (
                stt_local(
                    p,
                    lang=lang_hint or "it",
                    device=container.settings.compute.stt.device,
                ),
                lang_hint or "",
            )

            return stt_local_faster(p, lang_hint or "it", device=device), lang_hint or ""

        finally:
            if tmp_path:
                try:
                    tmp_path.unlink()
                except OSError:
                    logger.warning("Impossibile eliminare file temporaneo %s", tmp_path, exc_info=True)


    data_bytes: bytes
    try:
        if isinstance(path_or_bytes, (str, Path)):
            data_bytes = Path(path_or_bytes).read_bytes()
        else:
            data_bytes = path_or_bytes
    except (OSError, TypeError, ValueError) as e:
        logger.error("Errore lettura input audio: %s", e, exc_info=True)
        return None, ""
    key_hash = hashlib.sha1(data_bytes).hexdigest()
    cache_key = f"transcribe:{key_hash}:{lang_hint or ''}"
    cached = cache_get_json(cache_key)
    if cached:
        return cached.get('text', ''), cached.get('lang', '')


    model_to_use = stt_model
    if lang_hint in ("it", "en") and "{lang}" in stt_model:
        model_to_use = stt_model.replace("{lang}", lang_hint)


    try:
        kwargs: Dict[str, Any] = {
            "model": model_to_use,
            "response_format": "json",
        }
        if lang_hint in ("it", "en"):
            kwargs["language"] = lang_hint
        if isinstance(path_or_bytes, (str, Path)):
            with open(path_or_bytes, "rb") as f:
                kwargs["file"] = f
                tx = await _maybe_await(client.audio.transcriptions.create(**kwargs))
        else:
            kwargs["file"] = path_or_bytes
            tx = await _maybe_await(client.audio.transcriptions.create(**kwargs))
    except (openai.OpenAIError, TimeoutError, OSError) as e:
        logger.error("Errore OpenAI: %s", e, exc_info=True)
        return None, ""
    text = (getattr(tx, "text", "") or "").strip()
    lang = getattr(tx, "language", "") or ""
    if lang.startswith("it"):
        lang_code = "it"
    elif lang.startswith("en"):
        lang_code = "en"

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

    on_token: callable | None = None,
    llm_backend: str = "openai",
    llm_device: str = "cpu",
) -> Tuple[str | None, List[Dict[str, Any]]]:
    if llm_model == "local":
        from .service_container import container

        ans = await container.llm_batcher().generate(question)
        return ans, context or []

    instructions = (
        "Answer in English." if lang_hint == "en" else "Rispondi in italiano."
    )
    instructions += " Rispondi SOLO usando i passaggi; se non sono sufficienti, chiedi chiarimenti."
    if mode == "concise":
        instructions += " Stile conciso: 2-4 frasi e termina con una domanda di follow-up."
    else:
        instructions += " Struttura: 1) sintesi, 2) 2-3 dettagli puntuali, 3) fonti citate [1], [2], …"
    messages = _build_messages(question, context, history)

    if stream and hasattr(client.responses, "with_streaming_response"):
        with client.responses.with_streaming_response.create(
            model=llm_model, instructions=instructions, input=messages
        ) as resp_stream:
            for ev in resp_stream:
                if getattr(ev, "type", "") == "response.output_text.delta" and on_token:
                    on_token(getattr(ev, "delta", ""))
            return getattr(resp_stream, "output_text", None), context or []

    resp = client.responses.create(
        model=llm_model, instructions=instructions, input=messages
    )
    answer = getattr(resp, "output_text", None)
    if stream and answer and on_token:
        on_token(answer)
    return answer, context or []


def oracle_answer(

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

    if llm_model == "local":
        container.load_llm(llm_model, container.settings.compute.llm.device)
        ans = llm_local(
            question,
            device=container.settings.compute.llm.device,
            style_prompt=style_prompt,
            context=context,
            history=history,
            topic=topic,
            policy_prompt=policy_prompt,
            mode=mode,

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


            try:
                ans = local_llm.generate(
                    messages,
                    model_path=llm_model,
                    device=llm_device,
                    precision=container.settings.compute.llm.precision,
                )
            except Exception as e:  # pragma: no cover - runtime dependent
                logger.error("Errore LLM locale: %s", e, exc_info=True)
                ans = retry_with_backoff(do_request).strip()
        else:
            ans = retry_with_backoff(do_request).strip()
    except (openai.OpenAIError, TimeoutError) as e:
        logger.error("Errore OpenAI: %s", e, exc_info=True)
        return None, context or []


    cache_set_json(cache_key, ans, ttl=container.settings.cache_ttl)
    return ans, context or []

async def oracle_answer_async(


def oracle_answer(


async def oracle_answer_stream(

    question: str,
    lang_hint: str,
    client: Any,
    llm_model: str,
    style_prompt: str,
    *,

    context: List[Dict[str, Any]] | None = None,
    history: List[Dict[str, str]] | None = None,
    topic: str | None = None,
    policy_prompt: str = "",
    mode: str = "detailed",
    stream: bool = False,
    on_token: callable | None = None,
    llm_backend: str = "openai",
    llm_device: str = "cpu",
) -> Tuple[str | None, List[Dict[str, Any]]]:
    return run(
        oracle_answer_async,
        question,
        lang_hint,
        client,
        llm_model,
        style_prompt,
        context=context,
        history=history,
        topic=topic,
        policy_prompt=policy_prompt,
        mode=mode,
        stream=stream,
        on_token=on_token,
        llm_backend=llm_backend,
        llm_device=llm_device,
    )

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


async def oracle_answer_stream(
    question: str,
    lang_hint: str,
    client: Any,
    llm_model: str,
    style_prompt: str,
    *,
    context: List[Dict[str, Any]] | None = None,
    history: List[Dict[str, str]] | None = None,
    topic: str | None = None,
    policy_prompt: str = "",
    mode: str = "detailed",
    llm_backend: str = "openai",
    llm_device: str = "cpu",
) -> AsyncGenerator[Tuple[str, bool], None]:
    queue: asyncio.Queue[Tuple[str, bool]] = asyncio.Queue()

    def _on(tok: str) -> None:
        queue.put_nowait((tok, False))

    async def _runner() -> None:
        ans, _ = await oracle_answer_async(
            question,
            lang_hint,
            client,
            llm_model,
            style_prompt,
            context=context,
            history=history,
            topic=topic,
            policy_prompt=policy_prompt,
            mode=mode,
            stream=True,
            on_token=_on,
            llm_backend=llm_backend,
            llm_device=llm_device,
        )
        queue.put_nowait((ans or "", True))

    task = asyncio.create_task(_runner())
    try:
        while True:
            item = await queue.get()
            yield item
            if item[1]:
                break
    finally:
        await task


def fast_transcribe(*args, **kwargs):
    return ""


def synthesize(*args, **kwargs):
    return b""


async def transcribe_async(*args, **kwargs):
    return "", ""



async def fast_transcribe_async(*args, **kwargs):

) -> Path | None:
    logger.info("🎧 Sintesi vocale…")
    if tts_model == "local":
        container.load_tts_model()
        tts_local(
            text,
            out_path,
            device=container.settings.compute.tts.device,
        )
        logger.info("✅ Audio → %s", out_path.name)
        return out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    async def do_call() -> Path:
        try:
            async with client.audio.speech.with_streaming_response.create(
                model=tts_model, voice=tts_voice, input=text, response_format="wav"
            ) as resp:
                await resp.stream_to_file(out_path.as_posix())
            return out_path
        except TypeError:
            alt = out_path.with_suffix(".mp3") if out_path.suffix.lower() != ".mp3" else out_path
            async with client.audio.speech.with_streaming_response.create(
                model=tts_model, voice=tts_voice, input=text
            ) as resp:
                await resp.stream_to_file(alt.as_posix())
            return alt
    for attempt in range(3):
        try:
            final_path = await do_call()
            logger.info("✅ Audio → %s", final_path.name)
            return final_path
        except (openai.OpenAIError, TimeoutError, OSError) as e:
            logger.error("Errore OpenAI: %s", e, exc_info=True)
            if attempt < 2:
                await asyncio.sleep(0.5 * (2 ** attempt))
    logger.error("❌ Impossibile sintetizzare l'audio.")

    return None

