from __future__ import annotations

from datetime import datetime
import json
import re
import uuid
import csv
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Tuple, Callable

import asyncio
import logging
import tempfile
import openai
from .openai_async import run
from .local_audio import tts_local, stt_local
from .cache import cache_get_json, cache_set_json
from .service_container import container

logger = logging.getLogger(__name__)


async def _maybe_await(result: Any) -> Any:
    if asyncio.iscoroutine(result):
        return await result
    return result


async def fast_transcribe_async(
    path_or_bytes,
    client,
    stt_model: str,
    lang_hint: str | None = None,
) -> str | None:
    """Perform a single transcription call with optional language hint."""

    if stt_model == "local":
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
            if isinstance(path_or_bytes, (str, Path)):
                p = Path(path_or_bytes)
            else:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                    tmp.write(path_or_bytes)
                    tmp_path = Path(tmp.name)
                p = tmp_path
            return stt_local(p, lang_hint or "it")

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

            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(path_or_bytes)
                tmp_path = Path(tmp.name)
            with open(tmp_path, "rb") as f:
                tx = client.audio.transcriptions.create(
                    model=stt_model, file=f, **kwargs
                )
        return (getattr(tx, "text", "") or "").strip()

    except (openai.OpenAIError, OSError, TimeoutError) as e:
        logger.error("Errore trascrizione: %s", e, exc_info=True)
        return None
    finally:
        if tmp_path:
            try:
                tmp_path.unlink()
            except OSError:
                logger.warning("Impossibile eliminare file temporaneo %s", tmp_path, exc_info=True)


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
) -> Tuple[str | None, str]:
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
            return stt_local(p, lang_hint or "it"), lang_hint or ""
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
    try:
        kwargs: Dict[str, Any] = {
            "model": stt_model,
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
        lang_code = ""
    if debug and lang_code:
        logger.info("🌐 Lingua rilevata: %s", lang_code.upper())
    cache_set_json(
        cache_key,
        {'text': text, 'lang': lang_code},
        ttl=container.settings.cache_ttl,
    )
    return text, lang_code


def transcribe(
    path_or_bytes: str | Path | bytes,
    client,
    stt_model: str,
    *,
    debug: bool = False,
    lang_hint: str | None = None,
) -> Tuple[str | None, str]:
    return run(
        transcribe_async,
        path_or_bytes,
        client,
        stt_model,
        debug=debug,
        lang_hint=lang_hint,
    )


async def oracle_answer_async(
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
) -> Tuple[str | None, List[Dict[str, Any]]]:

    payload = {
        'q': question,
        'lang': lang_hint,
        'context': context,
        'history': history,
        'topic': topic,
        'policy': policy_prompt,
        'mode': mode,
    }
    key_hash = hashlib.sha1(
        json.dumps(payload, sort_keys=True, default=str).encode('utf-8')
    ).hexdigest()

    """Return an answer generated by the LLM client.

    The function composes a conversation with optional context and history,
    augments it with simple policy instructions and attempts the request up to
    three times to handle transient API errors gracefully.

    When ``stream`` is ``True`` the request is performed in streaming mode and
    ``on_token`` is invoked for every text delta received. The returned string
    still contains the full final answer.
    """
    payload = {'q': question, 'lang': lang_hint, 'context': context, 'history': history, 'topic': topic, 'policy': policy_prompt, 'mode': mode}
    key_hash = hashlib.sha1(json.dumps(payload, sort_keys=True, default=str).encode('utf-8')).hexdigest()

    cache_key = f'oracle:{key_hash}'
    cached = cache_get_json(cache_key)
    if cached is not None:
        return cached, context or []
    logger.info("✨ Interrogo l’Oracolo…")
    lang_clause = "Answer in English." if lang_hint == "en" else "Rispondi in italiano."
    topic_clause = (
        " Rispondi solo con informazioni coerenti al topic corrente; non mescolare altri temi a meno che l'utente lo chieda esplicitamente. Topic: "
        + topic
        if topic
        else ""
    )
    mode_clause = (
        " Stile conciso: 2-4 frasi e termina con una domanda di follow-up."
        if mode == "concise"
        else " Struttura: 1) sintesi, 2) 2-3 dettagli puntuali, 3) fonti citate [1], [2], …"
    )
    grounding_clause = (
        "Answer ONLY using the passages; if they are insufficient, ask for clarifications."
        if lang_hint == "en"
        else "Rispondi SOLO usando i passaggi; se non sono sufficienti, chiedi chiarimenti."
    )
    policy = (
        (policy_prompt or "")
        + topic_clause
        + " "
        + grounding_clause
        + " "
        + lang_clause
        + mode_clause
    )
    messages: List[Dict[str, str]] = []
    if style_prompt:
        messages.append({"role": "system", "content": style_prompt})
    if context:
        ctx_txt = "\n".join(
            f"[{i+1}] {c.get('text','')}" for i, c in enumerate(context) if c.get("text")
        )
        if ctx_txt:
            messages.append({"role": "system", "content": f"Fonti:\n{ctx_txt}"})
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": question})
    for attempt in range(3):
        try:
            resp = await _maybe_await(
                client.responses.create(
                    model=llm_model,
                    instructions=policy,
                    input=messages,
                )
            )
            ans = resp.output_text.strip()
            cache_set_json(cache_key, ans, ttl=container.settings.cache_ttl)
            return ans, context or []
        except (openai.OpenAIError, TimeoutError) as e:
            logger.error("Errore OpenAI: %s", e, exc_info=True)
            if attempt < 2:
                await asyncio.sleep(0.5 * (2 ** attempt))
    return None, context or []



    def do_request():
        if not stream:
            return client.responses.create(
                model=llm_model,
                instructions=policy,
                input=messages,
            ).output_text

        chunks: list[str] = []
        try:
            with client.responses.with_streaming_response.create(
                model=llm_model,
                instructions=policy,
                input=messages,
            ) as resp:
                for event in resp:
                    if getattr(event, "type", "") == "response.output_text.delta":
                        delta = getattr(event, "delta", "")
                        if delta:
                            if on_token:
                                on_token(delta)
                            chunks.append(delta)
            return "".join(chunks)
        except AttributeError:
            # Fallback to legacy chat.completions streaming
            stream_resp = client.chat.completions.create(
                model=llm_model,
                messages=messages,
                stream=True,
            )
            for chunk in stream_resp:
                delta = (
                    chunk.choices[0].delta.get("content", "")
                    if getattr(chunk.choices[0], "delta", None)
                    else ""
                )
                if delta:
                    if on_token:
                        on_token(delta)
                    chunks.append(delta)
            return "".join(chunks)

    try:
        ans = retry_with_backoff(do_request).strip()
        cache_set_json(cache_key, ans, ttl=container.settings.cache_ttl)
        return ans, context or []
    except (openai.OpenAIError, TimeoutError) as e:
        logger.error("Errore OpenAI: %s", e, exc_info=True)
        return None, context or []


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
    policy_prompt: str = "",
    mode: str = "detailed",
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
    )



async def synthesize_async(
    text: str,
    out_path: Path,
    client,
    tts_model: str,
    tts_voice: str,

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
):
    """Asynchronous generator yielding partial tokens and the final answer."""

    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[tuple[str, bool]] = asyncio.Queue()

    def _on_token(tok: str) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, (tok, False))

    async def _runner() -> None:
        ans, _ = await run_async(
            oracle_answer,
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
            on_token=_on_token,
        )
        await queue.put((ans or "", True))

    task = asyncio.create_task(_runner())

    try:
        while True:
            chunk, is_final = await queue.get()
            yield chunk, is_final
            if is_final:
                break
    finally:
        await task


def synthesize(
    text: str, out_path: Path, client, tts_model: str, tts_voice: str

) -> Path | None:
    logger.info("🎧 Sintesi vocale…")
    if tts_model == "local":
        tts_local(text, out_path)
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


def synthesize(
    text: str,
    out_path: Path,
    client,
    tts_model: str,
    tts_voice: str,
) -> Path | None:
    return run(synthesize_async, text, out_path, client, tts_model, tts_voice)


def export_audio_answer(
    text: str,
    out_path: Path,
    *,
    synth: Any | None = None,
    client: Any | None = None,
    tts_model: str = "",
    tts_voice: str = "",
) -> Path | None:
    """Export ``text`` as an audio file to ``out_path``."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if synth is not None:
        synth(text, out_path)
        return out_path
    else:
        if client is None:
            raise ValueError("client required if synth not provided")
        return synthesize(text, out_path, client, tts_model, tts_voice)


def format_citations(sources: list[dict[str, str]]) -> str:
    """Return a comma-separated string of source identifiers."""
    return ", ".join(s.get("id", "") for s in sources if s.get("id"))


def append_log(
    q: str,
    a: str,
    log_path: Path,
    *,
    session_id: str | None = None,
    lang: str = "",
    topic: str | None = None,
    sources: list[dict[str, str]] | None = None,
) -> str:
    """Append a log entry and return the session identifier used."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if session_id is None:
        session_id = uuid.uuid4().hex
    record = {
        "timestamp": ts,
        "session_id": session_id,
        "lang": lang,
        "topic": topic or "",
        "question": q,
        "answer": a,
        "summary": extract_summary(a),
        "sources": sources or [],
    }
    if log_path.suffix.lower() == ".jsonl":
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    else:
        is_new = not log_path.exists()
        with log_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, quoting=csv.QUOTE_ALL)
            if is_new:
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
            src_str = ";".join(
                f"{s.get('id','')}:{s.get('score',0):.2f}" for s in record["sources"]
            )
            writer.writerow(
                [
                    record["timestamp"],
                    record["session_id"],
                    record["lang"],
                    record["topic"],
                    record["question"],
                    record["answer"],
                    src_str,
                ]
            )
    return session_id


def extract_summary(answer: str) -> str:
    """Extract the summary part from a structured oracle answer."""
    match = re.search(
        r"1\)\s*(?:Sintesi:)?\s*(.*?)(?:\n\s*2\)|$)", answer, flags=re.S | re.I
    )
    if match:
        return match.group(1).strip()
    match = re.search(r"Sintesi:\s*(.*)", answer, flags=re.S | re.I)
    if match:
        return match.group(1).strip()
    return answer.strip()
