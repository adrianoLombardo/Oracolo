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
    sid = session_id or str(int(datetime.utcnow().timestamp()))
    entry = {
        "timestamp": int(datetime.utcnow().timestamp()),
        "session_id": sid,
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
