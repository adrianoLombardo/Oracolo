from __future__ import annotations

"""Background task helpers for realtime processing."""

import asyncio
from pathlib import Path
from typing import Any

from ..conversation import ConversationManager
from ..event_bus import event_bus
from ..oracle.core import transcribe, oracle_answer_async, synthesize_async
from .queue import task_queue


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


__all__ = [
    "enqueue_transcription",
    "enqueue_generate_reply",
    "enqueue_synthesize_voice",
    "transcribe_worker",
    "generate_reply_worker",
    "synthesize_voice_worker",
]
