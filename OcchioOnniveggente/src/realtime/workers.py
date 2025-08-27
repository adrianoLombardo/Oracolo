from __future__ import annotations

"""Background task helpers for realtime processing.

The functions in this module push jobs onto the shared :mod:`realtime.queue`
and define the corresponding worker coroutines that consume those jobs and
publish results on the global :data:`~OcchioOnniveggente.src.event_bus.event_bus`.
"""

from pathlib import Path
from typing import Any

from ..conversation import ConversationManager
from ..event_bus import event_bus
from ..oracle.core import oracle_answer_async, synthesize_async, transcribe
from .queue import task_queue


def enqueue_transcription(
    audio_path: Path,
    *,
    client: Any | None = None,
    model: str | None = None,
    conv: ConversationManager | None = None,
) -> None:
    """Queue a transcription job for asynchronous processing."""

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
    """Queue a job to generate a reply for ``question``."""

    task_queue.publish(
        "generate_reply",
        question=question,
        lang_hint=lang_hint,
        client=client,
        model=model,
        conv=conv,
    )


def enqueue_synthesize_voice(text: str) -> None:
    """Queue a text-to-speech job."""

    task_queue.publish("synthesize_voice", text=text)


async def transcribe_worker() -> None:
    """Consume transcription jobs and emit ``transcript_ready`` events."""

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
    """Consume reply generation jobs and emit ``response_ready`` events."""

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
    """Consume text-to-speech jobs and emit ``tts_ready`` events."""

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

