from __future__ import annotations

import asyncio
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterable, Callable

import numpy as np

try:  # pragma: no cover - optional torch
    import torch
except Exception:  # pragma: no cover
    class _CudaStub:
        @staticmethod
        def is_available() -> bool:
            return False

    class _TorchStub:
        cuda = _CudaStub()

    torch = _TorchStub()  # type: ignore

from src import metrics


@dataclass
class _TaskJob:
    func: Any
    args: tuple
    kwargs: dict
    submitted: float


class Orchestrator:
    """Simple orchestrator to balance STT (GPU) and TTS (CPU) tasks."""

    def __init__(self) -> None:
        self.gpu_available = torch.cuda.is_available()
        self._gpu_sem = asyncio.Semaphore(1 if self.gpu_available else 0)
        self._cpu_sem = asyncio.Semaphore(4)
        self.metrics: list[dict[str, Any]] = []
        self._metrics_task: asyncio.Task | None = None

    def _ensure_metrics_task(self) -> None:
        if self._metrics_task is None:
            self._metrics_task = asyncio.create_task(metrics.metrics_loop())

    async def run_stt(self, func, *args, **kwargs):
        self._ensure_metrics_task()
        job = _TaskJob(func, args, kwargs, time.time())
        device = metrics.resolve_device()
        sem = self._gpu_sem if device == "cuda" else self._cpu_sem
        async with sem:
            start = time.time()
            self.metrics.append(
                {
                    "task": "stt",
                    "queue_time": start - job.submitted,
                    "device": device,
                }
            )
            metrics.record_system_metrics()
            if "device" not in job.kwargs and "device" in getattr(getattr(job.func, "__code__", None), "co_varnames", []):
                job.kwargs["device"] = device
            return await asyncio.to_thread(job.func, *job.args, **job.kwargs)

    async def run_tts(self, coro_func, *args, **kwargs):
        self._ensure_metrics_task()
        job = _TaskJob(coro_func, args, kwargs, time.time())
        device = metrics.resolve_device()
        sem = self._gpu_sem if device == "cuda" else self._cpu_sem
        async with sem:
            start = time.time()
            self.metrics.append(
                {
                    "task": "tts",
                    "queue_time": start - job.submitted,
                    "device": device,
                }
            )
            metrics.record_system_metrics()
            if "device" not in job.kwargs and "device" in getattr(getattr(coro_func, "__code__", None), "co_varnames", []):
                job.kwargs["device"] = device
            return await coro_func(*job.args, **job.kwargs)


ORCH = Orchestrator()


def rms_level(pcm_bytes: bytes) -> float:
    if not pcm_bytes:
        return 0.0
    x = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(x * x)))


def write_wav(path: Path, sr: int, pcm: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm)


def off_topic_message(profile: str, keywords: list[str]) -> str:
    """Compose a helpful message when the question is off-topic."""
    tips = ", ".join(keywords[:3]) if keywords else ""
    if tips:
        return (
            f"La domanda non sembra pertinente rispetto al profilo «{profile}». "
            f"Puoi chiedermi, ad esempio, di {tips}."
        )
    return (
        f"La domanda non sembra pertinente rispetto al profilo «{profile}». "
        "Prova a riformularla o a specificare meglio il tema."
    )
