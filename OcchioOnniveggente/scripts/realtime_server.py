from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any
from dataclasses import dataclass
from typing import Any, AsyncIterable, Callable

import numpy as np
import websockets
import yaml
from dotenv import load_dotenv
try:
    import torch
except Exception:  # pragma: no cover - fallback when torch isn't installed
    class _CudaStub:
        @staticmethod
        def is_available() -> bool:
            return False

        @staticmethod
        def device_count() -> int:
            return 0

    class _TorchStub:
        cuda = _CudaStub()

    torch = _TorchStub()  # type: ignore

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.chat import ChatState
from src.config import Settings, get_openai_api_key
from src.domain import validate_question
from src.hotword import strip_hotword_prefix
from src.oracle import oracle_answer, transcribe
from src.audio import AudioPreprocessor
from src.profile_utils import get_active_profile, make_domain_settings

import wave


@dataclass
class _TaskJob:
    func: Any
    args: tuple
    kwargs: dict
    submitted: float


class Orchestrator:
    """Simple orchestrator to balance STT (GPU) and TTS (CPU) tasks."""

    def __init__(self) -> None:
        settings = Settings.model_validate_yaml(ROOT / "settings.yaml")
        self.gpu_available = torch.cuda.is_available()
        self.n_gpu = torch.cuda.device_count() if self.gpu_available else 0
        conc = max(1, int(getattr(settings.compute, "device_concurrency", 1)))
        self._gpu_sems = [asyncio.Semaphore(conc) for _ in range(self.n_gpu)]
        self._next_gpu = 0
        self._cpu_sem = asyncio.Semaphore(4)
        self.metrics: list[dict[str, Any]] = []

    async def run_stt(self, func, *args, **kwargs):
        job = _TaskJob(func, args, kwargs, time.time())
        if self.gpu_available and self._gpu_sems:
            start_idx = self._next_gpu
            gpu_id = start_idx
            for i in range(self.n_gpu):
                idx = (start_idx + i) % self.n_gpu
                if not self._gpu_sems[idx].locked():
                    gpu_id = idx
                    break
            self._next_gpu = (gpu_id + 1) % self.n_gpu
            sem = self._gpu_sems[gpu_id]
        else:
            sem = self._cpu_sem
            gpu_id = None

        async with sem:
            start = time.time()
            metric = {
                "task": "stt",
                "queue_time": start - job.submitted,
                "device": "cuda" if gpu_id is not None else "cpu",
            }
            if gpu_id is not None:
                metric["gpu_id"] = gpu_id
            self.metrics.append(metric)

            def _run() -> Any:
                if gpu_id is not None:
                    prev = os.environ.get("CUDA_VISIBLE_DEVICES")
                    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
                    try:
                        torch.cuda.set_device(gpu_id)
                    except Exception:
                        pass
                    try:
                        return job.func(*job.args, **job.kwargs)
                    finally:
                        if prev is None:
                            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
                        else:
                            os.environ["CUDA_VISIBLE_DEVICES"] = prev
                return job.func(*job.args, **job.kwargs)

            return await asyncio.to_thread(_run)

    async def run_tts(self, coro_func, *args, **kwargs):
        job = _TaskJob(coro_func, args, kwargs, time.time())
        async with self._cpu_sem:
            start = time.time()
            self.metrics.append(
                {
                    "task": "tts",
                    "queue_time": start - job.submitted,
                    "device": "cpu",
                }
            )
            return await job.func(*job.args, **job.kwargs)


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
