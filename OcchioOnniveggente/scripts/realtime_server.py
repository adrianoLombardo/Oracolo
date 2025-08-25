
"""Utility e placeholder per il server realtime."""

from __future__ import annotations

"""Minimal realtime server helpers used in tests."""
from __future__ import annotations


from typing import Dict, List, Tuple

from src.profile_utils import get_active_profile, make_domain_settings

import asyncio
import json
import os
import sys
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

        @staticmethod
        def device_count() -> int:
            return 0

    class _TorchStub:
        cuda = _CudaStub()

    torch = _TorchStub()  # type: ignore


from src import metrics


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
from src.utils.device import resolve_device

import wave



@dataclass
class _TaskJob:
    func: Any
    args: tuple
    kwargs: dict
    submitted: float
    future: asyncio.Future
    tag: str


class Orchestrator:
    """Orchestrates CPU-bound and GPU-bound work using queues."""


    def __init__(self, n_cpu: int = 4, n_gpu: int = 1) -> None:

    def __init__(self) -> None:

        settings = Settings.model_validate_yaml(ROOT / "settings.yaml")
        self.gpu_available = torch.cuda.is_available()
        self.n_gpu = torch.cuda.device_count() if self.gpu_available else 0
        conc = max(1, int(getattr(settings.compute, "device_concurrency", 1)))
        self._gpu_sems = [asyncio.Semaphore(conc) for _ in range(self.n_gpu)]
        self._next_gpu = 0

        self.gpu_available = resolve_device("auto") == "cuda"
        self.cpu_queue: asyncio.Queue[_TaskJob] = asyncio.Queue()
        self.gpu_queue: asyncio.Queue[_TaskJob] = asyncio.Queue()
        self.metrics: list[dict[str, Any]] = []

        self.cpu_workers = [asyncio.create_task(self._cpu_worker()) for _ in range(n_cpu)]
        self.gpu_workers: list[asyncio.Task] = []
        if self.gpu_available:
            self.gpu_workers = [asyncio.create_task(self._gpu_worker()) for _ in range(n_gpu)]

    async def _cpu_worker(self) -> None:
        while True:
            job = await self.cpu_queue.get()
            start = time.time()
            try:
                if asyncio.iscoroutinefunction(job.func):
                    res = await job.func(*job.args, **job.kwargs)
                else:
                    res = await asyncio.to_thread(job.func, *job.args, **job.kwargs)
                job.future.set_result(res)
            except Exception as exc:  # pragma: no cover - pass through
                job.future.set_exception(exc)
            finally:
                qlen = self.cpu_queue.qsize()
                sat = qlen / max(len(self.cpu_workers), 1)
                self.metrics.append(
                    {
                        "task": job.tag,
                        "queue_time": start - job.submitted,
                        "device": "cpu",
                        "saturation": sat,
                    }
                )
                self.cpu_queue.task_done()

    async def _gpu_worker(self) -> None:
        while True:
            job = await self.gpu_queue.get()
            start = time.time()
            try:
                if asyncio.iscoroutinefunction(job.func):
                    res = await job.func(*job.args, **job.kwargs)
                else:
                    res = await asyncio.to_thread(job.func, *job.args, **job.kwargs)
                job.future.set_result(res)
            except Exception as exc:  # pragma: no cover - pass through
                job.future.set_exception(exc)
            finally:
                qlen = self.gpu_queue.qsize()
                sat_base = len(self.gpu_workers) or 1
                sat = qlen / sat_base
                self.metrics.append(
                    {
                        "task": job.tag,
                        "queue_time": start - job.submitted,
                        "device": "cuda" if self.gpu_available else "cpu",
                        "saturation": sat,
                    }
                )
                self.gpu_queue.task_done()

    async def run_cpu(self, func, *args, tag="cpu", **kwargs):
        loop = asyncio.get_running_loop()
        fut = loop.create_future()
        job = _TaskJob(func, args, kwargs, time.time(), fut, tag)
        await self.cpu_queue.put(job)
        return await fut

    async def run_gpu(self, func, *args, tag="gpu", **kwargs):
        if not self.gpu_available:
            return await self.run_cpu(func, *args, tag=tag, **kwargs)
        loop = asyncio.get_running_loop()
        fut = loop.create_future()
        job = _TaskJob(func, args, kwargs, time.time(), fut, tag)
        await self.gpu_queue.put(job)
        return await fut

    async def run_stt(self, func, *args, **kwargs):
        return await self.run_gpu(func, *args, tag="stt", **kwargs)

    async def run_tts(self, coro_func, *args, **kwargs):
        return await self.run_cpu(coro_func, *args, tag="tts", **kwargs)

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



ORCH: Orchestrator | None = None


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
    """Genera un messaggio per input fuori contesto."""

    if keywords:
        kw = ", ".join(keywords)
        return f"La domanda non riguarda il profilo «{profile}». Prova con parole chiave come {kw}."
    return (
        f"La domanda non riguarda il profilo «{profile}». "
        "Per favore riformularla in tema."
    )



__all__ = ["off_topic_message"]


async def stream_tts_pcm(
    ws,
    client,
    text_stream: AsyncIterable[str] | str,
    tts_model: str,
    tts_voice: str,
    sr: int = 24000,
    chunk_ms: int = 20,
    *,
    stop: Callable[[], bool] | None = None,
) -> None:
    """Streamma TTS per testo incrementale, interrompendo in caso di barge-in."""

    def _wrap_stream(obj: AsyncIterable[str] | str) -> AsyncIterable[str]:
        if isinstance(obj, str):
            async def gen():
                yield obj
            return gen()
        return obj

    text_iter = _wrap_stream(text_stream)
    chunk_bytes = int(sr * 2 * chunk_ms / 1000)

    try:
        async for piece in text_iter:
            if stop and stop():
                break

            async with client.audio.speech.with_streaming_response.create(
                model=tts_model,
                voice=tts_voice,
                input=piece,
                response_format="pcm",
                sample_rate=sr,
            ) as resp:
                async for chunk in resp.aiter_bytes(chunk_size=chunk_bytes):
                    await ws.send(chunk)
                    await asyncio.sleep(0)
                    if stop and stop():
                        break
            if stop and stop():
                break
    except Exception:
        pass
        pass


            try:
                with client.audio.speech.with_streaming_response.create(
                    model=tts_model,
                    voice=tts_voice,
                    input=piece,
                    response_format="pcm",
                    sample_rate=sr,
                ) as resp:
                    for chunk in resp.iter_bytes(chunk_size=chunk_bytes):
                        await ws.send(chunk)
                        await asyncio.sleep(0)
                        if stop and stop():
                            break
                if stop and stop():
                    break
                continue
            except TypeError:
                pass

            import io, wave as _wave

        async with client.audio.speech.with_streaming_response.create(
            model=tts_model,
            voice=tts_voice,
            input=text,
            response_format="pcm",
            sample_rate=sr,
        ) as resp:
            async for chunk in resp.aiter_bytes(chunk_size=chunk_bytes):
                await ws.send(chunk)
                await asyncio.sleep(0)
        return
    except TypeError:
        pass
    async with client.audio.speech.with_streaming_response.create(
        model=tts_model, voice=tts_voice, input=text, response_format="wav"
    ) as resp:
        buf = io.BytesIO()
        async for chunk in resp.aiter_bytes(chunk_size=4096):
            buf.write(chunk)
            if buf.tell() > 48:
                break
        async for chunk in resp.aiter_bytes(chunk_size=8192):
            buf.write(chunk)

    buf.seek(0)
    with _wave.open(buf, "rb") as wf:
        assert wf.getsampwidth() == 2 and wf.getnchannels() == 1
        while True:
            frames = wf.readframes(block_frames)
            if not frames:

                break


class RTSession:
    def __init__(self, ws, setts: Settings, raw: dict) -> None:
        self.ws = ws
        self.SET = setts
        self.raw = raw
        self.client_sr = setts.audio.sample_rate
        self.preproc = AudioPreprocessor(
            self.client_sr,
            denoise=getattr(setts.audio, "denoise", False),
            echo_cancel=getattr(setts.audio, "echo_cancel", False),
        )
        self.buf = bytearray()
        self.state = "idle"
        self.ms_in_state = 0
        self.ms_since_voice = 0
        self.barge = False

        dom = raw.get("domain", {}) or {}
        self.profiles = dom.get("profiles", {})
        self.profile = dom.get("profile", "museo")
        prof = self.profiles.get(self.profile, {})
        self.docstore_path = prof.get(
            "docstore_path", getattr(setts, "docstore_path", "DataBase/index.json")
        )
        self.domain_keywords = prof.get("keywords", [])
        self.domain_weights = prof.get("weights", {})
        self.system_hint = prof.get("system_hint", "")
        self.retrieval_top_k = int(
            prof.get("retrieval_top_k") or getattr(setts, "retrieval_top_k", 3)
        )

        self.active_until = 0.0
        self.wake_phrases = []
        if setts.wake:
            self.wake_phrases = list(getattr(setts.wake, "it_phrases", [])) + list(
                getattr(setts.wake, "en_phrases", [])
            )
        self.idle_timeout = float(getattr(setts.wake, "idle_timeout", 50.0))

        self.chat_enabled = bool(
            getattr(getattr(setts, "chat", None), "enabled", False)
        )
        self.chat = ChatState(
            max_turns=int(getattr(getattr(setts, "chat", None), "max_turns", 10)),
            persist_jsonl=Path(
                getattr(
                    getattr(setts, "chat", None),
                    "persist_jsonl",
                    "data/logs/chat_sessions.jsonl",
                )
            )
            if self.chat_enabled
            else None,
        )

        self.tmp = ROOT / "data" / "temp"
        self.in_wav = self.tmp / "rt_input.wav"

    async def send_json(self, obj: dict) -> None:
        try:
            await self.ws.send(json.dumps(obj))
        except Exception:
            pass

    async def send_partial(self, text: str) -> None:
        await self.send_json({"type": "partial", "text": text})

    async def send_answer(self, text: str) -> None:
        await self.send_json({"type": "answer", "text": text})

    async def stream_file(self, path: Path, chunk_bytes: int = 960) -> None:
        """Invia il contenuto di ``path`` in piccoli chunk al client."""
        if not path.exists():
            return
        try:
            with path.open("rb") as f:
                while True:
                    data = f.read(chunk_bytes)
                    if not data:
                        break
                    await self.ws.send(data)
                    await asyncio.sleep(0.01)
        except Exception:
            pass



    async def stream_tts_pcm(self, text: str, client: Any) -> None:
        """Sintetizza e invia ``text`` in formato PCM con latenza ridotta."""
        self.state = "tts"
        self.barge = False
        try:
            chunk_bytes = int(
                self.client_sr
                * 2
                * self.SET.realtime_audio.chunk_ms
                / 1000
            )
            async with client.audio.speech.with_streaming_response.create(
                model=self.SET.openai.tts_model,
                voice=self.SET.openai.tts_voice,
                input=text,
                response_format="pcm",
                sample_rate=self.client_sr,
            ) as resp:
                async for chunk in resp.aiter_bytes(chunk_size=chunk_bytes):
                    await self.ws.send(chunk)
                    await asyncio.sleep(0)
                    if self.barge:
                        break
        except Exception:
            pass
        self.state = "idle"
        self.barge = False


    async def stream_sentences(self, text: str, client: Any) -> None:
        """Sintetizza ``text`` frase per frase, consentendo il barge-in."""
        import re

        sentences = [s.strip() for s in re.split(r"(?<=[.!?]) +", text) if s.strip()]
        if not sentences:
            return

        self.state = "tts"
        self.barge = False
        for sent in sentences:
            try:
                await stream_tts_pcm(
                    self.ws,
                    client,
                    sent,
                    self.SET.openai.tts_model,
                    self.SET.openai.tts_voice,
                    sr=self.client_sr,
                    chunk_ms=self.SET.realtime_audio.chunk_ms,
                    stop=lambda: self.barge,
                )
            except Exception:
                break
            if self.barge:
                break
        self.state = "idle"
        self.barge = False

    async def on_audio(self, data: bytes, frame_ms: int) -> None:
        self.buf.extend(data)
        level = rms_level(data)

        self.ms_in_state += frame_ms
        start_level = self.SET.realtime_audio.start_level
        end_sil_ms = self.SET.realtime_audio.end_sil_ms
        max_utt_ms = self.SET.realtime_audio.max_utt_ms
        if level > start_level * 0.6:
            self.ms_since_voice = 0
        else:
            self.ms_since_voice += frame_ms

        if self.state == "idle":
            if level >= start_level:
                print("🎤 rilevato parlato", flush=True)
                self.state = "talking"
                self.ms_in_state = 0
                self.ms_since_voice = 0
        elif self.state == "talking":
            if self.ms_since_voice >= end_sil_ms or self.ms_in_state >= max_utt_ms:
                await self._finalize_utterance()
                self.buf.clear()
                self.state = "idle"
                self.ms_in_state = 0
                self.ms_since_voice = 0

    async def _finalize_utterance(self) -> None:
        if not self.buf:
            return
        audio_bytes = bytes(self.buf)
        if self.preproc is not None:
            arr = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            arr = self.preproc.process(arr)
            audio_bytes = (np.clip(arr, -1, 1) * 32767).astype(np.int16).tobytes()
        # Convert the raw PCM buffer to a temporary WAV file so that
        # OpenAI's transcription API receives a supported format.
        write_wav(self.in_wav, self.client_sr, audio_bytes)
        self.barge = False

        from openai import AsyncOpenAI

        load_dotenv()
        api_key = get_openai_api_key(self.SET)
        client = AsyncOpenAI(api_key=api_key) if api_key else AsyncOpenAI()



        text, lang = await ORCH.run_stt(
            transcribe,
            self.in_wav,
            client,
            self.SET.openai.stt_model,
            debug=self.SET.debug,
        )

        text = ""
        async for chunk, done in transcribe_stream(
            self.in_wav, client, self.SET.openai.stt_model
        ):
            if self.barge:
                return
            text = chunk
        lang = "it"

        text, lang = await ORCH.run_gpu(
            transcribe_async,
            self.in_wav,
            client,
            self.SET.openai.stt_model,
            debug=self.SET.debug,
        )

        if not text.strip():
            await self.send_partial("…silenzio…")
            return

        print(f"🗣️ {text.strip()}", flush=True)

        now = time.time()
        if now > self.active_until:
            self.active_until = 0.0

        if self.active_until == 0.0:
            matched, remainder = strip_hotword_prefix(text, self.wake_phrases)
            if not matched:
                await self.send_partial("…attendo 'ciao oracolo' o 'hello oracle'.")
                return
            text = remainder
            if not text.strip():
                await self.send_partial("Dimmi pure…")
                self.active_until = now + self.idle_timeout
                return

        self.active_until = now + self.idle_timeout

        embed_model = getattr(self.SET.openai, "embed_model", None)

        settings_for_domain = make_domain_settings(
            self.SET,
            self.profile,
            {"keywords": self.domain_keywords, "weights": self.domain_weights},
        )

        ok, ctx, clarify, reason, _ = validate_question(
            text,
            lang,
            settings=settings_for_domain,
            client=client,
            docstore_path=self.docstore_path,
            top_k=self.retrieval_top_k,
            embed_model=embed_model,
            topic=self.profile,
            history=(self.chat.history if self.chat_enabled else None),
        )
        if not ok:
            if clarify:
                ans = "La domanda non è chiarissima per questo contesto: puoi riformularla brevemente?"
            else:
                ans = off_topic_message(self.profile, self.domain_keywords)
            await self.send_answer(ans)
            await ORCH.run_tts(self.stream_sentences, ans, client)
            return

        context_texts = [
            item.get("text", "") for item in (ctx or []) if isinstance(item, dict)
        ]
        profile_hint = self.system_hint
        base_system = self.SET.oracle_system
        if profile_hint:
            effective_system = f"{base_system}\n\n[Profilo: {self.profile}]\n{profile_hint}"
        else:
            effective_system = base_system

        ans, _ = await ORCH.run_gpu(
            oracle_answer_async,
            text,
            lang,
            client,
            self.SET.openai.llm_model,
            effective_system,
            context=context_texts,
            history=(self.chat.history if self.chat_enabled else None),
            topic=self.profile,
        )

        final = ""
        async for chunk, done in oracle_answer_stream(

            text,
            lang,
            client,
            self.SET.openai.llm_model,
            effective_system,
            context=context_texts,
            history=(self.chat.history if self.chat_enabled else None),
            topic=self.profile,
        ):
            if self.barge:
                return
            if done:
                final = chunk
            else:
                await self.send_partial(chunk)
        if self.chat_enabled and final:
            self.chat.push_user(text)
            self.chat.push_assistant(final)
        await self.send_answer(final)
        await ORCH.run_tts(self.stream_sentences, final, client)


async def handler(ws):
    global ORCH
    raw_cfg = yaml.safe_load((ROOT / "settings.yaml").read_text(encoding="utf-8")) or {}
    SET = Settings.model_validate(raw_cfg)
    if ORCH is None:
        ORCH = Orchestrator(SET.realtime.cpu_workers, SET.realtime.gpu_workers)
    sess = RTSession(ws, SET, raw_cfg)

    try:
        hello_raw = await ws.recv()
        if isinstance(hello_raw, (bytes, bytearray)):
            await ws.close(code=1002, reason="expected text hello")
            return
        try:
            hello = json.loads(hello_raw)
        except json.JSONDecodeError:
            await ws.close(code=1002, reason="invalid hello")
            return
        if hello.get("type") not in (None, "hello"):
            await ws.close(code=1002, reason="missing hello type")
            return
        sess.client_sr = int(hello.get("sr", SET.audio.sample_rate))
        print(f"🤝 handshake sr={sess.client_sr}", flush=True)

        await sess.send_json({"type": "ready"})
        await sess.send_partial("Sto capendo...")

        bytes_per_ms = max(int(sess.client_sr * 2 / 1000), 1)
        async for msg in ws:
            if isinstance(msg, (bytes, bytearray)):
                est_ms = max(int(len(msg) / bytes_per_ms), 1)
                await sess.on_audio(bytes(msg), est_ms)
            else:
                try:
                    data = json.loads(msg)
                except Exception:
                    continue
                if data.get("type") == "barge_in":
                    sess.barge = True
                elif data.get("type") == "reset":
                    sess.chat.reset()
                    await sess.send_json({"type": "reset_ok"})
                elif data.get("type") == "profile":
                    sess.profile = data.get("value", "")
                    prof = sess.profiles.get(sess.profile, {})
                    sess.docstore_path = prof.get(
                        "docstore_path",
                        getattr(sess.SET, "docstore_path", "DataBase/index.json"),
                    )
                    sess.topic = sess.profile
                    sess.domain_keywords = prof.get("keywords", [])
                    sess.domain_weights = prof.get("weights", {})
                    sess.system_hint = prof.get("system_hint", "")
                    sess.retrieval_top_k = int(
                        prof.get("retrieval_top_k")
                        or getattr(sess.SET, "retrieval_top_k", 3)
                    )
                    await sess.send_json(
                        {"type": "info", "text": f"Profilo attivo: {sess.profile}"}
                    )
    except websockets.ConnectionClosed:
        pass



def get_active_profile(settings: dict) -> tuple[str, dict]:
    """Restituisce il profilo attivo dalle impostazioni."""

    dom = settings.get("domain", {})
    name = dom.get("profile", "")
    profiles = dom.get("profiles", {})
    return name, profiles.get(name, {})



__all__.append("get_active_profile")


def off_topic_message(profile: str, keywords: List[str]) -> str:
    """Return a simple off-topic warning message."""
    if keywords:
        kw = ", ".join(keywords)
        return f"La domanda non rientra nel profilo «{profile}». Prova con questi temi: {kw}."
    return f"La domanda non rientra nel profilo «{profile}». Per favore riformularla."


__all__ = ["off_topic_message", "get_active_profile", "make_domain_settings"]





