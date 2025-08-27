from __future__ import annotations

"""Realtime WebSocket client for bidirectional audio streaming."""

import asyncio
import json
import queue
import threading
import time
from typing import Any, Callable

import numpy as np
import sounddevice as sd
import websockets


class RealtimeWSClient:
    """Gestisce la connessione WebSocket realtime con audio bidirezionale."""

    def __init__(
        self,
        url: str,
        sr: int,
        on_partial=lambda text, final: None,
        on_answer=lambda text: None,
        *,
        barge_threshold: float = 500.0,
        ping_interval: int = 20,
        ping_timeout: int = 20,
        auto_reconnect: bool = False,
        on_input_level=lambda level: None,
        on_output_level=lambda level: None,
        on_event=lambda evt: None,
        on_ping=lambda ms: None,
        profile_name: str = "museo",
    ) -> None:
        self.url = url
        self.sr = sr
        self.frame_samples = int(self.sr * 0.02)
        self.on_partial = on_partial
        self.on_answer = on_answer
        self.barge_threshold = barge_threshold
        self.ping_interval = ping_interval
        self.ping_timeout = ping_timeout
        self.auto_reconnect = auto_reconnect
        self.on_input_level = on_input_level
        self.on_output_level = on_output_level
        self.on_event = on_event
        self.on_ping = on_ping
        self.profile_name = profile_name
        self.send_q: "queue.Queue[bytes]" = queue.Queue()
        self.audio_q: "queue.Queue[bytes]" = queue.Queue()
        self.state: dict[str, Any] = {
            "tts_playing": False,
            "barge_sent": False,
            "barge_threshold": barge_threshold,
        }
        self.loop: asyncio.AbstractEventLoop | None = None
        self.thread: threading.Thread | None = None
        self.ws = None
        self.stop_event: asyncio.Event | None = None

    async def _mic_worker(self) -> None:
        assert self.ws is not None and self.stop_event is not None
        loop = asyncio.get_running_loop()

        def callback(indata, frames, time_info, status) -> None:  # type: ignore[override]
            data = bytes(indata)  # CFFI -> bytes

            if not self.state.get("tts_playing"):
                self.send_q.put_nowait(data)

            samples = np.frombuffer(data, dtype=np.int16).astype(np.float32)
            level = float(np.sqrt(np.mean(samples ** 2)))
            self.on_input_level(level / 32768.0)

            if self.state.get("tts_playing"):
                if level > self.state.get("barge_threshold", 500.0):
                    self.state["hot_frames"] = self.state.get("hot_frames", 0) + 1
                else:
                    self.state["hot_frames"] = 0
                now = time.monotonic()
                if (
                    self.state.get("hot_frames", 0) >= 10
                    and not self.state.get("barge_sent", False)
                    and now - self.state.get("last_barge_ts", 0) > 0.6
                ):
                    self.state["barge_sent"] = True
                    self.state["last_barge_ts"] = now
                    asyncio.run_coroutine_threadsafe(
                        self.ws.send(json.dumps({"type": "barge_in"})), loop
                    )

        with sd.RawInputStream(
            samplerate=self.sr,
            blocksize=self.frame_samples,
            channels=1,
            dtype="int16",
            callback=callback,
            latency="low",
        ):
            while not self.stop_event.is_set():
                await asyncio.sleep(0.1)

    async def _sender(self) -> None:
        assert self.ws is not None and self.stop_event is not None
        while not self.stop_event.is_set():
            data = await asyncio.get_running_loop().run_in_executor(None, self.send_q.get)
            await self.ws.send(data)

    async def _receiver(self) -> None:
        assert self.ws is not None and self.stop_event is not None
        async for msg in self.ws:
            if isinstance(msg, bytes):
                self.audio_q.put_nowait(msg)
                continue
            try:
                data = json.loads(msg)
            except json.JSONDecodeError:
                continue
            kind = data.get("type")
            text = data.get("text", "")
            if kind == "partial":
                self.on_partial(text, bool(data.get("final")))
            elif kind in ("final", "transcript"):
                self.on_partial(text, True)
            elif kind == "answer":
                self.on_answer(text)
            if self.stop_event.is_set():
                break

    async def _player(self) -> None:
        assert self.stop_event is not None

        def callback(outdata, frames, time_info, status) -> None:  # type: ignore[override]
            try:
                chunk = self.audio_q.get_nowait()
                n = len(outdata)
                vol = 0.3 if self.state.get("barge_sent") else 1.0
                data = np.frombuffer(chunk, dtype=np.int16).astype(np.float32)
                data = np.clip(data * vol, -32768, 32767).astype(np.int16).tobytes()
                if len(data) >= n:
                    outdata[:] = data[:n]
                else:
                    outdata[:len(data)] = data
                    outdata[len(data):] = b"\x00" * (n - len(data))
                samples = np.frombuffer(outdata, dtype=np.int16).astype(np.float32)
                level = float(np.sqrt(np.mean(samples ** 2)))
                self.on_output_level(level / 32768.0)
                self.state["tts_playing"] = True
            except queue.Empty:
                outdata[:] = b"\x00" * len(outdata)
                self.state["tts_playing"] = False
                self.state["barge_sent"] = False
                self.on_output_level(0.0)

        with sd.RawOutputStream(
            samplerate=self.sr,
            blocksize=self.frame_samples,
            channels=1,
            dtype="int16",
            callback=callback,
        ):
            while not self.stop_event.is_set():
                await asyncio.sleep(0.1)

    async def _pinger(self) -> None:
        assert self.ws is not None and self.stop_event is not None
        while not self.stop_event.is_set():
            start = time.perf_counter()
            try:
                pong = await self.ws.ping()
                await pong
                self.on_ping((time.perf_counter() - start) * 1000)
            except Exception:
                break
            await asyncio.sleep(self.ping_interval)

    async def _run(self) -> None:
        self.stop_event = asyncio.Event()
        while not self.stop_event.is_set():
            try:
                async with websockets.connect(
                    self.url, ping_interval=None, ping_timeout=self.ping_timeout
                ) as ws:
                    self.ws = ws
                    self.on_event("connected")
                    await ws.send(
                        json.dumps(
                            {"type": "hello", "sr": self.sr, "format": "pcm_s16le", "channels": 1}
                        )
                    )
                    try:
                        ready_raw = await asyncio.wait_for(ws.recv(), timeout=10)
                        data = json.loads(ready_raw) if isinstance(ready_raw, str) else {}
                        if data.get("type") != "ready":
                            raise RuntimeError("handshake")
                    except Exception as e:
                        self.on_event(f"handshake_error:{e}")
                        if not self.auto_reconnect:
                            return
                        await asyncio.sleep(2)
                        continue

                    self.on_event("handshake_ok")
                    try:
                        await ws.send(json.dumps({"type": "profile", "value": self.profile_name}))
                    except Exception:
                        pass

                    tasks = [
                        asyncio.create_task(self._mic_worker()),
                        asyncio.create_task(self._sender()),
                        asyncio.create_task(self._receiver()),
                        asyncio.create_task(self._player()),
                        asyncio.create_task(self._pinger()),
                    ]
                    await self.stop_event.wait()
                    for t in tasks:
                        t.cancel()
            except Exception as e:
                self.on_event(f"error:{e}")
                if not self.auto_reconnect:
                    break
                await asyncio.sleep(2)
            finally:
                self.ws = None
            if not self.auto_reconnect:
                break
        self.on_event("disconnected")

    def start(self) -> None:
        if self.thread and self.thread.is_alive():
            return
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_until_complete, args=(self._run(),), daemon=True)
        self.thread.start()

    def stop(self) -> None:
        if not self.loop:
            return
        if self.stop_event is not None:
            self.loop.call_soon_threadsafe(self.stop_event.set)
        if self.ws is not None:
            asyncio.run_coroutine_threadsafe(self.ws.close(), self.loop)
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1)
        self.thread = None
        self.loop = None
