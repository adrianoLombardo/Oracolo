from __future__ import annotations
import asyncio
import json
import os
import sys
import time
import wave
from pathlib import Path
from typing import Any

import numpy as np
import websockets
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import Settings
from src.hotword import strip_hotword_prefix
from src.oracle import fast_transcribe, oracle_answer
from src.retrieval import retrieve
from src.chat import ChatState
from langdetect import detect

CHUNK_MS = 20


CHUNK_MS = 20
main
# soglia più permissiva per captare parlato anche con microfoni poco sensibili
START_LEVEL = 300
END_SIL_MS = 700
MAX_UTT_MS = 15_000

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


async def stream_tts_pcm(ws, client, text: str, tts_model: str, tts_voice: str, sr: int = 24000) -> None:
    """Streamma TTS in PCM nativo se supportato, altrimenti decapsula WAV."""

    chunk_bytes = int(sr * 2 * CHUNK_MS / 1000)
    block_frames = int(sr * CHUNK_MS / 1000)

    try:
        with client.audio.speech.with_streaming_response.create(
            model=tts_model,
            voice=tts_voice,
            input=text,
            response_format="pcm",
            sample_rate=sr,
        ) as resp:
            for chunk in resp.iter_bytes(chunk_size=chunk_bytes):
                await ws.send(chunk)
                await asyncio.sleep(0)
        return
    except TypeError:
        pass

    import io, wave as _wave

    with client.audio.speech.with_streaming_response.create(
        model=tts_model, voice=tts_voice, input=text, response_format="wav"
    ) as resp:
        buf = io.BytesIO()
        for chunk in resp.iter_bytes(chunk_size=4096):
            buf.write(chunk)
            if buf.tell() > 48:
                break
        for chunk in resp.iter_bytes(chunk_size=8192):
            buf.write(chunk)

    buf.seek(0)
    with _wave.open(buf, "rb") as wf:
        assert wf.getsampwidth() == 2 and wf.getnchannels() == 1
        while True:
            frames = wf.readframes(block_frames)
            if not frames:
                break
            await ws.send(frames)
            await asyncio.sleep(0)

class RTSession:
    def __init__(self, ws, setts: Settings) -> None:
        self.ws = ws
        self.SET = setts
        self.client_sr = setts.audio.sample_rate
        self.buf = bytearray()
        self.state = "idle"
        self.ms_in_state = 0
        self.ms_since_voice = 0
        self.barge = False

        self.tmp = ROOT / "data" / "temp"
        self.in_wav = self.tmp / "rt_input.wav"

        self.active_until = 0.0
        self.wake_phrases = []
        if setts.wake:
            self.wake_phrases = list(getattr(setts.wake, "it_phrases", [])) + list(
                getattr(setts.wake, "en_phrases", [])
            )
        self.idle_timeout = float(getattr(setts.wake, "idle_timeout", 60.0))

        self.chat_enabled = bool(getattr(getattr(setts, "chat", None), "enabled", False))
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
            with client.audio.speech.with_streaming_response.create(
                model=self.SET.openai.tts_model,
                voice=self.SET.openai.tts_voice,
                input=text,
                response_format="pcm",
                sample_rate=self.client_sr,
            ) as resp:
                for chunk in resp.iter_bytes(chunk_size=960):
                    await self.ws.send(chunk)
                    await asyncio.sleep(0)
                    if self.barge:
                        break
        except Exception:
            pass
        self.state = "idle"
        self.barge = False

    async def stream_sentences(self, text: str, client: Any) -> None:
        """Sintetizza ``text`` frase per frase, consentendo il barge-in
        solo ai confini di frase.

        Se ``self.barge`` viene impostato a ``True`` dal client durante la
        riproduzione, la frase corrente viene terminata ma le successive non
        vengono inviate.
        """

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
        if level > START_LEVEL * 0.6:
            self.ms_since_voice = 0
        else:
            self.ms_since_voice += frame_ms

        if self.state == "idle":
            if level >= START_LEVEL:
                print("🎤 rilevato parlato", flush=True)
                self.state = "talking"
                self.ms_in_state = 0
                self.ms_since_voice = 0
        elif self.state == "talking":
            if self.ms_since_voice >= END_SIL_MS or self.ms_in_state >= MAX_UTT_MS:
                await self._finalize_utterance()
                self.buf.clear()
                self.state = "idle"
                self.ms_in_state = 0
                self.ms_since_voice = 0

    async def _finalize_utterance(self) -> None:
        if not self.buf:
            return
        write_wav(self.in_wav, self.client_sr, bytes(self.buf))

        from openai import OpenAI
        load_dotenv()
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

        text = fast_transcribe(
            self.in_wav, client, self.SET.openai.stt_model
        )
        if not text.strip():
            await self.send_partial("…silenzio…")
            return

        try:
            lang = detect(text)
        except Exception:
            lang = "it"
        if lang not in ("it", "en"):
            lang = "it"

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

        # RAG
        try:
            ctx = retrieve(
                text,
                getattr(self.SET, "docstore_path", "DataBase/index.json"),
                top_k=int(getattr(self.SET, "retrieval_top_k", 3)),
            )
        except Exception:
            ctx = []

        if self.chat_enabled:
            self.chat.push_user(text)
        ans, _ = oracle_answer(
            text,
            lang,
            client,
            self.SET.openai.llm_model,
            self.SET.oracle_system,
            context=ctx,
            history=(self.chat.history if self.chat_enabled else None),
        )
        if self.chat_enabled:
            self.chat.push_assistant(ans)

        first_words = " ".join(ans.split()[:5])
        await self.send_partial(first_words)
        tts_task = asyncio.create_task(self.stream_tts_pcm(ans, client))
        await self.send_answer(ans)
        await tts_task

async def handler(ws):
    SET = Settings.model_validate_yaml(ROOT / "settings.yaml")
    sess = RTSession(ws, SET)

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

        if hello.get("type") != "hello":
            await ws.close(code=1002, reason="missing hello type")
            return
        sess.client_sr = int(hello.get("sr", SET.audio.sample_rate))

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
    except websockets.ConnectionClosed:
        pass

async def main(host="127.0.0.1", port=8765):
    # keepalive più larghi e gestione close robusta
    async with websockets.serve(
        handler, host, port,
        ping_interval=20, ping_timeout=40, max_size=None
    ):
        print(f"WS Realtime server pronto su ws://{host}:{port}", flush=True)
        await asyncio.Future()

if __name__ == "__main__":
    try:
        import argparse
        p = argparse.ArgumentParser()
        p.add_argument("--host", default="127.0.0.1")
        p.add_argument("--port", type=int, default=8765)
        args = p.parse_args()
        asyncio.run(main(args.host, args.port))
    except KeyboardInterrupt:
        pass
