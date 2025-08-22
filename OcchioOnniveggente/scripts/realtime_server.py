from __future__ import annotations
import asyncio
import json
import os
import sys
import wave
from pathlib import Path
from typing import Any, Optional

import numpy as np
import websockets
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import Settings
from src.oracle import transcribe, oracle_answer, synthesize
from src.retrieval import retrieve

CHUNK_MS = 20
START_LEVEL = 600
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
        self.out_wav = self.tmp / "rt_answer.wav"

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
        if not path.exists():
            return
        self.state = "tts"
        self.barge = False
        try:
            with path.open("rb") as f:
                while True:
                    if self.barge:
                        break
                    data = f.read(chunk_bytes)
                    if not data:
                        break
                    await self.ws.send(data)
                    await asyncio.sleep(0.01)
        except Exception:
            pass
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

        text, lang = transcribe(self.in_wav, client, self.SET.openai.stt_model, debug=self.SET.debug)
        if not text.strip():
            await self.send_partial("…silenzio…")
            return

        # RAG
        try:
            ctx = retrieve(text, getattr(self.SET, "docstore_path", "DataBase/index.json"),
                           top_k=int(getattr(self.SET, "retrieval_top_k", 3)))
        except Exception:
            ctx = []

        ans = oracle_answer(text, lang, client, self.SET.openai.llm_model, self.SET.oracle_system, context=ctx)
        await self.send_answer(ans)

        synthesize(ans, self.out_wav, client, self.SET.openai.tts_model, self.SET.openai.tts_voice)
        await self.stream_file(self.out_wav, chunk_bytes=960)

async def handler(ws):
    SET = Settings.model_validate_yaml(ROOT / "settings.yaml")
    sess = RTSession(ws, SET)

    try:
        hello_raw = await ws.recv()
        if isinstance(hello_raw, (bytes, bytearray)):
            return
        hello = json.loads(hello_raw)
        sess.client_sr = int(hello.get("sr", SET.audio.sample_rate))

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
