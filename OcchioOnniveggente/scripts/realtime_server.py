from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import websockets
import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.chat import ChatState
from src.config import Settings, get_openai_api_key
from src.domain import validate_question
from src.hotword import strip_hotword_prefix
from src.oracle import oracle_answer, transcribe
from src.profile_utils import get_active_profile, make_domain_settings

import wave




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


async def stream_tts_pcm(
    ws,
    client,
    text: str,
    tts_model: str,
    tts_voice: str,
    sr: int = 24000,
    chunk_ms: int = 20,
) -> None:
    """Streamma TTS in PCM nativo se supportato, altrimenti decapsula WAV."""
    chunk_bytes = int(sr * 2 * chunk_ms / 1000)
    block_frames = int(sr * chunk_ms / 1000)
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
    def __init__(self, ws, setts: Settings, raw: dict) -> None:
        self.ws = ws
        self.SET = setts
        self.raw = raw
        self.client_sr = setts.audio.sample_rate
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
            with client.audio.speech.with_streaming_response.create(
                model=self.SET.openai.tts_model,
                voice=self.SET.openai.tts_voice,
                input=text,
                response_format="pcm",
                sample_rate=self.client_sr,
            ) as resp:
                for chunk in resp.iter_bytes(chunk_size=chunk_bytes):
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
        # Convert the raw PCM buffer to a temporary WAV file so that
        # OpenAI's transcription API receives a supported format.
        write_wav(self.in_wav, self.client_sr, audio_bytes)

        from openai import OpenAI

        load_dotenv()
        api_key = get_openai_api_key(self.SET)
        client = OpenAI(api_key=api_key) if api_key else OpenAI()

        text, lang = transcribe(
            self.in_wav, client, self.SET.openai.stt_model, debug=self.SET.debug
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
            await self.stream_sentences(ans, client)
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
        ans, _ = oracle_answer(
            text,
            lang,
            client,
            self.SET.openai.llm_model,
            effective_system,
            context=context_texts,
            history=(self.chat.history if self.chat_enabled else None),
            topic=self.profile,
        )
        if self.chat_enabled:
            self.chat.push_user(text)
            self.chat.push_assistant(ans)
        await self.send_answer(ans)
        await self.stream_sentences(ans, client)


async def handler(ws):
    raw_cfg = yaml.safe_load((ROOT / "settings.yaml").read_text(encoding="utf-8")) or {}
    SET = Settings.model_validate(raw_cfg)
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


async def main(host="127.0.0.1", port=8765):
    async with websockets.serve(
        handler, host, port, ping_interval=20, ping_timeout=40, max_size=None
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

