# src/realtime_oracolo.py
"""Client WebSocket realtime per l'Oracolo.

Questo modulo apre una connessione WebSocket verso un backend che accetta
audio PCM mono a 16 bit. I frame provenienti dal microfono vengono inviati al
server che risponde con trascrizioni parziali, la risposta finale e frame TTS
da riprodurre in tempo reale. Se l'utente parla durante il TTS viene inviato un
evento ``barge_in``: il server interrompe la risposta al termine della frase
corrente e il volume del TTS viene attenuato localmente per favorire
l'interruzione naturale.
"""

from __future__ import annotations

import asyncio
import json
import os
import queue
from typing import Any

import numpy as np
import sounddevice as sd
import websockets
from src.dialogue import DialogueManager, DialogState


SR = 24_000
WS_URL = os.getenv("ORACOLO_WS_URL", "ws://localhost:8765")


async def _mic_worker(
    ws, send_q: "queue.Queue[bytes]", *, sr: int, state: dict[str, Any]
) -> None:
    """Cattura audio dal microfono e lo mette in coda da inviare.

    Se durante la riproduzione del TTS rileva voce in ingresso invia un messaggio
    ``barge_in`` al server per interrompere la risposta.
    """

    loop = asyncio.get_running_loop()

    def callback(indata, frames, time_info, status) -> None:  # type: ignore[override]
        data = bytes(indata)
        send_q.put_nowait(data)

        if state.get("tts_playing") and not state.get("barge_sent"):
            samples = np.frombuffer(data, dtype=np.int16).astype(np.float32)
            level = float(np.sqrt(np.mean(samples ** 2)))
            if level > 500:  # soglia empirica
                state["barge_sent"] = True
                asyncio.run_coroutine_threadsafe(
                    ws.send(json.dumps({"type": "barge_in"})),
                    loop,
                )

    with sd.RawInputStream(
        samplerate=sr, channels=1, dtype="int16", callback=callback
    ):
        while True:
            await asyncio.sleep(0.1)


async def _sender(ws, q: "queue.Queue[bytes]") -> None:
    """Invia in background i frame audio presenti nella coda."""

    try:
        while True:
            data = await asyncio.get_running_loop().run_in_executor(None, q.get)
            await ws.send(data)
    except (websockets.ConnectionClosed, asyncio.CancelledError):
        return


async def _player(
    audio_q: "queue.Queue[bytes]", *, sr: int, state: dict[str, Any], dlg: DialogueManager
) -> None:
    """Riproduce i frame audio (binari) ricevuti dal server."""
    buf = bytearray()

    def callback(outdata, frames, time_info, status) -> None:  # type: ignore[override]
        nonlocal buf
        n = len(outdata)

        while len(buf) < n:
            try:
                buf.extend(audio_q.get_nowait())
                state["tts_playing"] = True
            except queue.Empty:
                break

        if buf:
            vol = 0.3 if state.get("barge_sent") else 1.0
            data = np.frombuffer(buf[:n], dtype=np.int16).astype(np.float32)
            data = np.clip(data * vol, -32768, 32767).astype(np.int16)
            outdata[:] = data.tobytes()
            buf = buf[n:]
        else:
            outdata[:] = b"\x00" * n
            state["tts_playing"] = False
            state["barge_sent"] = False
            dlg.transition(DialogState.LISTENING)

    blocksize = int(sr * 0.02)  # ~20ms di audio
    with sd.RawOutputStream(
        samplerate=sr, channels=1, dtype="int16", callback=callback, blocksize=blocksize
    ):
        while True:
            await asyncio.sleep(0.1)


async def _receiver(ws, audio_q: "queue.Queue[bytes]", dlg: DialogueManager) -> None:
    """Gestisce i messaggi provenienti dal server."""

    try:
        async for msg in ws:
            if isinstance(msg, bytes):
                audio_q.put_nowait(msg)
                continue
            try:
                data = json.loads(msg)
            except json.JSONDecodeError:
                continue
            kind = data.get("type")
            if kind == "partial":
                dlg.transition(DialogState.THINKING)
                print(f"… {data.get('text', '')}", flush=True)
            elif kind == "answer":
                dlg.transition(DialogState.SPEAKING)
                print(f"🔮 {data.get('text', '')}", flush=True)
    except (websockets.ConnectionClosed, asyncio.CancelledError):
        return


async def _run(url: str, sr: int) -> None:
    """Avvia la sessione realtime verso ``url``."""

    send_q: "queue.Queue[bytes]" = queue.Queue()
    audio_q: "queue.Queue[bytes]" = queue.Queue()
    state: dict[str, Any] = {"tts_playing": False, "barge_sent": False}
    dlg = DialogueManager(idle_timeout=60)
    dlg.transition(DialogState.LISTENING)

    async with websockets.connect(url, ping_interval=20, ping_timeout=20) as ws:
        await ws.send(
            json.dumps(
                {"type": "hello", "sr": sr, "format": "pcm_s16le", "channels": 1}
            )
        )
        try:
            ready_raw = await asyncio.wait_for(ws.recv(), timeout=10)
            if not isinstance(ready_raw, (bytes, str)):
                raise RuntimeError("Handshake non valido")
            data = json.loads(ready_raw) if isinstance(ready_raw, str) else {}
            if data.get("type") != "ready":
                raise RuntimeError("Handshake non valido")
        except Exception:
            print("Handshake non valido", flush=True)
            return

        print("✅ pronto a ricevere audio", flush=True)
        print(
            f"🔌 Realtime WS → {url}  (sr={sr}, in={sd.default.device[0]}, out={sd.default.device[1]})",
            flush=True,
        )

        tasks = [
            asyncio.create_task(_mic_worker(ws, send_q, sr=sr, state=state)),
            asyncio.create_task(_sender(ws, send_q)),
            asyncio.create_task(_receiver(ws, audio_q, dlg)),
            asyncio.create_task(_player(audio_q, sr=sr, state=state, dlg=dlg)),
        ]
        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            for t in tasks:
                t.cancel()
            raise


def main() -> None:
    """Punto di ingresso da CLI."""

    import argparse

    parser = argparse.ArgumentParser(description="Client WS realtime Oracolo")
    parser.add_argument("--url", default=WS_URL)
    parser.add_argument("--sr", type=int, default=SR)
    parser.add_argument(
        "--in-dev", dest="in_dev", default=None, help="Indice input device (o None)"
    )
    parser.add_argument(
        "--out-dev", dest="out_dev", default=None, help="Indice output device (o None)"
    )
    parser.add_argument(
        "--barge-threshold",
        type=float,
        default=500.0,
        help="Soglia barge-in RMS (non usata qui)",
    )
    # Flag ignorati (per compatibilità con l'interfaccia precedente)
    parser.add_argument("--autostart", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    def _parse_dev(v):
        if v is None or str(v).strip().lower() in {"none", ""}:
            return None
        try:
            return int(v)
        except Exception:
            return None

    sd.default.device = (_parse_dev(args.in_dev), _parse_dev(args.out_dev))

    asyncio.run(_run(args.url, args.sr))


if __name__ == "__main__":
    main()

