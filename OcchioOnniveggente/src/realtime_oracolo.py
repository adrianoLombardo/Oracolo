"""Client WebSocket realtime per l'Oracolo.

Questo modulo apre una connessione WebSocket verso un backend o API realtime
che accetta audio PCM a 16 bit mono. I frame audio vengono inviati dal
microfono in background. Dal server si ricevono messaggi testuali con
trascrizioni parziali, la risposta finale e frame audio (TTS) da riprodurre
subito. Se l'utente parla durante la riproduzione del TTS viene inviato un
messaggio ``barge_in`` per interrompere il parlato lato server.

Nota: il server deve inviare i frame audio come messaggi binari WS e le
trascrizioni/testo come messaggi JSON con il campo ``type``.
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

SR = 24_000
WS_URL = os.getenv("ORACOLO_WS_URL", "ws://localhost:8765")


async def _mic_worker(ws, send_q: "queue.Queue[bytes]", *, sr: int, state: dict[str, Any]) -> None:
    """Cattura audio dal microfono e lo inserisce nella coda da inviare.

    Durante la riproduzione del TTS analizza il livello del microfono e invia
    un evento ``barge_in`` se rileva nuova voce.
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
                    ws.send(json.dumps({"type": "barge_in"})), loop
                )

    with sd.RawInputStream(
        samplerate=sr, channels=1, dtype="int16", callback=callback
    ):
        while True:
            await asyncio.sleep(0.1)


async def _sender(ws, q: "queue.Queue[bytes]") -> None:
    while True:
        data = await asyncio.get_running_loop().run_in_executor(None, q.get)
        await ws.send(data)


async def _player(audio_q: "queue.Queue[bytes]", *, sr: int, state: dict[str, Any]) -> None:
    """Riproduce i frame audio ricevuti dal server."""

    def callback(outdata, frames, time_info, status) -> None:  # type: ignore[override]
        try:
            chunk = audio_q.get_nowait()
            outdata[:] = chunk
            state["tts_playing"] = True
        except queue.Empty:
            outdata.fill(0)
            state["tts_playing"] = False
            state["barge_sent"] = False

    with sd.RawOutputStream(
        samplerate=sr, channels=1, dtype="int16", callback=callback
    ):
        while True:
            await asyncio.sleep(0.1)


async def _receiver(ws, audio_q: "queue.Queue[bytes]") -> None:
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
            print(f"… {data.get('text', '')}")
        elif kind == "answer":
            print(f"🔮 {data.get('text', '')}")


async def main(url: str = WS_URL, sr: int = SR) -> None:
    """Avvia il client realtime."""
    send_q: "queue.Queue[bytes]" = queue.Queue()
    audio_q: "queue.Queue[bytes]" = queue.Queue()
    state: dict[str, Any] = {"tts_playing": False, "barge_sent": False}

    async with websockets.connect(url) as ws:
        tasks = [
            asyncio.create_task(_mic_worker(ws, send_q, sr=sr, state=state)),
            asyncio.create_task(_sender(ws, send_q)),
            asyncio.create_task(_receiver(ws, audio_q)),
            asyncio.create_task(_player(audio_q, sr=sr, state=state)),
        ]
        await asyncio.gather(*tasks)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
