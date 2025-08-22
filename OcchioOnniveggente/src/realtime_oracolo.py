# src/realtime_oracolo.py
from __future__ import annotations

"""
Client WebSocket realtime per l'Oracolo.

- Apre una connessione WS verso un backend che accetta audio PCM s16le mono.
- Invia frame audio dal microfono.
- Riceve trascrizioni parziali (JSON {"type":"partial","text":...}),
  risposte finali ({"type":"answer","text":...}) e frame TTS (messaggi binari).
- Se l'utente parla durante il TTS, invia {"type":"barge_in"} per interrompere.
"""

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
    """Cattura audio dal microfono e lo mette in coda da inviare. Se TTS è in corso,
    rileva voce in ingresso e invia 'barge_in' una sola volta."""
    loop = asyncio.get_running_loop()

    def callback(indata, frames, time_info, status) -> None:  # type: ignore[override]
        data = bytes(indata)  # indata è buffer CFFI -> bytes
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

    with sd.RawInputStream(samplerate=sr, channels=1, dtype="int16", callback=callback):
        while True:
            await asyncio.sleep(0.1)


async def _sender(ws, q: "queue.Queue[bytes]") -> None:
    try:
        while True:
            data = await asyncio.get_running_loop().run_in_executor(None, q.get)
            await ws.send(data)
    except (websockets.ConnectionClosed, asyncio.CancelledError):
        return


async def _player(audio_q: "queue.Queue[bytes]", *, sr: int, state: dict[str, Any]) -> None:
    """Riproduce i frame audio (binari) ricevuti dal server."""
    def callback(outdata, frames, time_info, status) -> None:  # type: ignore[override]
        try:
            chunk = audio_q.get_nowait()
            # Adatta il chunk alla dimensione del buffer di uscita
            n = len(outdata)
            if len(chunk) >= n:
                outdata[:] = chunk[:n]
            else:
                outdata[:len(chunk)] = chunk
                outdata[len(chunk):] = b"\x00" * (n - len(chunk))
            state["tts_playing"] = True
        except queue.Empty:
            outdata[:] = b"\x00" * len(outdata)
            state["tts_playing"] = False
            state["barge_sent"] = False

    with sd.RawOutputStream(samplerate=sr, channels=1, dtype="int16", callback=callback):
        while True:
            await asyncio.sleep(0.1)


async def _receiver(ws, audio_q: "queue.Queue[bytes]") -> None:
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
            if kind == "ready":
                # handshake ok
                continue
            if kind == "partial":
                print(f"… {data.get('text', '')}", flush=True)
            elif kind == "answer":
                print(f"🔮 {data.get('text', '')}", flush=True)
    except (websockets.ConnectionClosed, asyncio.CancelledError):
        return


async def _run(url: str, sr: int) -> None:
    send_q: "queue.Queue[bytes]" = queue.Queue()
    audio_q: "queue.Queue[bytes]" = queue.Queue()
    state: dict[str, Any] = {"tts_playing": False, "barge_sent": False}

       codex/review-project-files-and-websocket-scripts-ag8z2e
    async with websockets.connect(url) as ws:
        # Log utili per capire dove stiamo collegandoci
        print(f"🔌 Realtime WS → {url}  (sr={sr})", flush=True)

        # Invio handshake iniziale con il sample rate. Il server risponde
        # con "ready"; se manca chiudiamo la sessione.
        await ws.send(json.dumps({"type": "hello", "sr": sr}))
        try:
            ready_raw = await ws.recv()
            if json.loads(ready_raw).get("type") != "ready":
                print("Handshake non valido", flush=True)
                return

    async with websockets.connect(url, ping_interval=20, ping_timeout=20) as ws:
        # Handshake: annuncia parametri audio e attendi 'ready'
        await ws.send(json.dumps({"type": "hello", "sr": sr, "format": "pcm_s16le", "channels": 1}))
        try:
            ready_raw = await asyncio.wait_for(ws.recv(), timeout=10)
            if not isinstance(ready_raw, (bytes, str)):
                raise RuntimeError("Handshake non valido")
            data = json.loads(ready_raw) if isinstance(ready_raw, str) else {}
            if data.get("type") != "ready":
                raise RuntimeError("Handshake non valido")
        main
        except Exception:
            print("Handshake non valido", flush=True)
            return

        codex/review-project-files-and-websocket-scripts-ag8z2e
        print("✅ pronto a ricevere audio", flush=True)

        print(
            f"🔌 Realtime WS → {url}  (sr={sr}, in={sd.default.device[0]}, out={sd.default.device[1]})",
            flush=True,
        )
        main

        tasks = [
            asyncio.create_task(_mic_worker(ws, send_q, sr=sr, state=state)),
            asyncio.create_task(_sender(ws, send_q)),
            asyncio.create_task(_receiver(ws, audio_q)),
            asyncio.create_task(_player(audio_q, sr=sr, state=state)),
        ]
        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            for t in tasks:
                t.cancel()
            raise


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Client WS realtime Oracolo")
    parser.add_argument("--url", default=WS_URL)
    parser.add_argument("--sr", type=int, default=SR)
    parser.add_argument("--in-dev", dest="in_dev", default=None, help="Indice input device (o None)")
    parser.add_argument("--out-dev", dest="out_dev", default=None, help="Indice output device (o None)")
    parser.add_argument("--barge-threshold", type=float, default=500.0, help="Soglia barge-in RMS (non usata qui)")
    # Flags ignorati (per compatibilità)
    parser.add_argument("--autostart", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    # Applica i device se forniti
    def _parse_dev(v):
        if v is None or str(v).strip().lower() == "none" or str(v).strip() == "":
            return None
        try:
            return int(v)
        except Exception:
            return None

    sd.default.device = (_parse_dev(args.in_dev), _parse_dev(args.out_dev))

    asyncio.run(_run(args.url, args.sr))


if __name__ == "__main__":
    main()

