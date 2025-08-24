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
import time
from typing import Any

import numpy as np
import sounddevice as sd
import websockets
from src.dialogue import DialogueManager, DialogState


SR = 24_000
FRAME_MS = 10
BLOCKSIZE = SR * FRAME_MS // 1000  # 240 campioni per 10ms @24k
BYTES_PER_SAMPLE = 2  # int16
CHANNELS = 1
NEED_BYTES = BLOCKSIZE * BYTES_PER_SAMPLE * CHANNELS
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
        data = bytes(indata)  # CFFI -> bytes

        # HALF-DUPLEX: non streammare al server mentre il TTS sta parlando,
        # eviti eco e "false barge-in". Continui però a misurare il livello.
        if not state.get("tts_playing"):
            send_q.put_nowait(data)

        # BARGE-IN: richiedi ~250ms sopra soglia prima di inviare l'evento.
        # (25 callback da 10ms = 250ms)
        samples = np.frombuffer(data, dtype=np.int16).astype(np.float32)
        level = float(np.sqrt(np.mean(samples ** 2)))

        if state.get("tts_playing"):
            # isteresi + cooldown per non spammare
            if level > state.get("barge_threshold", 500.0):
                state["hot_frames"] = state.get("hot_frames", 0) + 1
                state["ducking"] = True
            else:
                state["hot_frames"] = 0
                state["ducking"] = False
            now = time.monotonic()
            if (
                state.get("hot_frames", 0) >= 25
                and not state.get("barge_sent", False)
                and now - state.get("last_barge_ts", 0) > 0.6
            ):
                state["barge_sent"] = True
                state["last_barge_ts"] = now
                asyncio.run_coroutine_threadsafe(
                    ws.send(json.dumps({"type": "barge_in"})),
                    loop
                )

    with sd.RawInputStream(
        samplerate=sr, channels=1, dtype="int16",
        callback=callback, blocksize=BLOCKSIZE, latency="low"
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
    audio_q: "queue.Queue[bytes]", *, sr: int, state: dict[str, Any]
) -> None:
    carry = bytearray()

    def callback(outdata, frames, time_info, status) -> None:  # type: ignore[override]
        need = frames * BYTES_PER_SAMPLE * CHANNELS
        written = 0

        # 1) consuma prima eventuale carry
        if carry:
            take = min(len(carry), need)
            outdata[:take] = carry[:take]
            del carry[:take]
            written += take

        # 2) pesca pacchetti finché riempi o finché la coda è vuota
        while written < need:
            try:
                chunk = audio_q.get_nowait()
            except queue.Empty:
                break
            take = min(len(chunk), need - written)
            outdata[written : written + take] = chunk[:take]
            written += take
            # se avanza spezzone, rimettilo in carry
            if take < len(chunk):
                carry.extend(chunk[take:])

        # 3) padding se siamo corti
        if written < need:
            outdata[written:need] = b"\x00" * (need - written)
            state["tts_playing"] = False
            state["barge_sent"] = False
        else:
            state["tts_playing"] = True

        if state.get("ducking"):
            arr = np.frombuffer(outdata, dtype=np.int16)
            arr[:] = (arr * 0.3).astype(np.int16)

    with sd.RawOutputStream(
        samplerate=sr,
        channels=1,
        dtype="int16",
        callback=callback,
        blocksize=BLOCKSIZE,
        latency="low",
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


async def _run(url: str, sr: int, barge_threshold: float) -> None:
    """Avvia la sessione realtime verso ``url``."""

    send_q: "queue.Queue[bytes]" = queue.Queue()
    audio_q: "queue.Queue[bytes]" = queue.Queue()
    state: dict[str, Any] = {
        "tts_playing": False,
        "barge_sent": False,
        "barge_threshold": barge_threshold,
        "ducking": False,
    }
    dlg = DialogueManager(idle_timeout=60)
    dlg.transition(DialogState.LISTENING)

    async with websockets.connect(url, ping_interval=20, ping_timeout=20) as ws:
        await ws.send(
            json.dumps(
                {
                    "type": "hello",
                    "sr": sr,
                    "format": "pcm_s16le",
                    "channels": CHANNELS,
                }
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
            asyncio.create_task(_player(audio_q, sr=sr, state=state)),
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
        help="Soglia barge-in RMS",
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

    asyncio.run(_run(args.url, args.sr, args.barge_threshold))


if __name__ == "__main__":
    main()

