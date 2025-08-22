# scripts/realtime_dummy_server.py
from __future__ import annotations
import asyncio
import json
import math
import numpy as np
import websockets

async def handler(ws):
    try:
        # primo messaggio: hello (JSON)
        raw = await ws.recv()
        if isinstance(raw, (bytes, bytearray)):
            return
        hello = json.loads(raw)
        sr = int(hello.get("sr", 24000))
        print("hello:", hello, flush=True)

        # feedback testo
        await ws.send(json.dumps({"type": "partial", "text": "Sto capendo..."}))
        await asyncio.sleep(0.4)
        await ws.send(json.dumps({"type": "answer", "text": "Ciao, sono l’Oracolo realtime (server di prova)!"}))

        # audio TTS fake: tono 440Hz per ~0.7s
        dur = 0.7
        t = np.arange(int(sr * dur)) / sr
        wave = (0.22 * np.sin(2 * math.pi * 440 * t)).astype(np.float32)
        pcm = (wave * 32767).astype(np.int16).tobytes()

        # invia a piccoli chunk per simulare streaming (10ms @ 24kHz ≈ 480 byte)
        chunk = 480
        for i in range(0, len(pcm), chunk):
            await ws.send(pcm[i:i + chunk])
            await asyncio.sleep(0.01)

        # poi consuma il resto dei messaggi senza rumore
        async for _ in ws:
            pass

    except websockets.ConnectionClosed:
        pass
    except Exception as e:
        print("handler error:", e, flush=True)

async def main(host="127.0.0.1", port=8765):
    async with websockets.serve(handler, host, port):
        print(f"WS dummy server in ascolto su ws://{host}:{port}", flush=True)
        await asyncio.Future()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
