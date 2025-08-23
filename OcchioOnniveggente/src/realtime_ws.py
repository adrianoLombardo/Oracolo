# src/realtime_ws.py
"""
Starter per Realtime API via WebSocket (ASR+TTS low-latency).
Non è integrato nel main loop; usalo per prove e poi lo innestiamo.
"""
import asyncio, json, os, sys, wave
import websockets

OPENAI_KEY = os.getenv("OPENAI_API_KEY")
REALTIME_MODEL = os.getenv("REALTIME_MODEL", "gpt-4o-realtime-preview")
WS_URL = f"wss://api.openai.com/v1/realtime?model={REALTIME_MODEL}"

async def run():
    async with websockets.connect(
        WS_URL,
        extra_headers={
            "Authorization": f"Bearer {OPENAI_KEY}",
            "OpenAI-Beta": "realtime=v1",
        },
        max_size=10_000_000,
        ping_interval=20,
        ping_timeout=20,
    ) as ws:
        # esempio: manda un “response.create” (solo testo)
        await ws.send(json.dumps({"type": "response.create", "response": {"instructions":"Say hello from realtime."}}))
        while True:
            msg = await ws.recv()
            print(msg)

if __name__ == "__main__":
    if not OPENAI_KEY:
        print("⚠️ Set OPENAI_API_KEY first.")
        sys.exit(1)
    asyncio.run(run())
