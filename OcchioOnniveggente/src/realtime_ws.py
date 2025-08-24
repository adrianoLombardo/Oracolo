# src/realtime_ws.py
"""
Starter per Realtime API via WebSocket (ASR+TTS low-latency).
Non è integrato nel main loop; usalo per prove e poi lo innestiamo.
"""
import asyncio, json, os, sys, wave
import websockets

from src.config import get_openai_api_key

REALTIME_MODEL = os.getenv("REALTIME_MODEL", "gpt-4o-realtime-preview")
WS_URL = f"wss://api.openai.com/v1/realtime?model={REALTIME_MODEL}"


async def run(api_key: str):
    async with websockets.connect(
        WS_URL,
        extra_headers={
            "Authorization": f"Bearer {api_key}",
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
    api_key = get_openai_api_key()
    if not api_key:
        print("⚠️ Set OPENAI_API_KEY first.")
        sys.exit(1)
    asyncio.run(run(api_key))
