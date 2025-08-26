# src/realtime_ws.py
"""Minimal WebSocket client for OpenAI's realtime API.

This helper is primarily used for manual experiments.  The function now pulls a
shared :class:`ConversationManager` and an asynchronous queue from the service
container so that messages received from the websocket are funnelled through the
same backend used by the HTTP gateway.
"""

import asyncio, json, os, sys
import websockets

from src.config import get_openai_api_key
from .service_container import container

REALTIME_MODEL = os.getenv("REALTIME_MODEL", "gpt-4o-realtime-preview")
WS_URL = f"wss://api.openai.com/v1/realtime?model={REALTIME_MODEL}"


async def run(
    api_key: str,
    *,
    conv=None,
    queue: asyncio.Queue | None = None,
):
    conv = conv or container.conversation_manager
    queue = queue or container.message_queue
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
        await ws.send(
            json.dumps(
                {
                    "type": "response.create",
                    "response": {"instructions": "Say hello from realtime."},
                }
            )
        )
        while True:
            msg = await ws.recv()
            await queue.put(msg)
            print(msg)


if __name__ == "__main__":
    api_key = get_openai_api_key()
    if not api_key:
        print("⚠️ Set OPENAI_API_KEY first.")
        sys.exit(1)
    asyncio.run(run(api_key))
