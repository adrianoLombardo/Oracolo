"""Esecuzione dell'Oracolo in modalità realtime.

Questo modulo utilizza la API Realtime di OpenAI via WebSocket per
inviare l'audio dell'utente e ricevere la risposta sintetizzata in
tempo reale. Per l'I/O audio e le luci vengono riutilizzate le funzioni
`record_until_silence`, `play_and_pulse` e le classi `SacnLight` e
`WledLight` già presenti nel progetto.

L'implementazione è volutamente semplice e pensata per funzionare in
ambienti offline durante i test: non effettua alcuna ottimizzazione e
accumula l'audio ricevuto prima di riprodurlo.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv
import numpy as np
import sounddevice as sd
import websockets

from .audio import load_audio_as_float, play_and_pulse, record_until_silence
from .config import Settings
from .lights import SacnLight, WledLight, color_from_text
from .main import debug_print_devices, pick_device


async def _run() -> None:
    print("Occhio Onniveggente · Realtime Oracolo ✨")

    load_dotenv()
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("⚠️ OPENAI_API_KEY mancante nel file .env")
        return

    try:
        SET = Settings.model_validate_yaml(Path("settings.yaml"))
    except Exception:
        SET = Settings()

    DEBUG = SET.debug

    AUDIO_CONF = SET.audio
    AUDIO_SR = AUDIO_CONF.sample_rate
    INPUT_WAV = Path(AUDIO_CONF.input_wav)
    OUTPUT_WAV = Path(AUDIO_CONF.output_wav)
    in_spec = AUDIO_CONF.input_device
    out_spec = AUDIO_CONF.output_device
    in_dev = pick_device(in_spec, "input")
    out_dev = pick_device(out_spec, "output")
    sd.default.device = (in_dev, out_dev)
    if DEBUG:
        debug_print_devices()

    PALETTES: Dict[str, Dict[str, Any]] = {
        k: v.model_dump() for k, v in SET.palette_keywords.items()
    }
    LIGHT_MODE = SET.lighting.mode

    recording_conf = SET.recording.model_dump()
    vad_conf = SET.vad.model_dump()
    lighting_conf = SET.lighting.model_dump()

    if LIGHT_MODE == "sacn":
        light = SacnLight(lighting_conf)
    elif LIGHT_MODE == "wled":
        light = WledLight(lighting_conf)
    else:
        light = WledLight(lighting_conf)

    model = "gpt-4o-realtime-preview"
    voice = SET.openai.tts_voice
    instructions = SET.oracle_system

    headers = {
        "Authorization": f"Bearer {api_key}",
        "OpenAI-Beta": "realtime=v1",
    }

    uri = f"wss://api.openai.com/v1/realtime?model={model}"

    async with websockets.connect(uri, extra_headers=headers) as ws:
        # Aggiorna la sessione con le istruzioni e la voce scelta.
        await ws.send(
            json.dumps(
                {
                    "type": "session.update",
                    "session": {"voice": voice, "instructions": instructions},
                }
            )
        )

        try:
            while True:
                cmd = input("\nPremi INVIO per fare una domanda (q per uscire)… ")
                if cmd.strip().lower() == "q":
                    break

                ok = record_until_silence(
                    INPUT_WAV,
                    AUDIO_SR,
                    vad_conf,
                    recording_conf,
                    debug=DEBUG,
                    input_device_id=in_dev,
                )
                if not ok:
                    continue

                # Converte l'audio WAV in PCM16 e lo codifica in base64
                y, _ = load_audio_as_float(INPUT_WAV, AUDIO_SR)
                pcm16 = (np.clip(y, -1, 1) * 32768).astype(np.int16)
                b64 = base64.b64encode(pcm16.tobytes()).decode()

                await ws.send(
                    json.dumps({"type": "input_audio_buffer.append", "audio": b64})
                )
                await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
                await ws.send(json.dumps({"type": "response.create"}))

                final_text = ""
                audio_buf = bytearray()

                while True:
                    msg = json.loads(await ws.recv())
                    ev_type = msg.get("type")
                    if ev_type == "response.output_text.delta":
                        final_text += msg.get("delta", "")
                    elif ev_type == "response.audio.delta":
                        audio_buf.extend(base64.b64decode(msg.get("delta", "")))
                    elif ev_type == "response.completed":
                        # Salva l'audio ricevuto e riproducilo con pulsazioni
                        OUTPUT_WAV.parent.mkdir(parents=True, exist_ok=True)
                        with open(OUTPUT_WAV, "wb") as f:
                            f.write(audio_buf)
                        if final_text:
                            print(f"🔮 Oracolo: {final_text}")
                            base = color_from_text(final_text, PALETTES)
                            if hasattr(light, "set_base_rgb"):
                                light.set_base_rgb(base)
                                light.idle()
                        play_and_pulse(
                            OUTPUT_WAV,
                            light,
                            AUDIO_SR,
                            lighting_conf,
                            output_device_id=out_dev,
                        )
                        await ws.send(json.dumps({"type": "input_audio_buffer.clear"}))
                        break
                    elif ev_type == "error":
                        print("\u26a0\ufe0f Errore:", msg)
                        await ws.send(json.dumps({"type": "input_audio_buffer.clear"}))
                        break
        finally:
            light.blackout()
            light.stop()
            print("\ud83d\udc41\ufe0f  Arrivederci.")


def main() -> None:
    """Entry point sincrono."""
    asyncio.run(_run())


if __name__ == "__main__":
    main()

