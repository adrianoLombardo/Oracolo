from __future__ import annotations

import asyncio
import base64
import json
import os
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf
import websockets
from dotenv import load_dotenv

from .config import Settings
from .filters import ProfanityFilter
from .audio import record_until_silence, play_and_pulse
from .lights import SacnLight, WledLight, color_from_text
from .main import pick_device, debug_print_devices


async def main() -> None:
    """Interfaccia in tempo reale con modello OpenAI via WebSocket."""
    print("Occhio Onniveggente · Oracolo Realtime ✨")

    load_dotenv()
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("⚠️ OPENAI_API_KEY non impostata.")
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

    recording_conf = SET.recording.model_dump()
    vad_conf = SET.vad.model_dump()
    lighting_conf = SET.lighting.model_dump()
    FILTER_MODE = SET.filter.mode
    PALETTES = {k: v.model_dump() for k, v in SET.palette_keywords.items()}
    LIGHT_MODE = SET.lighting.mode
    ORACLE_SYSTEM = SET.oracle_system
    TTS_VOICE = SET.openai.tts_voice

    PROF = ProfanityFilter(
        Path("data/filters/it_blacklist.txt"),
        Path("data/filters/en_blacklist.txt"),
    )

    if LIGHT_MODE == "sacn":
        light = SacnLight(lighting_conf)
    elif LIGHT_MODE == "wled":
        light = WledLight(lighting_conf)
    else:
        print("⚠️ lighting.mode non valido, uso WLED di default")
        light = WledLight(lighting_conf)

    model = os.environ.get("OPENAI_REALTIME_MODEL", "gpt-4o-realtime-preview-2024-12-17")
    url = f"wss://api.openai.com/v1/realtime?model={model}"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "OpenAI-Beta": "realtime=v1",
    }

    async with websockets.connect(url, extra_headers=headers) as ws:
        await ws.send(
            json.dumps(
                {
                    "type": "session.update",
                    "session": {"voice": TTS_VOICE, "instructions": ORACLE_SYSTEM},
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

                audio_bytes = INPUT_WAV.read_bytes()
                await ws.send(
                    json.dumps(
                        {
                            "type": "input_audio_buffer.append",
                            "audio": base64.b64encode(audio_bytes).decode("ascii"),
                        }
                    )
                )
                await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
                await ws.send(
                    json.dumps(
                        {
                            "type": "response.create",
                            "response": {"modalities": ["text", "audio"]},
                        }
                    )
                )

                out_text = ""
                out_audio = bytearray()
                while True:
                    msg = await ws.recv()
                    event = json.loads(msg)
                    t = event.get("type", "")
                    if t == "response.output_text.delta":
                        out_text += event.get("delta", "")
                    elif t == "response.output_audio.delta":
                        b64 = event.get("audio") or event.get("delta") or ""
                        out_audio.extend(base64.b64decode(b64))
                    elif t in ("response.completed", "response.stop", "response.done"):
                        break

                if not out_text:
                    continue

                if PROF.contains_profanity(out_text):
                    if FILTER_MODE == "block":
                        print("🚫 Linguaggio offensivo non ammesso. Riprova.")
                        continue
                    out_text = PROF.mask(out_text)
                    print("⚠️ Testo filtrato:", out_text)

                print(f"🔮 Oracolo: {out_text}")
                base = color_from_text(out_text, PALETTES)
                if hasattr(light, "set_base_rgb"):
                    light.set_base_rgb(base)
                    light.idle()

                audio_np = (
                    np.frombuffer(bytes(out_audio), dtype=np.int16).astype(np.float32)
                    / 32768.0
                )
                sf.write(OUTPUT_WAV.as_posix(), audio_np, AUDIO_SR)
                play_and_pulse(
                    OUTPUT_WAV,
                    light,
                    AUDIO_SR,
                    lighting_conf,
                    output_device_id=out_dev,
                )
        finally:
            light.blackout()
            light.stop()
            print("👁️  Arrivederci.")


if __name__ == "__main__":
    asyncio.run(main())

