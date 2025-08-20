import os
from pathlib import Path
from typing import Any

import sounddevice as sd
import yaml
from dotenv import load_dotenv
from openai import OpenAI

from src.filters import ProfanityFilter
from src.audio import record_until_silence, play_and_pulse
from src.lights import SacnLight, WledLight, color_from_text
from src.oracle import transcribe, oracle_answer, synthesize, append_log


def pick_device(spec: Any, kind: str) -> Any:
    devices = sd.query_devices()

    def _valid(info: dict) -> bool:
        ch_key = "max_input_channels" if kind == "input" else "max_output_channels"
        return info.get(ch_key, 0) > 0

    if spec in (None, ""):
        try:
            idx = sd.default.device[0 if kind == "input" else 1]
            if idx is not None and _valid(sd.query_devices(idx)):
                return None
        except Exception:
            pass
    else:
        if isinstance(spec, int) or str(spec).isdigit():
            idx = int(spec)
            if 0 <= idx < len(devices) and _valid(devices[idx]):
                return idx
        spec_str = str(spec).lower()
        for i, info in enumerate(devices):
            if spec_str in info.get("name", "").lower() and _valid(info):
                return i

    for i, info in enumerate(devices):
        if _valid(info):
            return i
    return None


def main() -> None:
    print("Occhio Onniveggente · Oracolo ✨")

    load_dotenv()
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    SET = yaml.safe_load(Path("settings.yaml").read_text(encoding="utf-8"))
    DEBUG = bool(SET.get("debug", False))

    AUDIO_CONF = SET["audio"]
    AUDIO_SR = int(AUDIO_CONF["sample_rate"])
    INPUT_WAV = Path(AUDIO_CONF["input_wav"])
    OUTPUT_WAV = Path(AUDIO_CONF["output_wav"])
    in_spec = AUDIO_CONF.get("input_device")
    out_spec = AUDIO_CONF.get("output_device")
    in_dev = pick_device(in_spec, "input")
    out_dev = pick_device(out_spec, "output")
    sd.default.device = (in_dev, out_dev)
    if DEBUG:
        print(sd.query_devices())

    STT_MODEL = SET["openai"]["stt_model"]
    LLM_MODEL = SET["openai"]["llm_model"]
    TTS_MODEL = SET["openai"]["tts_model"]
    TTS_VOICE = SET["openai"]["tts_voice"]
    ORACLE_SYSTEM = SET["oracle_system"]

    FILTER_MODE = (SET.get("filter", {}) or {}).get("mode", "block").lower()
    PALETTES = SET.get("palette_keywords", {})
    LIGHT_MODE = SET["lighting"]["mode"]
    LOG_PATH = Path("data/logs/dialoghi.csv")

    PROF = ProfanityFilter(
        Path("data/filters/it_blacklist.txt"),
        Path("data/filters/en_blacklist.txt"),
    )

    recording_conf = SET.get("recording", {})
    vad_conf = SET.get("vad", {})
    lighting_conf = SET["lighting"]

    if LIGHT_MODE == "sacn":
        light = SacnLight(lighting_conf)
    elif LIGHT_MODE == "wled":
        light = WledLight(lighting_conf)
    else:
        print("⚠️ lighting.mode non valido, uso WLED di default")
        light = WledLight(lighting_conf)

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

            q, lang = transcribe(INPUT_WAV, client, STT_MODEL, debug=DEBUG)
            if not q:
                continue

            if PROF.contains_profanity(q):
                if FILTER_MODE == "block":
                    print(
                        "🚫 Linguaggio offensivo/bestemmie non ammessi. Riformula in italiano o inglese."
                    )
                    continue
                else:
                    q = PROF.mask(q)
                    print("⚠️ Testo filtrato:", q)

            a = oracle_answer(q, lang or "it", client, LLM_MODEL, ORACLE_SYSTEM)

            if PROF.contains_profanity(a):
                if FILTER_MODE == "block":
                    print("⚠️ La risposta conteneva termini non ammessi, riformulo…")
                    a = client.responses.create(
                        model=LLM_MODEL,
                        instructions=ORACLE_SYSTEM + " Evita qualsiasi offesa o blasfemia.",
                        input=[{"role": "user", "content": "Riformula in modo poetico e non offensivo:\n" + a}],
                    ).output_text.strip()
                else:
                    a = PROF.mask(a)

            append_log(q, a, LOG_PATH)

            base = color_from_text(a, PALETTES)
            if hasattr(light, "set_base_rgb"):
                light.set_base_rgb(base)
                light.idle()

            synthesize(a, OUTPUT_WAV, client, TTS_MODEL, TTS_VOICE)
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
    main()

