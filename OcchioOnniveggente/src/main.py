import os
from pathlib import Path
from typing import Any

import sounddevice as sd
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import ValidationError

from src.config import Settings
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


def debug_print_devices() -> None:
    try:
        devices = sd.query_devices()
    except Exception as e:
        print(f"⚠️ Unable to query audio devices: {e}")
        return
    header = f"{'Idx':>3}  {'Device Name':<40}  {'In/Out'}"
    print(header)
    print("-" * len(header))
    for idx, info in enumerate(devices):
        name = info.get("name", "")
        in_ch = info.get("max_input_channels", 0)
        out_ch = info.get("max_output_channels", 0)
        print(f"{idx:>3}  {name:<40}  {in_ch}/{out_ch}")


def main() -> None:
    print("Occhio Onniveggente · Oracolo ✨")

    load_dotenv()
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    cfg_path = Path("settings.yaml")
    if cfg_path.exists():
        try:
            SET = Settings.model_validate_yaml(cfg_path)
        except ValidationError as e:
            print("⚠️ Configurazione non valida:", e)
            print("Uso impostazioni di default.")
            SET = Settings()
    else:
        print("⚠️ settings.yaml non trovato, uso impostazioni di default.")
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

    STT_MODEL = SET.openai.stt_model
    LLM_MODEL = SET.openai.llm_model
    TTS_MODEL = SET.openai.tts_model
    TTS_VOICE = SET.openai.tts_voice
    ORACLE_SYSTEM = SET.oracle_system
    ALLOWED_TOPICS = SET.allowed_topics
    REJECT_IT = SET.reject_answer_it
    REJECT_EN = SET.reject_answer_en

    FILTER_MODE = SET.filter.mode
    PALETTES = {k: v.model_dump() for k, v in SET.palette_keywords.items()}
    LIGHT_MODE = SET.lighting.mode
    LOG_PATH = Path("data/logs/dialoghi.csv")

    PROF = ProfanityFilter(
        Path("data/filters/it_blacklist.txt"),
        Path("data/filters/en_blacklist.txt"),
    )

    recording_conf = SET.recording.model_dump()
    vad_conf = SET.vad.model_dump()
    lighting_conf = SET.lighting.model_dump()

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

            if not any(t.lower() in q.lower() for t in ALLOWED_TOPICS):
                print(REJECT_EN if (lang or "it") == "en" else REJECT_IT)
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

            a = oracle_answer(
                q,
                lang or "it",
                client,
                LLM_MODEL,
                ORACLE_SYSTEM,
                ALLOWED_TOPICS,
                REJECT_IT,
                REJECT_EN,
            )

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

