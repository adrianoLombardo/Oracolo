# src/main.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import sys
import time
import argparse
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

# ---------- Console UTF-8 safe (Windows) ----------
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
SPARK = "✨"
try:
    "✓".encode(sys.stdout.encoding or "utf-8")
except Exception:
    SPARK = "*"


def pick_device(spec: Any, kind: str) -> Any:
    """Ritorna indice device valido per 'kind' ('input'|'output') o None per default OS."""
    try:
        devices = sd.query_devices()
    except Exception:
        return None

    def _valid(info: dict) -> bool:
        ch_key = "max_input_channels" if kind == "input" else "max_output_channels"
        return info.get(ch_key, 0) > 0

    if spec in (None, ""):
        try:
            idx = sd.default.device[0 if kind == "input" else 1]
            if idx is not None and _valid(sd.query_devices(idx)):
                return None  # usa default di sistema
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

    # fallback: primo device valido
    for i, info in enumerate(devices):
        if _valid(info):
            return i
    return None


def debug_print_devices() -> None:
    try:
        devices = sd.query_devices()
    except Exception as e:
        print(f"⚠️ Unable to query audio devices: {e}", flush=True)
        return
    header = f"{'Idx':>3}  {'Device Name':<40}  {'In/Out'}"
    print(header, flush=True)
    print("-" * len(header), flush=True)
    for idx, info in enumerate(devices):
        name = info.get("name", "")
        in_ch = info.get("max_input_channels", 0)
        out_ch = info.get("max_output_channels", 0)
        print(f"{idx:>3}  {name:<40}  {in_ch}/{out_ch}", flush=True)


def main() -> None:
    # ---- argomenti: autostart (niente prompt), quiet (meno stampe) ----
    ap = argparse.ArgumentParser(description="Occhio Onniveggente · Oracolo")
    ap.add_argument("--autostart", action="store_true",
                    help="Esegue il loop senza richiedere INVIO (per UI/headless).")
    ap.add_argument("--quiet", action="store_true",
                    help="Riduce il logging a console.")
    args = ap.parse_args()

    load_dotenv()
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    cfg_path = Path("settings.yaml")
    if cfg_path.exists():
        try:
            SET = Settings.model_validate_yaml(cfg_path)
        except ValidationError as e:
            print("⚠️ Configurazione non valida:", e, flush=True)
            print("Uso impostazioni di default.", flush=True)
            SET = Settings()
    else:
        print("⚠️ settings.yaml non trovato, uso impostazioni di default.", flush=True)
        SET = Settings()

    DEBUG = SET.debug and not args.quiet

    # Audio config
    AUDIO_CONF = SET.audio
    AUDIO_SR = AUDIO_CONF.sample_rate
    INPUT_WAV = Path(AUDIO_CONF.input_wav)
    OUTPUT_WAV = Path(AUDIO_CONF.output_wav)

    in_spec = AUDIO_CONF.input_device
    out_spec = AUDIO_CONF.output_device
    in_dev = pick_device(in_spec, "input")
    out_dev = pick_device(out_spec, "output")
    try:
        sd.default.device = (in_dev, out_dev)
    except Exception:
        pass
    if DEBUG:
        print(f"🎛️  Audio devices -> input={in_dev}  output={out_dev}", flush=True)
        debug_print_devices()

    # OpenAI models / prompt
    STT_MODEL = SET.openai.stt_model
    LLM_MODEL = SET.openai.llm_model
    TTS_MODEL = SET.openai.tts_model
    TTS_VOICE = SET.openai.tts_voice
    ORACLE_SYSTEM = SET.oracle_system

    # Filtro / palette / luci
    FILTER_MODE = SET.filter.mode
    PALETTES = {k: v.model_dump() for k, v in SET.palette_keywords.items()}
    LIGHT_MODE = SET.lighting.mode
    LOG_PATH = Path("data/logs/dialoghi.csv")

    history: list[tuple[str, str]] = []

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
        print("⚠️ lighting.mode non valido, uso WLED di default", flush=True)
        light = WledLight(lighting_conf)

    # Messaggi di avvio
    if args.autostart or not sys.stdin.isatty():
        print(f"Occhio Onniveggente · Oracolo {SPARK}  (modalità UI: nessun prompt)", flush=True)
        autostart = True
    else:
        print(f"Occhio Onniveggente · Oracolo {SPARK}", flush=True)
        autostart = False

    try:
        while True:
            if not autostart:
                try:
                    cmd = input("\nPremi INVIO per fare una domanda (q per uscire)… ")
                except (EOFError, KeyboardInterrupt):
                    cmd = "q"
                if cmd.strip().lower() == "q":
                    break
            else:
                # in modalità UI/headless non chiediamo nulla:
                # un breve respiro per non ciclare a vuoto
                time.sleep(0.05)

            # 1) Registrazione (VAD / dorme in silenzio)
            ok = record_until_silence(
                INPUT_WAV,
                AUDIO_SR,
                vad_conf,
                recording_conf,
                debug=DEBUG,
                input_device_id=in_dev,
            )
            if not ok:
                print("⚠️ Registrazione audio non riuscita, riprovo tra 2s…", flush=True)
                time.sleep(2)
                continue

            # 2) Trascrizione + lingua
            q, lang = transcribe(INPUT_WAV, client, STT_MODEL, debug=DEBUG)
            if not q:
                continue

            if q.strip().lower() == "!reset":
                history.clear()
                print("🔄 Conversazione azzerata.", flush=True)
                continue

            # 3) Filtro input utente
            if PROF.contains_profanity(q):
                if FILTER_MODE == "block":
                    print("🚫 Linguaggio offensivo/bestemmie non ammessi. Riformula in italiano o inglese.", flush=True)
                    continue
                else:
                    q = PROF.mask(q)
                    print("⚠️ Testo filtrato:", q, flush=True)

            # 4) Risposta oracolare (stessa lingua)
            print("✨ Interrogo l’Oracolo…", flush=True)
            a = oracle_answer(q, lang or "it", client, LLM_MODEL, ORACLE_SYSTEM, history)

            # 5) Filtro output oracolo
            if PROF.contains_profanity(a):
                if FILTER_MODE == "block":
                    print("⚠️ La risposta conteneva termini non ammessi, riformulo…", flush=True)
                    a = client.responses.create(
                        model=LLM_MODEL,
                        instructions=ORACLE_SYSTEM + " Evita qualsiasi offesa o blasfemia.",
                        input=[{"role": "user", "content": "Riformula in modo poetico e non offensivo:\n" + a}],
                    ).output_text.strip()
                else:
                    a = PROF.mask(a)

            history.append((q, a))

            # 6) Log
            try:
                append_log(q, a, LOG_PATH)
            except Exception:
                pass

            # 7) Colori base da testo
            base = color_from_text(a, PALETTES)
            if hasattr(light, "set_base_rgb"):
                light.set_base_rgb(base)
                light.idle()

            # 8) TTS → WAV
            print("🎧 Sintesi vocale…", flush=True)
            synthesize(a, OUTPUT_WAV, client, TTS_MODEL, TTS_VOICE)

            # 9) Riproduzione + pulsazioni luci
            play_and_pulse(
                OUTPUT_WAV,
                light,
                AUDIO_SR,
                lighting_conf,
                output_device_id=out_dev,
            )
    finally:
        try:
            light.blackout()
            light.stop()
        except Exception:
            pass
        print("👁️  Arrivederci.", flush=True)


if __name__ == "__main__":
    main()
