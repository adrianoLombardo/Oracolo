# src/main.py

import os
import sys
import re
import time
import difflib
import unicodedata
import argparse
from pathlib import Path
from typing import Any, Iterable

import sounddevice as sd
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import ValidationError

from src.config import Settings
from src.filters import ProfanityFilter
from src.audio import record_until_silence, play_and_pulse
from src.lights import SacnLight, WledLight, color_from_text
from src.oracle import transcribe, oracle_answer, synthesize, append_log
from src.domain import validate_question


# --------------------------- console helpers --------------------------- #
def _ensure_utf8_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass


def say(msg: str, quiet: bool = False) -> None:
    if quiet:
        return
    print(msg, flush=True)


# --------------------------- audio device pick ------------------------- #
def pick_device(spec: Any, kind: str) -> Any:
    devices = sd.query_devices()

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
        if isinstance(spec, int) or (isinstance(spec, str) and spec.isdigit()):
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


# --------------------------- hotword (testuale) ------------------------ #
def _strip_accents(s: str) -> str:
    nf = unicodedata.normalize("NFD", s or "")
    return "".join(ch for ch in nf if not unicodedata.combining(ch))


def _normalize_lang(s: str) -> str:
    return _strip_accents(s).lower()


def _phrase_to_regex_tokens(phrase: str) -> re.Pattern:
    """
    "ciao oracolo" -> regex tollerante a punteggiatura/spazi:
    r"\bciao[\W_]*oracolo\b"
    """
    phrase = _normalize_lang(phrase)
    tokens = re.split(r"\s+", phrase.strip())
    parts = [re.escape(t) for t in tokens if t]
    if not parts:
        parts = [re.escape(phrase.strip())]
    pattern = r"\b" + r"[\W_]*".join(parts) + r"\b"
    return re.compile(pattern, flags=re.IGNORECASE)


def _match_hotword(text: str, phrases: Iterable[str]) -> bool:
    """
    Match robusto:
    1) regex tollerante (spazi/punteggiatura/maiuscole/accenti)
    2) fallback fuzzy: consente piccoli errori (es. “oràcolo”, “oraccolo”)
    """
    norm = _normalize_lang(text or "")

    # 1) regex tollerante
    for p in phrases or []:
        if _phrase_to_regex_tokens(p).search(norm):
            return True

    # 2) fuzzy: confronta stringhe senza simboli
    letters = re.sub(r"[\W_]+", "", norm)
    if not letters:
        return False

    for p in phrases or []:
        cand = re.sub(r"[\W_]+", "", _normalize_lang(p))
        if not cand:
            continue
        if cand in letters:
            return True
        ratio = difflib.SequenceMatcher(None, letters, cand).ratio()
        if ratio >= 0.86:
            return True

    return False


def oracle_greeting(lang: str) -> str:
    if (lang or "").lower().startswith("en"):
        return "I am awake. Ask, and I will answer from the shore of starlight."
    return "Sono desto. Chiedi, e risponderò dalla riva della luce."


# --------------------------- main ------------------------------------- #
def main() -> None:
    _ensure_utf8_stdout()

    parser = argparse.ArgumentParser(description="Occhio Onniveggente · Oracolo")
    parser.add_argument("--autostart", action="store_true", help="Avvia direttamente senza prompt input()")
    parser.add_argument("--quiet", action="store_true", help="Riduce i log a schermo")
    args = parser.parse_args()

    say("Occhio Onniveggente · Oracolo ✨", quiet=args.quiet)

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

    DEBUG = bool(SET.debug) and (not args.quiet)

    # Audio
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

    # OpenAI models
    STT_MODEL = SET.openai.stt_model
    LLM_MODEL = SET.openai.llm_model
    TTS_MODEL = SET.openai.tts_model
    TTS_VOICE = SET.openai.tts_voice
    EMB_MODEL = getattr(SET.openai, "embed_model", "text-embedding-3-small")
    ORACLE_SYSTEM = SET.oracle_system

    # Filters / palettes / lights
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

    # Luci
    if LIGHT_MODE == "sacn":
        light = SacnLight(lighting_conf)
    elif LIGHT_MODE == "wled":
        light = WledLight(lighting_conf)
    else:
        print("⚠️ lighting.mode non valido, uso WLED di default")
        light = WledLight(lighting_conf)

    # Wake config: default ON e lista di frasi
    WAKE_IT = ["ciao oracolo", "ehi oracolo", "salve oracolo", "ciao, oracolo"]
    WAKE_EN = ["hello oracle", "hey oracle", "hi oracle", "hello, oracle"]
    WAKE_ENABLED = True
    WAKE_SINGLE_TURN = True
    IDLE_TIMEOUT = 60.0  # secondi di inattività prima di tornare a SLEEP

    try:
        if getattr(SET, "wake", None) is not None:
            WAKE_ENABLED = bool(SET.wake.enabled)
            WAKE_SINGLE_TURN = bool(SET.wake.single_turn)
            if SET.wake.it_phrases:
                WAKE_IT = list(SET.wake.it_phrases)
            if SET.wake.en_phrases:
                WAKE_EN = list(SET.wake.en_phrases)
            if getattr(SET.wake, "idle_timeout", None):
                try:
                    IDLE_TIMEOUT = float(SET.wake.idle_timeout)
                except Exception:
                    pass
    except Exception:
        # se non c'è la sezione wake, rimaniamo con i default sopra
        pass

    last_lang = "it"  # lingua da usare quando STT è incerto

    # --------------------------- STATE MACHINE --------------------------- #
    state = "SLEEP"      # SLEEP -> attende hotword. ACTIVE -> dialogo attivo
    active_deadline = 0  # timestamp (time.time()) di scadenza inattività

    try:
        while True:
            if not args.autostart:
                try:
                    cmd = input("\nPremi INVIO per ascoltare (q per uscire)… ")
                    if cmd.strip().lower() == "q":
                        break
                except EOFError:
                    pass  # lanciato da UI con stdin chiuso

            # ---------------------- SLEEP: attendo hotword ---------------------- #
            if WAKE_ENABLED and state == "SLEEP":
                say("💤 In ascolto della parola chiave…  IT: «ciao oracolo» | EN: «hello oracle»", quiet=args.quiet)

                ok = record_until_silence(
                    INPUT_WAV,
                    AUDIO_SR,
                    vad_conf,
                    recording_conf,
                    debug=DEBUG and (not args.quiet),
                    input_device_id=in_dev,
                )
                if not ok:
                    # niente parlato → continua a dormire
                    continue

                text, lang = transcribe(INPUT_WAV, client, STT_MODEL, debug=DEBUG and (not args.quiet))
                say(f"📝 Riconosciuto: {text}", quiet=args.quiet)

                if lang in ("it", "en"):
                    last_lang = lang

                is_it = _match_hotword(text, WAKE_IT)
                is_en = _match_hotword(text, WAKE_EN)
                if not (is_it or is_en):
                    # non è hotword → resta a dormire
                    if DEBUG:
                        say("…hotword non riconosciuta, continuo l'attesa.", quiet=False)
                    continue

                # hotword OK → saluta e passa ad ACTIVE
                wake_lang = "it" if is_it and not is_en else "en" if is_en and not is_it else (last_lang or "it")
                greet = oracle_greeting(wake_lang)
                # ▼▼ AGGIUNTA: log del saluto in viewport (anche se --quiet)
                print(f"🔮 Oracolo: {greet}", flush=True)
                synthesize(greet, OUTPUT_WAV, client, TTS_MODEL, TTS_VOICE)
                play_and_pulse(
                    OUTPUT_WAV,
                    light,
                    AUDIO_SR,
                    lighting_conf,
                    output_device_id=out_dev,
                )
                state = "ACTIVE"
                active_deadline = time.time() + IDLE_TIMEOUT
                continue

            # ---------------------- ACTIVE: dialogo attivo ---------------------- #
            # Se hotword è disattivata a config, restiamo sempre "attivi".
            if (not WAKE_ENABLED) or state == "ACTIVE":
                # timeout inattività → torna a SLEEP
                if WAKE_ENABLED and time.time() > active_deadline:
                    say("🌘 Torno al silenzio. Di' «ciao oracolo» per riattivarmi.", quiet=args.quiet)
                    state = "SLEEP"
                    continue

                # ascolto una domanda
                say(
                    "🎤 Parla pure (VAD energia, max %.1fs)…" % (SET.vad.max_ms / 1000.0),
                    quiet=args.quiet,
                )
                ok = record_until_silence(
                    INPUT_WAV,
                    AUDIO_SR,
                    vad_conf,
                    recording_conf,
                    debug=DEBUG and (not args.quiet),
                    input_device_id=in_dev,
                )

                if not ok:
                    # nessuna traccia valida → controlla timeout e riprova
                    if WAKE_ENABLED and time.time() > active_deadline:
                        say("🌘 Torno al silenzio. Di' «ciao oracolo» per riattivarmi.", quiet=args.quiet)
                        state = "SLEEP"
                    continue

                q, qlang = transcribe(INPUT_WAV, client, STT_MODEL, debug=DEBUG and (not args.quiet))
                say(f"📝 Domanda: {q}", quiet=args.quiet)
                if not q:
                    if WAKE_ENABLED and time.time() > active_deadline:
                        say("🌘 Torno al silenzio. Di' «ciao oracolo» per riattivarmi.", quiet=args.quiet)
                        state = "SLEEP"
                    continue

                eff_lang = qlang if qlang in ("it", "en") else last_lang

                # filtro input
                if PROF.contains_profanity(q):
                    if FILTER_MODE == "block":
                        say("🚫 Linguaggio offensivo/bestemmie non ammessi. Riformula in italiano o inglese.", quiet=args.quiet)
                        # non estendere il timer se l'input è rifiutato
                        continue
                    else:
                        q = PROF.mask(q)
                        say("⚠️ Testo filtrato: " + q, quiet=args.quiet)

                # risposta oracolare
                try:
                    DOCSTORE_PATH = getattr(SET, "docstore_path", "DataBase/index.json")
                    TOPK = int(getattr(SET, "retrieval_top_k", 3))
                except Exception:
                    DOCSTORE_PATH = "DataBase/index.json"
                    TOPK = 3

                ok, context = validate_question(
                    q,
                    client=client,
                    emb_model=EMB_MODEL,
                    docstore_path=DOCSTORE_PATH,
                    top_k=TOPK,
                )

                if not ok:
                    a = "Domanda non pertinente"
                else:
                    a = oracle_answer(
                        q,
                        eff_lang,
                        client,
                        LLM_MODEL,
                        ORACLE_SYSTEM,
                        context=context,
                    )

                # ▼▼ AGGIUNTA: log della risposta in viewport (anche se --quiet)
                print(f"🔮 Oracolo: {a}", flush=True)

                # filtro output
                if PROF.contains_profanity(a):
                    if FILTER_MODE == "block":
                        say("⚠️ La risposta conteneva termini non ammessi, riformulo…", quiet=args.quiet)
                        a = client.responses.create(
                            model=LLM_MODEL,
                            instructions=ORACLE_SYSTEM + " Evita qualsiasi offesa o blasfemia.",
                            input=[{"role": "user", "content": "Riformula in modo poetico e non offensivo:\n" + a}],
                        ).output_text.strip()
                    else:
                        a = PROF.mask(a)

                append_log(q, a, LOG_PATH)

                # luce di base dal contenuto
                base = color_from_text(a, {k: v for k, v in PALETTES.items()})
                if hasattr(light, "set_base_rgb"):
                    light.set_base_rgb(base)
                    light.idle()

                # TTS + playback con pulse
                synthesize(a, OUTPUT_WAV, client, TTS_MODEL, TTS_VOICE)
                play_and_pulse(
                    OUTPUT_WAV,
                    light,
                    AUDIO_SR,
                    lighting_conf,
                    output_device_id=out_dev,
                )

                # estendi timeout perché c'è stata attività
                active_deadline = time.time() + IDLE_TIMEOUT

                # se modalità "single turn", torna subito a SLEEP
                if WAKE_ENABLED and WAKE_SINGLE_TURN:
                    state = "SLEEP"
                    say("🌘 Torno al silenzio. Di' «ciao oracolo» per riattivarmi.", quiet=args.quiet)
                    continue

                # altrimenti resta ACTIVE e continua ad ascoltare
                continue

            # Fallback: se per qualche motivo non rientra nei rami, vai a SLEEP
            state = "SLEEP"

    finally:
        try:
            light.blackout()
            light.stop()
        except Exception:
            pass
        print("👁️  Arrivederci.")


if __name__ == "__main__":
    main()

