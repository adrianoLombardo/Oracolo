# src/main.py

import os
import sys
import re
import time
import difflib
import unicodedata
import argparse
import threading
from pathlib import Path
from typing import Any, Iterable

import numpy as np
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
from src.chat import ChatState


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
    ORACLE_POLICY = getattr(SET, "oracle_policy", "")
    ANSWER_MODE = getattr(SET, "answer_mode", "detailed")

    # Chat state
    CHAT_ENABLED = bool(getattr(getattr(SET, "chat", None), "enabled", False))
    chat = ChatState(
        max_turns=int(getattr(getattr(SET, "chat", None), "max_turns", 10)),
        persist_jsonl=Path(
            getattr(
                getattr(SET, "chat", None),
                "persist_jsonl",
                "data/logs/chat_sessions.jsonl",
            )
        )
        if CHAT_ENABLED
        else None,
    )
    CHAT_RESET_ON_WAKE = bool(
        getattr(getattr(SET, "chat", None), "reset_on_hotword", True)
    )

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

    session_lang = "it"  # lingua bloccata per la sessione

    # --------------------------- STATE MACHINE --------------------------- #
        codex/improve-oracolo-chatbot-functionality-w0v4hl

        codex/improve-oracolo-chatbot-functionality-gnbrya
        main
    state = "SLEEP"
    active_deadline = 0.0
    is_processing = False
    wake_lang = "it"
    pending_q = ""
    pending_lang = ""
    pending_topic = None
    pending_history = None
    pending_answer = ""
    pending_sources: list[dict[str, str]] = []

    if not WAKE_ENABLED:
        state = "LISTENING"

    def _monitor_barge(evt: threading.Event, sr: int, thresh: float, device: Any | None) -> None:
        frame = int(sr * 0.03)
        try:
            with sd.InputStream(samplerate=sr, blocksize=frame, channels=1, dtype="float32", device=device) as stream:
                while not evt.is_set():
                    block, _ = stream.read(frame)
                    level = float(np.sqrt(np.mean(block[:, 0] ** 2)))
                    if level > thresh:
                        evt.set()
                        break
        except Exception:
            pass
        codex/improve-oracolo-chatbot-functionality-w0v4hl


    state = "SLEEP"      # SLEEP -> attende hotword. ACTIVE -> dialogo attivo
    active_deadline = 0  # timestamp (time.time()) di scadenza inattività
    processing_turn = False
        main
        main

    try:
        while True:
            if not args.autostart:
                try:
                    cmd = input("\nPremi INVIO per ascoltare (q per uscire)… ")
                    if cmd.strip().lower() == "q":
                        break
                except EOFError:
                    pass

            now = time.time()

            # timeout inattività globale
            if WAKE_ENABLED and state != "SLEEP" and now > active_deadline:
                say("🌘 Torno al silenzio. Di' «ciao oracolo» per riattivarmi.", quiet=args.quiet)
                state = "SLEEP"
                continue

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
                    continue
                text, lang = transcribe(INPUT_WAV, client, STT_MODEL, debug=DEBUG and (not args.quiet))
                say(f"📝 Riconosciuto: {text}", quiet=args.quiet)
                if lang in ("it", "en"):
                    session_lang = lang
                is_it = _match_hotword(text, WAKE_IT)
                is_en = _match_hotword(text, WAKE_EN)
                if not (is_it or is_en):
                    if DEBUG:
                        say("…hotword non riconosciuta, continuo l'attesa.", quiet=False)
                    continue
                wake_lang = "it" if is_it and not is_en else "en" if is_en and not is_it else (session_lang or "it")
                state = "AWAKE"
                continue

            # ---------------------- AWAKE: saluta ---------------------- #
            if state == "AWAKE":
                greet = oracle_greeting(wake_lang)
                print(f"🔮 Oracolo: {greet}", flush=True)
                synthesize(greet, OUTPUT_WAV, client, TTS_MODEL, TTS_VOICE)
                evt = threading.Event()
                mon = threading.Thread(
                    target=_monitor_barge,
                    args=(evt, AUDIO_SR, float(recording_conf.get("min_speech_level", 0.01)), in_dev),
                    daemon=True,
                )
                mon.start()
                play_and_pulse(
                    OUTPUT_WAV,
                    light,
                    AUDIO_SR,
                    lighting_conf,
                    output_device_id=out_dev,
                    duck_event=evt,
                )
                evt.set()
                mon.join()
                if CHAT_ENABLED and CHAT_RESET_ON_WAKE:
                    chat.reset()
                active_deadline = time.time() + IDLE_TIMEOUT
                session_lang = wake_lang
                state = "LISTENING"
                continue

        codex/improve-oracolo-chatbot-functionality-w0v4hl
            # ---------------------- LISTENING: attesa domanda ---------------------- #
            if state == "LISTENING":

        codex/improve-oracolo-chatbot-functionality-gnbrya
            # ---------------------- LISTENING: attesa domanda ---------------------- #
            if state == "LISTENING":

            # ---------------------- ACTIVE: dialogo attivo ---------------------- #
            # Se hotword è disattivata a config, restiamo sempre "attivi".
            if (not WAKE_ENABLED) or state == "ACTIVE":
                # timeout inattività → torna a SLEEP
                if WAKE_ENABLED and time.time() > active_deadline:
                    say("🌘 Torno al silenzio. Di' «ciao oracolo» per riattivarmi.", quiet=args.quiet)
                    state = "SLEEP"
                    continue

                if processing_turn:
                    # stiamo ancora elaborando il turno precedente
                    continue

                # ascolto una domanda
        main
        main
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
                    continue
                q, qlang = transcribe(INPUT_WAV, client, STT_MODEL, debug=DEBUG and (not args.quiet))
                say(f"📝 Domanda: {q}", quiet=args.quiet)
                if not q:
                    continue
        codex/improve-oracolo-chatbot-functionality-w0v4hl

        codex/improve-oracolo-chatbot-functionality-gnbrya
        main
                active_deadline = time.time() + IDLE_TIMEOUT
                if session_lang not in ("it", "en"):
                    session_lang = qlang if qlang in ("it", "en") else "it"
                eff_lang = session_lang
                low_q = q.lower()
                if qlang != session_lang:
                    if "inglese" in low_q or "english" in low_q:
                        session_lang = "en"
                        eff_lang = "en"
                    elif "italiano" in low_q or "italian" in low_q:
                        session_lang = "it"
                        eff_lang = "it"
       codex/improve-oracolo-chatbot-functionality-w0v4hl


                # rinnova timer appena rilevato parlato valido
                active_deadline = time.time() + IDLE_TIMEOUT

                eff_lang = qlang if qlang in ("it", "en") else last_lang

                # filtro input
        main
        main
                if PROF.contains_profanity(q):
                    if FILTER_MODE == "block":
                        say("🚫 Linguaggio offensivo/bestemmie non ammessi. Riformula in italiano o inglese.", quiet=args.quiet)
                        continue
                    else:
                        q = PROF.mask(q)
                        say("⚠️ Testo filtrato: " + q, quiet=args.quiet)
                pending_q = q
                pending_lang = eff_lang
                if CHAT_ENABLED:
                    chat.push_user(q)
                    changed = chat.update_topic(q, client, EMB_MODEL)
                    if changed:
                        say("🔀 Cambio tema.", quiet=args.quiet)
                    pending_topic = chat.topic_text
                    pending_history = chat.history
                else:
                    pending_topic = None
                    pending_history = None
                is_processing = True
                state = "THINKING"
                continue

            # ---------------------- THINKING: genera risposta ---------------------- #
            if state == "THINKING":
                try:
                    DOCSTORE_PATH = getattr(SET, "docstore_path", "DataBase/index.json")
                    TOPK = int(getattr(SET, "retrieval_top_k", 3))
                except Exception:
                    DOCSTORE_PATH = "DataBase/index.json"
                    TOPK = 3
        codex/improve-oracolo-chatbot-functionality-w0v4hl
                ok, context, clarify, reason = validate_question(
                    pending_q,

        codex/improve-oracolo-chatbot-functionality-gnbrya
                ok, context, clarify, reason = validate_question(
                    pending_q,


                ok, context, clarify = validate_question(
                    q,
        main
        main
                    client=client,
                    emb_model=EMB_MODEL,
                    docstore_path=DOCSTORE_PATH,
                    top_k=TOPK,
                    topic=pending_topic,
                    history=pending_history,
                )
        codex/improve-oracolo-chatbot-functionality-w0v4hl

        codex/improve-oracolo-chatbot-functionality-gnbrya
        main
                if DEBUG:
                    say(f"[VAL] {reason}", quiet=False)
                if not ok and clarify:
                    say("🤔 Puoi chiarire meglio la tua domanda?", quiet=args.quiet)
                    is_processing = False
                    state = "LISTENING"
                    continue
        codex/improve-oracolo-chatbot-functionality-w0v4hl



                if not ok and clarify:
                    say("🤔 Puoi chiarire meglio la tua domanda?", quiet=args.quiet)
                    processing_turn = False
                    continue

        main
        main
                if not ok:
                    pending_answer = "Domanda non pertinente"
                    if CHAT_ENABLED:
                        chat.push_assistant(pending_answer)
                else:
        codex/improve-oracolo-chatbot-functionality-w0v4hl
                    pending_answer, pending_sources = oracle_answer(
                        pending_q,
                        pending_lang,

        codex/improve-oracolo-chatbot-functionality-gnbrya
                    pending_answer, pending_sources = oracle_answer(
                        pending_q,
                        pending_lang,

                    if CHAT_ENABLED:
                        chat.push_user(q)
                        chat.update_topic(q, client, EMB_MODEL)
                        topic_txt = chat.topic_text
                    else:
                        topic_txt = None
                    processing_turn = True
                    a = oracle_answer(
                        q,
                        eff_lang,
        main
        main
                        client,
                        LLM_MODEL,
                        ORACLE_SYSTEM,
                        context=context,
        codex/improve-oracolo-chatbot-functionality-w0v4hl

        codex/improve-oracolo-chatbot-functionality-gnbrya
        main
                        history=pending_history,
                        topic=pending_topic,
                        policy_prompt=ORACLE_POLICY,
                        mode=ANSWER_MODE,
        codex/improve-oracolo-chatbot-functionality-w0v4hl


                        history=(chat.history if CHAT_ENABLED else None),
                        topic=topic_txt,
        main
        main
                    )
                    if CHAT_ENABLED:
                        chat.push_assistant(pending_answer)
                state = "SPEAKING"
                continue

            # ---------------------- SPEAKING: TTS risposta ---------------------- #
            if state == "SPEAKING":
                print(f"🔮 Oracolo: {pending_answer}", flush=True)
                if PROF.contains_profanity(pending_answer):
                    if FILTER_MODE == "block":
                        say("⚠️ La risposta conteneva termini non ammessi, riformulo…", quiet=args.quiet)
                        pending_answer = client.responses.create(
                            model=LLM_MODEL,
                            instructions=ORACLE_SYSTEM + " Evita qualsiasi offesa o blasfemia.",
                            input=[{"role": "user", "content": "Riformula in modo poetico e non offensivo:\n" + pending_answer}],
                        ).output_text.strip()
                    else:
                        pending_answer = PROF.mask(pending_answer)
                append_log(pending_q, pending_answer, LOG_PATH)
                base = color_from_text(pending_answer, {k: v for k, v in PALETTES.items()})
                if hasattr(light, "set_base_rgb"):
                    light.set_base_rgb(base)
                    light.idle()
                synthesize(pending_answer, OUTPUT_WAV, client, TTS_MODEL, TTS_VOICE)
                evt = threading.Event()
                mon = threading.Thread(
                    target=_monitor_barge,
                    args=(evt, AUDIO_SR, float(recording_conf.get("min_speech_level", 0.01)), in_dev),
                    daemon=True,
                )
                mon.start()
                play_and_pulse(
                    OUTPUT_WAV,
                    light,
                    AUDIO_SR,
                    lighting_conf,
                    output_device_id=out_dev,
                    duck_event=evt,
                )
                evt.set()
                mon.join()
                active_deadline = time.time() + IDLE_TIMEOUT
        codex/improve-oracolo-chatbot-functionality-w0v4hl
                is_processing = False

        codex/improve-oracolo-chatbot-functionality-gnbrya
                is_processing = False

                processing_turn = False

                # se modalità "single turn", torna subito a SLEEP
        main
        main
                if WAKE_ENABLED and WAKE_SINGLE_TURN:
                    state = "SLEEP"
                    say("🌘 Torno al silenzio. Di' «ciao oracolo» per riattivarmi.", quiet=args.quiet)
                else:
                    state = "LISTENING"
                continue

            # fallback
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

