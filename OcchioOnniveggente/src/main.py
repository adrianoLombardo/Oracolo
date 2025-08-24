# src/main.py

import os
import sys
import re
import time
import difflib
import unicodedata
import argparse
import threading
import uuid
from pathlib import Path
from typing import Any, Iterable
from collections import defaultdict

import numpy as np
import sounddevice as sd
import yaml
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import ValidationError

from src.config import Settings
from src.filters import ProfanityFilter
from src.audio import record_until_silence, play_and_pulse
from src.lights import SacnLight, WledLight, color_from_text
from src.oracle import (
    transcribe,
    fast_transcribe,
    oracle_answer,
    synthesize,
    append_log,
    extract_summary,
)
from src.domain import validate_question
from src.chat import ChatState
from src.dialogue import DialogueManager, DialogState
from src.logging_utils import setup_logging


# --------------------------- console helpers --------------------------- #
def _ensure_utf8_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass


def say(msg: str) -> None:
    """Print a message intended for the user conversation."""
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
        return "Hello, I am the Oracle. Ask your question."
    return "Ciao, sono l'Oracolo. Fai pure la tua domanda?"


def get_active_profile(SETTINGS):
    if isinstance(SETTINGS, dict):
        dom = SETTINGS.get("domain", {}) or {}
        prof_name = dom.get("profile", "museo")
        profiles = dom.get("profiles", {}) or {}
        prof = profiles.get(prof_name, {})
    else:
        dom = getattr(SETTINGS, "domain", None)
        prof_name = getattr(dom, "profile", "museo") if dom else "museo"
        profiles = getattr(dom, "profiles", {}) if dom else {}
        prof = profiles.get(prof_name, {})
    return prof_name, prof


def make_domain_settings(base_settings, prof_name):
    if isinstance(base_settings, dict):
        new_s = dict(base_settings)
        dom = dict(new_s.get("domain") or {})
        dom["enabled"] = True
        dom["profile"] = prof_name
        new_s["domain"] = dom
        return new_s
    else:
        try:
            base_settings.domain.enabled = True
            base_settings.domain.profile = prof_name
        except Exception:
            pass
        return base_settings


# --------------------------- main ------------------------------------- #
def main() -> None:
    _ensure_utf8_stdout()


    session_id = uuid.uuid4().hex
    listener = setup_logging(Path("data/logs/oracolo.log"), session_id=session_id)

    parser = argparse.ArgumentParser(description="Occhio Onniveggente · Oracolo")
    parser.add_argument("--autostart", action="store_true", help="Avvia direttamente senza prompt input()")
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Nasconde i log dalla console (vista conversazione pulita)",
    )
    args = parser.parse_args()

    listener = setup_logging(Path("data/logs/oracolo.log"), console=not args.quiet)

    say("Occhio Onniveggente · Oracolo ✨")

    load_dotenv()

    cfg_path = Path("settings.yaml")
    raw_settings = {}
    if cfg_path.exists():
        try:
            raw_settings = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
            SET = Settings.model_validate(raw_settings)
        except (ValidationError, yaml.YAMLError) as e:
            print("⚠️ Configurazione non valida:", e)
            print("Uso impostazioni di default.")
            raw_settings = {}
            SET = Settings()
    else:
        print("⚠️ settings.yaml non trovato, uso impostazioni di default.")
        raw_settings = {}
        SET = Settings()

    api_key = os.environ.get("OPENAI_API_KEY") or getattr(SET.openai, "api_key", "")
    client = OpenAI(api_key=api_key) if api_key else OpenAI()

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

    STYLE_ENABLED = os.getenv("ORACOLO_STYLE", "poetic").lower() != "plain"
    ANSWER_MODE = os.getenv("ORACOLO_ANSWER_MODE", ANSWER_MODE)
    LANG_PREF = os.getenv("ORACOLO_LANG", "auto").lower()

    # Chat state
    CHAT_ENABLED = bool(getattr(getattr(SET, "chat", None), "enabled", False))

    def _new_chat():
        return ChatState(
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

    chat_histories: defaultdict[str, ChatState] = defaultdict(_new_chat)
    chat: ChatState | None = None
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

    is_tts_playing = threading.Event()

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
    WAKE_SINGLE_TURN = False
    IDLE_TIMEOUT = 50.0  # secondi di inattività prima di tornare a SLEEP

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

    session_lang = "it"
    if LANG_PREF in ("it", "en"):
        session_lang = LANG_PREF
    
    # --------------------------- STATE MACHINE --------------------------- #
    dlg = DialogueManager(IDLE_TIMEOUT)
    wake_lang = "it"
    pending_q = ""
    pending_lang = ""
    pending_topic = None
    pending_history = None
    pending_answer = ""
    pending_full_answer = ""
    pending_sources: list[dict[str, str]] = []
    processing_turn = 0
    countdown_last = -1

    if not WAKE_ENABLED:
        dlg.state = DialogState.LISTENING

    def _monitor_barge(
        evt: threading.Event,
        sr: int,
        thresh: float,
        device: Any | None,
        skip_ms: float = 300.0,
        tts_event: threading.Event | None = None,
    ) -> None:
        frame = int(sr * 0.03)
        hot_frames = 0
        try:
            with sd.InputStream(
                samplerate=sr,
                blocksize=frame,
                channels=1,
                dtype="float32",
                device=device,
            ) as stream:
                if skip_ms > 0:
                    time.sleep(skip_ms / 1000.0)
                while not evt.is_set():
                    block, _ = stream.read(frame)
                    level = float(np.sqrt(np.mean(block[:, 0] ** 2)))
                    eff_thresh = thresh * 3 if tts_event is not None and tts_event.is_set() else thresh
                    if level > eff_thresh:
                        hot_frames += 1
                        if hot_frames >= 10:
                            evt.set()
                            break
                    else:
                        hot_frames = 0
        except Exception:
            pass

    try:
        while True:
            try:
                CURRENT_CFG = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
            except Exception:
                CURRENT_CFG = {}
            prof_name, prof = get_active_profile(CURRENT_CFG)
            if CHAT_ENABLED:
                chat = chat_histories[prof_name]
            else:
                chat = None
            if not args.autostart:
                try:
                    cmd = input("\nPremi INVIO per ascoltare (q per uscire)… ")
                    dlg.refresh_deadline()
                    if cmd.strip().lower() == "q":
                        break
                except EOFError:
                    pass

            now = time.time()

            if dlg.state != DialogState.SLEEP:
                remain = int(dlg.active_deadline - now)
                if remain != countdown_last and not args.quiet:
                    print(f"\r⏳ Inattività: {max(remain,0):02d}s", end="", flush=True)
                    countdown_last = remain

            # timeout inattività globale
            if WAKE_ENABLED and dlg.timed_out(now):
                say("🌘 Torno al silenzio. Di' «ciao oracolo» per riattivarmi.")
                dlg.transition(DialogState.SLEEP)
                countdown_last = -1
                if not args.quiet:
                    print()
                continue

            # ---------------------- SLEEP: attendo hotword ---------------------- #
            if WAKE_ENABLED and dlg.state == DialogState.SLEEP:
                if dlg.is_processing:
                    continue
                say("💤 In ascolto della parola chiave…  IT: «ciao oracolo» | EN: «hello oracle»")
                ok = record_until_silence(
                    INPUT_WAV,
                    AUDIO_SR,
                    vad_conf,
                    recording_conf,
                    debug=DEBUG and (not args.quiet),
                    input_device_id=in_dev,
                    tts_playing=is_tts_playing,
                )
                if not ok:
                    continue
                # prova trascrizione forzando IT/EN per maggiore robustezza
                text_it = fast_transcribe(INPUT_WAV, client, STT_MODEL, lang_hint="it")
                text_en = fast_transcribe(INPUT_WAV, client, STT_MODEL, lang_hint="en")
                if _match_hotword(text_it, WAKE_IT):
                    text, lang = text_it, "it"
                elif _match_hotword(text_en, WAKE_EN):
                    text, lang = text_en, "en"
                else:
                    text = text_it or text_en
                    lang = "it" if text_it else "en" if text_en else ""
                say(f"📝 Riconosciuto: {text}")
                if lang in ("it", "en"):
                    session_lang = lang
                is_it = _match_hotword(text, WAKE_IT)
                is_en = _match_hotword(text, WAKE_EN)
                if not (is_it or is_en):
                    if DEBUG:
                        say("…hotword non riconosciuta, continuo l'attesa.")
                    continue
                wake_lang = "it" if is_it and not is_en else "en" if is_en and not is_it else (session_lang or "it")
                dlg.transition(DialogState.AWAKE)
                continue

            # ---------------------- AWAKE: saluta ---------------------- #
            if dlg.state == DialogState.AWAKE:
                greet = oracle_greeting(wake_lang)
                print(f"🔮 Oracolo: {greet}", flush=True)
                synthesize(greet, OUTPUT_WAV, client, TTS_MODEL, TTS_VOICE)
                evt = threading.Event()
                mon = threading.Thread(
                    target=_monitor_barge,
                    args=(evt, AUDIO_SR, AUDIO_CONF.barge_rms_threshold, in_dev),
                    kwargs={"tts_event": is_tts_playing},
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
                    tts_event=is_tts_playing,
                )
                evt.set()
                mon.join()
                time.sleep(recording_conf.get("hold_off_after_tts_ms", 500) / 1000.0)
                if CHAT_ENABLED and CHAT_RESET_ON_WAKE and chat is not None:
                    chat.reset()
                dlg.refresh_deadline()
                session_lang = wake_lang
                dlg.transition(DialogState.LISTENING)
                continue

            # ---------------------- LISTENING: attesa domanda ---------------------- #
            if dlg.state == DialogState.LISTENING:
                if dlg.is_processing:
                    continue
                say(
                    "🎤 Parla pure (VAD energia, max %.1fs)…" % (SET.vad.max_ms / 1000.0),
                )
                ok = record_until_silence(
                    INPUT_WAV,
                    AUDIO_SR,
                    vad_conf,
                    recording_conf,
                    debug=DEBUG and (not args.quiet),
                    input_device_id=in_dev,
                    tts_playing=is_tts_playing,
                )
                if not ok:
                    continue
                dlg.refresh_deadline()
                q, qlang = transcribe(
                    INPUT_WAV,
                    client,
                    STT_MODEL,
                    debug=DEBUG and (not args.quiet),
                    lang_hint=session_lang,
                )
                say(f"📝 Domanda: {q}")
                if not q:
                    continue
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
                if PROF.contains_profanity(q):
                    if FILTER_MODE == "block":
                        say("🚫 Linguaggio offensivo/bestemmie non ammessi. Riformula in italiano o inglese.")
                        continue
                    else:
                        q = PROF.mask(q)
                        say("⚠️ Testo filtrato: " + q)
                pending_q = q
                pending_lang = eff_lang
                if CHAT_ENABLED and chat is not None:
                    chat.push_user(q)
                    changed = chat.update_topic(q, client, EMB_MODEL)
                    if changed:
                        say("🔀 Cambio tema.")
                    pending_topic = chat.topic_text
                    pending_history = chat.history
                else:
                    pending_topic = None
                    pending_history = None
                processing_turn = dlg.start_processing()
                dlg.transition(DialogState.THINKING)
                continue

            # ---------------------- THINKING: genera risposta ---------------------- #
            if dlg.state == DialogState.THINKING:
                embed_model = getattr(SET.openai, "embed_model", None)
                effective_docstore = prof.get("docstore_path") or getattr(
                    SET, "docstore_path", "DataBase/index.json"
                )
                effective_top_k = int(
                    prof.get("retrieval_top_k") or getattr(SET, "retrieval_top_k", 3)
                )
                settings_for_domain = make_domain_settings(SET, prof_name)
                ok, context, clarify, reason, suggestion = validate_question(
                    pending_q,
                    pending_lang,
                    settings=settings_for_domain,
                    client=client,
                    docstore_path=effective_docstore,
                    top_k=effective_top_k,
                    embed_model=embed_model,
                    topic=prof_name,
                    history=pending_history,
                )
                if DEBUG:
                    say(f"[VAL] {reason}")
                if not ok:
                    if clarify:
                        pending_answer = (
                            "La domanda non è chiarissima per questo contesto: puoi riformularla brevemente?"
                        )
                    else:
                        pending_answer = (
                            f"Tema non pertinente rispetto al profilo «{prof.get('label', prof_name)}»."
                        )
                    pending_full_answer = pending_answer
                    if CHAT_ENABLED and chat is not None:
                        chat.push_assistant(pending_full_answer)
                    if dlg.turn_id != processing_turn:
                        continue
                    dlg.transition(DialogState.SPEAKING)
                    continue
                context_texts = [
                    item.get("text", "") for item in (context or []) if isinstance(item, dict)
                ]
                profile_hint = prof.get("system_hint", "")
                if profile_hint:
                    effective_system = (
                        f"{ORACLE_SYSTEM}\n\n[Profilo: {prof.get('label', prof_name)}]\n{profile_hint}"
                    )
                else:
                    effective_system = ORACLE_SYSTEM
                pending_answer, pending_sources = oracle_answer(
                    pending_q,
                    pending_lang,
                    client,
                    LLM_MODEL,
                    effective_system if STYLE_ENABLED else "",
                    context=context_texts,
                    history=pending_history,
                    topic=prof_name,
                    policy_prompt=ORACLE_POLICY,
                    mode=ANSWER_MODE,
                )
                pending_full_answer = pending_answer
                if CHAT_ENABLED and chat is not None:
                    chat.push_assistant(pending_full_answer)
                pending_answer = extract_summary(pending_full_answer)
                if dlg.turn_id != processing_turn:
                    continue
                dlg.transition(DialogState.SPEAKING)
                continue

            # ---------------------- SPEAKING: TTS risposta ---------------------- #
            if dlg.state == DialogState.SPEAKING:
                print(f"🔮 Oracolo: {pending_answer}", flush=True)
                if PROF.contains_profanity(pending_answer):
                    if FILTER_MODE == "block":
                        say("⚠️ La risposta conteneva termini non ammessi, riformulo…")
                        pending_answer = client.responses.create(
                            model=LLM_MODEL,
                            instructions=ORACLE_SYSTEM + " Evita qualsiasi offesa o blasfemia.",
                            input=[{"role": "user", "content": "Riformula in modo poetico e non offensivo:\n" + pending_answer}],
                        ).output_text.strip()
                    else:
                        pending_answer = PROF.mask(pending_answer)
                append_log(
                    pending_q,
                    pending_full_answer or pending_answer,
                    LOG_PATH,
                    lang=pending_lang,
                    topic=pending_topic,
                    sources=pending_sources,
                )
                if pending_sources:
                    print("📚 Fonti:")
                    for i, src in enumerate(pending_sources, 1):
                        title = src.get("title") or src.get("id", "")
                        sid = src.get("id", "")
                        score = src.get("score", 0.0)
                        snippet = src.get("text", "").replace("\n", " ")[:200]
                        print(f"[{i}] {title} ({sid}, score={score:.2f})")
                        print(f"    {snippet}")
                base = color_from_text(pending_answer, {k: v for k, v in PALETTES.items()})
                if hasattr(light, "set_base_rgb"):
                    light.set_base_rgb(base)
                    light.idle()
                synthesize(pending_answer, OUTPUT_WAV, client, TTS_MODEL, TTS_VOICE)
                evt = threading.Event()
                mon = threading.Thread(
                    target=_monitor_barge,
                    args=(evt, AUDIO_SR, AUDIO_CONF.barge_rms_threshold, in_dev),
                    kwargs={"tts_event": is_tts_playing},
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
                    tts_event=is_tts_playing,
                )
                interrupted = evt.is_set()
                evt.set()
                mon.join()
                time.sleep(recording_conf.get("hold_off_after_tts_ms", 500) / 1000.0)
                dlg.end_processing()
                processing_turn = dlg.turn_id
                if interrupted:
                    dlg.transition(DialogState.INTERRUPTED)
                    say("⚠️ Interrotto.")
                    dlg.transition(DialogState.LISTENING)
                    continue
                dlg.refresh_deadline()
                if WAKE_ENABLED and WAKE_SINGLE_TURN:
                    dlg.transition(DialogState.SLEEP)
                    say("🌘 Torno al silenzio. Di' «ciao oracolo» per riattivarmi.")
                    countdown_last = -1
                    if not args.quiet:
                        print()
                else:
                    dlg.transition(DialogState.LISTENING)
                continue

            # fallback
            dlg.transition(DialogState.SLEEP)
            countdown_last = -1
            if not args.quiet:
                print()

    finally:
        try:
            light.blackout()
            light.stop()
        except Exception:
            pass
        print("👁️  Arrivederci.")
        listener.stop()


if __name__ == "__main__":
    main()

