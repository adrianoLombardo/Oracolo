# src/main.py

import os
import sys
import re
import time
import difflib
import argparse
import threading
import uuid
from pathlib import Path
from typing import Any
from collections import defaultdict

import asyncio
import numpy as np
import sounddevice as sd
import yaml
from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic import ValidationError
import logging

from src.config import Settings, get_openai_api_key
from src.filters import ProfanityFilter
from src.audio import AudioPreprocessor, record_until_silence, play_and_pulse
from src.lights import SacnLight, WledLight, color_from_text
from src.oracle import (
    transcribe,
    fast_transcribe,
    oracle_answer,
    synthesize,
    append_log,
    extract_summary,
    detect_language,
)
from src.domain import validate_question
from src.hotword import is_wake, matches_hotword_text, strip_hotword_prefix
from src.hotword import is_wake
from src.chat import ChatState
from src.conversation import ConversationManager, DialogState
from src.logging_utils import setup_logging
from src.language_session import update_language
from src.cli import _ensure_utf8_stdout, say, oracle_greeting, default_response
from src.audio_device import pick_device, debug_print_devices
from src.profile_utils import get_active_profile, make_domain_settings

logger = logging.getLogger(__name__)



# --------------------------- main ------------------------------------- #
def main() -> None:
    _ensure_utf8_stdout()


    session_id = uuid.uuid4().hex

    parser = argparse.ArgumentParser(description="Occhio Onniveggente · Oracolo")
    parser.add_argument("--autostart", action="store_true", help="Avvia direttamente senza prompt input()")
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Nasconde i log dalla console (vista conversazione pulita)",
    )
    parser.add_argument("--persona", type=str, default=None, help="Personalità iniziale")
    args = parser.parse_args()

    listener = setup_logging(
        Path("data/logs/oracolo.log"),
        session_id=session_id,
        console=not args.quiet,
    )

    say("Occhio Onniveggente · Oracolo ✨", role="system")

    load_dotenv()

    cfg_path = Path("settings.yaml")
    raw_settings = {}
    if cfg_path.exists():
        try:
            raw_settings = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
            SET = Settings.model_validate(raw_settings)
        except (ValidationError, yaml.YAMLError) as e:
            logger.warning("⚠️ Configurazione non valida: %s", e)
            logger.warning("Uso impostazioni di default.")
            raw_settings = {}
            SET = Settings()
    else:
        logger.warning("⚠️ settings.yaml non trovato, uso impostazioni di default.")
        raw_settings = {}
        SET = Settings()

    api_key = get_openai_api_key(SET)
    if not api_key:
        logger.error(
            "❌ È necessaria una API key OpenAI.\n"
            "   Imposta la variabile d'ambiente OPENAI_API_KEY"
            " oppure aggiungi openai.api_key a settings.yaml."
        )
        return

    client = AsyncOpenAI(api_key=api_key)

    session_profile_name, _ = get_active_profile(raw_settings)
    session_persona_name = args.persona or raw_settings.get("persona", {}).get("current") or getattr(getattr(SET, "persona", None), "current", "standard")

    DEBUG = bool(SET.debug) and (not args.quiet)
    tone = getattr(SET.chat, "tone", "informal")

    # Audio
    AUDIO_CONF = SET.audio
    AUDIO_SR = AUDIO_CONF.sample_rate
    INPUT_WAV = Path(AUDIO_CONF.input_wav)
    OUTPUT_WAV = Path(AUDIO_CONF.output_wav)
    PREPROC = AudioPreprocessor(
        AUDIO_SR,
        denoise=getattr(AUDIO_CONF, "denoise", False),
        echo_cancel=getattr(AUDIO_CONF, "echo_cancel", False),
    )
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
    CHAT_TURNS = int(getattr(getattr(SET, "chat", None), "max_turns", 10))
    CHAT_HISTORY_LIMIT = CHAT_TURNS * 2

    def _new_chat():
        return ChatState(
            max_turns=0,
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
        logger.warning("⚠️ lighting.mode non valido, uso WLED di default")
        light = WledLight(lighting_conf)

    # Wake config: default ON e lista di frasi
    WAKE_IT = ["ciao oracolo", "ehi oracolo", "salve oracolo", "ciao, oracolo"]
    WAKE_EN = ["hello oracle", "hey oracle", "hi oracle", "hello, oracle"]
    WAKE_ENABLED = True
    WAKE_SINGLE_TURN = False
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

    session_lang: str | None = None
    if LANG_PREF in ("it", "en"):
        session_lang = LANG_PREF
    
    # --------------------------- STATE MACHINE --------------------------- #
    dlg = ConversationManager(IDLE_TIMEOUT)
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
            prof_name, prof = get_active_profile(CURRENT_CFG, session_profile_name)
            persona_cfg = CURRENT_CFG.get("persona", {}) or {}
            persona_name = persona_cfg.get("current", session_persona_name)
            personas = persona_cfg.get("profiles", {}) or {}
            persona = personas.get(persona_name, {})
            if CHAT_ENABLED:
                chat = chat_histories[session_profile_name]
                dlg.chat = chat
                dlg.max_history = CHAT_HISTORY_LIMIT
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
                    logger.debug("⏳ Inattività: %02ds", max(remain, 0))
                    countdown_last = remain

            # timeout inattività globale
            if WAKE_ENABLED and dlg.timed_out(now):
                say("🌘 Torno al silenzio. Di' «ciao oracolo» per riattivarmi.")
                dlg.transition(DialogState.SLEEP)
                countdown_last = -1
                if not args.quiet:
                    logger.debug("")
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
                    preprocessor=PREPROC,
                )
                if not ok:
                    continue

                text = asyncio.run(
                    fast_transcribe(INPUT_WAV, client, STT_MODEL)
                )
                wake, lang = is_wake(text, WAKE_IT, WAKE_EN)
                if not wake:
                    text_it = asyncio.run(
                        fast_transcribe(
                            INPUT_WAV, client, STT_MODEL, lang_hint="it"
                        )
                    )
                    wake, lang = is_wake(text_it, WAKE_IT, WAKE_EN)
                    if wake:
                        text = text_it
                if not wake:
                    text_en = asyncio.run(
                        fast_transcribe(
                            INPUT_WAV, client, STT_MODEL, lang_hint="en"
                        )
                    )
                    wake, lang = is_wake(text_en, WAKE_IT, WAKE_EN)
                    if wake:
                        text = text_en
                if not wake:
                    if DEBUG:
                        say("…hotword non riconosciuta, continuo l'attesa.")
                    logging.info("Hotword non riconosciuta: %s", text)
                    continue
                session_lang = update_language(session_lang, lang, "")
                wake_lang = session_lang or lang or "it"

                # prova trascrizione forzando IT/EN per maggiore robustezza
                text_it = asyncio.run(
                    fast_transcribe(INPUT_WAV, client, STT_MODEL, lang_hint="it")
                )
                text_en = asyncio.run(
                    fast_transcribe(INPUT_WAV, client, STT_MODEL, lang_hint="en")
                )
                wake_it, lang_it = is_wake(text_it, WAKE_IT, WAKE_EN)
                wake_en, lang_en = is_wake(text_en, WAKE_IT, WAKE_EN)
                is_it = wake_it and lang_it == "it"
                is_en = wake_en and lang_en == "en"
                if session_lang in ("it", "en"):
                    text = text_en if session_lang == "en" else text_it
                else:
                    if wake_en:
                        text = text_en
                        session_lang = lang_en or "en"
                    elif wake_it:
                        text = text_it
                        session_lang = lang_it or "it"
                    else:
                        text = text_en or text_it
                        if text_en:
                            session_lang = "en"
                        elif text_it:
                            session_lang = "it"
                say(f"📝 Riconosciuto: {text}")
                logging.info(
                    "Hotword riconosciuta: %s (lang=%s)", text, session_lang
                )
                if not (wake_it or wake_en):
                    if DEBUG:
                        say("…hotword non riconosciuta, continuo l'attesa.")
                    logging.info("Hotword non riconosciuta: %s", text)
                    continue
                wake_lang = session_lang or (lang_en if wake_en and not wake_it else "it")

                dlg.transition(DialogState.AWAKE)
                continue

            # ---------------------- AWAKE: saluta ---------------------- #
            if dlg.state == DialogState.AWAKE:
                greet = oracle_greeting(wake_lang, tone)
                say(greet, role="assistant")
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
                    preprocessor=PREPROC,
                )
                if not ok:
                    continue
                dlg.refresh_deadline()
                q = asyncio.run(
                    transcribe(
                        INPUT_WAV,
                        client,
                        STT_MODEL,
                        lang_hint=session_lang,
                    )
                )
                qlang = detect_language(q or "")
                matched, remainder = strip_hotword_prefix(q, WAKE_IT + WAKE_EN)
                if matched:
                    q = remainder
                say(f"📝 Domanda: {q}")
                if not q:
                    continue
                low_q = q.lower()
                session_lang = update_language(session_lang, qlang, q)
                eff_lang = session_lang
                if PROF.contains_profanity(q):
                    if FILTER_MODE == "block":
                        say(default_response("profanity", eff_lang, tone))
                        continue
                    else:
                        q = PROF.mask(q)
                        say(default_response("filtered", eff_lang, tone) + q)
                m = re.search(r"cambia profilo in\s+(.+)", low_q)
                if m:
                    new_prof = m.group(1).strip().lower()
                    profiles = CURRENT_CFG.get("profiles", {})
                    target = None
                    for name in profiles:
                        if new_prof == name.lower():
                            target = name
                            break
                    if target is None:
                        matches = difflib.get_close_matches(new_prof, list(profiles.keys()), n=1, cutoff=0.6)
                        if matches:
                            target = matches[0]
                    if target:
                        CURRENT_CFG.setdefault("profile", {})["current"] = target
                        CURRENT_CFG.setdefault("domain", {})["profile"] = profiles.get(target, {}).get("topic") or target
                        try:
                            cfg_path.write_text(
                                yaml.safe_dump(CURRENT_CFG, allow_unicode=True),
                                encoding="utf-8",
                            )
                        except Exception:
                            pass
                        pending_answer = f"Profilo cambiato in {profiles.get(target, {}).get('label', target)}."
                    else:
                        pending_answer = "Profilo non trovato."
                    pending_full_answer = pending_answer
                    dlg.transition(DialogState.SPEAKING)
                    continue
                m = re.search(r"cambia personalit[aà] in\s+(.+)", low_q)
                if not m:
                    m = re.search(r"change persona to\s+(.+)", low_q)
                if m:
                    new_pers = m.group(1).strip().lower()
                    personas = CURRENT_CFG.get("persona", {}).get("profiles", {})
                    target = None
                    for name in personas:
                        if new_pers == name.lower():
                            target = name
                            break
                    if target is None:
                        matches = difflib.get_close_matches(new_pers, list(personas.keys()), n=1, cutoff=0.6)
                        if matches:
                            target = matches[0]
                    if target:
                        CURRENT_CFG.setdefault("persona", {})["current"] = target
                        try:
                            cfg_path.write_text(
                                yaml.safe_dump(CURRENT_CFG, allow_unicode=True),
                                encoding="utf-8",
                            )
                        except Exception:
                            pass
                        pending_answer = f"Personalità cambiata in {target}."
                        session_persona_name = target
                    else:
                        pending_answer = "Personalità non trovata."
                    pending_full_answer = pending_answer
                    dlg.transition(DialogState.SPEAKING)
                    continue
                pending_q = q
                pending_lang = eff_lang
                if CHAT_ENABLED and chat is not None:
                    dlg.push_user(q)
                    changed = chat.update_topic(q, client, EMB_MODEL)
                    if changed:
                        say("🔀 Cambio tema.")
                    pending_topic = chat.topic_text
                    pending_history = chat.messages_for_llm()
                    pending_history = dlg.messages

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

                settings_for_domain = make_domain_settings(SET, prof_name, prof)
                topic = getattr(getattr(settings_for_domain, "domain", {}), "profile", None)
                pending_topic = topic

                ok, context, clarify, reason, suggestion = validate_question(
                    pending_q,
                    pending_lang,
                    settings=settings_for_domain,
                    client=client,
                    docstore_path=effective_docstore,
                    top_k=effective_top_k,
                    embed_model=embed_model,
                    topic=topic,
                    history=pending_history,
                )
                if DEBUG:
                    say(f"[VAL] {reason}")
                if not ok:
                    if clarify:
                        pending_answer = (
                            "Domanda un po’ fuori ambito per questo profilo. Puoi riformularla in tema, oppure dire “cambia profilo”?"
                        )
                    else:

                        pending_answer = (
                            "Questa domanda è fuori dall’ambito selezionato. Per proseguire, resta sul tema o dì “cambia profilo”."
                        )

                        if suggestion:
                            pending_answer = (
                                f"Questa domanda è fuori dall'ambito {session_profile_name}. "
                                f"Vuoi passare a {suggestion}? (di' \"cambia profilo\" per confermare)"
                            )
                        else:
                            pending_answer = (
                                f"Questa domanda è fuori dall'ambito {session_profile_name}."
                            )

                    pending_full_answer = pending_answer
                    if CHAT_ENABLED and chat is not None:
                        dlg.push_assistant(pending_full_answer)
                    if dlg.turn_id != processing_turn:
                        continue
                    dlg.transition(DialogState.SPEAKING)
                    continue
                context_texts = [
                    item.get("text", "") for item in (context or []) if isinstance(item, dict)
                ]
                profile_hint = prof.get("system_hint", "")
                effective_system = ORACLE_SYSTEM
                if profile_hint:
                    effective_system = (
                        f"{effective_system}\n\n[Profilo: {prof.get('label', prof_name)}]\n{profile_hint}"
                    )
                if persona:
                    persona_prompt = f"Tono: {persona.get('tone', '')}. Stile: {persona.get('style', '')}."
                    effective_system = (
                        f"{effective_system}\n\n[Personalità: {persona_name}]\n{persona_prompt}"
                    )
                pending_answer, pending_sources = oracle_answer(
                    pending_q,
                    pending_lang,
                    client,
                    LLM_MODEL,
                    effective_system if STYLE_ENABLED else "",
                    tone=tone,
                    context=context_texts,
                    history=pending_history,
                    topic=topic,
                    policy_prompt=ORACLE_POLICY,
                    mode=ANSWER_MODE,
                )
                pending_full_answer = pending_answer
                if CHAT_ENABLED and chat is not None:
                    dlg.push_assistant(pending_full_answer)
                pending_answer = extract_summary(pending_full_answer)
                if dlg.turn_id != processing_turn:
                    continue
                dlg.transition(DialogState.SPEAKING)
                continue

            # ---------------------- SPEAKING: TTS risposta ---------------------- #
            if dlg.state == DialogState.SPEAKING:
                say(pending_answer, role="assistant")
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
                    session_id=session_id,
                    lang=pending_lang,
                    topic=pending_topic,
                    sources=pending_sources,
                )
                if pending_sources:
                    logger.info("📚 Fonti:")
                    for i, src in enumerate(pending_sources, 1):
                        title = src.get("title") or src.get("id", "")
                        sid = src.get("id", "")
                        score = src.get("score", 0.0)
                        snippet = src.get("text", "").replace("\n", " ")[:200]
                        logger.info("[%s] %s (%s, score=%.2f)", i, title, sid, score)
                        logger.info("    %s", snippet)
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
                        logger.info("")
                else:
                    dlg.transition(DialogState.LISTENING)
                continue

            # fallback
            dlg.transition(DialogState.SLEEP)
            countdown_last = -1
            if not args.quiet:
                logger.info("")

    finally:
        try:
            light.blackout()
            light.stop()
        except Exception:
            pass
        say("👁️  Arrivederci.", role="assistant")
        listener.stop()


if __name__ == "__main__":
    main()

