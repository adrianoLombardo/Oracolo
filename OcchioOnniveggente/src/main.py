# src/main.py
import os, time, threading, collections, re
from pathlib import Path
from datetime import datetime

import numpy as np
import yaml

import sounddevice as sd
import soundfile as sf
from dotenv import load_dotenv
from openai import OpenAI

# luci
import sacn
from requests import exceptions as req_exc
from src.wled_client import WLED

# lingua + filtro
from langdetect import detect, DetectorFactory
from src.filters import ProfanityFilter
DetectorFactory.seed = 42  # stabilizza langdetect

# === DEBUG SWITCH ===
DEBUG = True  # metti False in mostra

# ==== BOOTSTRAP ====
load_dotenv()
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

SET = yaml.safe_load(Path("settings.yaml").read_text(encoding="utf-8"))

# ---- audio / file ----
AUDIO_SR = int(SET["audio"]["sample_rate"])
ASK_SECONDS = int(SET["audio"]["ask_seconds"])  # usato nel fallback
INPUT_WAV = Path(SET["audio"]["input_wav"])
OUTPUT_WAV = Path(SET["audio"]["output_wav"])

# ---- modelli ----
STT_MODEL     = SET["openai"]["stt_model"]
LLM_MODEL     = SET["openai"]["llm_model"]
TTS_MODEL     = SET["openai"]["tts_model"]
TTS_VOICE     = SET["openai"]["tts_voice"]
ORACLE_SYSTEM = SET["oracle_system"]

# ---- filtro / palette / luci ----
FILTER_MODE = (SET.get("filter", {}) or {}).get("mode", "block").lower()  # "block" | "mask"
PALETTES  = SET.get("palette_keywords", {})
LIGHT_MODE = SET["lighting"]["mode"]
LOG_PATH = Path("data/logs/dialoghi.csv")

PROF = ProfanityFilter(
    Path("data/filters/it_blacklist.txt"),
    Path("data/filters/en_blacklist.txt"),
)

# =====================================================================
#                         DRIVER LUCI
# =====================================================================
class SacnLight:
    def __init__(self, conf):
        sacn_conf = conf["sacn"]
        self.universe = int(sacn_conf["universe"])
        self.dest_ip = sacn_conf["destination_ip"]
        self.rgb = sacn_conf["rgb_channels"]
        self.idle_level = int(sacn_conf["idle_level"])
        self.peak_level = int(sacn_conf["peak_level"])
        self.sender = sacn.sACNsender(); self.sender.start()
        self.sender.activate_output(self.universe)
        self.sender[self.universe].multicast = False
        self.sender[self.universe].destination = self.dest_ip
        self.frame = [0]*512
        self.idle()

    def set_rgb(self, r, g, b):
        r = int(np.clip(r, 0, 255))
        g = int(np.clip(g, 0, 255))
        b = int(np.clip(b, 0, 255))
        self.frame[self.rgb[0]-1] = r
        self.frame[self.rgb[1]-1] = g
        self.frame[self.rgb[2]-1] = b
        self.sender[self.universe].dmx_data = tuple(self.frame)

    def idle(self):
        self.set_rgb(self.idle_level, self.idle_level, self.idle_level)

    def blackout(self):
        self.set_rgb(0, 0, 0)

    def stop(self):
        try: self.sender.stop()
        except Exception: pass

class WledLight:
    def __init__(self, conf):
        host = conf["wled"]["host"]
        self.w = WLED(host)
        self.base_rgb = (180, 180, 200)
        self.w.set_color(*self.base_rgb, brightness=40)

    def set_base_rgb(self, rgb):
        self.base_rgb = tuple(int(x) for x in rgb)

    def pulse(self, level):
        try:
            self.w.pulse_by_level(level, base_rgb=self.base_rgb)
        except req_exc.RequestException:
            pass

    def idle(self):
        self.w.set_color(*self.base_rgb, brightness=30)

    def blackout(self):
        self.w.set_color(0,0,0,brightness=0)

    def stop(self): pass

# =====================================================================
#                         UTILS
# =====================================================================
def append_log(q: str, a: str):
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    def clean(s): return s.replace('"', "'")
    line = f'"{ts}","{clean(q)}","{clean(a)}"\n'
    if not LOG_PATH.exists():
        LOG_PATH.write_text('"timestamp","question","answer"\n', encoding="utf-8")
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(line)

def color_from_text(text: str):
    t = text.lower()
    for kw, cfg in PALETTES.items():
        if kw in t:
            return tuple(cfg["rgb"])
    return (180, 180, 200)

def load_audio_as_float(path: Path, target_sr: int) -> tuple[np.ndarray, int]:
    # WAV: via soundfile (nessun warning)
    if path.suffix.lower() != ".mp3":
        y, sr = sf.read(path.as_posix(), dtype='float32', always_2d=False)
        if y.ndim > 1: y = y.mean(axis=1)
        return y, sr

    # MP3 fallback (lazy import)
    try:
        from pydub import AudioSegment
    except Exception:
        raise RuntimeError("File MP3 ma pydub/ffmpeg non disponibili. Usa WAV oppure installa ffmpeg.")
    audio = AudioSegment.from_mp3(path.as_posix())
    audio = audio.set_channels(1).set_frame_rate(target_sr)
    y = np.array(audio.get_array_of_samples()).astype(np.float32) / 32768.0
    return y, target_sr

# =====================================================================
#                   VAD SEMPLICE (attende fine frase)
# =====================================================================
def record_wav(path: Path, seconds: int, sr: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n🎤 Registra (max {seconds}s)…")
    audio = sd.rec(int(seconds*sr), samplerate=sr, channels=1, dtype='float32')
    sd.wait()
    sf.write(path.as_posix(), audio, sr)
    print("✅ Fatto.")

def record_until_silence(path: Path) -> bool:
    """
    VAD semplice a energia:
      - Ascolta a SR = AUDIO_SR
      - Parte quando rileva parlato per ~START_MS consecutivi
      - Termina quando rileva ~END_MS di silenzio consecutivi
      - Pre-roll per non tagliare le prime sillabe
      - Soglie adattive sul rumore di fondo
    Ritorna True se ha scritto un file nuovo; False altrimenti (e non lascia file vecchi).
    """
    # Elimina un eventuale file vecchio per evitare trascrizioni "fantasma"
    try:
        if path.exists():
            path.unlink()
    except Exception:
        pass

    FRAME_MS   = 30    # 10/20/30 ms
    START_MS   = 150   # parlato continuo per partire (ridotto)
    END_MS     = 800   # silenzio continuo per fermarsi (aumentato)
    MAX_MS     = 15000 # stop di sicurezza
    PREROLL_MS = 300   # audio pre-start
    NOISE_WIN_MS = 800

    assert FRAME_MS in (10,20,30)
    frame_samples = int(AUDIO_SR * FRAME_MS / 1000)

    print(f"\n🎤 Parla pure (VAD energia, max {MAX_MS/1000:.1f}s)…")

    # buffer per pre-roll e stima rumore
    pre_frames = PREROLL_MS // FRAME_MS
    preroll = collections.deque(maxlen=pre_frames)
    noise_frames_needed = max(1, NOISE_WIN_MS // FRAME_MS)
    noise_rms_history = collections.deque(maxlen=noise_frames_needed)

    started = False
    speech_streak = 0
    silence_streak = 0
    total_ms = 0
    recorded_blocks = []  # liste di np.float32

    def rms(block_f32: np.ndarray) -> float:
        if block_f32.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(block_f32**2)))

    with sd.InputStream(samplerate=AUDIO_SR, blocksize=frame_samples,
                        channels=1, dtype='float32') as stream:
        while total_ms < MAX_MS:
            audio_block, _ = stream.read(frame_samples)  # float32 (N,1)
            block = audio_block[:,0]
            total_ms += FRAME_MS

            # stima rumore (prima dello start)
            if not started and len(noise_rms_history) < noise_frames_needed:
                noise_rms_history.append(rms(block))

            # soglie adattive (più permissive)
            noise_floor = np.median(noise_rms_history) if noise_rms_history else 0.003
            start_thresh = max(noise_floor * 1.8, 0.006)
            end_thresh   = max(noise_floor * 1.3, 0.0035)

            level = rms(block)
            if DEBUG and total_ms % 300 == 0 and not started:
                print(f"   [DBG] level={level:.4f} start_th={start_thresh:.4f} noise={noise_floor:.4f}")

            if not started:
                preroll.append(block.copy())
                if level >= start_thresh:
                    speech_streak += FRAME_MS
                    if speech_streak >= START_MS:
                        started = True
                        if len(preroll):
                            recorded_blocks.extend([b.copy() for b in preroll])
                        recorded_blocks.append(block.copy())
                        silence_streak = 0
                else:
                    speech_streak = 0
            else:
                recorded_blocks.append(block.copy())
                if level < end_thresh:
                    silence_streak += FRAME_MS
                    if silence_streak >= END_MS:
                        break
                else:
                    silence_streak = 0

    if not recorded_blocks:
        print("⚠️ Non ho rilevato parlato. Riprova.")
        # Fallback opzionale: registrazione temporizzata (evita turni a vuoto)
        # Commenta le prossime 3 righe se preferisci saltare il turno.
        print("↻ Avvio fallback: registrazione temporizzata.")
        record_wav(path, seconds=ASK_SECONDS, sr=AUDIO_SR)
        return True

    y = np.concatenate(recorded_blocks, axis=0).astype(np.float32)
    sf.write(path.as_posix(), y, AUDIO_SR)
    dur = len(y) / AUDIO_SR
    print(f"✅ Registrazione completata ({dur:.2f}s).")
    return True

# =====================================================================
#                 TRASCRIZIONE: IT/EN con scelta robusta
# =====================================================================
_IT_SW = {
    "di","e","che","il","la","un","una","non","per","con","come","sono","sei","è","siamo","siete","sono",
    "io","tu","lui","lei","noi","voi","loro","questo","questa","quello","quella","qui","lì","dove","quando",
    "perché","come","cosa","tutto","anche","ma","se","nel","nella","dei","delle","degli","agli","alle","allo",
    "fare","andare","venire","dire","vedere","può","posso","devo","voglio","grazie","ciao"
}
_EN_SW = {
    "the","and","to","of","in","that","it","is","you","i","we","they","this","these","those","for","with",
    "on","at","as","but","if","not","are","be","was","were","have","has","do","does","did","what","when",
    "where","why","how","all","also","can","could","should","would","hello","hi","thanks","please"
}

def _score_lang(text: str, lang: str) -> float:
    if not text:
        return 0.0
    toks = re.findall(r"[a-zàèéìòóù]+", text.lower())
    if not toks:
        return 0.0
    sw = _IT_SW if lang == "it" else _EN_SW
    hits = sum(1 for t in toks if t in sw)
    coverage = hits / max(len(toks), 1)            # 0..1
    try:
        ld = detect(text)
        bonus = 0.5 if (ld == lang) else 0.0
    except Exception:
        bonus = 0.0
    length_bonus = min(len(toks), 30) / 30 * 0.2   # fino a +0.2
    score = coverage + bonus + length_bonus
    if DEBUG:
        short = (text[:80] + "…") if len(text) > 80 else text
        print(f"   [DBG] {lang.upper()} score={score:.3f} text='{short}'")
    return score

def transcribe(path: Path) -> tuple[str, str]:
    """
    Doppio passaggio robusto:
      A) prova 'auto' (senza language) come hint
      B) SEMPRE forzata IT e EN, poi sceglie punteggio migliore
    Ritorna (testo, 'it'|'en') oppure ("","") se nulla di valido.
    """
    # A) auto (hint)
    try:
        with open(path, "rb") as f:
            print("🧠 Trascrizione (auto)…")
            tx_auto = client.audio.transcriptions.create(
                model=STT_MODEL,
                file=f,
                prompt="Language is either Italian or English. Ignore any other language."
            )
        text_auto = (tx_auto.text or "").strip()
    except Exception:
        text_auto = ""

    # B1) forza IT
    with open(path, "rb") as f_it:
        print("↻ Trascrizione forzata IT…")
        tx_it = client.audio.transcriptions.create(
            model=STT_MODEL,
            file=f_it,
            language="it",
            prompt="Lingua: italiano. Dominio: arte, luce, mare, destino."
        )
    text_it = (tx_it.text or "").strip()

    # B2) forza EN
    with open(path, "rb") as f_en:
        print("↻ Trascrizione forzata EN…")
        tx_en = client.audio.transcriptions.create(
            model=STT_MODEL,
            file=f_en,
            language="en",
            prompt="Language: English. Domain: art, light, sea, destiny."
        )
    text_en = (tx_en.text or "").strip()

    # Punteggi
    s_it = _score_lang(text_it, "it")
    s_en = _score_lang(text_en, "en")

    # Boost se l'auto concorda
    try:
        auto_lang = detect(text_auto) if text_auto else None
    except Exception:
        auto_lang = None
    if auto_lang == "it":
        s_it += 0.05
    elif auto_lang == "en":
        s_en += 0.05

    if s_it == 0 and s_en == 0:
        print("⚠️ Per favore parla in italiano o inglese.")
        return "", ""
    if s_it >= s_en:
        print("🌐 Lingua rilevata: IT")
        return text_it, "it"
    else:
        print("🌐 Lingua rilevata: EN")
        return text_en, "en"

# =====================================================================
#                      ORACOLO + TTS + LUCI
# =====================================================================
def oracle_answer(question: str, lang_hint: str) -> str:
    """
    Genera la risposta 'oracolare' nella stessa lingua dell'utente.
    """
    print("✨ Interrogo l’Oracolo…")
    lang_clause = "Answer in English." if lang_hint == "en" else "Rispondi in italiano."
    resp = client.responses.create(
        model=LLM_MODEL,
        instructions=(ORACLE_SYSTEM + " " + lang_clause),
        input=[{"role": "user", "content": question}],
    )
    ans = resp.output_text.strip()
    print(f"🔮 Oracolo: {ans}")
    return ans

def synthesize(text: str, out_path: Path):
    print("🎧 Sintesi vocale…")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        # preferito: WAV diretto
        with client.audio.speech.with_streaming_response.create(
            model=TTS_MODEL, voice=TTS_VOICE, input=text, response_format="wav"
        ) as resp:
            resp.stream_to_file(out_path.as_posix())
    except TypeError:
        # fallback: spesso MP3
        with client.audio.speech.with_streaming_response.create(
            model=TTS_MODEL, voice=TTS_VOICE, input=text
        ) as resp:
            if out_path.suffix.lower() != ".mp3":
                out_path = out_path.with_suffix(".mp3")
            resp.stream_to_file(out_path.as_posix())
    print(f"✅ Audio → {out_path.name}")

def play_and_pulse(path: Path, light, sr: int):
    y, sr = load_audio_as_float(path, sr)
    stop = False
    def worker():
        win = int(0.02 * sr)  # 20ms
        pos = 0
        cur = 0.1
        while not stop and pos < len(y):
            seg = y[pos:pos+win]
            rms = float(np.sqrt(np.mean(seg**2))) if len(seg) else 0.0
            level = max(0.0, min(1.0, rms*6.0))
            cur = 0.7*cur + 0.3*level
            if isinstance(light, SacnLight):
                idle = SET["lighting"]["sacn"]["idle_level"]
                peak = SET["lighting"]["sacn"]["peak_level"]
                v = int(idle + (peak-idle)*cur)
                light.set_rgb(v//3, v//3, v)  # bianco freddo tendente al blu
            else:
                light.pulse(cur)
            time.sleep(win/sr)
            pos += win
    t = threading.Thread(target=worker, daemon=True); t.start()
    sd.play(y, sr, blocking=True)
    stop = True; t.join()

def record_until_silence(path: Path) -> bool:
    """
    VAD semplice a energia con parametri da settings.yaml.
    Ritorna True se ha salvato un file valido (parlato presente), False altrimenti.
    Non fa fallback automatico se 'fallback_to_timed' è False.
    """
    # elimina eventuale file precedente per evitare trascrizioni “fantasma”
    try:
        if path.exists():
            path.unlink()
    except Exception:
        pass

    VCONF = (SET.get("vad", {}) or {})
    FRAME_MS      = int(VCONF.get("frame_ms", 30))
    START_MS      = int(VCONF.get("start_ms", 150))
    END_MS        = int(VCONF.get("end_ms", 800))
    MAX_MS        = int(VCONF.get("max_ms", 15000))
    PREROLL_MS    = int(VCONF.get("preroll_ms", 300))
    NOISE_WIN_MS  = int(VCONF.get("noise_window_ms", 800))
    START_MULT    = float(VCONF.get("start_mult", 1.8))
    END_MULT      = float(VCONF.get("end_mult", 1.3))
    BASE_START    = float(VCONF.get("base_start", 0.006))
    BASE_END      = float(VCONF.get("base_end", 0.0035))

    assert FRAME_MS in (10, 20, 30)
    frame_samples = int(AUDIO_SR * FRAME_MS / 1000)

    print(f"\n🎤 Parla pure (VAD energia, max {MAX_MS/1000:.1f}s)…")

    # buffer per pre-roll e stima rumore
    pre_frames = PREROLL_MS // FRAME_MS
    preroll = collections.deque(maxlen=pre_frames)
    noise_frames_needed = max(1, NOISE_WIN_MS // FRAME_MS)
    noise_rms_history = collections.deque(maxlen=noise_frames_needed)

    started = False
    speech_streak = 0
    silence_streak = 0
    total_ms = 0
    recorded_blocks = []  # liste di np.float32

    def rms(block_f32: np.ndarray) -> float:
        if block_f32.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(block_f32**2)))

    with sd.InputStream(samplerate=AUDIO_SR,
                        blocksize=frame_samples,
                        channels=1,
                        dtype='float32',
                        device=INPUT_DEVICE_ID) as stream:  # se non usi i device helper, rimuovi 'device=INPUT_DEVICE_ID'
        while total_ms < MAX_MS:
            audio_block, _ = stream.read(frame_samples)  # float32 (N,1)
            block = audio_block[:, 0]
            total_ms += FRAME_MS

            # stima rumore (pre-start)
            if not started and len(noise_rms_history) < noise_frames_needed:
                noise_rms_history.append(rms(block))

            noise_floor = np.median(noise_rms_history) if noise_rms_history else 0.003
            start_thresh = max(noise_floor * START_MULT, BASE_START)
            end_thresh   = max(noise_floor * END_MULT,  BASE_END)

            level = rms(block)
            if DEBUG and total_ms % 300 == 0 and not started:
                print(f"   [DBG] level={level:.4f} start_th={start_thresh:.4f} noise={noise_floor:.4f}")

            if not started:
                preroll.append(block.copy())
                if level >= start_thresh:
                    speech_streak += FRAME_MS
                    if speech_streak >= START_MS:
                        started = True
                        if len(preroll):
                            recorded_blocks.extend([b.copy() for b in preroll])
                        recorded_blocks.append(block.copy())
                        silence_streak = 0
                else:
                    speech_streak = 0
            else:
                recorded_blocks.append(block.copy())
                if level < end_thresh:
                    silence_streak += FRAME_MS
                    if silence_streak >= END_MS:
                        break
                else:
                    silence_streak = 0

    if not recorded_blocks:
        print("😴 Silenzio rilevato. Resto in attesa.")
        # NIENTE fallback se configurato così
        if FALLBACK_TIMED:
            print("↻ Fallback: registrazione temporizzata.")
            record_wav(path, seconds=TIMED_SECONDS, sr=AUDIO_SR)
            return True
        return False

    y = np.concatenate(recorded_blocks, axis=0).astype(np.float32)

    # Scarta registrazioni “vuote” (picco troppo basso)
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak < MIN_SPEECH_LEVEL:
        if DEBUG:
            print(f"   [DBG] Registrazione scartata: peak={peak:.4f} < min={MIN_SPEECH_LEVEL:.4f}")
        print("😴 Non ho colto parole distinte. Resto in attesa.")
        return False

    sf.write(path.as_posix(), y, AUDIO_SR)
    dur = len(y) / AUDIO_SR
    print(f"✅ Registrazione completata ({dur:.2f}s).")
    return True


# =====================================================================
#                           MAIN LOOP
# =====================================================================
def main():
    print("Occhio Onniveggente · Oracolo ✨")

    # driver luci
    if LIGHT_MODE == "sacn":
        light = SacnLight(SET["lighting"])
    elif LIGHT_MODE == "wled":
        light = WledLight(SET["lighting"])
    else:
        print("⚠️ lighting.mode non valido, uso WLED di default")
        light = WledLight(SET["lighting"])

    try:
        while True:
            cmd = input("\nPremi INVIO per fare una domanda (q per uscire)… ")
            if cmd.strip().lower() == "q": break

            # Registrazione che attende la fine frase
            ok = record_until_silence(INPUT_WAV)
            if not ok:
                # Se hai commentato il fallback nella funzione, qui saltiamo il turno
                continue

            # Trascrizione + lingua
            q, lang = transcribe(INPUT_WAV)
            if not q:
                continue

            # Filtro INPUT
            if PROF.contains_profanity(q):
                if FILTER_MODE == "block":
                    print("🚫 Linguaggio offensivo/bestemmie non ammessi. Riformula in italiano o inglese.")
                    continue
                else:
                    q = PROF.mask(q)
                    print("⚠️ Testo filtrato:", q)

            # Risposta nella stessa lingua
            a = oracle_answer(q, lang or "it")

            # Filtro OUTPUT (by safety)
            if PROF.contains_profanity(a):
                if FILTER_MODE == "block":
                    print("⚠️ La risposta conteneva termini non ammessi, riformulo…")
                    a = client.responses.create(
                        model=LLM_MODEL,
                        instructions=ORACLE_SYSTEM + " Evita qualsiasi offesa o blasfemia.",
                        input=[{"role":"user","content":"Riformula in modo poetico e non offensivo:\n"+a}],
                    ).output_text.strip()
                else:
                    a = PROF.mask(a)

            append_log(q, a)

            # Colore base dalla semantica della risposta
            base = color_from_text(a)
            if isinstance(light, WledLight):
                light.set_base_rgb(base); light.idle()

            # TTS + luci sincronizzate
            synthesize(a, OUTPUT_WAV)
            play_and_pulse(OUTPUT_WAV, light, AUDIO_SR)

    finally:
        light.blackout()
        light.stop()
        print("👁️  Arrivederci.")

if __name__ == "__main__":
    main()
