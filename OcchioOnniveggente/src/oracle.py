# src/oracle.py
from __future__ import annotations

import re
import time
from datetime import datetime
from pathlib import Path
from typing import Tuple, Any, Optional

import openai
from langdetect import DetectorFactory, detect

# determinismo per langdetect
DetectorFactory.seed = 42

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


def _score_lang(text: str, lang: str, *, debug: bool = False) -> float:
    if not text:
        return 0.0
    toks = re.findall(r"[a-zàèéìòóù]+", text.lower())
    if not toks:
        return 0.0
    sw = _IT_SW if lang == "it" else _EN_SW
    hits = sum(1 for t in toks if t in sw)
    coverage = hits / max(len(toks), 1)
    try:
        ld = detect(text)
        bonus = 0.5 if (ld == lang) else 0.0
    except Exception:
        bonus = 0.0
    length_bonus = min(len(toks), 30) / 30 * 0.2
    score = coverage + bonus + length_bonus
    if debug:
        short = (text[:80] + "…") if len(text) > 80 else text
        print(f"   [DBG] {lang.upper()} score={score:.3f} text='{short}'", flush=True)
    return score


def transcribe(path: Path, client, stt_model: str, *, debug: bool = False) -> Tuple[str, str]:
    """Trascrive il file audio (auto/IT/EN) e sceglie la lingua migliore."""
    def _call_transcription(**kwargs) -> str:
        for _ in range(3):
            try:
                with open(path, "rb") as f:
                    tx = client.audio.transcriptions.create(model=stt_model, file=f, **kwargs)
                return (getattr(tx, "text", "") or "").strip()
            except (openai.APIError, openai.RateLimitError, openai.APIConnectionError, openai.BadRequestError, Exception) as e:
                print(f"Errore OpenAI: {e}", flush=True)
                time.sleep(1)
        return ""

    print("🧠 Trascrizione (auto)…", flush=True)
    text_auto = _call_transcription(
        prompt=(
            "Language is either Italian or English. Focus on neuroscience, "
            "neuroaesthetics, contemporary art, the universe, and "
            "neuroscientific AI. Ignore any other language."
        )
    )

    print("↻ Trascrizione forzata IT…", flush=True)
    text_it = _call_transcription(
        language="it",
        prompt=(
            "Lingua: italiano. Dominio: neuroscienze, neuroestetica, "
            "arte contemporanea, universo e IA neuroscientifica."
        ),
    )

    print("↻ Trascrizione forzata EN…", flush=True)
    text_en = _call_transcription(
        language="en",
        prompt=(
            "Language: English. Domain: neuroscience, neuroaesthetics, "
            "contemporary art, universe, and neuroscientific AI."
        ),
    )

    s_it = _score_lang(text_it, "it", debug=debug)
    s_en = _score_lang(text_en, "en", debug=debug)

    try:
        auto_lang = detect(text_auto) if text_auto else None
    except Exception:
        auto_lang = None
    if auto_lang == "it":
        s_it += 0.05
    elif auto_lang == "en":
        s_en += 0.05

    if s_it == 0 and s_en == 0:
        print("⚠️ Per favore parla in italiano o inglese.", flush=True)
        return "", ""
    if s_it >= s_en:
        print("🌐 Lingua rilevata: IT", flush=True)
        return text_it, "it"
    else:
        print("🌐 Lingua rilevata: EN", flush=True)
        return text_en, "en"


def _normalize_context(context: Any) -> str:
    """
    Accetta:
      - stringa
      - lista/tupla di stringhe
      - lista/tupla di dict con chiave 'text' (o simili)
    Ritorna un blocco testuale unico (tagliato a ~4000 char).
    """
    if not context:
        return ""
    chunks: list[str] = []
    if isinstance(context, (list, tuple)):
        for item in context:
            if isinstance(item, str):
                t = item.strip()
            elif isinstance(item, dict):
                t = str(item.get("text") or item.get("content") or "").strip()
            else:
                t = str(item).strip()
            if t:
                chunks.append(t)
    elif isinstance(context, str):
        chunks.append(context.strip())
    else:
        chunks.append(str(context).strip())

    joined = "\n\n---\n\n".join(chunks)
    return joined[:4000]


def _responses_create_safely(client, model: str, messages: list[dict], temperature: Optional[float]):
    """Chiama Responses API; se 'temperature' non è supportato, ritenta senza."""
    try:
        kwargs = {"model": model, "input": messages}
        if temperature is not None:
            kwargs["temperature"] = temperature
        return client.responses.create(**kwargs)
    except openai.BadRequestError as e:
        # se il modello non supporta 'temperature', ritenta senza
        msg = str(e).lower()
        if "temperature" in msg:
            return client.responses.create(model=model, input=messages)
        raise


def _chat_create_safely(client, model: str, messages: list[dict], temperature: Optional[float]):
    """Chiama chat.completions; se 'temperature' non è supportato, ritenta senza."""
    try:
        kwargs = {"model": model, "messages": messages}
        if temperature is not None:
            kwargs["temperature"] = temperature
        return client.chat.completions.create(**kwargs)
    except openai.BadRequestError as e:
        msg = str(e).lower()
        if "temperature" in msg:
            return client.chat.completions.create(model=model, messages=messages)
        raise


def oracle_answer(question, lang, client, model, system, *, context=None, **kwargs):
    """
    Genera la risposta dell'Oracolo.
    - Preferisce Responses API; fallback a chat.completions.
    - 'context' può essere stringa o lista di frammenti.
    - Se il modello non supporta 'temperature', la rimuove automaticamente.
    """
    ctx = _normalize_context(context)
    sys_prompt = system if not ctx else f"{system}\n\n[Contesto]\n{ctx}"

    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": question},
    ]

    # Permetti di passare temperature via kwargs, ma gestisci l'errore
    temperature = kwargs.get("temperature", None)

    # Responses API
    try:
        if hasattr(client, "responses"):
            resp = _responses_create_safely(client, model, messages, temperature)
            text = (getattr(resp, "output_text", None) or "").strip()
            if text:
                return text
    except (openai.APIError, openai.RateLimitError, openai.APIConnectionError, openai.BadRequestError) as e:
        print(f"Errore OpenAI (responses): {e}", flush=True)

    # Fallback a chat.completions
    resp = _chat_create_safely(client, model, messages, temperature)
    return resp.choices[0].message.content.strip()


def synthesize(text: str, out_path: Path, client, tts_model: str, tts_voice: str) -> None:
    """TTS con streaming su file; fallback a MP3 se 'wav' non è supportato."""
    print("🎧 Sintesi vocale…", flush=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for _ in range(3):
        try:
            with client.audio.speech.with_streaming_response.create(
                model=tts_model, voice=tts_voice, input=text, response_format="wav"
            ) as resp:
                resp.stream_to_file(out_path.as_posix())
            print(f"✅ Audio → {out_path.name}", flush=True)
            return
        except TypeError:
            # Alcuni modelli non accettano response_format: forza MP3
            with client.audio.speech.with_streaming_response.create(
                model=tts_model, voice=tts_voice, input=text
            ) as resp:
                if out_path.suffix.lower() != ".mp3":
                    out_path = out_path.with_suffix(".mp3")
                resp.stream_to_file(out_path.as_posix())
            print(f"✅ Audio → {out_path.name}", flush=True)
            return
        except (openai.APIError, openai.RateLimitError, openai.APIConnectionError, openai.BadRequestError, Exception) as e:
            print(f"Errore OpenAI: {e}", flush=True)
            time.sleep(1)
    print("❌ Impossibile sintetizzare l'audio.", flush=True)


def append_log(q: str, a: str, log_path: Path) -> None:
    """Appende domanda/risposta a CSV con timestamp."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def clean(s: str) -> str:
        return (s or "").replace('"', "'")

    line = f'"{ts}","{clean(q)}","{clean(a)}"\n'
    if not log_path.exists():
        log_path.write_text('"timestamp","question","answer"\n', encoding="utf-8")
    with log_path.open("a", encoding="utf-8") as f:
        f.write(line)
