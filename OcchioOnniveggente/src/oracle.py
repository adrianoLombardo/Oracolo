from __future__ import annotations

import re
import time
from datetime import datetime
from pathlib import Path
from typing import Tuple

import openai
from langdetect import DetectorFactory, detect

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
        print(f"   [DBG] {lang.upper()} score={score:.3f} text='{short}'")
    return score


def transcribe(path: Path, client, stt_model: str, *, debug: bool = False) -> Tuple[str, str]:
    def _call_transcription(**kwargs) -> str:
        for _ in range(3):
            try:
                with open(path, "rb") as f:
                    tx = client.audio.transcriptions.create(model=stt_model, file=f, **kwargs)
                return (tx.text or "").strip()
            except openai.OpenAIError as e:
                print(f"Errore OpenAI: {e}")
                time.sleep(1)
        return ""

    print("🧠 Trascrizione (auto)…")
    text_auto = _call_transcription(
        prompt=(
            "Language is either Italian or English. Focus on neuroscience, "
            "neuroaesthetics, contemporary art, the universe, and "
            "neuroscientific AI. Ignore any other language."
        )
    )

    print("↻ Trascrizione forzata IT…")
    text_it = _call_transcription(
        language="it",
        prompt=(
            "Lingua: italiano. Dominio: neuroscienze, neuroestetica, "
            "arte contemporanea, universo e IA neuroscientifica."
        ),
    )

    print("↻ Trascrizione forzata EN…")
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
        print("⚠️ Per favore parla in italiano o inglese.")
        return "", ""
    if s_it >= s_en:
        print("🌐 Lingua rilevata: IT")
        return text_it, "it"
    else:
        print("🌐 Lingua rilevata: EN")
        return text_en, "en"


def oracle_answer(
    question: str,
    lang_hint: str,
    client,
    llm_model: str,
    oracle_system: str,
    *,
    context: list[dict] | None = None,
    history: list[dict[str, str]] | None = None,
    topic: str | None = None,
) -> str:
    print("✨ Interrogo l’Oracolo…")
    lang_clause = "Answer in English." if lang_hint == "en" else "Rispondi in italiano."

    messages: list[dict[str, str]] = []
    if context:
        ctx_txt = "\n\n".join(str(c.get("text", "")) for c in context if c.get("text"))
        if ctx_txt:
            messages.append({"role": "system", "content": f"Contesto:\n{ctx_txt}"})
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": question})

    topic_clause = (
        " Rispondi solo con informazioni coerenti al tema seguente e non mescolare altri argomenti a meno che l'utente lo chieda esplicitamente. Tema: "
        + topic
        if topic
        else ""
    )
    sys_prompt = (
        oracle_system
        + topic_clause
        + " Se la domanda non riguarda neuroscienze, neuroestetica, arte contemporanea o l'universo, rispondi: 'Domanda non pertinente'. "
        + lang_clause
    )

    for _ in range(3):
        try:
            resp = client.responses.create(
                model=llm_model,
                instructions=sys_prompt,
                input=messages,
            )
            ans = resp.output_text.strip()
            return ans
        except openai.OpenAIError as e:
            print(f"Errore OpenAI: {e}")
            time.sleep(1)
    return ""


def synthesize(text: str, out_path: Path, client, tts_model: str, tts_voice: str) -> None:
    print("🎧 Sintesi vocale…")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for _ in range(3):
        try:
            with client.audio.speech.with_streaming_response.create(
                model=tts_model, voice=tts_voice, input=text, response_format="wav"
            ) as resp:
                resp.stream_to_file(out_path.as_posix())
            print(f"✅ Audio → {out_path.name}")
            return
        except TypeError:
            with client.audio.speech.with_streaming_response.create(
                model=tts_model, voice=tts_voice, input=text
            ) as resp:
                if out_path.suffix.lower() != ".mp3":
                    out_path = out_path.with_suffix(".mp3")
                resp.stream_to_file(out_path.as_posix())
            print(f"✅ Audio → {out_path.name}")
            return
        except openai.OpenAIError as e:
            print(f"Errore OpenAI: {e}")
            time.sleep(1)
    print("❌ Impossibile sintetizzare l'audio.")


def append_log(q: str, a: str, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    def clean(s: str) -> str:
        return s.replace('"', "'")
    line = f'"{ts}","{clean(q)}","{clean(a)}"\n'
    if not log_path.exists():
        log_path.write_text('"timestamp","question","answer"\n', encoding="utf-8")
    with log_path.open("a", encoding="utf-8") as f:
        f.write(line)


