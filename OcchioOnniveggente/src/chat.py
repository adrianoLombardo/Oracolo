from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional, Any, Iterator
import time, json
import numpy as np
from openai import OpenAI
from .config import Settings, get_openai_api_key


def summarize_history(prev_summary: str, msgs: List[Dict[str, str]]) -> str:
    """Summarize *msgs* with a lightweight LLM.

    Falls back to simple concatenation if the API call fails or no key is
    configured. ``prev_summary`` is included so the model can produce an
    updated cumulative summary.
    """

    if not msgs:
        return prev_summary

    settings = Settings()
    model = settings.chat.summary_model
    max_tokens = settings.chat.summary_max_tokens
    lines = [f"{m.get('role', '')}: {m.get('content', '')}" for m in msgs]
    prompt = (
        "Aggiorna il seguente riassunto della conversazione con i nuovi messaggi.\n"
        f"Riassunto precedente:\n{prev_summary}\n\nMessaggi:\n" + "\n".join(lines)
    )

    try:
        api_key = get_openai_api_key(settings)
        client = OpenAI(api_key=api_key)
        try:
            resp = client.responses.create(
                model=model, input=prompt, max_output_tokens=max_tokens
            )
            txt = getattr(resp, "output_text", "").strip()
        except Exception:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
            )
            txt = resp.choices[0].message.content.strip()  # type: ignore[index]
        if txt:
            return txt
    except Exception:
        pass

    # Fallback: append raw text
    snippet = " ".join(lines)
    if prev_summary:
        return (prev_summary + " " + snippet).strip()
    return snippet


@dataclass
class ChatState:
    max_turns: int = 10
    persist_jsonl: Optional[Path] = None
    session_id: str = str(int(time.time()))
    history: List[Dict[str, str]] = field(default_factory=list)
    topic_emb: Optional[np.ndarray] = field(default=None, repr=False)
    topic_text: Optional[str] = None
    topic_locked: bool = False
    pinned: List[str] = field(default_factory=list)
    summary: str = ""
    language: Optional[str] = None
    pinned_limit: int = 5

    def reset(self) -> None:
        self.history.clear()
        self.summary = ""

    def push_message(self, role: str, text: str) -> None:
        self.history.append({"role": role, "content": text})
        self._trim()
        self._persist(role, text)

    def push_user(self, text: str) -> None:
        self.push_message("user", text)

    def push_assistant(self, text: str) -> None:
        self.push_message("assistant", text)

    def stream_assistant(self, tokens: "Iterator[str]") -> str:
        """Consume ``tokens`` updating the last assistant message incrementally.

        The generator is iterated token by token; after each token the partial
        text is stored in :attr:`history`.  The full assistant message is
        returned once the stream is exhausted so that callers can further
        process it if needed.
        """

        text = ""
        for chunk in tokens:
            text += chunk
            if self.history and self.history[-1].get("role") == "assistant":
                self.history[-1]["content"] = text
            else:
                self.history.append({"role": "assistant", "content": text})
        return text

    def pin_message(self, text: str) -> None:
        if not text:
            return
        self.pinned.append(text)
        if self.pinned_limit > 0 and len(self.pinned) > self.pinned_limit:
            self.pinned = self.pinned[-self.pinned_limit:]
        self._persist("pinned", text)

    def pin_last_user(self) -> None:
        for msg in reversed(self.history):
            if msg.get("role") == "user":
                self.pin_message(msg.get("content", ""))
                break

    def pinned_shortlist(self) -> List[str]:
        if self.pinned_limit > 0:
            return self.pinned[-self.pinned_limit:]
        return list(self.pinned)

    def _trim(self) -> None:
        if self.max_turns > 0:
            excess = len(self.history) - (2 * self.max_turns)
            if excess > 0:
                self.summary = summarize_history(self.summary, self.history[:excess])
                self.history = self.history[excess:]

    def soft_reset(self) -> None:
        self.summary = summarize_history(self.summary, self.history)
        self.history.clear()
        self.topic_emb = None
        self.topic_text = None

    def _persist(self, role: str, text: str) -> None:
        if not self.persist_jsonl:
            return
        rec = {
            "ts": int(time.time()),
            "session_id": self.session_id,
            "role": role,
            "text": text,
        }
        self.persist_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with self.persist_jsonl.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def export_history(self, path: Path) -> None:
        """Export the current chat history to *path*.

        The output format is determined by the file extension:

        - ``.json`` → JSON array of messages
        - ``.md``   → simple Markdown transcript
        - otherwise → plain text transcript
        """
        ext = path.suffix.lower()
        if ext == ".json":
            path.write_text(
                json.dumps(self.history, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            return

        lines = []
        for msg in self.history:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if ext == ".md":
                lines.append(f"**{role}:** {content}")
            else:
                lines.append(f"{role}: {content}")
        path.write_text("\n".join(lines), encoding="utf-8")

    # ----------------- topic tracking -----------------
    def update_topic(
        self,
        text: str,
        client: Any,
        emb_model: str,
        threshold: float = 0.65,
    ) -> bool:
        """Aggiorna l'argomento corrente.

        Ritorna True se è stato rilevato un cambio di tema rispetto al
        topic precedente.
        """
        if self.topic_locked:
            return False
        if not text or client is None or not emb_model:
            return False
        try:
            resp = client.embeddings.create(model=emb_model, input=[text])  # type: ignore[attr-defined]
            vec = np.array(resp.data[0].embedding, dtype=np.float32)
        except Exception:
            # se embedding fallisce, aggiorna solo il testo
            if self.topic_text is None:
                self.topic_text = text
            return False

        if self.topic_emb is None:
            self.topic_emb = vec
            self.topic_text = text
            return False

        sim = float(np.dot(self.topic_emb, vec) / (np.linalg.norm(self.topic_emb) * np.linalg.norm(vec) + 1e-9))
        changed = sim < threshold
        if changed:
            last = self.history[-1:]  # conserva ultimo messaggio
            self.soft_reset()
            self.history.extend(last)
            self.topic_text = text
            self.topic_emb = vec
        else:
            # aggiorna embedding medio per seguire l'evoluzione del tema
            self.topic_emb = (self.topic_emb + vec) / 2.0
        return changed
