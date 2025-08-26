from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional
import time, json
from uuid import uuid4

import numpy as np
from openai import OpenAI

from ..DataBase.conversation_store import ConversationStore
from .config import Settings, get_openai_api_key
from .dialogue import DialogueManager, DialogState


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

    if model == "local":
        from .service_container import container
        try:
            return container.llm_batcher().generate_sync(prompt)
        except Exception:
            pass
    else:
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
    session_id: str = field(default_factory=lambda: str(uuid4()))
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

    def summarize_history(self) -> None:
        """Summarize the current conversation into ``summary`` and clear history."""

        if not self.history:
            return
        self.summary = summarize_history(self.summary, self.history)
        self.history.clear()

    def stream_assistant(self, tokens: Iterator[str]) -> str:
        """Consume ``tokens`` updating the last assistant message incrementally."""

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
        """Export the current chat history to *path*."""

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

    def messages_for_llm(self) -> List[Dict[str, str]]:
        """Return the history preceded by the cumulative summary if present."""

        msgs: List[Dict[str, str]] = []
        if self.summary:
            msgs.append({"role": "system", "content": self.summary})
        msgs.extend(self.history)
        return msgs

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

        sim = float(
            np.dot(self.topic_emb, vec)
            / (np.linalg.norm(self.topic_emb) * np.linalg.norm(vec) + 1e-9)
        )
        changed = sim < threshold
        if changed:
            last = self.history[-1:]
            self.soft_reset()
            self.history.extend(last)
            self.topic_text = text
            self.topic_emb = vec
        else:
            # aggiorna embedding medio per seguire l'evoluzione del tema
            self.topic_emb = (self.topic_emb + vec) / 2.0
        return changed


@dataclass
class ConversationManager:
    """Unified wrapper around :class:`ChatState` and :class:`DialogueManager`.

    It exposes helper methods to push user/assistant messages while keeping
    track of dialogue state and processing turns.
    """

    idle_timeout: float = 60.0
    max_history: int = 10
    chat: ChatState = field(default_factory=ChatState)
    store: ConversationStore = field(default_factory=ConversationStore)

    dlg: DialogueManager = field(init=False)
    is_processing: bool = False
    turn_id: int = 0

    def __post_init__(self) -> None:  # pragma: no cover - simple delegation
        self.dlg = DialogueManager(self.idle_timeout)
        # ensure initial state is persisted
        self._save_state()

    # ------------------------- dialogue proxies -------------------------
    @property
    def state(self) -> DialogState:
        return self.dlg.state

    @state.setter
    def state(self, value: DialogState) -> None:
        self.dlg.state = value

    def refresh_deadline(self) -> None:
        self.dlg.refresh_deadline()

    @property
    def active_deadline(self) -> float:
        """Proxy access to the underlying dialogue deadline."""
        return self.dlg.active_deadline

    @active_deadline.setter
    def active_deadline(self, value: float) -> None:
        self.dlg.active_deadline = value

    def transition(self, new_state: DialogState) -> None:
        self.dlg.transition(new_state)

    def timed_out(self, now: float) -> bool:
        return self.dlg.timed_out(now)

    # --------------------------- turn helpers ---------------------------
    def start_processing(self) -> int:
        """Mark that a user turn is being processed and return turn id."""
        self.is_processing = True
        self.turn_id += 1
        return self.turn_id

    def end_processing(self) -> None:
        """Clear processing flag after finishing the turn."""
        self.is_processing = False

    # --------------------------- chat helpers ---------------------------
    def push_user(self, text: str) -> None:
        self.chat.push_user(text)
        self._trim_history()
        self._save_state()

    def push_assistant(self, text: str) -> None:
        self.chat.push_assistant(text)
        self._trim_history()
        self._save_state()

    def _trim_history(self) -> None:
        excess = len(self.chat.history) - self.max_history
        if excess > 0:
            self.chat.summary = summarize_history(
                self.chat.summary, self.chat.history[:excess]
            )
            self.chat.history = self.chat.history[excess:]

    def summarize_history(self) -> None:
        """Force summarization of the entire conversation history."""
        if self.chat.history:
            self.chat.summary = summarize_history(self.chat.summary, self.chat.history)
            self.chat.history.clear()
            self._save_state()

    @property
    def messages(self) -> List[Dict[str, str]]:
        msgs: List[Dict[str, str]] = []
        if self.chat.summary:
            msgs.append({"role": "system", "content": self.chat.summary})
        msgs.extend(self.chat.history)
        return msgs

    def messages_for_llm(self) -> list[dict[str, str]]:
        """Return recent messages including any accumulated summary."""

        return self.chat.messages_for_llm()

    # ------------------------- persistence -------------------------
    def _state_dict(self) -> Dict[str, Any]:
        data = asdict(self.chat)
        if self.chat.topic_emb is not None:
            data["topic_emb"] = self.chat.topic_emb.tolist()
        if self.chat.persist_jsonl is not None:
            data["persist_jsonl"] = str(self.chat.persist_jsonl)
        return data

    def _save_state(self) -> None:
        self.store.save_state(self.chat.session_id, self._state_dict())

    def load_session(self, session_id: str) -> None:
        """Load a session from the store into ``chat``."""
        data = self.store.load_state(session_id)
        if data is None:
            self.chat = ChatState(session_id=session_id)
        else:
            self.chat = ChatState(**data)
        self._save_state()


__all__ = ["ChatState", "ConversationManager", "DialogState", "summarize_history"]

