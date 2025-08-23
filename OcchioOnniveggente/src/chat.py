from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional
import time, json


@dataclass
class ChatState:
    max_turns: int = 10
    persist_jsonl: Optional[Path] = None
    session_id: str = str(int(time.time()))
    history: List[Dict[str, str]] = field(default_factory=list)

    def reset(self) -> None:
        self.history.clear()

    def push_user(self, text: str) -> None:
        self.history.append({"role": "user", "content": text})
        self._trim()
        self._persist("user", text)

    def push_assistant(self, text: str) -> None:
        self.history.append({"role": "assistant", "content": text})
        self._trim()
        self._persist("assistant", text)

    def _trim(self) -> None:
        if self.max_turns > 0:
            excess = len(self.history) - (2 * self.max_turns)
            if excess > 0:
                self.history = self.history[excess:]

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
