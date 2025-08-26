from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


class ConversationStore:
    """Simple SQLite-backed store for chat sessions."""

    def __init__(self, db_path: str | Path = "chat_sessions.sqlite") -> None:
        self.db_path = Path(db_path)
        self.conn = sqlite3.connect(str(self.db_path))
        self._create_schema()

    def _create_schema(self) -> None:
        with self.conn:
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    data TEXT NOT NULL
                )
                """
            )

    # --------------------------- persistence ---------------------------
    def save_state(self, session_id: str, data: Dict[str, Any]) -> None:
        """Persist *data* for ``session_id``."""
        serial = json.dumps(data, default=self._serialize, ensure_ascii=False)
        with self.conn:
            self.conn.execute(
                "INSERT OR REPLACE INTO sessions (session_id, data) VALUES (?, ?)",
                (session_id, serial),
            )

    def load_state(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Load and return the data for ``session_id`` if present."""
        cur = self.conn.execute(
            "SELECT data FROM sessions WHERE session_id=?", (session_id,)
        )
        row = cur.fetchone()
        if not row:
            return None
        data = json.loads(row[0])
        if data.get("topic_emb") is not None:
            data["topic_emb"] = np.array(data["topic_emb"], dtype=np.float32)
        if data.get("persist_jsonl"):
            data["persist_jsonl"] = Path(data["persist_jsonl"])
        return data

    # ------------------------------ utils ------------------------------
    def _serialize(self, obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, Path):
            return str(obj)
        raise TypeError(f"Object of type {type(obj)!r} not serializable")


__all__ = ["ConversationStore"]
