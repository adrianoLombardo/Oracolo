"""Data models for the retrieval package."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Protocol


@dataclass
class Chunk:
    """A small piece of a document returned by the search layer."""

    doc_id: str
    text: str
    meta: Dict[str, Any]


@dataclass
class Question:
    """Representation of a single question entry.

    Only a subset of the fields used in the full project is modelled here as
    the tests exercise just a few of them.
    """

    id: str | None = None
    domanda: str = ""
    type: str = ""
    follow_up: str | None = None
    opera: str | None = None
    artista: str | None = None
    location: str | None = None
    tag: List[str] | None = None

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def __post_init__(self) -> None:
        if not self.domanda:
            raise ValueError("domanda must not be empty")
        if not self.type:
            raise ValueError("type must not be empty")
        self.type = self.type.lower()
        if self.tag:
            seen: set[str] = set()
            normalized: List[str] = []
            for t in self.tag:
                s = str(t).lower()
                if s not in seen:
                    seen.add(s)
                    normalized.append(s)
            self.tag = normalized

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "id": self.id,
            "domanda": self.domanda,
            "type": self.type,
        }
        if self.follow_up is not None:
            data["follow_up"] = self.follow_up
        if self.opera is not None:
            data["opera"] = self.opera
        if self.artista is not None:
            data["artista"] = self.artista
        if self.location is not None:
            data["location"] = self.location
        if self.tag:
            data["tag"] = list(self.tag)
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Question":
        tags = data.get("tag")
        if isinstance(tags, list):
            tag_list = [str(t) for t in tags]
        elif tags is None:
            tag_list = None
        else:
            tag_list = [str(tags)]
        qid = data.get("id")
        follow_up = data.get("follow_up")
        if follow_up is not None:
            follow_up = str(follow_up)
        return cls(
            id=str(qid) if qid is not None else None,
            domanda=str(data.get("domanda", "")),
            type=str(data.get("type", "")),
            follow_up=follow_up,
            opera=data.get("opera"),
            artista=data.get("artista"),
            location=data.get("location"),
            tag=tag_list,
        )


class QuestionProvider(Protocol):
    """Interface for pluggable question sources."""

    name: str

    def load(self) -> Dict[str, List[Question]]:
        """Return the questions grouped by category."""


class Context(str, Enum):
    """Enumeration of thematic contexts for question datasets."""

    GENERIC = "generic"
    CONFERENZA_DIDATTICA = "conferenza_didattica"
    MOSTRA = "mostra"

    @classmethod
    def from_str(cls, value: str) -> "Context":
        """Return the matching ``Context`` for ``value`` (case insensitive)."""

        try:
            return cls(value.lower())
        except ValueError:
            return cls.GENERIC


__all__ = [
    "Chunk",
    "Question",
    "QuestionProvider",
    "Context",
]

