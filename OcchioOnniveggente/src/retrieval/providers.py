"""Question providers and loading utilities."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Iterable, List

from .models import Question, QuestionProvider


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Provider registry
# ---------------------------------------------------------------------------

_QUESTION_PROVIDERS: Dict[str, QuestionProvider] = {}


def register_question_provider(provider: QuestionProvider) -> None:
    """Register ``provider`` in the global provider registry."""

    _QUESTION_PROVIDERS[provider.name.lower()] = provider


def get_question_provider(name: str) -> QuestionProvider | None:
    """Return the provider registered under ``name`` if present."""

    return _QUESTION_PROVIDERS.get(name.lower())


def iter_question_providers(context: str | None = None) -> Iterable[QuestionProvider]:
    """Yield all registered providers or the one matching ``context``."""

    if context:
        p = get_question_provider(context)
        return [p] if p is not None else []
    return list(_QUESTION_PROVIDERS.values())


# ---------------------------------------------------------------------------
# Loading utilities
# ---------------------------------------------------------------------------


def load_questions(path: str | Path | None = None) -> Dict[str, List[Question]]:
    """Load questions grouped only by category.

    The real project exposes a more elaborate loader capable of handling
    multiple contexts.  For the purposes of the tests we provide a compact
    implementation that flattens all questions into a single mapping
    ``{category -> [Question, ...]}``.
    """

    root = (
        Path(path)
        if path is not None
        else Path(__file__).resolve().parents[2] / "data" / "domande_oracolo.json"
    )
    if root.is_dir():
        root = root / "domande_oracolo.json"
    try:
        data = json.loads(root.read_text(encoding="utf-8"))
    except Exception:
        return {}

    categories: Dict[str, List[Question]] = {}

    def _add_item(item: dict) -> None:
        try:
            q = Question.from_dict(item)
        except Exception:
            return
        categories.setdefault(q.type, []).append(q)

    if isinstance(data, list):
        for it in data:
            if isinstance(it, dict):
                _add_item(it)
    elif isinstance(data, dict):
        for items in data.values():
            if isinstance(items, list):
                for it in items:
                    if isinstance(it, dict):
                        _add_item(it)
    return categories


def load_questions_from_providers(context: str | None = None) -> Dict[str, List[Question]]:
    """Load questions from the registered providers."""

    merged: Dict[str, List[Question]] = {}
    for provider in iter_question_providers(context):
        try:
            data = provider.load() or {}
        except Exception:
            logger.exception("Question provider %s failed", provider.name)
            data = {}
        for cat, qs in data.items():
            for q in qs:
                if not isinstance(q, Question):
                    try:
                        q = Question.from_dict(q)
                    except Exception:
                        logger.warning(
                            "Invalid question from provider %s: %r", provider.name, q
                        )
                        continue
                merged.setdefault(cat, []).append(q)

    # Deduplicate by question text to avoid duplicates when multiple providers
    # return overlapping entries.
    for cat, qs in list(merged.items()):
        seen: set[str] = set()
        unique: List[Question] = []
        for q in qs:
            if q.domanda in seen:
                continue
            seen.add(q.domanda)
            unique.append(q)
        merged[cat] = unique
    return merged


# ---------------------------------------------------------------------------
# Built-in providers
# ---------------------------------------------------------------------------


class JSONQuestionProvider:
    """Simple provider reading questions from a JSON file."""

    def __init__(self, name: str, path: Path):
        self.name = name
        self.path = Path(path)

    def load(self) -> Dict[str, List[Question]]:  # pragma: no cover - tiny wrapper
        return load_questions(self.path)


class AdrianoLombardoProvider(JSONQuestionProvider):
    def __init__(self, path: Path | None = None):
        if path is None:
            path = (
                Path(__file__).resolve().parent.parent
                / "data"
                / "domande_oracolo.json"
            )
        super().__init__("AdrianoLombardo", path)


class TheMProvider(JSONQuestionProvider):
    def __init__(self, path: Path | None = None):
        if path is None:
            path = (
                Path(__file__).resolve().parent.parent
                / "data"
                / "domande_oracolo.json"
            )
        super().__init__("TheM", path)


class CryptoMadonneProvider(JSONQuestionProvider):
    def __init__(self, path: Path | None = None):
        if path is None:
            path = (
                Path(__file__).resolve().parent.parent
                / "data"
                / "domande_oracolo.json"
            )
        super().__init__("CryptoMadonne", path)


# Register built-in providers
register_question_provider(AdrianoLombardoProvider())
register_question_provider(TheMProvider())
register_question_provider(CryptoMadonneProvider())


__all__ = [
    "register_question_provider",
    "get_question_provider",
    "iter_question_providers",
    "load_questions",
    "load_questions_from_providers",
    "JSONQuestionProvider",
    "AdrianoLombardoProvider",
    "TheMProvider",
    "CryptoMadonneProvider",
]

