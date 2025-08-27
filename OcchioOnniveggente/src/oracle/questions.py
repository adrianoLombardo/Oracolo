from __future__ import annotations

"""Utilities for loading and serving questions by category and context."""

from typing import Dict, List
import random

from ..retrieval import Question, load_questions, Context

# Load questions once at import time. ``load_questions`` may return either a
# mapping ``{category: [Question, ...]}`` or ``{Context: {category: [...]}}``.
_loaded = load_questions()
if _loaded and isinstance(next(iter(_loaded.values())), list):  # type: ignore[arg-type]
    _QUESTIONS_BY_CONTEXT: Dict[Context, Dict[str, List[Question]]] = {
        Context.GENERIC: _loaded  # type: ignore[assignment]
    }
else:
    _QUESTIONS_BY_CONTEXT = _loaded  # type: ignore[assignment]

# Track which questions have been served per category to avoid immediate
# repetitions.
_USED_QUESTIONS: Dict[str, set[int]] = {}


def get_questions(context: Context | str | None = None) -> Dict[str, List[Question]]:
    """Return questions grouped by category for ``context``."""

    if context is None:
        ctx = Context.GENERIC
    elif isinstance(context, Context):
        ctx = context
    else:
        ctx = Context.from_str(context)
    return _QUESTIONS_BY_CONTEXT.get(ctx, {})


def random_question(category: str, context: Context | str | None = None) -> Question | None:
    """Return a random question from ``category`` without immediate repeats."""

    questions = get_questions(context).get(category, [])
    if not questions:
        return None

    used = _USED_QUESTIONS.setdefault(category, set())
    available = [i for i in range(len(questions)) if i not in used]
    if not available:
        used.clear()
        available = list(range(len(questions)))
    idx = random.choice(available)
    used.add(idx)
    return questions[idx]


__all__ = ["get_questions", "random_question", "_USED_QUESTIONS"]
