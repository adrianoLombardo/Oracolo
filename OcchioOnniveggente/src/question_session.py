from __future__ import annotations

"""Session helper combining question rotation and answer tracking."""

from dataclasses import dataclass, field
import random
from typing import Dict, List, Optional

from .oracle import get_questions
from .retrieval import Question


@dataclass
class QuestionSession:
    """Maintain rotation state and store conversation history."""

    questions: Optional[Dict[str, List[Question]]] = None
    question: Optional[str] = None
    follow_up: Optional[str] = None
    weights: Optional[Dict[str, float]] = None
    answers: List[str] = field(default_factory=list)
    replies: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.questions is None:
            self.questions = get_questions()
        # Normalise categories
        self.questions = {k.lower(): list(v) for k, v in self.questions.items()}
        self._categories = list(self.questions.keys())
        self._index = 0
        self._used: Dict[str, set[int]] = {cat: set() for cat in self._categories}

    def next_question(
        self, category: str | None = None, tags: set[str] | None = None
    ) -> Question | None:
        """Return a question, cycling categories and avoiding repeats.

        When ``tags`` is provided only questions containing *all* the
        requested tags are considered.  If no question in the selected
        category matches, another category is tried unless ``category`` was
        explicitly supplied.
        """

        user_specified = category is not None
        tried: set[str] = set()
        for _ in range(len(self._categories)):
            if category is None:
                if self.weights:
                    cats = [
                        c
                        for c in self._categories
                        if self.weights.get(c, 0) > 0 and c not in tried
                    ]
                    weights = [self.weights.get(c, 0) for c in cats]
                    if not cats:
                        return None
                    category = random.choices(cats, weights=weights, k=1)[0]
                else:
                    category = self._categories[self._index]
                    self._index = (self._index + 1) % len(self._categories)
                    if category in tried:
                        category = None
                        continue
            cat = category.lower()
            qs = self.questions.get(cat, [])
            if tags:
                qs = [q for q in qs if q.tag and tags <= set(q.tag)]
            if qs:
                used = self._used.setdefault(cat, set())
                remaining = [i for i in range(len(qs)) if i not in used]
                if not remaining:
                    used.clear()
                    remaining = list(range(len(qs)))
                idx = random.choice(remaining)
                used.add(idx)
                return qs[idx]
            if user_specified:
                return None
            tried.add(category)
            category = None
        return None

    def record_answer(self, answer: str, reply: str | None = None) -> None:
        """Store ``answer`` and optional user ``reply``."""

        self.answers.append(answer)
        if reply is not None:
            self.replies.append(reply)
