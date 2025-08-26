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

    def add_question(self, category: str, question: Question) -> None:
        """Add ``question`` to ``category``, creating it if needed."""

        cat = category.lower()
        if cat not in self.questions:
            self.questions[cat] = []
            self._categories.append(cat)
            self._used[cat] = set()
        self.questions[cat].append(question)

    def remove_question(self, category: str, question_id: int) -> None:
        """Remove question by index from ``category``.

        If the category becomes empty it is removed as well.
        """

        cat = category.lower()
        qs = self.questions.get(cat)
        if qs is None:
            return
        if 0 <= question_id < len(qs):
            qs.pop(question_id)
            used = self._used.get(cat, set())
            self._used[cat] = {
                i - 1 if i > question_id else i
                for i in used
                if i != question_id
            }
            if not qs:
                del self.questions[cat]
                self._categories.remove(cat)
                self._used.pop(cat, None)
                if self._categories:
                    self._index %= len(self._categories)
                else:
                    self._index = 0

    def reset_category(self, category: str) -> None:
        """Clear used-question state for ``category``."""

        cat = category.lower()
        if cat in self._used:
            self._used[cat].clear()

    def next_question(self, category: str | None = None) -> Question | None:
        """Return a question, cycling categories and avoiding repeats."""

        if category is None:
            if self.weights:
                cats = [c for c in self._categories if self.weights.get(c, 0) > 0]
                weights = [self.weights.get(c, 0) for c in cats]
                if not cats:
                    return None
                category = random.choices(cats, weights=weights, k=1)[0]
            else:
                category = self._categories[self._index]
                self._index = (self._index + 1) % len(self._categories)
        cat = category.lower()
        qs = self.questions.get(cat, [])
        if not qs:
            return None
        used = self._used.setdefault(cat, set())
        remaining = [i for i in range(len(qs)) if i not in used]
        if not remaining:
            used.clear()
            remaining = list(range(len(qs)))
        idx = random.choice(remaining)
        used.add(idx)
        return qs[idx]

    def record_answer(self, answer: str, reply: str | None = None) -> None:
        """Store ``answer`` and optional user ``reply``."""

        self.answers.append(answer)
        if reply is not None:
            self.replies.append(reply)
