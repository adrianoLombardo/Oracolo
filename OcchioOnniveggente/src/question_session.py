from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

"""Utilities for serving non-repeating questions within a session."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List
import random

from .retrieval import Question



@dataclass
class QuestionSession:

    """Simple container tracking a question and user answers.

    The real project keeps a more sophisticated structure but for the purposes
    of the tests we only need to remember the answers provided by the oracle and
    any replies from the user.  Each call to :meth:`record_answer` appends the
    pair to the internal lists so that callers can later inspect them.
    """

    question: str
    follow_up: str | None = None
    answers: List[str] = field(default_factory=list)
    replies: List[str] = field(default_factory=list)

    def record_answer(self, answer: str, reply: str | None = None) -> None:
        """Store ``answer`` and optional user ``reply``.

        Parameters
        ----------
        answer:
            Text produced by the oracle in response to the main question.
        reply:
            The user's follow-up answer.  When provided it is stored in the
            :attr:`replies` list so the session retains the full interaction.
        """

        self.answers.append(answer)
        if reply is not None:
            self.replies.append(reply)

    """Maintain per-category question rotation state.

    Parameters
    ----------
    questions:
        Mapping from category name to the list of :class:`Question` objects
        belonging to that category. Category keys are treated case-insensitively.
    """

    questions: Dict[str, List[Question]]
    _asked_ids: Dict[str, set[int]] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        # Normalise categories to lowercase and initialise tracking sets
        self.questions = {cat.lower(): list(qs) for cat, qs in self.questions.items()}
        self._asked_ids = {cat: set() for cat in self.questions}

    def next_question(self, category: str) -> Question | None:
        """Return a question from ``category`` avoiding immediate repeats.

        Once all questions in the category have been served the internal pool is
        reset so that a new cycle can begin.
        """

        cat = category.lower()
        qs = self.questions.get(cat)
        if not qs:
            return None

        asked = self._asked_ids.setdefault(cat, set())
        remaining = [i for i in range(len(qs)) if i not in asked]
        if not remaining:
            asked.clear()
            remaining = list(range(len(qs)))
        idx = random.choice(remaining)
        asked.add(idx)
        return qs[idx]

