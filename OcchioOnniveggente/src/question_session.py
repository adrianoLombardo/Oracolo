from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


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
