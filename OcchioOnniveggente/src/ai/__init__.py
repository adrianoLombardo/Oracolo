from typing import Iterable, Tuple


def is_relevant(question: str, allowed_topics: Iterable[str], refusal_message: str) -> Tuple[bool, str]:
    """Return True if question contains any allowed topic, else False and refusal message."""
    lowered = question.lower()
    for topic in allowed_topics:
        if topic.lower() in lowered:
            return True, ""
    return False, refusal_message
