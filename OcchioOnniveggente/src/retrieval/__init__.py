"""Retrieval package exposing search utilities and data models."""

from .models import Chunk, Context, Question, QuestionProvider
from .providers import (
    JSONQuestionProvider,
    AdrianoLombardoProvider,
    CryptoMadonneProvider,
    TheMProvider,
    get_question_provider,
    iter_question_providers,
    load_questions,
    load_questions_from_providers,
    register_question_provider,
)
from .search import _embed_texts, _simple_sentences, retrieve

__all__ = [
    "Chunk",
    "Context",
    "Question",
    "QuestionProvider",
    "JSONQuestionProvider",
    "AdrianoLombardoProvider",
    "CryptoMadonneProvider",
    "TheMProvider",
    "get_question_provider",
    "iter_question_providers",
    "load_questions",
    "load_questions_from_providers",
    "register_question_provider",
    "_embed_texts",
    "_simple_sentences",
    "retrieve",
]
