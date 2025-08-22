"""Domain relevance utilities for the Oracle.

This module provides helpers to decide whether a user's question is
pertinent to the topics covered by the Oracle (neuroscience,
neuroaesthetics, contemporary art and the universe).  The validation is
performed in three steps:

1. **Keyword overlap** – the question must contain at least a minimal
   amount of domain specific words.
2. **Embedding similarity** – OpenAI embeddings are used to compare the
   question with a short description of the allowed domain.
3. **Retrieval validation** – ``retrieval.retrieve`` is called and the
   question is considered off–topic when no document with a reasonable
   score is found.

Each step is intentionally lightweight; failing any of them marks the
question as not pertinent and avoids a potentially expensive LLM call.
"""

from __future__ import annotations

import math
import re
from typing import Iterable, List, Tuple

import openai

from .retrieval import retrieve

# ---------------------------------------------------------------------------
# Keyword based filtering
# ---------------------------------------------------------------------------

# A minimal vocabulary for the accepted domain.  The list is purposely short
# and can easily be extended in the future or loaded from configuration.
DOMAIN_KEYWORDS = {
    "neuroscienza",
    "neuroscienze",
    "neuroestetica",
    "arte",
    "arte contemporanea",
    "universo",
    "cosmo",
    "cervello",
    "brain",
    "neuroscience",
    "contemporary",
    "universe",
}


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[\w']+", text.lower())


def keyword_score(question: str, keywords: Iterable[str] = DOMAIN_KEYWORDS) -> float:
    """Return the ratio of tokens in *question* that appear in *keywords*."""

    toks = _tokenize(question)
    if not toks:
        return 0.0
    kw = {k.lower() for k in keywords}
    hits = sum(1 for t in toks if t in kw)
    return hits / max(len(toks), 1)


def keyword_relevant(question: str, min_overlap: float = 0.1) -> bool:
    return keyword_score(question) >= min_overlap


# ---------------------------------------------------------------------------
# Embedding similarity
# ---------------------------------------------------------------------------

def _cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def embedding_relevant(
    question: str,
    *,
    client,
    model: str,
    threshold: float = 0.3,
    domain_text: str = "neuroscienze, neuroestetica, arte contemporanea, universo",
) -> bool:
    """Compare embeddings between the question and ``domain_text``.

    Any failure in the embedding API results in a permissive ``True`` so that
    the system can fall back to other checks or continue working offline.
    """

    try:
        q_emb = client.embeddings.create(model=model, input=question).data[0].embedding
        d_emb = client.embeddings.create(model=model, input=domain_text).data[0].embedding
        sim = _cosine(q_emb, d_emb)
        return sim >= threshold
    except Exception:
        # Embeddings are optional; in case of network/API errors assume the
        # question might be relevant and let other checks decide.
        return True


# ---------------------------------------------------------------------------
# High level validator
# ---------------------------------------------------------------------------

def validate_question(
    question: str,
    *,
    client,
    emb_model: str,
    docstore_path: str | None,
    top_k: int = 3,
    kw_threshold: float = 0.1,
    emb_threshold: float = 0.3,
    retr_threshold: float = 0.1,
) -> Tuple[bool, List[dict]]:
    """Return ``(is_valid, context)`` for *question*.

    ``context`` is the list returned by :func:`retrieve` when the question is
    accepted.  When any check fails ``context`` is an empty list.
    """

    if not keyword_relevant(question, kw_threshold):
        return False, []

    if not embedding_relevant(
        question, client=client, model=emb_model, threshold=emb_threshold
    ):
        return False, []

    if not docstore_path:
        return True, []

    try:
        ctx = retrieve(question, docstore_path, top_k=top_k)
    except Exception:
        ctx = []

    if not ctx:
        return False, []

    best = float(ctx[0].get("score", 0.0))
    if best < retr_threshold:
        return False, []

    return True, ctx

