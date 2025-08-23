# src/domain.py
from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Any, Iterable, Tuple

import numpy as np

# Usa il tuo retriever se disponibile
try:
    from src.retrieval import retrieve
except Exception:
    def retrieve(query: str, docstore_path: str | Path, top_k: int = 3) -> list[str]:
        return []


# ------------------------- utilità testo ------------------------- #
_WORD_RE = re.compile(r"[a-zàèéìòóù]+", re.IGNORECASE)

def _strip_accents(s: str) -> str:
    nf = unicodedata.normalize("NFD", s or "")
    return "".join(ch for ch in nf if not unicodedata.combining(ch))

def _norm(s: str) -> str:
    return _strip_accents((s or "").lower())

def _tokens(s: str) -> list[str]:
    return _WORD_RE.findall(_norm(s))

def _keyword_overlap_score(text: str, keywords: Iterable[str]) -> float:
    toks = set(_tokens(text))
    kw_tokens: set[str] = set()
    for k in (keywords or []):
        for t in _tokens(k):
            kw_tokens.add(t)
            if len(t) > 4:
                kw_tokens.add(t[:-1])  # micro-stem (es. neuroscienz*)
    if not toks or not kw_tokens:
        return 0.0
    inter = toks.intersection(kw_tokens)
    return len(inter) / max(len(toks), 1)


# ------------------------- embeddings ------------------------- #
def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

def _embed_texts(client: Any, model: str, texts: list[str]) -> list[np.ndarray]:
    if not texts:
        return []
    resp = client.embeddings.create(model=model, input=texts)  # type: ignore[attr-defined]
    vecs: list[np.ndarray] = []
    for item in resp.data:
        emb = np.array(getattr(item, "embedding", []), dtype=np.float32)
        vecs.append(emb)
    return vecs

def _normalize_context_list(ctx: Any) -> list[str]:
    out: list[str] = []
    if not ctx:
        return out
    if isinstance(ctx, str):
        t = ctx.strip()
        if t:
            out.append(t[:1000])
        return out
    if isinstance(ctx, (list, tuple)):
        for it in ctx:
            if isinstance(it, str):
                t = it.strip()
            elif isinstance(it, dict):
                t = str(it.get("text") or it.get("content") or "").strip()
            else:
                t = str(it).strip()
            if t:
                out.append(t[:1000])
        return out
    out.append(str(ctx).strip()[:1000])
    return out


# ------------------------- funzione principale ------------------------- #
def validate_question(
    question: str,
    lang: str | None = None,           # opzionale
    *,
    settings: Any = None,
    client: Any = None,
    docstore_path: str | Path | None = None,
    top_k: int = 3,
    emb_model: str | None = None,      # compat
    embed_model: str | None = None,    # alias
    **_: Any,                          # ignora altri kwargs
) -> Tuple[bool, list[str]]:
    """
    Ritorna (is_ok, context_list).

    Politica:
    - Se settings.domain manca o non ha keywords -> NON filtriamo (sempre OK).
    - Boost se la domanda contiene termini ovvi (neuro, cervell*, brain, IA…).
    - Overlap parole chiave (soglia bassa).
    - Opzionale: similarità embedding con i frammenti recuperati.

    Soglie default (se non presenti in settings.domain):
      kw_min_overlap = 0.04
      use_embeddings = False
      emb_min_sim = 0.22
    """
    q_norm = _norm(question)

    # ---- leggi settings.domain ----
    dom = None
    if settings is not None:
        try:
            dom = getattr(settings, "domain", None)
        except Exception:
            try:
                dom = (settings or {}).get("domain")  # type: ignore[attr-defined]
            except Exception:
                dom = None

    enabled = True
    keywords: list[str] = []
    kw_min_overlap = 0.04
    use_embeddings = False
    emb_min_sim = 0.22

    if dom:
        # enabled (default True)
        try:
            enabled = bool(getattr(dom, "enabled", True))
        except Exception:
            try:
                enabled = bool(dom.get("enabled", True))  # type: ignore[attr-defined]
            except Exception:
                enabled = True
        # keywords
        try:
            keywords = list(getattr(dom, "keywords", []) or [])
        except Exception:
            try:
                keywords = list(dom.get("keywords", []) or [])  # type: ignore[attr-defined]
            except Exception:
                keywords = []
        # soglie
        for (attr, var_name, cast, default) in [
            ("kw_min_overlap", "kw_min_overlap", float, kw_min_overlap),
            ("use_embeddings", "use_embeddings", bool, use_embeddings),
            ("emb_min_sim", "emb_min_sim", float, emb_min_sim),
        ]:
            try:
                val = getattr(dom, attr, default)
            except Exception:
                try:
                    val = dom.get(attr, default)  # type: ignore[attr-defined]
                except Exception:
                    val = default
            locals()[var_name] = cast(val)  # type: ignore[misc]

    # Se il filtro non è abilitato → sempre OK
    if not enabled:
        ctx = _try_retrieve(question, settings, docstore_path, top_k)
        return True, ctx

    # Se NON ci sono keywords → non filtriamo (sempre OK)
    if not keywords:
        ctx = _try_retrieve(question, settings, docstore_path, top_k)
        return True, ctx

    # ---- Boost terms (accettazione immediata se compaiono) ----
    boost_terms = [
        "neuro", "neuroscienze", "neuroscience", "cervello", "sinaps",
        "brain", "cortex", "neurone", "neuroni", "neuroestetica",
        "arte", "estetica", "percezione",
        "ia", "intelligenza artificiale", "ai",
        "cosmo", "universo", "stella", "stelle", "mare"
    ]
    if any(bt in q_norm for bt in boost_terms):
        ctx = _try_retrieve(question, settings, docstore_path, top_k)
        return True, ctx

    # ---- Overlap parole chiave ----
    kw_score = _keyword_overlap_score(question, keywords)
    ok = kw_score >= kw_min_overlap

    # ---- Recupera contesto ----
    ctx = _try_retrieve(question, settings, docstore_path, top_k)

    # ---- Embeddings (opzionale) ----
    emb_model = emb_model or embed_model
    if use_embeddings and client is not None and emb_model:
        try:
            to_embed = [question] + ctx[:3]
            vecs = _embed_texts(client, emb_model, to_embed)
            if len(vecs) >= 2:
                qv = vecs[0]
                sims = [_cosine(qv, v) for v in vecs[1:]]
                best_sim = max(sims) if sims else 0.0
                ok = ok or (best_sim >= emb_min_sim)
        except Exception:
            pass

    return ok, ctx


# ------------------------- helper retrieval ------------------------- #
def _try_retrieve(
    question: str,
    settings: Any,
    docstore_path: str | Path | None,
    top_k: int,
) -> list[str]:
    if docstore_path is None and settings is not None:
        try:
            docstore_path = getattr(settings, "docstore_path", None)
        except Exception:
            try:
                docstore_path = settings.get("docstore_path")  # type: ignore[attr-defined]
            except Exception:
                pass
    if not docstore_path:
        return []
    try:
        raw_ctx = retrieve(question, docstore_path, top_k=top_k)
    except Exception:
        raw_ctx = []
    # normalizza
    out: list[str] = []
    for it in raw_ctx or []:
        if isinstance(it, str):
            t = it.strip()
        elif isinstance(it, dict):
            t = str(it.get("text") or it.get("content") or "").strip()
        else:
            t = str(it).strip()
        if t:
            out.append(t[:1000])
    return out
