# src/domain.py
from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Any, Iterable, Tuple, cast
import time

import numpy as np

# Usa il tuo retriever se disponibile
try:
    from src.retrieval import retrieve
except Exception:
    def retrieve(
        query: str,
        docstore_path: str | Path,
        top_k: int = 3,
        *,
        topic: str | None = None,
        client: Any | None = None,
        embed_model: str | None = None,
    ) -> list[dict]:
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
    lang: str | None = None,
    *,
    settings: Any = None,
    client: Any = None,
    docstore_path: str | Path | None = None,
    top_k: int = 3,
    emb_model: str | None = None,
    embed_model: str | None = None,
    topic: str | None = None,
    history: list[dict[str, str]] | None = None,
    log_path: str | Path | None = "data/logs/validator.log",
    **_: Any,
) -> Tuple[bool, list[dict], bool, str]:
    """Valida la domanda usando tre segnali e logga la decisione.

    Ritorna (is_ok, context_list, needs_clarification, reason).
    """
    q_norm = _norm(question)

    # ---- leggi settings.domain ----
    dom = None
    if settings is not None:
        if isinstance(settings, dict):
            dom = settings.get("domain")
        else:
            dom = getattr(settings, "domain", None)

    enabled = True
    keywords: list[str] = []
    weights: dict[str, float] = {"kw": 0.4, "emb": 0.3, "rag": 0.3}
    accept_threshold = 0.5
    clarify_margin = 0.15
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
        try:
            keywords = list(getattr(dom, "keywords", []) or [])
        except Exception:
            try:
                keywords = list(dom.get("keywords", []) or [])  # type: ignore[attr-defined]
            except Exception:
                keywords = []
        try:
            weights = dict(getattr(dom, "weights", weights))
        except Exception:
            try:
                weights = dict(dom.get("weights", weights))  # type: ignore[attr-defined]
            except Exception:
                weights = weights
        for attr, var_name, caster, default in [
            ("accept_threshold", "accept_threshold", float, accept_threshold),
            ("clarify_margin", "clarify_margin", float, clarify_margin),
            ("emb_min_sim", "emb_min_sim", float, emb_min_sim),
        ]:
            val = default
            if isinstance(dom, dict):
                val = dom.get(attr, default)
            else:
                val = getattr(dom, attr, default)
            try:
                val = caster(val)
            except Exception:
                val = default
            if var_name == "accept_threshold":
                accept_threshold = cast(float, val)
            elif var_name == "clarify_margin":
                clarify_margin = cast(float, val)
            elif var_name == "emb_min_sim":
                emb_min_sim = cast(float, val)

    # Se il filtro non è abilitato → sempre OK
    if not enabled:
        ctx = _try_retrieve(question, settings, docstore_path, top_k, topic, client, emb_model or embed_model)
        return True, ctx, False, "disabled"

    # Se NON ci sono keywords → non filtriamo (sempre OK)
    if not keywords:
        ctx = _try_retrieve(question, settings, docstore_path, top_k, topic, client, emb_model or embed_model)
        return True, ctx, False, "no keywords"

    # ---- Boost terms (accettazione immediata se compaiono) ----
    boost_terms = [
        "neuro", "neuroscienze", "neuroscience", "cervello", "sinaps",
        "brain", "cortex", "neurone", "neuroni", "neuroestetica",
        "arte", "estetica", "percezione",
        "ia", "intelligenza artificiale", "ai",
        "cosmo", "universo", "stella", "stelle", "mare"
    ]
    if any(bt in q_norm for bt in boost_terms):
        ctx = _try_retrieve(question, settings, docstore_path, top_k, topic, client, emb_model or embed_model)
        return True, ctx, False, "boost"

    # ---- Overlap parole chiave ----
    kw_score = _keyword_overlap_score(question, keywords)

    # segnali dinamici
    hist_tokens: set[str] = set()
    if history:
        for turn in history[-6:]:
            if turn.get("role") == "user":
                hist_tokens.update(_tokens(turn.get("content", "")))
    if hist_tokens & set(_tokens(" ".join(keywords))):
        weights["kw"] += 0.1

    ctx = _try_retrieve(question, settings, docstore_path, top_k, topic, client, emb_model or embed_model)
    rag_hits = len(ctx)
    rag_score = rag_hits / float(top_k or 1)

    emb_model = emb_model or embed_model
    emb_sim = 0.0
    if client is not None and emb_model and keywords:
        try:
            vecs = _embed_texts(client, emb_model, [question, " ".join(keywords)])
            if len(vecs) == 2:
                emb_sim = _cosine(vecs[0], vecs[1])
        except Exception:
            pass

    score = weights.get("kw", 0.0) * kw_score + weights.get("emb", 0.0) * emb_sim + weights.get("rag", 0.0) * rag_score
    ok = score >= accept_threshold
    clarify = (not ok) and (score >= accept_threshold - clarify_margin)
    reason = f"kw={kw_score:.2f} emb={emb_sim:.2f} rag={rag_score:.2f} score={score:.2f}"

    if log_path:
        try:
            Path(log_path).parent.mkdir(parents=True, exist_ok=True)
            with Path(log_path).open("a", encoding="utf-8") as f:
                f.write(f"{int(time.time())}\t{int(ok)}\t{int(clarify)}\t{reason}\t{question}\n")
        except Exception:
            pass

    return ok, ctx, clarify, reason


# ------------------------- helper retrieval ------------------------- #
def _try_retrieve(
    question: str,
    settings: Any,
    docstore_path: str | Path | None,
    top_k: int,
    topic: str | None,
    client: Any,
    emb_model: str | None,
) -> list[dict]:
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
        raw_ctx = retrieve(
            question,
            docstore_path,
            top_k=top_k,
            topic=topic,
            client=client,
            embed_model=emb_model,
        )
    except Exception:
        raw_ctx = []
    # normalizza
    out: list[dict] = []
    for it in raw_ctx or []:
        if isinstance(it, dict):
            t = str(it.get("text") or it.get("content") or "").strip()
            if not t:
                continue
            it = dict(it)
            it["text"] = t[:1000]
            out.append(it)
        else:
            t = str(it).strip()
            if t:
                out.append({"text": t[:1000]})
    return out
