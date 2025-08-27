# src/domain.py
from __future__ import annotations

import re
import time
import unicodedata
from pathlib import Path
from typing import Any, Iterable, Tuple

import numpy as np
from .openai_async import run

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


def _get_field(obj: Any, name: str, default: Any | None = None) -> Any:
    """Access ``name`` from ``obj`` treating dicts and objects uniformly."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    val = getattr(obj, name, default)
    if val is default and hasattr(obj, "__getitem__"):
        try:
            return obj[name]  # type: ignore[index]
        except Exception:
            return default
    return val


def _keyword_overlap_score(text: str, keywords: Iterable[str]) -> float:
    toks = set(_tokens(text))
    kw_tokens: set[str] = set()
    for k in (keywords or []):
        for t in _tokens(k):
            kw_tokens.add(t)
            if len(t) > 4:
                # micro-stem grezzo (es. neuroscienz*):
                kw_tokens.add(t[:-1])
    if not toks or not kw_tokens:
        return 0.0
    inter = toks.intersection(kw_tokens)
    # Normalizza sull'ampiezza più piccola tra domanda e keyword per non
    # penalizzare frasi lunghe o molto corte.
    denom = max(min(len(toks), len(kw_tokens)), 1)
    return len(inter) / denom


from .utils.math_utils import cosine_similarity


# ------------------------- embeddings ------------------------- #


def _embed_texts(client: Any, model: str, texts: list[str]) -> list[np.ndarray]:
    if not texts:
        return []
    # OpenAI embeddings API style
    resp = run(
        client.embeddings.create,  # type: ignore[attr-defined]
        model=model,
        input=texts,
    )
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


def _read_config(settings: Any, top_k: int | None, topic: str | None):
    top_k = int(top_k if top_k is not None else _get_field(settings, "retrieval_top_k", 3))
    dom = _get_field(settings, "domain")
    enabled = True
    keywords: list[str] = []
    weights: dict[str, float] = {"kw": 0.7, "emb": 0.15, "rag": 0.15}
    accept_threshold = 0.75
    clarify_margin = 0.10
    fallback_accept_threshold = 0.4
    emb_min_sim = 0.22
    dom_topic = ""
    dom_profile = ""
    profiles: dict[str, Any] = {}
    always_accept_wake = True
    if dom:
        enabled = bool(_get_field(dom, "enabled", True))
        dom_profile = str(_get_field(dom, "profile", "") or "")
        keywords = list(_get_field(dom, "keywords", []) or [])
        if not keywords and dom_profile:
            profiles = _get_field(dom, "profiles", {}) or {}
            prof_conf = profiles.get(dom_profile, {}) if isinstance(profiles, dict) else {}
            keywords = list(_get_field(prof_conf, "keywords", []) or [])
        weights = dict(_get_field(dom, "weights", weights) or weights)
        accept_threshold = float(_get_field(dom, "accept_threshold", accept_threshold))
        clarify_margin = float(_get_field(dom, "clarify_margin", clarify_margin))
        fallback_accept_threshold = float(
            _get_field(dom, "fallback_accept_threshold", fallback_accept_threshold)
        )
        emb_min_sim = float(_get_field(dom, "emb_min_sim", emb_min_sim))
        dom_topic = str(_get_field(dom, "topic", dom_profile or ""))
        profiles = dict(_get_field(dom, "profiles", {}) or {})
        always_accept_wake = bool(_get_field(dom, "always_accept_wake", True))
    if not keywords and dom_profile and profiles:
        prof_obj = profiles.get(dom_profile, {})
        keywords = list(_get_field(prof_obj, "keywords", []) or [])
        w = _get_field(prof_obj, "weights")
        if w:
            weights = dict(w)
        at = _get_field(prof_obj, "accept_threshold")
        if at is not None:
            accept_threshold = float(at)
        cm = _get_field(prof_obj, "clarify_margin")
        if cm is not None:
            clarify_margin = float(cm)
        fat = _get_field(prof_obj, "fallback_accept_threshold")
        if fat is not None:
            fallback_accept_threshold = float(fat)
    if not dom_topic:
        dom_topic = dom_profile
    use_topic = dom_topic or topic
    wake_phrases: list[str] = []
    w = _get_field(settings, "wake")
    if w:
        for lst in ("it_phrases", "en_phrases"):
            wake_phrases.extend(_get_field(w, lst, []) or [])
    return (
        top_k,
        enabled,
        keywords,
        weights,
        accept_threshold,
        clarify_margin,
        fallback_accept_threshold,
        emb_min_sim,
        use_topic,
        always_accept_wake,
        wake_phrases,
    )


def _embedding_score(question: str, keywords: list[str], client: Any, model: str | None) -> float:
    if client is None or not model or not keywords:
        return 0.0
    try:
        vecs = _embed_texts(client, model, [question, " ".join(keywords)])
        if len(vecs) == 2:
            return cosine_similarity(vecs[0], vecs[1])
    except Exception:
        pass
    return 0.0


def _retrieval_score(
    question: str,
    settings: Any,
    docstore_path: str | Path | None,
    top_k: int,
    topic: str | None,
    client: Any,
    emb_model: str | None,
) -> tuple[list[dict], float, int]:
    ctx = _try_retrieve(question, settings, docstore_path, top_k, topic, client, emb_model)
    hits = len(ctx)
    score = hits / float(top_k or 1)
    return ctx, score, hits


def _adapt_thresholds(
    question: str,
    history: list[dict[str, str]] | None,
    keywords: list[str],
    kw_score: float,
    rag_hits: int,
    emb_sim: float,
    accept_threshold: float,
    fallback_accept_threshold: float,
    weights: dict[str, float],
) -> tuple[float, dict[str, float]]:
    hist_tokens: set[str] = set()
    if history:
        for turn in history[-6:]:
            if turn.get("role") == "user":
                hist_tokens.update(_tokens(turn.get("content", "")))
    if hist_tokens & set(_tokens(" ".join(keywords))):
        weights["kw"] = min(1.0, weights.get("kw", 0.4) + 0.1)
    if history:
        recent_msgs = [t.get("content", "") for t in history[-4:] if t.get("role") == "user"]
        if recent_msgs:
            overlaps = [_keyword_overlap_score(m, keywords) for m in recent_msgs]
            div = sum(1 for s in overlaps if s < 0.2)
            coh = sum(1 for s in overlaps if s > 0.6)
            accept_threshold += 0.05 * div - 0.05 * coh
            accept_threshold = min(max(accept_threshold, 0.1), 0.9)
    q_tok_count = len(_tokens(question))
    if q_tok_count <= 4 and kw_score > 0:
        accept_threshold -= 0.05 * (5 - q_tok_count)
    if emb_sim == 0.0 and rag_hits == 0 and kw_score > 0:
        accept_threshold = min(accept_threshold, fallback_accept_threshold)
    return accept_threshold, weights


# ------------------------- funzione principale ------------------------- #
def validate_question(
    question: str,
    lang: str | None = None,
    *,
    settings: Any = None,
    client: Any = None,
    docstore_path: str | Path | None = None,
    top_k: int | None = None,
    emb_model: str | None = None,
    embed_model: str | None = None,
    topic: str | None = None,
    history: list[dict[str, str]] | None = None,
    log_path: str | Path | None = "data/logs/validator.log",
    **_: Any,
) -> Tuple[bool, list[dict], bool, str, str | None]:
    """
    Valida la domanda usando tre segnali: overlap parole-chiave (kw), similarità
    embedding (emb) e recupero documenti (rag). Ritorna:
      (is_ok, context_list, needs_clarification, reason_str, topic_suggestion or None).
    """
    q_norm = _norm(question)
    (
        top_k,
        enabled,
        keywords,
        weights,
        accept_threshold,
        clarify_margin,
        fallback_accept_threshold,
        emb_min_sim,
        use_topic,
        always_accept_wake,
        wake_phrases,
    ) = _read_config(settings, top_k, topic)
    emb_model = emb_model or embed_model
    if always_accept_wake and wake_phrases:
        cleaned = re.sub(r"[\W_]+", "", q_norm)
        for phr in wake_phrases:
            if re.sub(r"[\W_]+", "", _norm(phr)) == cleaned:
                ctx, _, _ = _retrieval_score(
                    question, settings, docstore_path, top_k, use_topic, client, emb_model
                )
                return True, ctx, False, "wake", None
    if not enabled:
        ctx, _, _ = _retrieval_score(
            question, settings, docstore_path, top_k, use_topic, client, emb_model
        )
        return True, ctx, False, "disabled", None
    if not keywords:
        ctx, _, _ = _retrieval_score(
            question, settings, docstore_path, top_k, use_topic, client, emb_model
        )
        return True, ctx, False, "no keywords", None
    kw_score = _keyword_overlap_score(question, keywords)
    ctx, rag_score, rag_hits = _retrieval_score(
        question, settings, docstore_path, top_k, use_topic, client, emb_model
    )
    emb_sim = _embedding_score(question, keywords, client, emb_model)
    accept_threshold, weights = _adapt_thresholds(
        question,
        history,
        keywords,
        kw_score,
        rag_hits,
        emb_sim,
        accept_threshold,
        fallback_accept_threshold,
        weights,
    )
    score = (
        weights.get("kw", 0.0) * kw_score
        + weights.get("emb", 0.0) * emb_sim
        + weights.get("rag", 0.0) * rag_score
    )
    thr = accept_threshold
    ok = score >= thr
    clarify = (not ok) and (score >= (thr - clarify_margin))
    reason = f"kw={kw_score:.2f} emb={emb_sim:.2f} rag={rag_score:.2f} score={score:.2f} thr={thr:.2f}"
    topic_suggestion = ""
    if kw_score == 0 and rag_hits == 0:
        ok = False
        clarify = score >= (thr - clarify_margin)
        ctx = []
        reason += f" kw0 rag_hits={rag_hits}"
    if clarify and docstore_path:
        alt_ctx = _try_retrieve(
            question, settings, docstore_path, top_k, None, client, emb_model
        )
        for it in alt_ctx:
            t = str(it.get("topic") or "")
            if t and t != use_topic:
                topic_suggestion = t
                break
        if topic_suggestion:
            reason += f" suggest={topic_suggestion}"
    if log_path:
        try:
            Path(log_path).parent.mkdir(parents=True, exist_ok=True)
            with Path(log_path).open("a", encoding="utf-8") as f:
                f.write(f"{int(time.time())}\t{int(ok)}\t{int(clarify)}\t{reason}\t{question}\n")
        except Exception:
            pass
    return ok, ctx, clarify, reason, topic_suggestion


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
        docstore_path = _get_field(settings, "docstore_path")
    if not docstore_path:
        return []
    rewrite_model = _get_field(settings, "retrieval_rewrite_model")
    rerank_model = _get_field(settings, "retrieval_rerank_model")
    try:
        raw_ctx = retrieve(
            question,
            docstore_path,
            top_k=top_k,
            topic=topic,
            client=client,
            embed_model=emb_model,
            rewrite_model=rewrite_model,
            rerank_model=rerank_model,
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
