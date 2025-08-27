"""Search and ranking utilities for document retrieval."""

from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

from ..cache import cache_get_json, cache_set_json
from ..openai_async import run
from ..utils.math_utils import cosine_similarity
from .models import Chunk

try:  # pragma: no cover - optional dependency
    from ..metadata_store import MetadataStore  # type: ignore
except Exception:  # pragma: no cover - optional
    MetadataStore = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from rank_bm25 import BM25Okapi  # type: ignore
except Exception:  # pragma: no cover
    BM25Okapi = None

try:  # pragma: no cover - optional dependency
    from rapidfuzz import fuzz  # type: ignore
except Exception:  # pragma: no cover
    fuzz = None

try:  # pragma: no cover - optional dependency
    from sentence_transformers import CrossEncoder  # type: ignore
except Exception:  # pragma: no cover
    CrossEncoder = None

logger = logging.getLogger(__name__)

_CROSS_ENCODER_CACHE: Dict[str, Any] = {}


def _simple_sentences(txt: str) -> List[str]:
    """Split text into simple sentences."""
    parts = re.split(r"(?<=[\.\!\?])\s+|\n{2,}", txt)
    return [p.strip() for p in parts if p and not p.isspace()]


def _tokenize(s: str) -> List[str]:
    return [t for t in re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9]+", s.lower()) if t]


def _configured_docstore_path() -> Path | None:
    """Return the docstore path from settings files, if any."""
    root = Path(__file__).resolve().parent.parent
    for name in ("settings.local.yaml", "settings.yaml"):
        cfg = root / name
        if not cfg.exists():
            continue
        try:
            data = yaml.safe_load(cfg.read_text(encoding="utf-8")) or {}
        except Exception:
            continue
        path = data.get("docstore_path")
        if path:
            candidate = (cfg.parent / path).resolve()
            if candidate.exists():
                return candidate
    return None


def _load_index(path: str | Path) -> List[Dict]:
    p = Path(path)
    if not p.exists():
        cfg_path = _configured_docstore_path()
        if cfg_path is not None:
            p = cfg_path
        if not p.exists():
            logger.warning(
                "Index file not found at %s. Run `scripts/ingest_docs.py` to populate it or set "
                "`docstore_path` in settings.yaml.",
                p,
            )
            try:
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text("[]", encoding="utf-8")
            except Exception:
                pass
            return []
    data = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(data, list):
        documents = data
    elif isinstance(data, dict) and "documents" in data:
        documents = data["documents"]
    else:
        documents = []
    if not documents:
        logger.warning("Docstore is empty: %s", p)
    return documents


def _make_chunks(text: str, max_chars: int = 800, overlap_ratio: float = 0.1) -> List[str]:
    """Split text into semantically coherent chunks."""
    if not text.strip():
        return []

    paras = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    chunks: List[str] = []
    buf = ""
    for p in paras:
        if not buf:
            buf = p
        elif len(buf) + 1 + len(p) <= max_chars:
            buf += "\n" + p
        else:
            chunks.append(buf)
            ov = int(len(buf) * overlap_ratio)
            buf = (buf[-ov:] + "\n" + p) if ov > 0 else p
    if buf:
        chunks.append(buf)

    out: List[str] = []
    for ch in chunks:
        if len(ch) <= max_chars:
            out.append(ch)
            continue
        sents = _simple_sentences(ch)
        tmp = ""
        for s in sents:
            if not tmp:
                tmp = s
            elif len(tmp) + 1 + len(s) <= max_chars:
                tmp += " " + s
            else:
                out.append(tmp)
                ov = int(len(tmp) * overlap_ratio)
                tmp = (tmp[-ov:] + " " + s) if ov > 0 else s
        if tmp:
            out.append(tmp)
    return out


def _embed_texts(
    client: Any, model: str, texts: List[str], *, cache_dir: str | Path | None = None
) -> List[np.ndarray]:
    if not texts:
        return []

    results: List[Optional[np.ndarray]] = [None] * len(texts)
    to_compute: List[Tuple[int, str, str, Path | None]] = []
    cdir = Path(cache_dir) if cache_dir else None
    if cdir:
        cdir.mkdir(parents=True, exist_ok=True)
    for idx, txt in enumerate(texts):
        key = hashlib.sha1(txt.encode("utf-8")).hexdigest()
        cache_key = f"emb:{model}:{key}"
        cached_vec = cache_get_json(cache_key)
        if cached_vec is not None:
            results[idx] = np.array(cached_vec, dtype=np.float32)
            continue
        if cdir:
            h = hashlib.sha1(txt.encode("utf-8")).hexdigest()
            fp = cdir / f"{h}.npy"
            if fp.exists():
                try:
                    results[idx] = np.load(fp)
                    continue
                except Exception:
                    pass
            to_compute.append((idx, txt, cache_key, fp))
        else:
            to_compute.append((idx, txt, cache_key, None))

    if to_compute:
        resp = run(
            client.embeddings.create,  # type: ignore[attr-defined]
            model=model,
            input=[t for _, t, _, _ in to_compute],
        )
        for (idx, _txt, cache_key, fp), item in zip(to_compute, resp.data):
            vec = np.array(getattr(item, "embedding", []), dtype=np.float32)
            results[idx] = vec
            cache_set_json(cache_key, vec.tolist(), ttl=86400)
            if fp is not None:
                try:
                    np.save(fp, vec)
                except Exception:
                    pass

    return [r if r is not None else np.zeros(0, dtype=np.float32) for r in results]


def _rewrite_query(client: Any, model: str, query: str, n: int = 2) -> List[str]:
    """Generate ``n`` short rewrites for ``query`` using ``model``."""
    raw_key = f"{model}:{n}:{query}".encode("utf-8")
    cache_key = "rq:" + hashlib.sha1(raw_key).hexdigest()
    cached = cache_get_json(cache_key)
    if isinstance(cached, list) and all(isinstance(x, str) for x in cached):
        return cached[:n]

    prompt = (
        "Fornisci {n} riformulazioni concise della seguente query in italiano o inglese, una per riga.\n"
        "Query: {q}"
    ).format(n=n, q=query)
    txt = ""
    try:
        resp = run(
            client.responses.create,  # type: ignore[attr-defined]
            model=model,
            input=prompt,
        )
        txt = getattr(resp, "output_text", "")
    except Exception:
        try:
            resp = run(
                client.chat.completions.create,  # type: ignore[attr-defined]
                model=model,
                messages=[{"role": "user", "content": prompt}],
            )
            txt = resp.choices[0].message.get("content", "")  # type: ignore[attr-defined]
        except Exception:
            return []

    rewrites = [t.strip() for t in txt.splitlines() if t.strip()]
    cache_set_json(cache_key, rewrites, ttl=3600)
    return rewrites[:n]


def _score_fallback(query: str, chunks: List[Chunk], top_k: int) -> List[Tuple[Chunk, float]]:
    if fuzz is None:
        scores = [(c, 1.0) for c in chunks]
    else:
        scores = [(c, float(fuzz.token_set_ratio(query, c.text))) for c in chunks]
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]


def _compress_passage(text: str, query: str, max_sents: int = 5) -> str:
    sents = _simple_sentences(text)
    if not sents:
        return text
    if fuzz is None:
        sents.sort(key=lambda s: -len(s))
    else:
        sents.sort(key=lambda s: fuzz.partial_ratio(query, s), reverse=True)
    return " ".join(sents[:max_sents])


def _rerank_cross_encoder(
    query: str, items: List[Tuple[Chunk, float]], model: str
) -> List[Tuple[Chunk, float]]:
    if CrossEncoder is None:
        return items
    if model not in _CROSS_ENCODER_CACHE:
        try:
            _CROSS_ENCODER_CACHE[model] = CrossEncoder(model)  # type: ignore[call-arg]
        except Exception:
            return items
    ce: Any = _CROSS_ENCODER_CACHE[model]
    pairs = [[query, c.text] for c, _ in items]
    scores = ce.predict(pairs)  # type: ignore[call-arg]
    reranked = [(c, float(s)) for (c, _), s in zip(items, scores)]
    reranked.sort(key=lambda x: x[1], reverse=True)
    return reranked


def retrieve(
    query: str,
    docstore_path: str | Path,
    *,
    top_k: int = 3,
    topic: str | None = None,
    client: Any | None = None,
    embed_model: str | None = None,
    rewrite_model: str | None = None,
    rerank_model: str | None = None,
    store: Any | None = None,
) -> List[Dict]:
    """Hybrid BM25 + embedding retrieval returning metadata for citations."""
    if client is not None and rewrite_model:
        try:
            rewrites = _rewrite_query(client, rewrite_model, query)
            if rewrites:
                query = query + " " + " ".join(rewrites)
        except Exception:
            pass

    if store is not None:
        docs = store.get_documents() if hasattr(store, "get_documents") else []
    else:
        docs = _load_index(docstore_path)
    if topic:
        tnorm = str(topic).lower()
        docs = [d for d in docs if str(d.get("topic", "")).lower() == tnorm]
    if not docs or not query.strip():
        return []

    chunks: List[Chunk] = []
    for d in docs:
        try:
            did = str(d.get("id", ""))
            txt = str(d.get("text", ""))
        except Exception:
            continue
        meta = {k: d.get(k) for k in ("title", "lang", "date", "topic")}
        for c in _make_chunks(txt):
            chunks.append(Chunk(did, c, meta))

    if not chunks:
        return []

    if BM25Okapi is not None:
        tokenized = [_tokenize(c.text) for c in chunks]
        bm = BM25Okapi(tokenized)  # type: ignore
        scores = bm.get_scores(_tokenize(query))  # type: ignore
        pairs = list(zip(chunks, [float(s) for s in scores]))
        pairs.sort(key=lambda x: x[1], reverse=True)
        best = pairs[: max(top_k * 4, top_k)]
    else:
        best = _score_fallback(query, chunks, max(top_k * 4, top_k))

    if client is not None and embed_model:
        try:
            cache_dir = Path(docstore_path).with_suffix(".embcache")
            qv = _embed_texts(client, embed_model, [query])[0]
            texts = [c.text for c, _ in best]
            vecs = _embed_texts(client, embed_model, texts, cache_dir=cache_dir)
            sims = [cosine_similarity(qv, v) for v in vecs]
            combo: List[Tuple[Chunk, float]] = []
            for (ch, bm_sc), sim in zip(best, sims):
                combo.append((ch, 0.5 * bm_sc + 0.5 * sim))
            combo.sort(key=lambda x: x[1], reverse=True)
            best = combo[: max(top_k * 4, top_k)]
        except Exception:
            best = best[: max(top_k * 4, top_k)]
    else:
        best = best[: max(top_k * 4, top_k)]

    if rerank_model:
        best = _rerank_cross_encoder(query, best, rerank_model)

    best = best[:top_k]

    out = []
    for ch, sc in best:
        txt = _compress_passage(ch.text, query)
        item = {"id": ch.doc_id, "text": txt, "score": float(sc)}
        item.update(ch.meta or {})
        out.append(item)
    return out


__all__ = ["retrieve", "_simple_sentences", "_embed_texts"]
