# src/retrieval.py
from __future__ import annotations
import json, math, re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple, Iterable, Optional, Any
import numpy as np

# Prova librerie migliori, ma non sono obbligatorie
try:
    from rank_bm25 import BM25Okapi  # pip install rank-bm25
except Exception:
    BM25Okapi = None

try:
    from rapidfuzz import fuzz  # pip install rapidfuzz
except Exception:
    fuzz = None


@dataclass
class Chunk:
    doc_id: str
    text: str
    meta: Dict[str, Any]


def _simple_sentences(txt: str) -> List[str]:
    # split robusto su righe/punteggiatura
    parts = re.split(r'(?<=[\.\!\?])\s+|\n{2,}', txt)
    return [p.strip() for p in parts if p and not p.isspace()]


def _tokenize(s: str) -> List[str]:
    return [t for t in re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9]+", s.lower()) if t]


def _load_index(path: str | Path) -> List[Dict]:
    p = Path(path)
    if not p.exists():
        return []
    data = json.loads(p.read_text(encoding="utf-8"))
    # attesi: [{"id": "...", "text": "..."}]
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "documents" in data:
        return data["documents"]
    return []


def _make_chunks(text: str, max_chars: int = 800, overlap: int = 80) -> List[str]:
    sents = _simple_sentences(text)
    chunks: List[str] = []
    buf = ""
    for s in sents:
        if not buf:
            buf = s
        elif len(buf) + 1 + len(s) <= max_chars:
            buf += " " + s
        else:
            chunks.append(buf)
            buf = (buf[-overlap:] + " " + s) if overlap and len(buf) > overlap else s
    if buf:
        chunks.append(buf)
    if not chunks and text.strip():
        raw = text.strip()
        step = max_chars - overlap
        for i in range(0, len(raw), step):
            chunks.append(raw[i : i + max_chars])
    return chunks


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _embed_texts(client: Any, model: str, texts: List[str]) -> List[np.ndarray]:
    if not texts:
        return []
    resp = client.embeddings.create(model=model, input=texts)  # type: ignore[attr-defined]
    vecs: List[np.ndarray] = []
    for item in resp.data:
        vec = np.array(getattr(item, "embedding", []), dtype=np.float32)
        vecs.append(vec)
    return vecs


def _score_fallback(query: str, chunks: List[Chunk], top_k: int) -> List[Tuple[Chunk, float]]:
    # Fallback super semplice: token overlap + (opz.) fuzzy
    qtok = set(_tokenize(query))
    scored: List[Tuple[Chunk, float]] = []
    for ch in chunks:
        toks = set(_tokenize(ch.text))
        inter = len(qtok & toks)
        score = float(inter)
        if fuzz is not None:
            # aggiungi un pizzico di similarità fuzzy (max con maggiore peso)
            score += 0.01 * fuzz.partial_ratio(query, ch.text)
        if score > 0:
            scored.append((ch, score))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]


def retrieve(
    query: str,
    docstore_path: str | Path,
    top_k: int = 3,
    *,
    topic: str | None = None,
    client: Any | None = None,
    embed_model: str | None = None,
) -> List[Dict]:
    """Hybrid BM25 + embedding retrieval returning metadata for citations."""
    docs = _load_index(docstore_path)
    if topic:
        tnorm = str(topic).lower()
        docs = [d for d in docs if str(d.get("topic", "")).lower() == tnorm]
    if not docs or not query.strip():
        return []

    # prepara chunk
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

    # BM25 se disponibile
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
            to_embed = [query] + [c.text for c, _ in best]
            vecs = _embed_texts(client, embed_model, to_embed)
            if len(vecs) >= 2:
                qv = vecs[0]
                sims = [_cosine(qv, v) for v in vecs[1:]]
                combo: List[Tuple[Chunk, float]] = []
                for (ch, bm_sc), sim in zip(best, sims):
                    combo.append((ch, 0.5 * bm_sc + 0.5 * sim))
                combo.sort(key=lambda x: x[1], reverse=True)
                best = combo[:top_k]
            else:
                best = best[:top_k]
        except Exception:
            best = best[:top_k]
    else:
        best = best[:top_k]

    out = []
    for ch, sc in best:
        item = {"id": ch.doc_id, "text": ch.text, "score": float(sc)}
        item.update(ch.meta or {})
        out.append(item)
    return out
