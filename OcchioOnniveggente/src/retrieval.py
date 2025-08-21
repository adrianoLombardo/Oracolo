# src/retrieval.py
from __future__ import annotations
import json, math, re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple, Iterable, Optional

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


def _make_chunks(text: str, max_chars: int = 700) -> List[str]:
    # spezza in frasi e ricompone in blocchi ~700 caratteri
    sents = _simple_sentences(text)
    chunks, buf = [], ""
    for s in sents:
        if not buf:
            buf = s
        elif len(buf) + 1 + len(s) <= max_chars:
            buf += " " + s
        else:
            chunks.append(buf)
            buf = s
    if buf:
        chunks.append(buf)
    # fallback se non ci sono punti
    if not chunks and text.strip():
        raw = text.strip()
        chunks = [raw[i:i+max_chars] for i in range(0, len(raw), max_chars)]
    return chunks


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


def retrieve(query: str, docstore_path: str | Path, top_k: int = 3) -> List[Dict]:
    """
    Ritorna una lista di dict:
      [{"id": ..., "text": ..., "score": ...}, ...]
    """
    docs = _load_index(docstore_path)
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
        for c in _make_chunks(txt, max_chars=700):
            chunks.append(Chunk(did, c))

    if not chunks:
        return []

    # BM25 se disponibile
    if BM25Okapi is not None:
        tokenized = [_tokenize(c.text) for c in chunks]
        bm = BM25Okapi(tokenized)  # type: ignore
        scores = bm.get_scores(_tokenize(query))  # type: ignore
        pairs = list(zip(chunks, [float(s) for s in scores]))
        pairs.sort(key=lambda x: x[1], reverse=True)
        best = pairs[:top_k]
    else:
        best = _score_fallback(query, chunks, top_k)

    out = []
    for ch, sc in best:
        out.append({"id": ch.doc_id, "text": ch.text, "score": float(sc)})
    return out
