# src/retrieval.py
from __future__ import annotations
import json, math, re, hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple, Iterable, Optional, Any
import numpy as np


try:
    from .metadata_store import MetadataStore
except Exception:  # pragma: no cover - optional
    MetadataStore = None  # type: ignore


from .cache import cache_get_json, cache_set_json
from .openai_async import run


# Prova librerie migliori, ma non sono obbligatorie
try:
    from rank_bm25 import BM25Okapi  # pip install rank-bm25
except Exception:
    BM25Okapi = None

try:
    from rapidfuzz import fuzz  # pip install rapidfuzz
except Exception:
    fuzz = None

# Reranker opzionale basato su mini cross-encoder
try:
    from sentence_transformers import CrossEncoder  # pip install sentence-transformers
except Exception:  # pragma: no cover - libreria facoltativa
    CrossEncoder = None

_CROSS_ENCODER_CACHE: Dict[str, Any] = {}


logger = logging.getLogger(__name__)


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
        logger.warning("Index file not found: %s", p)
        return []
    data = json.loads(p.read_text(encoding="utf-8"))
    # attesi: [{"id": "...", "text": "..."}]
    if isinstance(data, list):
        documents = data
    elif isinstance(data, dict) and "documents" in data:
        documents = data["documents"]
    else:
        documents = []
    if not documents:
        logger.warning("Docstore is empty: %s", p)
    return documents



def load_questions(path: str | Path) -> Dict[str, List[Dict[str, Any]]]:
    """Load oracle questions and categorize off-topic entries.

    The JSON file is expected to contain a list of question objects.  Entries
    marked with ``"type": "off_topic"`` are grouped by their ``categoria``
    field (e.g. ``poetica`` or ``didattica``).  All other entries are returned
    under the ``"good"`` key.


def load_questions(path: str | Path) -> Dict[str, Any]:
    """Load oracle questions from ``path`` grouping off-topic ones by category.

    The JSON file is expected to contain a list of objects.  Entries without
    ``type`` or where ``type`` is different from ``off_topic`` are considered
    regular questions and returned in the ``good`` list.  Entries tagged with
    ``off_topic`` must also provide a ``categoria`` field; these are grouped
    under ``off_topic`` using the category as key.

def load_questions(
    path: str | Path | None = None,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]], List[str]]:
    """Read oracle questions and follow ups from ``path``.


    Parameters
    ----------
    path:

        Location of ``domande_oracolo.json``.

    Returns
    -------
    dict
        ``{"good": [...], "off_topic": {"cat": [...]}}``

    """

    p = Path(path)
    if not p.exists():

        logger.warning("Questions file not found: %s", p)

        return {"good": [], "off_topic": {}}

    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:


        logger.exception("Invalid JSON in questions file: %s", p)

        return {"good": [], "off_topic": {}}

    good: List[Dict[str, Any]] = []
    off_topic: Dict[str, List[Dict[str, Any]]] = {}


    if isinstance(data, list):
        for item in data:
            if not isinstance(item, dict):
                continue
            qtype = str(item.get("type", "")).lower()
            if qtype == "off_topic":
                cat = str(item.get("categoria", "")).lower()
                off_topic.setdefault(cat, []).append(item)
            else:
                good.append(item)

    return {"good": good, "off_topic": off_topic}


    for item in data if isinstance(data, list) else []:
        if item.get("type") == "off_topic":
            cat = str(item.get("categoria", "")) or "unknown"
            off_topic.setdefault(cat, []).append(item)
        else:
            good.append(item)

    return {"good": good, "off_topic": off_topic}

        Optional custom location of the JSON file.  When ``None`` the
        function looks for ``data/domande_oracolo.json`` relative to the
        project root.

    Returns
    -------
    tuple
        ``(good, off_topic, follow_ups)`` where ``good`` and ``off_topic`` are
        lists of dictionaries containing at least ``question`` and
        ``response_type``.  ``follow_ups`` is a list with all the
        ``follow_up`` strings present in the file.  Empty lists are returned
        when the file cannot be read.
    """

    p = (
        Path(path)
        if path is not None
        else Path(__file__).resolve().parent.parent / "data" / "domande_oracolo.json"
    )
    if not p.exists():
        logger.warning("Questions file not found: %s", p)
        return [], [], []
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        logger.exception("Failed to read questions file: %s", p)
        return [], [], []
    good = data.get("good", []) if isinstance(data, dict) else []
    off_topic = data.get("off_topic", []) if isinstance(data, dict) else []
    follow_ups = [q.get("follow_up", "") for q in good + off_topic if q.get("follow_up")]
    return good, off_topic, follow_ups




def _make_chunks(text: str, max_chars: int = 800, overlap_ratio: float = 0.1) -> List[str]:
    """Split text into semantically coherent chunks.

    La funzione prova prima a spezzare su paragrafi/titoli (doppie nuove
    linee). Se un paragrafo supera ``max_chars`` viene ulteriormente spezzato
    in frasi con un overlap dinamico proporzionale alla lunghezza del chunk
    precedente.
    """

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

    # Gestisci eventuali chunk ancora troppo lunghi spezzandoli in frasi
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


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


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
    """Ottiene riformulazioni della query tramite un piccolo modello LLM.

    I risultati sono memorizzati in cache usando ``model``, ``query`` e ``n``
    come parte della chiave; modificare uno di questi parametri invalida la
    cache. Il contenuto viene mantenuto per circa un'ora.
    """

    # calcola chiave cache stabile basata sui parametri principali
    raw_key = f"{model}:{n}:{query}".encode("utf-8")
    cache_key = "rq:" + hashlib.sha1(raw_key).hexdigest()
    cached = cache_get_json(cache_key)
    if isinstance(cached, list) and all(isinstance(x, str) for x in cached):
        return cached[:n]

    prompt = (
        "Fornisci {n} riformulazioni concise della seguente query in italiano o inglese, una per riga.\nQuery: {q}"
    ).format(n=n, q=query)
    txt = ""
    try:
        # API 'responses' (OpenAI>=2024)
        resp = run(
            client.responses.create,  # type: ignore[attr-defined]
            model=model,
            input=prompt,
        )
        txt = getattr(resp, "output_text", "")
    except Exception:
        try:
            # compatibilità con chat.completions
            resp = run(
                client.chat.completions.create,  # type: ignore[attr-defined]
                model=model,
                messages=[{"role": "user", "content": prompt}],
            )
            txt = resp.choices[0].message.content  # type: ignore[index]
        except Exception:
            return []
    lines = [l.strip("- •\t") for l in txt.splitlines() if l.strip()]
    lines = lines[:n]
    if lines:
        cache_set_json(cache_key, lines, ttl=3600)
    return lines


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


def _compress_passage(text: str, query: str, max_sents: int = 5) -> str:
    """Reduce ``text`` to at most ``max_sents`` sentences relevant to ``query``.

    Sentences are scored by simple token overlap with the query and the top
    ones are kept in their original order.
    """
    sents = _simple_sentences(text)
    if len(sents) <= max_sents:
        return " ".join(sents)
    qtok = set(_tokenize(query))
    scored: List[Tuple[int, int, str]] = []
    for idx, s in enumerate(sents):
        stok = set(_tokenize(s))
        score = len(qtok & stok)
        scored.append((score, idx, s))
    scored.sort(key=lambda x: x[0], reverse=True)
    selected = sorted(scored[:max_sents], key=lambda x: x[1])
    return " ".join(s for _, _, s in selected)


def _rerank_cross_encoder(
    query: str, pairs: List[Tuple[Chunk, float]], model_name: str
) -> List[Tuple[Chunk, float]]:
    """Rerank usando un mini cross-encoder (se disponibile)."""
    if CrossEncoder is None:
        return pairs
    try:  # cache per evitare reload del modello ad ogni chiamata
        ce = _CROSS_ENCODER_CACHE.get(model_name)
        if ce is None:
            ce = CrossEncoder(model_name)
            _CROSS_ENCODER_CACHE[model_name] = ce
        texts = [c.text for c, _ in pairs]
        scores = ce.predict([(query, t) for t in texts])
        reranked = [(c, float(s)) for (c, _), s in zip(pairs, scores)]
        reranked.sort(key=lambda x: x[1], reverse=True)
        return reranked
    except Exception:
        return pairs


def retrieve(
    query: str,
    docstore_path: str | Path,
    top_k: int = 3,
    *,
    topic: str | None = None,
    client: Any | None = None,
    embed_model: str | None = None,
    rewrite_model: str | None = None,
    rerank_model: str | None = None,
    store: Any | None = None) -> List[Dict]:
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
            cache_dir = Path(docstore_path).with_suffix(".embcache")
            qv = _embed_texts(client, embed_model, [query])[0]
            texts = [c.text for c, _ in best]
            vecs = _embed_texts(client, embed_model, texts, cache_dir=cache_dir)
            sims = [_cosine(qv, v) for v in vecs]
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
