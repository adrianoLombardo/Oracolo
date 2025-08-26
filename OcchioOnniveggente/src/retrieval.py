# src/retrieval.py
from __future__ import annotations
import json, math, re, hashlib
import logging
import yaml
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Iterable, Optional, Any, Protocol
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


@dataclass
class Question:
    domanda: str
    type: str
    follow_up: str | None = None
    opera: str | None = None
    artista: str | None = None
    location: str | None = None
    tag: List[str] | None = None

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)


class QuestionProvider(Protocol):
    """Interface for pluggable question sources.

    Each provider exposes a ``name`` attribute and a :meth:`load` method
    returning questions grouped by category.  Providers can retrieve
    questions from any backend (JSON files, APIs, databases...) allowing the
    application to be easily extended.
    """

    name: str

    def load(self) -> Dict[str, List[Question]]:
        """Return the questions grouped by category."""


_QUESTION_PROVIDERS: Dict[str, QuestionProvider] = {}


def register_question_provider(provider: QuestionProvider) -> None:
    """Register ``provider`` in the global provider registry."""

    _QUESTION_PROVIDERS[provider.name.lower()] = provider


def get_question_provider(name: str) -> QuestionProvider | None:
    """Return the provider registered under ``name`` if present."""

    return _QUESTION_PROVIDERS.get(name.lower())


def iter_question_providers(context: str | None = None) -> Iterable[QuestionProvider]:
    """Yield all registered providers or the one matching ``context``."""

    if context:
        p = get_question_provider(context)
        return [p] if p is not None else []
    return list(_QUESTION_PROVIDERS.values())


def _simple_sentences(txt: str) -> List[str]:
    # split robusto su righe/punteggiatura
    parts = re.split(r'(?<=[\.\!\?])\s+|\n{2,}', txt)
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



def load_questions(path: str | Path | None = None) -> Dict[str, List[Question]]:
    """Read the entire questions dataset and group entries by category.

    Parameters
    ----------
    path:
        Optional location of ``domande_oracolo.json``. When ``None`` the file
        is searched relative to the project root.

    Returns
    -------
    dict
        Mapping each question ``type`` to the list of question objects.
    """

    p = (
        Path(path)
        if path is not None
        else Path(__file__).resolve().parent.parent / "data" / "domande_oracolo.json"
    )
    if not p.exists():
        logger.warning("Questions file not found: %s", p)
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        logger.exception("Invalid JSON in questions file: %s", p)
        return {}

    categories: Dict[str, List[Question]] = {}
    if isinstance(data, list):
        for idx, item in enumerate(data):
            if not isinstance(item, dict):
                logger.warning("Invalid question at index %s: %r", idx, item)
                continue
            domanda = item.get("domanda")
            qtype = item.get("type")
            if not isinstance(domanda, str) or not isinstance(qtype, str):
                logger.warning(
                    "Question missing required fields at index %s: %r", idx, item
                )
                continue
            follow_up = item.get("follow_up")
            if follow_up is not None and not isinstance(follow_up, str):
                follow_up = str(follow_up)
            opera = item.get("opera")
            if opera is not None and not isinstance(opera, str):
                opera = str(opera)
            artista = item.get("artista")
            if artista is not None and not isinstance(artista, str):
                artista = str(artista)
            location = item.get("location")
            if location is not None and not isinstance(location, str):
                location = str(location)
            tag_field = item.get("tag")
            tags: List[str] | None
            if tag_field is None:
                tags = None
            elif isinstance(tag_field, list):
                tags = [str(t) for t in tag_field if isinstance(t, (str, int, float))]
            else:
                tags = [str(tag_field)]
            cat = qtype.lower()
            categories.setdefault(cat, []).append(
                Question(
                    domanda=domanda,
                    type=cat,
                    follow_up=follow_up,
                    opera=opera,
                    artista=artista,
                    location=location,
                    tag=tags,
                )
            )

    return categories


def load_questions_from_providers(context: str | None = None) -> Dict[str, List[Question]]:
    """Load questions from the registered providers.

    Parameters
    ----------
    context:
        Optional name of a specific provider.  When ``None`` all registered
        providers are queried and their results merged.
    """

    merged: Dict[str, List[Question]] = {}
    for provider in iter_question_providers(context):
        try:
            data = provider.load() or {}
        except Exception:
            logger.exception("Question provider %s failed", provider.name)
            data = {}
        for cat, qs in data.items():
            merged.setdefault(cat, []).extend(qs)
    # Deduplicate by question text to avoid duplicates when multiple providers
    # return overlapping entries.
    for cat, qs in list(merged.items()):
        seen: set[str] = set()
        unique: List[Question] = []
        for q in qs:
            text = q.domanda if isinstance(q, Question) else q.get("domanda", "")
            if text in seen:
                continue
            seen.add(text)
            unique.append(q)
        merged[cat] = unique
    return merged


class JSONQuestionProvider:
    """Simple provider reading questions from a JSON file."""

    def __init__(self, name: str, path: Path):
        self.name = name
        self.path = Path(path)

    def load(self) -> Dict[str, List[Question]]:  # pragma: no cover - tiny wrapper
        return load_questions(self.path)


class AdrianoLombardoProvider(JSONQuestionProvider):
    def __init__(self, path: Path | None = None):
        if path is None:
            path = Path(__file__).resolve().parent.parent / "data" / "domande_oracolo.json"
        super().__init__("AdrianoLombardo", path)


class TheMProvider(JSONQuestionProvider):
    def __init__(self, path: Path | None = None):
        if path is None:
            path = Path(__file__).resolve().parent.parent / "data" / "domande_oracolo.json"
        super().__init__("TheM", path)


class CryptoMadonneProvider(JSONQuestionProvider):
    def __init__(self, path: Path | None = None):
        if path is None:
            path = Path(__file__).resolve().parent.parent / "data" / "domande_oracolo.json"
        super().__init__("CryptoMadonne", path)


# Register built-in providers
register_question_provider(AdrianoLombardoProvider())
register_question_provider(TheMProvider())
register_question_provider(CryptoMadonneProvider())




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
