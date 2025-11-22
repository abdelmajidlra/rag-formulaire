from __future__ import annotations

import json
import logging
import os
import pickle
from pathlib import Path
from typing import List

import chromadb
from rank_bm25 import BM25Okapi

try:  # pragma: no cover - heavy deps
    from sentence_transformers import SentenceTransformer
except Exception:  # noqa: BLE001
    SentenceTransformer = None

from sklearn.feature_extraction.text import TfidfVectorizer

from . import config
from .data_models import FormChunk

FRENCH_STOP_WORDS = [
    "alors",
    "au",
    "aucuns",
    "aussi",
    "autre",
    "avant",
    "avec",
    "avoir",
    "bon",
    "car",
    "ce",
    "cela",
    "ces",
    "ceux",
    "chaque",
    "ci",
    "comme",
    "comment",
    "dans",
    "des",
    "du",
    "elle",
    "elles",
    "en",
    "encore",
    "est",
    "et",
    "eu",
    "fait",
    "fois",
    "font",
    "hors",
    "ici",
    "il",
    "ils",
    "je",
    "juste",
    "la",
    "le",
    "les",
    "leur",
    "là",
    "ma",
    "mais",
    "me",
    "même",
    "mes",
    "mine",
    "moins",
    "mon",
    "mot",
    "ne",
    "ni",
    "nommés",
    "notre",
    "nous",
    "nouveaux",
    "ou",
    "où",
    "par",
    "parce",
    "pas",
    "peut",
    "peu",
    "plupart",
    "pour",
    "pourquoi",
    "quand",
    "que",
    "quel",
    "quelle",
    "quelles",
    "quels",
    "qui",
    "sa",
    "sans",
    "ses",
    "seulement",
    "si",
    "sien",
    "son",
    "sont",
    "sous",
    "soyez",
    "sujet",
    "sur",
    "ta",
    "tandis",
    "tellement",
    "tels",
    "tes",
    "ton",
    "tous",
    "tout",
    "trop",
    "très",
    "tu",
    "voient",
    "vont",
    "votre",
    "vous",
    "vu",
]
logger = logging.getLogger(__name__)


class EmbeddingBackend:
    def __init__(self, vectorizer: TfidfVectorizer | None = None):
        self.vectorizer = vectorizer
        self._fitted = vectorizer is not None
        offline = os.getenv("HF_HUB_OFFLINE", "1").lower() in {"1", "true", "yes"}
        try:  # pragma: no cover - heavy dependency
            self.model = (
                SentenceTransformer(config.EMBEDDING_MODEL_NAME)
                if (SentenceTransformer and vectorizer is None and not offline)
                else None
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Chargement SentenceTransformer impossible (%s), fallback TF-IDF.", exc)
            self.model = None

        if self.model is None and self.vectorizer is None:
            self.vectorizer = TfidfVectorizer(stop_words=FRENCH_STOP_WORDS)
            self._fitted = False

    def encode(self, texts: List[str]):
        if self.model is not None:
            return self.model.encode(texts, normalize_embeddings=True)
        if not getattr(self, "_fitted", False):
            self.vectorizer.fit(texts)
            self._fitted = True
        return self.vectorizer.transform(texts).toarray()


class IndexStore:
    def __init__(self, bm25, bm25_corpus, chroma_collection, chunk_map: dict, embedding_backend=None):
        self.bm25 = bm25
        self.bm25_corpus = bm25_corpus
        self.chroma = chroma_collection
        self.chunk_map = chunk_map
        self.embedding_backend = embedding_backend



def _tokenize(text: str) -> List[str]:
    return [t for t in text.lower().split() if t.isalpha() or t.isalnum()]


def _tfidf_path() -> Path:
    return config.INDEX_DIR / "tfidf_vectorizer.pkl"


def build_indexes(chunks: List[FormChunk]) -> IndexStore:
    config.BM25_DIR.mkdir(parents=True, exist_ok=True)
    config.CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    corpus = [f"{c.content} {c.form_code} {c.section_title}" for c in chunks]
    tokenized_corpus = [_tokenize(doc) for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus, k1=config.BM25_K1, b=config.BM25_B)

    emb_backend = EmbeddingBackend()
    embeddings = emb_backend.encode(corpus)

    tfidf_path = _tfidf_path()
    tfidf_path.parent.mkdir(parents=True, exist_ok=True)
    if emb_backend.model is None and emb_backend.vectorizer is not None:
        with open(tfidf_path, "wb") as f:
            pickle.dump(emb_backend.vectorizer, f)
    elif tfidf_path.exists():
        tfidf_path.unlink()

    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    collection = client.get_or_create_collection("forms")
    try:
        collection.delete(where={})
    except Exception:
        pass
    ids = [c.chunk_id for c in chunks]
    metadata = [
        {
            "form_code": c.form_code,
            "section_title": c.section_title,
            "form_title": c.form_title,
            "page_number": c.page_number,
        }
        for c in chunks
    ]
    collection.add(documents=corpus, embeddings=list(embeddings), ids=ids, metadatas=metadata)

    chunk_map = {c.chunk_id: c for c in chunks}

    with open(config.BM25_DIR / "bm25.pkl", "wb") as f:
        pickle.dump({"bm25": bm25, "corpus": corpus}, f)
    with open(config.BM25_DIR / "chunks.json", "w", encoding="utf-8") as f:
        json.dump({cid: c.__dict__ for cid, c in chunk_map.items()}, f, ensure_ascii=False)

    return IndexStore(
        bm25=bm25,
        bm25_corpus=corpus,
        chroma_collection=collection,
        chunk_map=chunk_map,
        embedding_backend=emb_backend,
    )


def load_indexes() -> IndexStore:
    with open(config.BM25_DIR / "bm25.pkl", "rb") as f:
        bm25_data = pickle.load(f)
    with open(config.BM25_DIR / "chunks.json", "r", encoding="utf-8") as f:
        raw_chunks = json.load(f)
    chunk_map = {cid: FormChunk(**data) for cid, data in raw_chunks.items()}

    tfidf_path = _tfidf_path()
    vectorizer = None
    if tfidf_path.exists():
        with open(tfidf_path, "rb") as f:
            vectorizer = pickle.load(f)

    emb_backend = EmbeddingBackend(vectorizer=vectorizer) if vectorizer else None

    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    collection = client.get_or_create_collection("forms")

    return IndexStore(
        bm25=bm25_data["bm25"],
        bm25_corpus=bm25_data["corpus"],
        chroma_collection=collection,
        chunk_map=chunk_map,
        embedding_backend=emb_backend,
    )
