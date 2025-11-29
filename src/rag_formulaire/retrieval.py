from __future__ import annotations

import logging
import re
from typing import Dict, List, Tuple

import numpy as np

from . import config
from .chunking import ContextualChunkEnhancer
from .data_models import ContextualizedChunk, FormChunk
from .indexing import IndexStore, EmbeddingBackend, _tokenize

logger = logging.getLogger(__name__)


def _bm25_search(index: IndexStore, query: str, top_k: int) -> List[Tuple[str, float]]:
    tokens = _tokenize(query)
    scores = index.bm25.get_scores(tokens)
    ranked = np.argsort(scores)[::-1][:top_k]
    return [(list(index.chunk_map.keys())[i], float(scores[i])) for i in ranked]


def _vector_search(index: IndexStore, query: str, top_k: int) -> List[Tuple[str, float]]:
    emb_backend = index.embedding_backend or EmbeddingBackend()
    query_vec = emb_backend.encode([query])[0]
    results = index.chroma.query(query_embeddings=[query_vec], n_results=top_k)
    ids = results.get("ids", [[]])[0]
    dists = results.get("distances", [[]])[0]
    scores = [1 - d for d in dists]
    return list(zip(ids, scores))


def _smart_vector_search(index: IndexStore, query: str, top_k: int, filter_dict: Dict | None = None) -> List[Tuple[str, float]]:
    """Vector search with optional ChromaDB metadata filtering."""
    emb_backend = index.embedding_backend or EmbeddingBackend()
    query_vec = emb_backend.encode([query])[0]

    # Appel à ChromaDB avec le paramètre 'where' pour le filtrage
    # C'est ici que le filtrage "form_code" se fait
    results = index.chroma.query(
        query_embeddings=[query_vec],
        n_results=top_k,
        where=filter_dict
    )

    ids = results.get("ids", [[]])[0]
    dists = results.get("distances", [[]])[0]

    # Conversion distance -> score (1 - distance cosine)
    scores = [1 - d for d in dists]
    return list(zip(ids, scores))


def _rrf_fusion(bm25_res: List[Tuple[str, float]], vec_res: List[Tuple[str, float]], k: int = config.RRF_K):
    fused = {}
    for rank, (cid, _) in enumerate(bm25_res):
        fused[cid] = fused.get(cid, 0) + 1 / (k + rank + 1)
    for rank, (cid, _) in enumerate(vec_res):
        fused[cid] = fused.get(cid, 0) + 1 / (k + rank + 1)
    return sorted(fused.items(), key=lambda x: x[1], reverse=True)


class HybridRetriever:
    def __init__(self, index: IndexStore):
        self.index = index

    def retrieve(self, query: str, manifest=None, top_k_sparse: int | None = None, top_k_dense: int | None = None) -> List[ContextualizedChunk]:
        k_sparse = top_k_sparse or config.BM25_TOP_K
        k_dense = top_k_dense or config.VECTOR_TOP_K

        # --- A. DÉTECTION DU CODE FORMULAIRE (ex: "IMM 5476") ---
        pattern = r"(IMM|CIT)\s?[-_]?\s?(\d{3,4})"
        match = re.search(pattern, query, re.IGNORECASE)

        specific_filter = None
        if match:
            # Normalisation : "imm5476" -> "IMM 5476"
            prefix = match.group(1).upper()
            number = match.group(2)
            target_code = f"{prefix} {number}"

            logger.info("🎯 CIBLE DÉTECTÉE : %s -> Filtrage strict activé.", target_code)
            specific_filter = {"form_code": target_code}

        # --- B. RECHERCHE VECTORIELLE AVEC FILTRE ---
        # On utilise notre nouvelle fonction _smart_vector_search
        vec_res = _smart_vector_search(self.index, query, k_dense, filter_dict=specific_filter)

        # --- C. RECHERCHE BM25 (LEXICALE) ---
        # BM25 ne filtre pas nativement, on doit filtrer les résultats après coup
        # On double le k pour avoir assez de candidats après filtrage
        bm25_res = _bm25_search(self.index, query, k_sparse * 2)

        if specific_filter:
            target = specific_filter["form_code"]
            # Filtrage manuel des résultats BM25 : on ne garde que ceux du bon formulaire
            filtered_bm25 = []
            for cid, score in bm25_res:
                # On récupère le chunk pour vérifier son code
                chunk = self.index.chunk_map.get(cid)
                if chunk and chunk.form_code == target:
                    filtered_bm25.append((cid, score))
            bm25_res = filtered_bm25[:k_sparse]

        # --- D. FUSION ET CONTEXTE (Logique originale conservée) ---
        fused_ids = _rrf_fusion(bm25_res, vec_res)

        # Récupération des objets chunks complets
        chunk_candidates = []
        for cid, _ in fused_ids:
            if cid in self.index.chunk_map:
                chunk_candidates.append(self.index.chunk_map[cid])

        # Ajout du contexte (avant/après)
        contextualized = ContextualChunkEnhancer.enhance_with_context(chunk_candidates, manifest=manifest)
        return contextualized
