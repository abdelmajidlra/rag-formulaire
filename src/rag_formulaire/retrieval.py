from __future__ import annotations

import logging
from typing import List, Tuple

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

        bm25_res = _bm25_search(self.index, query, k_sparse)
        vec_res = _vector_search(self.index, query, k_dense)
        fused_ids = _rrf_fusion(bm25_res, vec_res)
        chunk_candidates = [self.index.chunk_map[cid] for cid, _ in fused_ids]
        contextualized = ContextualChunkEnhancer.enhance_with_context(chunk_candidates, manifest=manifest)
        return contextualized
