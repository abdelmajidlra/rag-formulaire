from __future__ import annotations

import logging
from typing import List

try:  # pragma: no cover - heavy dependency
    from sentence_transformers import CrossEncoder
except Exception:  # noqa: BLE001
    CrossEncoder = None

from . import config
from .data_models import ContextualizedChunk

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    def __init__(self):
        try:  # pragma: no cover - heavy dependency
            # Force CPU to keep GPU VRAM available for the generator (avoids OOM)
            self.model = CrossEncoder(config.RERANK_MODEL_NAME, device="cpu") if CrossEncoder else None
        except Exception as exc:  # noqa: BLE001
            logger.warning("Chargement du reranker impossible (%s); utilisation d'un score heuristique.", exc)
            self.model = None

    def rerank(self, query: str, candidates: List[ContextualizedChunk], top_n: int | None = None) -> List[ContextualizedChunk]:
        if self.model is None:
            scored = sorted(candidates, key=lambda c: len(c.base_chunk.content), reverse=True)
        else:
            pairs = [[query, c.combined_text()] for c in candidates]
            scores = self.model.predict(pairs)
            scored = [c for _, c in sorted(zip(scores, candidates), key=lambda p: p[0], reverse=True)]
        if top_n:
            scored = scored[:top_n]
        return scored
