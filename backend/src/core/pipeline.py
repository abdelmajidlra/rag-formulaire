from __future__ import annotations

import logging
import re
from functools import lru_cache
from typing import Any, Dict, List

from rag_formulaire.pipeline import RAGPipeline

logger = logging.getLogger(__name__)

FORM_CODE_PATTERN = re.compile(r"\b(?:IMM|CIT)\s?-?\s?\d{4}\b", re.IGNORECASE)


@lru_cache()
def get_pipeline() -> RAGPipeline:
    """Return a singleton instance of the underlying RAG pipeline."""

    logger.info("Loading shared RAG pipeline instance")
    return RAGPipeline()


def _extract_forms(answer: str, evidence: List[Any]) -> List[str]:
    codes = {match.group(0).replace(" ", "").upper() for match in FORM_CODE_PATTERN.finditer(answer or "")}
    for chunk in evidence:
        code = getattr(chunk.base_chunk, "form_code", None)
        if code:
            codes.add(code.upper())
    return sorted(codes)


def _extract_sources(evidence: List[Any]) -> List[str]:
    sources: List[str] = []
    for chunk in evidence:
        metadata = getattr(chunk, "form_metadata", None)
        base_chunk = getattr(chunk, "base_chunk", None)
        path = None
        if metadata and getattr(metadata, "local_path", None):
            path = str(metadata.local_path)
        elif base_chunk and getattr(base_chunk, "form_code", None):
            path = f"FORM:{base_chunk.form_code}"
        if path:
            sources.append(path)
    return sources


def _confidence_score(evidence: List[Any]) -> float:
    if not evidence:
        return 0.2
    if len(evidence) >= 5:
        return 0.9
    if len(evidence) >= 3:
        return 0.8
    return 0.65


def answer_question(question: str, lang: str = "fr") -> Dict[str, Any]:
    """
    Utilize the existing RAG pipeline to answer a question.

    Returns a dictionary compatible with the API response schema.
    """

    pipeline = get_pipeline()
    raw = pipeline.ask_question(question)
    evidence = raw.get("evidence", []) if isinstance(raw, dict) else []
    answer = raw.get("answer", "") if isinstance(raw, dict) else ""
    route = raw.get("route") if isinstance(raw, dict) else None
    expansions = raw.get("expansions", []) if isinstance(raw, dict) else []

    forms = _extract_forms(answer, evidence)
    sources = _extract_sources(evidence)
    confidence = _confidence_score(evidence)

    if not answer:
        answer = "Je n'ai pas pu trouver une réponse fiable pour le moment."
        confidence = 0.2

    return {
        "answer": answer,
        "confidence": confidence,
        "forms": forms,
        "sources": sources,
        "meta": {"route": route, "expansions": expansions},
    }
