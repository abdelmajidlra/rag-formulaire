from __future__ import annotations

import logging
import re
from typing import List

from . import config
from .data_models import ContextualizedChunk
from .llm import LocalLLM

logger = logging.getLogger(__name__)


class CRAGEvaluator:
    def __init__(self):
        self.llm = LocalLLM()

    def is_evidence_strong(self, scores: List[float], chunks: List[ContextualizedChunk]) -> bool:
        if not scores:
            return False
        max_score = max(scores)
        mean_top = sum(scores[: min(5, len(scores))]) / min(5, len(scores))
        distinct_forms = len({c.base_chunk.form_code for c in chunks[:5]})
        return (
            max_score >= config.CRAG_MIN_SCORE
            and mean_top >= config.CRAG_MEAN_TOPK
            and distinct_forms >= config.CRAG_MIN_DISTINCT_FORMS
        )

    def fallback_message(self) -> str:
        return (
            "Je ne peux pas répondre de façon fiable à partir des formulaires IRCC indexés. "
            "Veuillez vérifier directement le formulaire officiel ou consulter un professionnel qualifié."
        )


class AdvancedSelfReflector:
    def __init__(self):
        self.llm = LocalLLM()

    def reflect(self, question: str, answer: str, evidence: List[ContextualizedChunk]) -> str:
        snippets = "\n".join([c.base_chunk.content[:200] for c in evidence[:3]])
        prompt = (
            "Question: "
            + question
            + "\nRéponse actuelle: "
            + answer
            + "\nExtraits de preuves: "
            + snippets
            + "\nAnalyse: indique si la réponse est incertaine ou spéculative."
        )
        critique = self.llm.generate(prompt, max_new_tokens=128)
        if any(word in critique.lower() for word in ["incertain", "specul", "faible"]):
            return (
                "Réponse prudente: les informations ne sont pas entièrement confirmées par les extraits fournis. "
                "Veuillez consulter les formulaires officiels."
            )
        return answer


def _normalize_text(text: str) -> str:
    return re.sub(r"[^a-z0-9\s]", "", text.lower())


def verify_response_against_evidence(answer: str, evidence: List[ContextualizedChunk]) -> bool:
    normalized_evidence = _normalize_text(" ".join([c.base_chunk.content for c in evidence]))
    tokens = answer.split()
    n = config.HALLUCINATION_NGRAM
    for i in range(len(tokens) - n):
        span = " ".join(tokens[i : i + n])
        if len(span) < 10:
            continue
        if _normalize_text(span) not in normalized_evidence:
            return False
    return True
