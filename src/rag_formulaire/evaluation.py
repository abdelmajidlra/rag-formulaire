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
        
        # Extract form codes from evidence for verification
        evidence_forms = ", ".join(sorted(set(c.base_chunk.form_code for c in evidence)))
        
        # Enhanced reflection prompt with form code verification
        prompt = (
            f"Question: {question}\n"
            f"Réponse générée: {answer}\n\n"
            f"Formulaires dans les preuves: {evidence_forms}\n"
            f"Extraits de preuves: {snippets}\n\n"
            "Analyse critique (réponds en français):\n"
            "1. La réponse mentionne-t-elle des codes de formulaire (IMM XXXX, CIT XXXX)?\n"
            "2. Ces codes correspondent-ils EXACTEMENT aux formulaires dans les preuves?\n"
            "3. Y a-t-il des informations qui ne sont PAS supportées par les extraits?\n"
            "4. La réponse est-elle incertaine ou spéculative?\n\n"
            "Indique 'PROBLEME' si tu détectes une incohérence, sinon 'OK'."
        )
        
        critique = self.llm.generate(prompt, max_new_tokens=128)
        
        # More aggressive detection of issues
        problematic_keywords = ["incertain", "specul", "faible", "probleme", "problème", 
                                "incohéren", "erreur", "incorrect", "faux"]
        if any(word in critique.lower() for word in problematic_keywords):
            logger.info(f"Self-reflection detected issue: {critique[:100]}")
            return (
                "Réponse prudente: les informations ne sont pas entièrement confirmées par les extraits fournis. "
                "Veuillez consulter les formulaires officiels."
            )
        return answer


def _normalize_text(text: str) -> str:
    return re.sub(r"[^a-z0-9\s]", "", text.lower())


def verify_response_against_evidence(answer: str, evidence: List[ContextualizedChunk]) -> bool:
    """
    Verify answer against evidence. Always validates form codes.
    Optionally performs strict n-gram verification if enabled.
    """
    # Always validate form codes to prevent hallucination
    if not _validate_form_codes_in_answer(answer, evidence):
        return False
    
    # Check if strict verification is enabled (n-gram matching)
    if not config.ENABLE_STRICT_VERIFICATION:
        return True  # Lenient mode after form code check
    
    # Strict mode: verify n-grams exist in evidence
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


def _validate_form_codes_in_answer(answer: str, evidence: List[ContextualizedChunk]) -> bool:
    """
    Validate that form codes (IMM/CIT XXXX) mentioned in answer exist in evidence.
    Prevents hallucination of non-existent form codes.
    """
    # Extract form codes from answer
    answer_codes = set(re.findall(r'(IMM|CIT)\s*\d{4}', answer, re.IGNORECASE))
    
    if not answer_codes:
        return True  # No form codes to validate
    
    # Extract form codes from evidence
    evidence_codes = {chunk.base_chunk.form_code for chunk in evidence}
    
    # Normalize codes (handle spacing variations)
    answer_codes_normalized = {re.sub(r'\s+', ' ', code.upper()) for code in answer_codes}
    
    # Check if all mentioned codes are in evidence
    for code in answer_codes_normalized:
        if code not in evidence_codes:
            logger.warning(f"Hallucinated form code detected in answer: {code} (not in evidence: {evidence_codes})")
            return False
    
    return True
