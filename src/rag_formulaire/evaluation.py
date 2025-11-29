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
        snippets = "\n".join([c.base_chunk.content[:300] for c in evidence[:5]])
        
        # Extract form codes from evidence for verification
        evidence_forms = ", ".join(sorted(set(c.base_chunk.form_code for c in evidence[:10])))
        
        # Enhanced reflection prompt with confidence scoring
        prompt = (
            f"Question: {question}\n"
            f"Réponse générée: {answer}\n\n"
            f"Formulaires dans les preuves: {evidence_forms}\n"
            f"Extraits de preuves: {snippets}\n\n"
            "Analyse critique (réponds en français):\n"
            "1. La réponse mentionne-t-elle des codes de formulaire?\n"
            "2. Si oui, ces codes sont-ils présents dans les preuves?\n"
            "3. Les informations sont-elles supportées par les extraits?\n"
            "4. La réponse est-elle confiante et précise?\n\n"
            "Donne un score de confiance de 0.0 à 1.0:\n"
            "- 0.0-0.3: réponse incorrecte ou non supportée\n"
            "- 0.4-0.6: réponse partiellement supportée\n"
            "- 0.7-1.0: réponse bien supportée\n\n"
            "Réponds au format: CONFIANCE: <score>\nExplication: <raison>"
        )
        
        critique = self.llm.generate(prompt, max_new_tokens=200)
        
        # Extract confidence score from critique
        confidence_match = re.search(r'CONFIANCE:\s*([0-9]*\.?[0-9]+)', critique, re.IGNORECASE)
        if confidence_match:
            try:
                confidence = float(confidence_match.group(1))
            except ValueError:
                confidence = 0.5  # Default to moderate confidence if parsing fails
        else:
            # Fallback: keyword-based confidence estimation
            problematic_keywords = ["incorrect", "faux", "erreur", "probleme", "problème"]
            supportive_keywords = ["correct", "supporté", "confirmé", "valide", "ok"]
            
            critique_lower = critique.lower()
            if any(word in critique_lower for word in problematic_keywords):
                confidence = 0.2
            elif any(word in critique_lower for word in supportive_keywords):
                confidence = 0.8
            else:
                confidence = 0.5
        
        logger.info(f"Self-reflection confidence: {confidence:.2f} | Critique: {critique[:150]}")
        
        # Only reject if confidence is very low (more lenient threshold)
        if confidence < 0.3:
            logger.warning(f"Low confidence ({confidence:.2f}), returning cautious response")
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
    Uses fuzzy matching to handle spacing variations (e.g., "IMM5476" vs "IMM 5476").
    """
    # Extract form codes from answer - FIXED: capture full code with \d{4}
    # Pattern now captures the ENTIRE code, not just the prefix
    answer_codes = set(re.findall(r'(?:IMM|CIT)\s*\d{4}', answer, re.IGNORECASE))
    
    if not answer_codes:
        return True  # No form codes to validate
    
    # Extract form codes from evidence chunks
    evidence_codes = {chunk.base_chunk.form_code for chunk in evidence}
    
    # Also extract codes from evidence content for robustness
    evidence_content = " ".join([chunk.base_chunk.content for chunk in evidence])
    evidence_codes_in_content = set(re.findall(r'(?:IMM|CIT)\s*\d{4}', evidence_content, re.IGNORECASE))
    
    # Combine both sources of evidence codes
    all_evidence_codes = evidence_codes | evidence_codes_in_content
    
    # Normalize all codes: uppercase, single space between prefix and number
    def normalize_code(code: str) -> str:
        """Normalize form code to format 'IMM 1234' or 'CIT 1234'"""
        code = code.upper()
        # Handle various formats: IMM1234, IMM 1234, imm  1234
        match = re.match(r'(IMM|CIT)\s*(\d{4})', code, re.IGNORECASE)
        if match:
            return f"{match.group(1).upper()} {match.group(2)}"
        return code
    
    answer_codes_normalized = {normalize_code(code) for code in answer_codes}
    evidence_codes_normalized = {normalize_code(code) for code in all_evidence_codes}
    
    # Check if all mentioned codes are in evidence (with fuzzy matching)
    hallucinated_codes = answer_codes_normalized - evidence_codes_normalized
    
    if hallucinated_codes:
        logger.warning(
            f"Hallucinated form codes detected in answer: {hallucinated_codes} "
            f"(not in evidence: {evidence_codes_normalized})"
        )
        return False
    
    logger.debug(
        f"Form code validation passed. Answer codes: {answer_codes_normalized}, "
        f"Evidence codes: {evidence_codes_normalized}"
    )
    return True
