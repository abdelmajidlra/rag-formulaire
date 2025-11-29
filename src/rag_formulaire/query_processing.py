from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List

from langdetect import detect

from .llm import LocalLLM

logger = logging.getLogger(__name__)


@dataclass
class QueryRoutingDecision:
    query_original: str
    query_french: str
    route: str
    expansions: List[str]
    subqueries: List[str]


class MultilingualQueryHandler:
    def __init__(self):
        self.llm = LocalLLM()

    def normalize(self, query: str) -> tuple[str, str]:
        try:
            lang = detect(query)
        except Exception:  # noqa: BLE001
            lang = "fr"
        if lang != "fr":
            logger.info("Traduction approximative vers le français depuis %s", lang)
            return query, f"[traduction approximative] {query}"
        return query, query


class AgenticQueryRouter:
    def __init__(self):
        self.llm = LocalLLM()

    def route(self, query_fr: str) -> str:
        if any(word in query_fr.lower() for word in ["quel formulaire", "lequel", "quel document"]):
            return "DOCUMENT_LOOKUP"
        if len(query_fr.split()) > 25:
            return "MULTI_STEP"
        if any(word in query_fr.lower() for word in ["comment", "pourquoi"]):
            return "MULTI_STEP"
        return "SIMPLE_FACT"


class QueryExpander:
    def __init__(self):
        self.llm = LocalLLM()
    
    def expand(self, query: str, n: int = 3) -> List[str]:
        """Expand query with synonyms and context-specific terms."""
        variants = [query]
        
        # Add COVID-19 specific expansions
        if any(term in query.lower() for term in ["covid", "coronavirus", "pandémie", "pandemic"]):
            variants.append(query + " décrets quarantaine")
            variants.append(query + " famille élargie")
            variants.append(query.replace("COVID-19", "pandémie").replace("covid", "pandémie"))
            variants.append(query + " loi mise en quarantaine")
        
        # Immigration-specific synonyms
        synonyms = ["permis", "demande", "formulaire", "document", "déclaration", "attestation"]
        
        for i in range(1, min(n, len(synonyms)) + 1):
            variant = f"{query} {synonyms[i % len(synonyms)]}"
            if variant not in variants:
                variants.append(variant)
        
        # Return unique variants, limit to reasonable number
        return list(dict.fromkeys(variants))[:n + 3]  # Allow extra for COVID expansions


class QueryDecomposer:
    def __init__(self):
        self.llm = LocalLLM()

    def decompose(self, query_fr: str) -> List[str]:
        return self.llm.decompose(query_fr)
