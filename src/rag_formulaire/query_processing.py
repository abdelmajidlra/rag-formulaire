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

    def expand(self, query_fr: str, n: int = 3) -> List[str]:
        return self.llm.expand_queries(query_fr, n=n)


class QueryDecomposer:
    def __init__(self):
        self.llm = LocalLLM()

    def decompose(self, query_fr: str) -> List[str]:
        return self.llm.decompose(query_fr)
