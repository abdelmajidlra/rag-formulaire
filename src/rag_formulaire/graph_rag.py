from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict, List, Set, Tuple

from .data_models import FormChunk

logger = logging.getLogger(__name__)


class GraphRAG:
    def __init__(self):
        self.graph: Dict[str, Set[str]] = defaultdict(set)

    def add_relation(self, form_code: str, concept: str):
        self.graph[form_code].add(concept)
        self.graph[concept].add(form_code)

    def build_from_chunks(self, chunks: List[FormChunk]):
        for chunk in chunks:
            text = chunk.content.lower()
            for keyword in ["conjoint", "enfant", "permis", "travail", "études", "citoyenneté"]:
                if keyword in text:
                    self.add_relation(chunk.form_code, keyword)

    def neighbors(self, node: str) -> Set[str]:
        return self.graph.get(node, set())

    def suggest_related_forms(self, form_code: str) -> List[str]:
        related = set()
        for neighbor in self.neighbors(form_code):
            related.update([n for n in self.neighbors(neighbor) if n != form_code and n.startswith("IMM")])
        return list(related)
