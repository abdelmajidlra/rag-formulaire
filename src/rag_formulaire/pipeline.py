from __future__ import annotations

import logging
import torch
from typing import Any, Dict, List

from . import config
from .evaluation import AdvancedSelfReflector, CRAGEvaluator, verify_response_against_evidence
from .indexing import load_indexes
from .llm import LocalLLM
from .query_processing import (
    AgenticQueryRouter,
    MultilingualQueryHandler,
    QueryDecomposer,
    QueryExpander,
)
from .reranker import CrossEncoderReranker
from .retrieval import HybridRetriever

logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    Encapsulates the full RAG pipeline:
    Query -> Normalize -> Route -> Expand/Decompose -> Retrieve -> Rerank -> Generate -> Reflect
    """

    def __init__(self):
        logger.info(f"Initializing RAG Pipeline with model: {config.GEN_MODEL_NAME}")
        
        self.index_store = load_indexes()
        self.query_handler = MultilingualQueryHandler()
        self.router = AgenticQueryRouter()
        self.expander = QueryExpander()
        self.decomposer = QueryDecomposer()
        self.retriever = HybridRetriever(self.index_store)
        self.reranker = CrossEncoderReranker()
        self.evaluator = CRAGEvaluator()
        self.reflector = AdvancedSelfReflector()
        self.llm = LocalLLM()  # Singleton

        logger.info("RAG Pipeline initialized successfully.")

    def ask_question(self, question: str, evidence_k: int = config.FINAL_EVIDENCE_K) -> Dict[str, Any]:
        """
        Main entry point for asking a question.
        """
        q_orig, q_fr = self.query_handler.normalize(question)
        route = self.router.route(q_fr)
        expansions = self.expander.expand(q_fr, n=3)
        subqueries = self.decomposer.decompose(q_fr) if route == "MULTI_STEP" else [q_fr]

        candidates = []
        for sub in subqueries:
            for variant in expansions:
                candidates.extend(self.retriever.retrieve(variant, manifest=None))

        reranked = self.reranker.rerank(q_fr, candidates, top_n=config.RERANK_TOP_N)
        
        if not reranked:
            return {
                "route": route,
                "answer": "Aucun extrait trouvé.",
                "evidence": [],
                "expansions": expansions
            }

        scores = list(range(len(reranked), 0, -1))
        if not self.evaluator.is_evidence_strong(scores, reranked):
            return {
                "route": route,
                "answer": self.evaluator.fallback_message(),
                "evidence": [],
                "expansions": expansions
            }

        evidence_texts = [
            f"[{c.base_chunk.form_code}] {c.base_chunk.section_title}: {c.base_chunk.content}"
            for c in reranked[:evidence_k]
        ]
        
        system_prompt = (
            "Vous êtes un assistant spécialisé dans les formulaires IRCC. Répondez uniquement en français en vous basant sur les "
            "extraits fournis. Citez le code du formulaire et la section."
        )
        user_prompt = q_fr + "\n\nExtraits:\n" + "\n---\n".join(evidence_texts)
        
        # Memory cleanup before generation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        answer = None
        oom_occurred = False
        
        try:
            answer = self.llm.chat(system_prompt, user_prompt, max_new_tokens=256)
        except RuntimeError as e:
            if "out of memory" in str(e):
                oom_occurred = True
                logger.warning("OOM detected during generation. Preparing to retry with reduced context.")
            else:
                raise e
        
        # Handle retry outside the except block to ensure 'e' reference is released
        if oom_occurred:
            # Force aggressive cleanup
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            
            # Retry with drastically reduced context (1/4 of evidence)
            # If evidence is small, fallback to just the question
            reduced_len = max(1, len(evidence_texts) // 4)
            if reduced_len < 1:
                # Extreme fallback: no evidence, just question (to avoid crash)
                logger.warning("Extreme OOM fallback: generating without evidence.")
                user_prompt = q_fr
            else:
                reduced_evidence = evidence_texts[:reduced_len]
                user_prompt = q_fr + "\n\nExtraits (Reduced):\n" + "\n---\n".join(reduced_evidence)
            
            try:
                answer = self.llm.chat(system_prompt, user_prompt, max_new_tokens=256)
            except RuntimeError as e:
                logger.error(f"OOM retry failed: {e}")
                answer = "Erreur: Mémoire insuffisante pour générer une réponse (OOM)."

        # Validation des codes de formulaire (empêche hallucinations)
        if verify_response_against_evidence(answer, reranked):
            answer = self.reflector.reflect(q_fr, answer, reranked)
        else:
            answer = self.evaluator.fallback_message()

        return {
            "route": route,
            "expansions": expansions,
            "answer": answer,
            "evidence": reranked[:evidence_k],
        }
