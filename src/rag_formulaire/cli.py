from __future__ import annotations

import json
import logging

from . import config
from .evaluation import AdvancedSelfReflector, CRAGEvaluator, verify_response_against_evidence
from .indexing import load_indexes
from .query_processing import AgenticQueryRouter, MultilingualQueryHandler, QueryDecomposer, QueryExpander
from .reranker import CrossEncoderReranker
from .retrieval import HybridRetriever
from .llm import LocalLLM

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DISCLAIMER = (
    "Cette réponse est fournie à titre informatif et ne constitue pas un avis juridique ou un conseil en immigration. "
    "Veuillez vérifier les formulaires officiels et, au besoin, consulter un professionnel qualifié."
)


def _load_chunks():
    chunks = []
    with open(config.CHUNKS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks


def run_cli():  # pragma: no cover - interactive
    index_store = load_indexes()
    chunk_map = index_store.chunk_map

    query_handler = MultilingualQueryHandler()
    router = AgenticQueryRouter()
    expander = QueryExpander()
    decomposer = QueryDecomposer()
    retriever = HybridRetriever(index_store)
    reranker = CrossEncoderReranker()
    evaluator = CRAGEvaluator()
    reflector = AdvancedSelfReflector()
    llm = LocalLLM()

    while True:
        question = input("Question (ou 'quit'): ")
        if question.lower() in {"quit", "exit"}:
            break

        q_orig, q_fr = query_handler.normalize(question)
        route = router.route(q_fr)
        expansions = expander.expand(q_fr, n=3)
        subqueries = decomposer.decompose(q_fr) if route == "MULTI_STEP" else [q_fr]

        candidates = []
        for sub in subqueries:
            for variant in expansions:
                candidates.extend(retriever.retrieve(variant, manifest=None))
        reranked = reranker.rerank(q_fr, candidates, top_n=config.RERANK_TOP_N)
        scores = list(range(len(reranked), 0, -1))

        if not evaluator.is_evidence_strong(scores, reranked):
            print(evaluator.fallback_message())
            continue

        evidence_texts = [f"[{c.base_chunk.form_code}] {c.base_chunk.section_title}: {c.base_chunk.content}" for c in reranked[: config.FINAL_EVIDENCE_K]]
        system_prompt = (
            "Vous êtes un assistant spécialisé dans les formulaires IRCC. Répondez uniquement en français en vous basant sur les "
            "extraits fournis. Citez le code du formulaire et la section."
        )
        user_prompt = q_fr + "\nExtraits:\n" + "\n".join(evidence_texts)
        answer = llm.chat(system_prompt, user_prompt, max_new_tokens=256)

        if not verify_response_against_evidence(answer, reranked):
            answer = evaluator.fallback_message()
        else:
            answer = reflector.reflect(q_fr, answer, reranked)
        print(answer)
        print(DISCLAIMER)


if __name__ == "__main__":  # pragma: no cover
    run_cli()
