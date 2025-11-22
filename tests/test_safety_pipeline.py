from rag_formulaire.evaluation import CRAGEvaluator


def test_crag_fallback_when_no_scores():
    evaluator = CRAGEvaluator()
    assert evaluator.is_evidence_strong([], []) is False
    assert "Je ne peux pas" in evaluator.fallback_message()
