import types

import pytest

from core import pipeline


class DummyChunk:
    def __init__(self, form_code: str, local_path: str = ""):
        self.base_chunk = types.SimpleNamespace(form_code=form_code, section_title="", content="")
        self.form_metadata = types.SimpleNamespace(local_path=local_path) if local_path else None


class DummyRAG:
    def __init__(self, answer: str, evidence_count: int):
        self.answer = answer
        self.evidence_count = evidence_count

    def ask_question(self, question: str):
        evidence = [DummyChunk("IMM 1234", "/tmp/IMM1234.pdf") for _ in range(self.evidence_count)]
        return {"answer": self.answer, "evidence": evidence, "route": "SINGLE", "expansions": [question]}


@pytest.mark.parametrize(
    "question,answer,evidence_count,expected_forms",
    [
        ("Quel formulaire pour permis d'études ?", "Utilisez IMM 5476.", 3, ["IMM1234", "IMM5476"]),
        ("Question hors sujet", "", 0, []),
    ],
)
def test_answer_question(monkeypatch, question, answer, evidence_count, expected_forms):
    dummy = DummyRAG(answer, evidence_count)
    monkeypatch.setattr(pipeline, "get_pipeline", lambda: dummy)

    result = pipeline.answer_question(question)
    for code in expected_forms:
        assert code in result["forms"]

    if evidence_count == 0:
        assert result["confidence"] <= 0.2
    else:
        assert result["confidence"] >= 0.65


def test_missing_question_handled(monkeypatch):
    dummy = DummyRAG("", 0)
    monkeypatch.setattr(pipeline, "get_pipeline", lambda: dummy)

    result = pipeline.answer_question("")
    assert result["answer"]
    assert result["confidence"] == 0.2
