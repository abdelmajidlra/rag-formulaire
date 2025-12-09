from importlib import reload
from pathlib import Path

import pytest

from rag_formulaire import config
from rag_formulaire import llm
from rag_formulaire import pipeline
from rag_formulaire.data_models import FormChunk
from rag_formulaire.indexing import build_indexes


class DummyTokenizer:
    eos_token_id = 0
    pad_token_id = 0

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        content = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
        if add_generation_prompt:
            content += "\nassistant:"
        return content


@pytest.fixture(autouse=True)
def reset_singleton(monkeypatch):
    llm._SHARED_LLM = None
    monkeypatch.delenv("RAG_FORM_GEN_ENDPOINT", raising=False)
    monkeypatch.delenv("RAG_FORM_DATA_DIR", raising=False)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    yield
    llm._SHARED_LLM = None
    reload(config)


def _build_small_index(tmp_path, monkeypatch):
    monkeypatch.setenv("RAG_FORM_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    reload(config)
    chunks = [
        FormChunk(
            chunk_id="IMM0001-1",
            form_code="IMM 0001",
            form_title="Demande",
            section_title="Section A",
            question_label=None,
            question_id=None,
            page_number=1,
            content="Nom du demandeur et informations",
            position_in_form=0,
        )
    ]
    return build_indexes(chunks)


def test_pipeline_generation_calls_llama(monkeypatch, tmp_path):
    monkeypatch.setenv("RAG_FORM_GEN_MAX_NEW_TOKENS", "200")
    monkeypatch.setenv("RAG_FORM_GEN_ENDPOINT", "https://llama.example.com")
    reload(config)
    reload(pipeline)

    class DummyReranker:
        def rerank(self, query, candidates, top_n):
            return candidates[:top_n]

    monkeypatch.setattr(pipeline, "CrossEncoderReranker", lambda: DummyReranker())

    index_store = _build_small_index(tmp_path, monkeypatch)
    monkeypatch.setattr(pipeline, "load_indexes", lambda: index_store)
    monkeypatch.setattr(llm.LocalLLM, "_prepare_tokenizer", lambda self: setattr(self, "tokenizer", DummyTokenizer()))

    captured = {}
    expected_output = Path("tests/fixtures/llama_sample_output.txt").read_text().strip()

    def fake_remote(self, prompt: str, max_new_tokens: int):
        captured["prompt"] = prompt
        captured["max_new_tokens"] = max_new_tokens
        return expected_output

    monkeypatch.setattr(llm.LocalLLM, "_call_remote", fake_remote)

    rag = pipeline.RAGPipeline()
    response = rag.ask_question("Quels documents fournir pour IMM 0001 ?", evidence_k=1)

    assert response["answer"] == expected_output
    assert response["evidence"], "evidence should include retrieved chunks"
    assert "Extraits" in captured["prompt"]
    assert captured["max_new_tokens"] == config.GEN_MAX_NEW_TOKENS


def test_chat_prompt_keeps_llama_format(monkeypatch):
    dummy = DummyTokenizer()
    instance = llm.LocalLLM.__new__(llm.LocalLLM)
    instance.tokenizer = dummy

    prompt = instance._format_prompt("system", "user")

    assert "system" in prompt
    assert "user" in prompt
    assert prompt.strip().endswith("assistant:")
