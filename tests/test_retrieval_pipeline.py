from importlib import reload
from pathlib import Path

from rag_formulaire import config
from rag_formulaire.data_models import FormChunk
from rag_formulaire.indexing import build_indexes
from rag_formulaire.retrieval import HybridRetriever


def test_retrieval_returns_chunks(tmp_path, monkeypatch):
    monkeypatch.setenv("RAG_FORM_DATA_DIR", str(tmp_path / "data"))
    reload(config)

    chunks = [
        FormChunk(
            chunk_id=f"IMM000{i}",
            form_code="IMM 0000",
            form_title="Test",
            section_title="Renseignements",
            question_label=None,
            question_id=None,
            page_number=1,
            content=f"Nom du demandeur {i}",
            position_in_form=i,
        )
        for i in range(3)
    ]
    index_store = build_indexes(chunks)
    retriever = HybridRetriever(index_store)
    results = retriever.retrieve("Nom du demandeur")
    assert results
