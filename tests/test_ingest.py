import os
from pathlib import Path

from rag_formulaire import config
from rag_formulaire.downloader import download_french_ircc_forms


def test_downloader_generates_forms(tmp_path, monkeypatch):
    monkeypatch.setenv("RAG_FORM_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("RAG_FORM_MIN_FORMS", "3")
    # reload config paths
    from importlib import reload

    reload(config)
    forms = download_french_ircc_forms(min_count=3)
    assert len(forms) >= 3
    for meta in forms[:3]:
        assert Path(meta.local_path).exists()
