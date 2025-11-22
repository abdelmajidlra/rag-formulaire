import os
from pathlib import Path

from rag_formulaire import config
from rag_formulaire.downloader import download_french_ircc_forms


def test_downloader_fetches_real_forms(tmp_path, monkeypatch):
    monkeypatch.setenv("RAG_FORM_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("RAG_FORM_MIN_FORMS", "1")
    # reload config paths
    from importlib import reload

    reload(config)
    try:
        forms = download_french_ircc_forms(min_count=1)
    except RuntimeError:
        import pytest

        pytest.skip("Téléchargement réseau indisponible")
    assert len(forms) >= 1
    assert Path(forms[0].local_path).exists()
