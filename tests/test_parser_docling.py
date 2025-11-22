from pathlib import Path

import pytest

from rag_formulaire.parser_docling import parse_pdf_to_docling


def test_parse_invalid_pdf_returns_empty_pages(tmp_path: Path, caplog: pytest.LogCaptureFixture):
    invalid_pdf = tmp_path / "invalid.pdf"
    invalid_pdf.write_text("not a pdf")

    with caplog.at_level("WARNING"):
        doc = parse_pdf_to_docling(str(invalid_pdf))

    assert doc == {"pages": []}
    assert any("Échec de lecture du PDF" in message for message in caplog.messages)
