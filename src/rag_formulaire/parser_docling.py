from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import List

try:  # pragma: no cover - heavy dependency
    from docling.document_converter import DocumentConverter
    from docling.datamodel.document import Document as DoclingDocument
    _DOCLING_AVAILABLE = True
except Exception:  # noqa: BLE001
    DoclingDocument = object  # type: ignore
    _DOCLING_AVAILABLE = False

import pdfplumber

from .data_models import FormChunk

logger = logging.getLogger(__name__)


def parse_pdf_to_docling(pdf_path: str):
    path = Path(pdf_path)
    try:
        if _DOCLING_AVAILABLE:
            converter = DocumentConverter()
            return converter.convert(path)
        # Fallback: simple object with pages text
        pages = []
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                pages.append({"text": page.extract_text() or ""})
        return {"pages": pages}
    except Exception as exc:  # noqa: BLE001
        logger.warning("Échec de lecture du PDF %s: %s", path, exc)
        return {"pages": []}


def _extract_sections(text: str) -> List[str]:
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    sections = []
    buffer = []
    for line in lines:
        if re.match(r"^[A-ZÉÈÀÙÂÊÎÔÛÄËÏÖÜ\d][A-ZÉÈÀÙÂÊÎÔÛÄËÏÖÜ\d\s\-:]{4,}$", line):
            if buffer:
                sections.append("\n".join(buffer))
                buffer = []
            sections.append(line)
        else:
            buffer.append(line)
    if buffer:
        sections.append("\n".join(buffer))
    return sections


def parse_chunks_from_doc(doc, form_code: str, title: str) -> List[FormChunk]:
    chunks: List[FormChunk] = []
    if _DOCLING_AVAILABLE and hasattr(doc, "pages"):
        pages = doc.pages
    else:
        pages = doc.get("pages", []) if isinstance(doc, dict) else []

    position = 0
    for page_number, page in enumerate(pages, start=1):
        if _DOCLING_AVAILABLE:
            text = page.get_text() or ""
        else:
            text = page.get("text", "") if isinstance(page, dict) else ""
        sections = _extract_sections(text) if text else []
        if not sections:
            sections = [text]
        for section in sections:
            position += 1
            section_title = section.split("\n")[0][:120] if section else "Section"
            chunks.append(
                FormChunk(
                    chunk_id=f"{form_code}-{page_number}-{position}",
                    form_code=form_code,
                    form_title=title,
                    section_title=section_title,
                    question_label=None,
                    question_id=None,
                    page_number=page_number,
                    content=section,
                    position_in_form=position,
                )
            )
    return chunks
