from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import List

try:
    from docling.document_converter import DocumentConverter
    from docling.datamodel.document import Document as DoclingDocument
    _DOCLING_AVAILABLE = True
except Exception:
    DoclingDocument = object
    _DOCLING_AVAILABLE = False

import pdfplumber

logger = logging.getLogger(__name__)

def parse_pdf_to_docling(pdf_path: str):
    path = Path(pdf_path)
    
    # Try primary method: Docling
    try:
        if _DOCLING_AVAILABLE:
            converter = DocumentConverter()
            result = converter.convert(path)
            # Validate that we got actual content
            if hasattr(result, 'pages') and result.pages:
                return result
            logger.warning("Docling returned empty result for %s, trying fallback", path)
    except Exception as exc:
        logger.warning("Docling failed for %s: %s, trying fallback", path, exc)
    
    # Fallback 1: pdfplumber (good for form-based PDFs)
    try:
        pages = []
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                text = page.extract_text() or ""
                if text.strip():  # Only add pages with actual content
                    pages.append({"text": text})
        
        if pages:
            logger.info("Extracted %d pages from %s using pdfplumber", len(pages), path.name)
            return {"pages": pages}
    except Exception as exc:
        logger.warning("pdfplumber failed for %s: %s", path, exc)
    
    # Fallback 2: pypdf (handles encrypted/protected PDFs)
    try:
        import pypdf
        pages = []
        with open(path, 'rb') as file:
            reader = pypdf.PdfReader(file)
            for page in reader.pages:
                text = page.extract_text() or ""
                if text.strip():
                    pages.append({"text": text})
        
        if pages:
            logger.info("Extracted %d pages from %s using pypdf", len(pages), path.name)
            return {"pages": pages}
    except Exception as exc:
        logger.warning("pypdf failed for %s: %s", path, exc)
    
    # Fallback 3: pymupdf/fitz (OCR-like extraction for scanned PDFs)
    try:
        import fitz  # pymupdf
        pages = []
        doc = fitz.open(path)
        for page_num in range(doc.page_count):
            page = doc[page_num]
            text = page.get_text() or ""
            if text.strip():
                pages.append({"text": text})
        doc.close()
        
        if pages:
            logger.info("Extracted %d pages from %s using pymupdf", len(pages), path.name)
            return {"pages": pages}
    except Exception as exc:
        logger.warning("pymupdf failed for %s: %s", path, exc)
    
    # All methods failed - return empty but log the failure
    logger.error(
        "All PDF extraction methods failed for %s. This PDF may be corrupt, "
        "encrypted, or require specific Adobe Reader features. Skipping.", 
        path.name
    )
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

def parse_chunks_from_doc(doc, form_code: str, title: str):
    from .data_models import FormChunk
    
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
            # Filter out low-quality chunks
            section_clean = section.strip()
            
            # Skip empty or very short sections
            if len(section_clean) < 50:
                continue
            
            # Skip sections that are just form codes or page numbers
            if re.match(r'^(IMM|CIT)\s*\d{4}(\s*\(\d{2}-\d{4}\)\s*F)?\.{3,}$', section_clean):
                logger.debug(f"Skipping truncated form code chunk: {section_clean[:50]}")
                continue
            
            # Skip Adobe Reader error messages
            if "adobe reader" in section_clean.lower() and len(section_clean) < 300:
                logger.debug(f"Skipping Adobe error message chunk from {form_code}")
                continue
            
            # Skip sections with only metadata/page numbers
            if re.match(r'^Page\s+\d+\s+de\s+\d+$', section_clean, re.IGNORECASE):
                logger.debug(f"Skipping page number only chunk: {section_clean}")
                continue
            
            position += 1
            section_title = section_clean.split("\n")[0][:120] if section_clean else "Section"
            chunks.append(
                FormChunk(
                    chunk_id=f"{form_code}-{page_number}-{position}",
                    form_code=form_code,
                    form_title=title,
                    section_title=section_title,
                    question_label=None,
                    question_id=None,
                    page_number=page_number,
                    content=section_clean,
                    position_in_form=position,
                )
            )
    return chunks