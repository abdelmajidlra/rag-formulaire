from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class FormMetadata:
    form_code: str
    title_fr: str
    pdf_url: str
    local_path: Path
    category: Optional[str] = None
    last_updated: Optional[str] = None


@dataclass
class FormChunk:
    chunk_id: str
    form_code: str
    form_title: str
    section_title: str
    question_label: Optional[str]
    question_id: Optional[str]
    page_number: int
    content: str
    position_in_form: int
    category: Optional[str] = None
    last_updated: Optional[str] = None


@dataclass
class ContextualizedChunk:
    base_chunk: FormChunk
    before: Optional[str] = None
    after: Optional[str] = None
    form_metadata: Optional[FormMetadata] = None

    def combined_text(self) -> str:
        parts = [self.before or "", self.base_chunk.content, self.after or ""]
        return "\n".join([p for p in parts if p])


@dataclass
class RetrievalResult:
    query: str
    chunks: List[ContextualizedChunk]
    scores: List[float] = field(default_factory=list)
