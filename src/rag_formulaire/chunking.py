from __future__ import annotations

import logging
import re
from typing import List

from .data_models import ContextualizedChunk, FormChunk, FormMetadata

logger = logging.getLogger(__name__)


class AdaptiveSectionAwareChunkSplitter:
    @staticmethod
    def split(form_chunks: List[FormChunk]) -> List[FormChunk]:
        # Already chunked; optionally refine long chunks
        refined: List[FormChunk] = []
        for chunk in form_chunks:
            if len(chunk.content.split()) > 180:
                segments = re.split(r"\n\s*\n|\d+\)|\d+\.\s", chunk.content)
                position = chunk.position_in_form
                for seg in segments:
                    text = seg.strip()
                    if not text:
                        continue
                    position += 1
                    refined.append(
                        FormChunk(
                            chunk_id=f"{chunk.chunk_id}-{position}",
                            form_code=chunk.form_code,
                            form_title=chunk.form_title,
                            section_title=chunk.section_title,
                            question_label=chunk.question_label,
                            question_id=chunk.question_id,
                            page_number=chunk.page_number,
                            content=text,
                            position_in_form=position,
                            category=chunk.category,
                            last_updated=chunk.last_updated,
                        )
                    )
            else:
                refined.append(chunk)
        return refined


class ContextualChunkEnhancer:
    @staticmethod
    def enhance_with_context(chunks: List[FormChunk], window: int = 1, manifest: List[FormMetadata] | None = None) -> List[ContextualizedChunk]:
        manifest_map = {m.form_code: m for m in manifest or []}
        contextualized: List[ContextualizedChunk] = []
        for idx, chunk in enumerate(chunks):
            before = chunks[idx - 1].content if idx - 1 >= 0 and chunks[idx - 1].form_code == chunk.form_code else None
            after = chunks[idx + 1].content if idx + 1 < len(chunks) and chunks[idx + 1].form_code == chunk.form_code else None
            contextualized.append(
                ContextualizedChunk(
                    base_chunk=chunk,
                    before=before,
                    after=after,
                    form_metadata=manifest_map.get(chunk.form_code),
                )
            )
        return contextualized
