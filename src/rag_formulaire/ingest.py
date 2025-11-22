from __future__ import annotations

import json
import logging
from pathlib import Path

from tqdm import tqdm

from . import config
from .chunking import AdaptiveSectionAwareChunkSplitter
from .downloader import download_french_ircc_forms
from .graph_rag import GraphRAG
from .indexing import build_indexes
from .parser_docling import parse_chunks_from_doc, parse_pdf_to_docling

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def ingest_pipeline(min_forms: int | None = None):
    data_dir = config.DATA_DIR
    data_dir.mkdir(parents=True, exist_ok=True)
    manifest = download_french_ircc_forms(min_forms)

    all_chunks = []
    for meta in tqdm(manifest, desc="Parsing"):
        doc = parse_pdf_to_docling(str(meta.local_path))
        chunks = parse_chunks_from_doc(doc, meta.form_code, meta.title_fr)
        refined = AdaptiveSectionAwareChunkSplitter.split(chunks)
        for c in refined:
            c.category = meta.category
            c.last_updated = meta.last_updated
        all_chunks.extend(refined)
        logger.debug("%s -> %s chunks", meta.form_code, len(refined))

    logger.info("Total des chunks: %s", len(all_chunks))

    # Persist chunk map for CLI loading
    config.CHUNKS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(config.CHUNKS_PATH, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk.__dict__, ensure_ascii=False) + "\n")

    index_store = build_indexes(all_chunks)

    if config.ENABLE_GRAPHRAG:
        graph = GraphRAG()
        graph.build_from_chunks(all_chunks)
        with open(config.DATA_DIR / "graph.json", "w", encoding="utf-8") as f:
            json.dump({k: list(v) for k, v in graph.graph.items()}, f, ensure_ascii=False, indent=2)

    logger.info("Ingestion terminée: %s formulaires, %s chunks", len(manifest), len(all_chunks))
    return index_store


if __name__ == "__main__":  # pragma: no cover
    ingest_pipeline()
