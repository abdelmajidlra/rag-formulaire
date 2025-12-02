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


def cleanup_corrupt_pdfs():
    """Remove any corrupt PDFs from previous downloads."""
    logger.info("Step 1/3: Cleaning up corrupt PDFs...")

    if not config.RAW_FORMS_DIR.exists():
        logger.info("No existing PDFs to clean up")
        return 0

    corrupt_count = 0
    for pdf_file in config.RAW_FORMS_DIR.glob("*.pdf"):
        try:
            with open(pdf_file, 'rb') as f:
                header = f.read(1024).lower()

            is_corrupt = (
                not header.startswith(b'%pdf-') or
                b'<!doctype' in header or
                b'<html' in header or
                pdf_file.stat().st_size < 5120
            )

            if is_corrupt:
                logger.warning(f"Removing corrupt PDF: {pdf_file.name}")
                pdf_file.unlink()
                corrupt_count += 1
        except Exception as e:
            logger.error(f"Error checking {pdf_file.name}: {e}")

    logger.info(f"Removed {corrupt_count} corrupt PDFs")
    return corrupt_count


def validate_index():
    """Validate the index quality."""
    logger.info("Step 3/3: Validating index quality...")

    # Check that index directories exist and have content
    checks = {
        "BM25 index": config.BM25_DIR / "bm25.pkl",
        "Vector index": config.CHROMA_DIR,
        "Chunks file": config.CHUNKS_PATH,
        "Manifest": config.MANIFEST_PATH,
    }

    all_ok = True
    for name, path in checks.items():
        if path.exists():
            if path.is_file():
                size = path.stat().st_size
                logger.info(f"✓ {name}: {size:,} bytes")
            else:
                logger.info(f"✓ {name}: directory exists")
        else:
            logger.error(f"✗ {name}: NOT FOUND at {path}")
            all_ok = False

    return all_ok


def complete_reindex():
    """Execute complete re-indexing workflow."""
    logger.info("="*70)
    logger.info("RAG FORMULAIRE - COMPLETE RE-INDEXING")
    logger.info("="*70)

    try:
        # Step 1: Cleanup
        cleanup_corrupt_pdfs()

        # Step 2: Full Ingestion (Download, Parse, Index)
        logger.info("Step 2/3: Running full ingestion pipeline (Download -> Parse -> Index)...")
        ingest_pipeline()
        logger.info("✓ Ingestion pipeline completed")

        # Step 3: Validate
        if validate_index():
            logger.info("")
            logger.info("="*70)
            logger.info("✓ RE-INDEXING COMPLETED SUCCESSFULLY")
            logger.info("="*70)
        else:
            logger.error("Index validation failed!")
            raise RuntimeError("Index validation failed")

    except Exception as e:
        logger.error(f"Re-indexing failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":  # pragma: no cover
    complete_reindex()
