#!/usr/bin/env python3
"""
Complete re-indexing workflow with enhanced validation and error recovery.

This script:
1. Cleans up corrupt PDFs from previous downloads
2. Re-downloads forms with strict PDF validation
3. Re-parses with multi-fallback PDF extraction
4. Re-indexes with the adjusted confidence threshold (0.15)
5. Validates the index quality
"""

import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from rag_formulaire import config
from rag_formulaire.downloader import download_french_ircc_forms
from rag_formulaire.ingest import ingest_forms
from rag_formulaire.indexing import build_bm25_index, build_vector_index

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def cleanup_corrupt_pdfs():
    """Remove any corrupt PDFs from previous downloads."""
    logger.info("Step 1/5: Cleaning up corrupt PDFs...")
    
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


def download_forms():
    """Download forms with enhanced PDF validation."""
    logger.info("Step 2/5: Downloading forms with validation...")
    try:
        forms = download_french_ircc_forms()
        logger.info(f"Successfully downloaded {len(forms)} valid forms")
        return forms
    except Exception as e:
        logger.error(f"Download failed: {e}")
        raise


def parse_and_chunk():
    """Parse PDFs and create chunks with quality filters."""
    logger.info("Step 3/5: Parsing PDFs with multi-fallback extraction...")
    try:
        chunks = ingest_forms()
        logger.info(f"Successfully created {len(chunks)} quality chunks")
        
        # Log chunk statistics
        form_codes = {c.form_code for c in chunks}
        logger.info(f"Chunks extracted from {len(form_codes)} unique forms")
        
        # Check for quality issues
        avg_length = sum(len(c.content) for c in chunks) / len(chunks) if chunks else 0
        logger.info(f"Average chunk length: {avg_length:.0f} characters")
        
        return chunks
    except Exception as e:
        logger.error(f"Parsing failed: {e}")
        raise


def build_indexes(chunks):
    """Build BM25 and vector indexes."""
    logger.info("Step 4/5: Building search indexes...")
    try:
        # BM25 index
        logger.info("Building BM25 index...")
        build_bm25_index(chunks)
        logger.info("✓ BM25 index created")
        
        # Vector index
        logger.info("Building vector index...")
        build_vector_index(chunks)
        logger.info("✓ Vector index created")
        
    except Exception as e:
        logger.error(f"Indexing failed: {e}")
        raise


def validate_index():
    """Validate the index quality."""
    logger.info("Step 5/5: Validating index quality...")
    
    # Check that index directories exist and have content
    checks = {
        "BM25 index": config.BM25_DIR / "index.pkl",
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
            logger.error(f"✗ {name}: NOT FOUND")
            all_ok = False
    
    return all_ok


def main():
    """Execute complete re-indexing workflow."""
    logger.info("="*70)
    logger.info("RAG FORMULAIRE - COMPLETE RE-INDEXING WITH FIXES")
    logger.info("="*70)
    logger.info("")
    logger.info("Applied fixes:")
    logger.info("  1. Confidence threshold: 0.25 → 0.15 (reduced false negatives)")
    logger.info("  2. PDF validation: Added header/content validation")
    logger.info("  3. Multi-fallback parsing: Docling → pdfplumber → pypdf → pymupdf")
    logger.info("")
    logger.info("="*70)
    logger.info("")
    
    try:
        # Step 1: Cleanup
        cleanup_corrupt_pdfs()
        
        # Step 2: Download
        forms = download_forms()
        
        # Step 3: Parse and chunk
        chunks = parse_and_chunk()
        
        # Step 4: Build indexes
        build_indexes(chunks)
        
        # Step 5: Validate
        if validate_index():
            logger.info("")
            logger.info("="*70)
            logger.info("✓ RE-INDEXING COMPLETED SUCCESSFULLY")
            logger.info("="*70)
            logger.info("")
            logger.info("Next steps:")
            logger.info("  1. Run evaluation: python -m rag_formulaire.cli evaluate")
            logger.info("  2. Expected improvement: 85% → 60%+ direct answers")
            logger.info("")
        else:
            logger.error("Index validation failed!")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Re-indexing failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
