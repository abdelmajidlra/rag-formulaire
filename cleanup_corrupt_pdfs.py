#!/usr/bin/env python3
"""
Cleanup utility to remove corrupt/invalid PDF files from the data directory.
This should be run before re-indexing to ensure only valid PDFs are processed.
"""

import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def validate_pdf(file_path: Path) -> tuple[bool, str]:
    """
    Validate if a file is a valid PDF.
    Returns (is_valid, reason).
    """
    try:
        # Check file size
        if file_path.stat().st_size < 5120:
            return False, f"too small ({file_path.stat().st_size} bytes)"
        
        # Read first 1KB to check header
        with open(file_path, 'rb') as f:
            header = f.read(1024).lower()
        
        # Check for PDF magic bytes
        if not header.startswith(b'%pdf-'):
            return False, "missing PDF header"
        
        # Check for HTML error pages
        if b'<!doctype' in header or b'<html' in header:
            return False, "HTML content (error page)"
        
        return True, "valid"
        
    except Exception as e:
        return False, f"read error: {e}"


def cleanup_corrupt_pdfs(data_dir: str = "data/raw/forms", dry_run: bool = True):
    """
    Scan data directory and remove corrupt PDFs.
    
    Args:
        data_dir: Directory containing PDF files
        dry_run: If True, only report issues without deleting
    """
    data_path = Path(data_dir)
    
    if not data_path.exists():
        logger.warning(f"Directory not found: {data_path}")
        return
    
    pdf_files = list(data_path.glob("*.pdf"))
    logger.info(f"Scanning {len(pdf_files)} PDF files in {data_path}")
    
    corrupt_files = []
    valid_count = 0
    
    for pdf_file in pdf_files:
        is_valid, reason = validate_pdf(pdf_file)
        
        if is_valid:
            valid_count += 1
            logger.debug(f"✓ {pdf_file.name}: {reason}")
        else:
            corrupt_files.append((pdf_file, reason))
            logger.warning(f"✗ {pdf_file.name}: {reason}")
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info(f"Scan Results:")
    logger.info(f"  Valid PDFs:   {valid_count}")
    logger.info(f"  Corrupt PDFs: {len(corrupt_files)}")
    logger.info(f"{'='*60}\n")
    
    if corrupt_files:
        if dry_run:
            logger.info("DRY RUN mode - files will NOT be deleted")
            logger.info("Run with --clean to actually remove corrupt files\n")
        
        logger.info("Corrupt files to be removed:")
        for pdf_file, reason in corrupt_files:
            logger.info(f"  - {pdf_file.name}: {reason}")
        
        if not dry_run:
            logger.info(f"\nDeleting {len(corrupt_files)} corrupt files...")
            for pdf_file, reason in corrupt_files:
                try:
                    pdf_file.unlink()
                    logger.info(f"  Deleted: {pdf_file.name}")
                except Exception as e:
                    logger.error(f"  Failed to delete {pdf_file.name}: {e}")
            logger.info("Cleanup complete!")
    else:
        logger.info("No corrupt files found. All PDFs are valid!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Clean up corrupt PDF files")
    parser.add_argument("--data-dir", default="data/raw/forms", help="Directory containing PDFs")
    parser.add_argument("--clean", action="store_true", help="Actually delete files (default is dry-run)")
    
    args = parser.parse_args()
    
    cleanup_corrupt_pdfs(
        data_dir=args.data_dir,
        dry_run=not args.clean
    )
