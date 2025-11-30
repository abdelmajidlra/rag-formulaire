import shutil
import os
import sys
from pathlib import Path

# Add src to path so we can import rag_formulaire
current_dir = Path(__file__).resolve().parent
src_dir = current_dir / "src"
sys.path.append(str(src_dir))

from rag_formulaire.config import CHROMA_DIR, INDEX_DIR
from rag_formulaire.ingest import ingest_pipeline

def main():
    print("🚀 Starting Re-indexing Fix Script")
    
    # 1. Delete old ChromaDB
    if CHROMA_DIR.exists():
        print(f"🗑️  Deleting old ChromaDB at: {CHROMA_DIR}")
        try:
            shutil.rmtree(CHROMA_DIR)
            print("✅ Old index deleted successfully")
        except Exception as e:
            print(f"❌ Error deleting ChromaDB: {e}")
            return
    else:
        print(f"ℹ️  No existing ChromaDB found at {CHROMA_DIR}")

    # Also clear BM25 if it exists to be safe
    bm25_dir = INDEX_DIR / "bm25"
    if bm25_dir.exists():
        print(f"🗑️  Deleting old BM25 index at: {bm25_dir}")
        shutil.rmtree(bm25_dir)
        print("✅ Old BM25 index deleted")

    # 2. Run Ingestion
    print("\n🔄 Starting Ingestion Pipeline (this may take a few minutes)...")
    try:
        ingest_pipeline()
        print("\n✅✅ RE-INDEXING COMPLETE! ✅✅")
        print("The system is now using the clean index with aggressive chunk filtering.")
        print("Please re-run your 20-question test now.")
    except Exception as e:
        print(f"\n❌ Error during ingestion: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
