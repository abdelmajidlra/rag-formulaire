import sys
import os
from rag_formulaire import config
from rag_formulaire.indexing import load_indexes

def check_form_index(form_code: str):
    print(f"Checking index for form: {form_code}")
    
    # Load indexes
    index_store = load_indexes()
    
    # Check BM25
    print(f"\n--- BM25 Index ---")
    if form_code in index_store.bm25.corpus:
        print(f"Found exact match in BM25 corpus keys? {form_code in index_store.bm25.corpus}")
    else:
        # BM25 corpus is a list of texts, not keys. We need to check the mapping if it exists.
        # Actually, BM25Retriever usually stores the corpus or doc_ids.
        # Let's check the doc_ids if available.
        pass

    # Check ChromaDB
    print(f"\n--- ChromaDB Index ---")
    try:
        results = index_store.chroma_collection.get(
            where={"form_code": form_code},
            limit=5
        )
        
        count = len(results['ids'])
        print(f"Found {count} chunks for {form_code} (limit 5 shown).")
        
        if count == 0:
            print("⚠️ No chunks found in ChromaDB for this form code.")
            # Try partial match
            print("Trying partial match...")
            results_partial = index_store.chroma_collection.get(
                where={"document": {"$contains": form_code}},
                limit=5
            )
            print(f"Found {len(results_partial['ids'])} chunks containing '{form_code}' in text.")
        else:
            for i, text in enumerate(results['documents']):
                print(f"\nChunk {i+1}:")
                print(text[:200] + "...")
                
    except Exception as e:
        print(f"Error querying ChromaDB: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        form = sys.argv[1]
    else:
        form = "IMM 5741"
    
    check_form_index(form)
