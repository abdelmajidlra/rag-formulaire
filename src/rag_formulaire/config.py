import os
from pathlib import Path

# Base directories
BASE_DIR = Path(os.getenv("RAG_FORM_BASE_DIR", Path(__file__).resolve().parents[2]))
DATA_DIR = Path(os.getenv("RAG_FORM_DATA_DIR", BASE_DIR / "data"))
RAW_FORMS_DIR = DATA_DIR / "raw" / "forms"
PARSED_DIR = DATA_DIR / "parsed"
INDEX_DIR = DATA_DIR / "index"
BM25_DIR = INDEX_DIR / "bm25"
CHROMA_DIR = INDEX_DIR / "chroma"
MANIFEST_PATH = DATA_DIR / "forms_manifest.json"
CHUNKS_PATH = DATA_DIR / "chunks.jsonl"

# Model configuration
EMBEDDING_MODEL_NAME = os.getenv("RAG_FORM_EMBED_MODEL", "intfloat/multilingual-e5-base")
RERANK_MODEL_NAME = os.getenv("RAG_FORM_RERANK_MODEL", "BAAI/bge-reranker-large")
GEN_MODEL_NAME = os.getenv("RAG_FORM_GEN_MODEL", "mistralai/Mistral-7B-Instruct-v0.2")

# Retrieval params
BM25_K1 = 1.5
BM25_B = 0.75
BM25_TOP_K = 25
VECTOR_TOP_K = 30
RERANK_TOP_N = 12
FINAL_EVIDENCE_K = 6
RRF_K = 60

# Gating thresholds
CRAG_MIN_SCORE = 0.05
CRAG_MEAN_TOPK = 0.02
CRAG_MIN_DISTINCT_FORMS = 1

HALLUCINATION_NGRAM = 5

# GraphRAG flag
ENABLE_GRAPHRAG = os.getenv("RAG_FORM_ENABLE_GRAPHRAG", "false").lower() == "true"

# Ingestion controls
MIN_FORMS = int(os.getenv("RAG_FORM_MIN_FORMS", "100"))
MAX_SYNTHETIC_FORMS = int(os.getenv("RAG_FORM_MAX_SYNTH", "120"))
