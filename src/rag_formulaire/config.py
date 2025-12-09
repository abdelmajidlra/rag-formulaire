import os
from pathlib import Path

# Set PyTorch memory configuration to avoid fragmentation - MUST BE DONE BEFORE IMPORTING TORCH
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

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
# Llama 3.1 8B Instruct est le modèle par défaut (meilleure compréhension des instructions)
GEN_MODEL_NAME = os.getenv("RAG_FORM_GEN_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
# Chargement 4 bits facultatif pour tenir sur des GPUs type Tesla (désactivé si non disponible)
GEN_LOAD_4BIT = os.getenv("RAG_FORM_GEN_4BIT", "true").lower() == "true"

# Retrieval params (tunable via env for constrained GPU like Colab T4)
BM25_K1 = 1.5
BM25_B = 0.75
BM25_TOP_K = int(os.getenv("RAG_FORM_BM25_TOP_K", "25"))
VECTOR_TOP_K = int(os.getenv("RAG_FORM_VECTOR_TOP_K", "30"))
RERANK_TOP_N = int(os.getenv("RAG_FORM_RERANK_TOP_N", "12"))
FINAL_EVIDENCE_K = int(os.getenv("RAG_FORM_FINAL_EVIDENCE_K", "6"))
RRF_K = int(os.getenv("RAG_FORM_RRF_K", "60"))

# Generation params
GEN_MAX_NEW_TOKENS = int(os.getenv("RAG_FORM_GEN_MAX_NEW_TOKENS", "256"))

# Gating thresholds
CRAG_MIN_SCORE = 0.05
CRAG_MEAN_TOPK = 0.02
CRAG_MIN_DISTINCT_FORMS = 1

# Hallucination detection
HALLUCINATION_NGRAM = 5
# Strict verification mode: if enabled, performs n-gram matching (may block valid natural language)
# If disabled, only validates form codes (recommended)
ENABLE_STRICT_VERIFICATION = os.getenv("RAG_FORM_STRICT_VERIFICATION", "false").lower() == "true"

# Chunking configuration  
CHUNK_SIZE = int(os.getenv("RAG_FORM_CHUNK_SIZE", "400"))  # Increased from 200 for better context
CHUNK_OVERLAP = int(os.getenv("RAG_FORM_CHUNK_OVERLAP", "80"))  # Increased from 30

# GraphRAG flag
ENABLE_GRAPHRAG = os.getenv("RAG_FORM_ENABLE_GRAPHRAG", "false").lower() == "true"

# Ingestion controls
# On exige au moins 40 formulaires réels (français) téléchargés automatiquement.
MIN_FORMS = int(os.getenv("RAG_FORM_MIN_FORMS", "40"))
