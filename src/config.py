"""
Configuration settings for Delta Airlines RAG system.
"""
import os
from pathlib import Path
from dotenv import load_dotenv
import yaml

load_dotenv()

# Directory paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
VECTOR_STORE_PATH = DATA_DIR / "faiss.index"
META_PATH = DATA_DIR / "meta.pkl"
BM25_CORPUS_PATH = DATA_DIR / "bm25_corpus.pkl"
METADATA_STORE_PATH = DATA_DIR / "metadata_store.json"
PROMPTS_PATH = BASE_DIR / "prompts.yaml"

# Model configuration
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
OLLAMA_URL = "http://localhost:11434/v1"
OLLAMA_MODEL = "llama3.2:3b"

# Retrieval settings
TOP_K = 5
HYBRID_TOP_K = 10       # Candidates from hybrid BM25+FAISS search
RERANK_TOP_N = 3        # Final chunks after cross-encoder reranking
RRF_K = 60              # Reciprocal Rank Fusion constant
RELEVANCE_THRESHOLD = 0.01  # Min cross-encoder score for relevance

# Debug mode
DEBUG = os.getenv("DEBUG", "False").lower() == "true"


def load_prompts():
    """Load versioned prompts from prompts.yaml."""
    if not PROMPTS_PATH.exists():
        raise FileNotFoundError(f"Prompts file not found: {PROMPTS_PATH}")
    with open(PROMPTS_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def validate_config():
    """Verify Ollama configuration."""
    print(f"Ollama configuration: {OLLAMA_MODEL} at {OLLAMA_URL}")
    prompts = load_prompts()
    print(f"Prompts version: {prompts.get('version', 'unknown')}")
    return True


if __name__ == "__main__":
    validate_config()
