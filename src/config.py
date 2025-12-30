"""
Configuration settings for Delta Airlines RAG system.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Directory paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
VECTOR_STORE_PATH = DATA_DIR / "faiss.index"
META_PATH = DATA_DIR / "meta.pkl"

# Model configuration
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
OLLAMA_URL = "http://localhost:11434/v1"
OLLAMA_MODEL = "llama3.2:3b"

# Retrieval settings
TOP_K = 5

# Debug mode
DEBUG = os.getenv("DEBUG", "False").lower() == "true"


def validate_config():
    """Verify Ollama configuration."""
    print(f"Ollama configuration: {OLLAMA_MODEL} at {OLLAMA_URL}")
    return True


if __name__ == "__main__":
    validate_config()
