"""
Build FAISS index from Delta Airlines policy documents.
"""
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
from src.config import *
from src.parse_delta_policies import load_delta_policies


def build_index():
    """Create FAISS index from policy documents and save to disk."""
    print("Building FAISS index...")
    
    # Load documents
    print("Loading documents...")
    texts = load_delta_policies()
    
    if len(texts) < 5:
        raise ValueError(f"Insufficient documents: found {len(texts)}, need at least 5")
    
    # Generate embeddings
    print(f"Encoding {len(texts)} chunks...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        batch_size=16
    ).astype("float32")
    
    # Create FAISS index with cosine similarity (inner product on normalized vectors)
    print("Creating FAISS index...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    faiss.normalize_L2(embeddings)  # Normalize for cosine similarity
    index.add(embeddings)
    
    # Save index and metadata
    print("Saving index and metadata...")
    DATA_DIR.mkdir(exist_ok=True)
    faiss.write_index(index, str(VECTOR_STORE_PATH))
    
    with open(META_PATH, "wb") as f:
        pickle.dump({"texts": texts}, f)
    
    print(f"Index built: {len(texts)} chunks")
    print(f"Saved to: {VECTOR_STORE_PATH}")
    print(f"Run application: python -m src.app_cli")


if __name__ == "__main__":
    build_index()
