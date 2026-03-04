"""
Build FAISS index from Delta Airlines policy documents.
"""
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
import json
from src.config import (
    DATA_DIR, VECTOR_STORE_PATH, META_PATH, BM25_CORPUS_PATH,
    EMBEDDING_MODEL_NAME, METADATA_STORE_PATH
)
from src.parse_delta_policies import load_delta_policies


def build_index():
    """Create FAISS index from policy documents and save to disk."""
    print("Building FAISS index...")
    
    # Load documents mapping
    print("Loading documents...")
    chunks_data = load_delta_policies()
    
    if len(chunks_data) < 5:
        raise ValueError(f"Insufficient documents: found {len(chunks_data)}, need at least 5")
    
    # Extract plain text for embedding
    texts = [chunk["text"] for chunk in chunks_data]
    
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
    
    # Save tokenized corpus for BM25 hybrid search
    print("Saving BM25 corpus...")
    tokenized_corpus = [doc.lower().split() for doc in texts]
    with open(BM25_CORPUS_PATH, "wb") as f:
        pickle.dump(tokenized_corpus, f)
        
    # Save structured metadata mapping (chunk_id -> metadata)
    print("Saving metadata store...")
    metadata_store = {chunk["chunk_id"]: chunk for chunk in chunks_data}
    with open(METADATA_STORE_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata_store, f, indent=2)
    
    print(f"Index built: {len(texts)} chunks")
    print(f"FAISS index: {VECTOR_STORE_PATH}")
    print(f"BM25 corpus: {BM25_CORPUS_PATH}")
    print(f"Metadata store: {METADATA_STORE_PATH}")
    print(f"Run application: python -m src.app_cli")


if __name__ == "__main__":
    build_index()
