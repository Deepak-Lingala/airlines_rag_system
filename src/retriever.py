"""
Hybrid Retriever: BM25 + FAISS with Cross-Encoder Reranking.

Combines sparse keyword search (BM25) with dense vector search (FAISS)
using Reciprocal Rank Fusion, then reranks with a cross-encoder model.
"""
import pickle
import numpy as np
import faiss
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
from src.config import (
    VECTOR_STORE_PATH, META_PATH, BM25_CORPUS_PATH, METADATA_STORE_PATH,
    EMBEDDING_MODEL_NAME, CROSS_ENCODER_MODEL,
    HYBRID_TOP_K, RERANK_TOP_N, RRF_K, RELEVANCE_THRESHOLD,
)
import json


class HybridRetriever:
    """
    Production-grade retriever combining BM25 keyword search with
    FAISS dense vector search, merged via Reciprocal Rank Fusion (RRF),
    and reranked with a cross-encoder model.
    """

    def __init__(self):
        """Load all retrieval components: embeddings, FAISS, BM25, cross-encoder."""
        print("Loading hybrid retriever...")

        # Dense retrieval — sentence-transformer + FAISS
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        self.index = faiss.read_index(str(VECTOR_STORE_PATH))

        # Metadata (original format)
        with open(META_PATH, "rb") as f:
            meta = pickle.load(f)
        self.texts = meta["texts"]

        # Structured Metadata (chunk_id -> metadata mapping)
        with open(METADATA_STORE_PATH, "r", encoding="utf-8") as f:
            self.metadata_store = json.load(f)
            
        # Create a reverse mapping from document index to chunk_id 
        # (Assuming texts list matches chunks exactly)
        self.idx_to_chunk_id = list(self.metadata_store.keys())

        # Sparse retrieval — BM25
        if BM25_CORPUS_PATH.exists():
            with open(BM25_CORPUS_PATH, "rb") as f:
                tokenized_corpus = pickle.load(f)
        else:
            # Fallback: tokenize from metadata texts
            tokenized_corpus = [doc.lower().split() for doc in self.texts]

        self.bm25 = BM25Okapi(tokenized_corpus)

        # Cross-encoder reranker
        self.cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)

        print(f"  FAISS index: {self.index.ntotal} vectors")
        print(f"  BM25 corpus: {len(self.texts)} documents")
        print(f"  Cross-encoder: {CROSS_ENCODER_MODEL}")
        print("Hybrid retriever ready.")

    # ------------------------------------------------------------------
    # Individual search methods
    # ------------------------------------------------------------------

    def faiss_search(self, query: str, k: int = HYBRID_TOP_K) -> List[Tuple[int, float]]:
        """
        Dense vector search using FAISS.

        Returns:
            List of (doc_index, similarity_score) tuples.
        """
        query_embedding = self.embedding_model.encode([query]).astype("float32")
        faiss.normalize_L2(query_embedding)
        distances, indices = self.index.search(query_embedding, k)

        results = []
        for idx, score in zip(indices[0], distances[0]):
            if 0 <= idx < len(self.texts):
                results.append((int(idx), float(score)))
        return results

    def bm25_search(self, query: str, k: int = HYBRID_TOP_K) -> List[Tuple[int, float]]:
        """
        Sparse keyword search using BM25.

        Returns:
            List of (doc_index, bm25_score) tuples, sorted by score descending.
        """
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)

        # Get top-k indices by score
        top_indices = np.argsort(scores)[::-1][:k]
        results = [
            (int(idx), float(scores[idx]))
            for idx in top_indices
            if scores[idx] > 0
        ]
        return results

    # ------------------------------------------------------------------
    # Reciprocal Rank Fusion
    # ------------------------------------------------------------------

    def reciprocal_rank_fusion(
        self,
        faiss_results: List[Tuple[int, float]],
        bm25_results: List[Tuple[int, float]],
        k: int = RRF_K,
    ) -> List[Tuple[int, float]]:
        """
        Merge FAISS and BM25 results using Reciprocal Rank Fusion.

        RRF score = sum(1 / (k + rank_i)) across all result lists.
        This is robust to score scale differences between retrievers.

        Args:
            faiss_results: (doc_index, score) from dense search
            bm25_results: (doc_index, score) from sparse search
            k: RRF constant (default 60, standard in literature)

        Returns:
            Merged list of (doc_index, rrf_score) sorted by score descending.
        """
        rrf_scores: Dict[int, float] = {}

        for rank, (doc_idx, _) in enumerate(faiss_results):
            rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0) + 1.0 / (k + rank + 1)

        for rank, (doc_idx, _) in enumerate(bm25_results):
            rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0) + 1.0 / (k + rank + 1)

        # Sort by RRF score descending
        merged = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return merged

    # ------------------------------------------------------------------
    # Hybrid search (BM25 + FAISS + RRF)
    # ------------------------------------------------------------------

    def hybrid_search(self, query: str, k: int = HYBRID_TOP_K) -> List[Tuple[int, float]]:
        """
        Perform hybrid search: BM25 + FAISS merged via RRF.

        Returns:
            Top-k (doc_index, rrf_score) tuples.
        """
        faiss_results = self.faiss_search(query, k)
        bm25_results = self.bm25_search(query, k)
        merged = self.reciprocal_rank_fusion(faiss_results, bm25_results)
        return merged[:k]

    # ------------------------------------------------------------------
    # Cross-Encoder Reranking
    # ------------------------------------------------------------------

    def rerank(
        self,
        query: str,
        candidate_indices: List[Tuple[int, float]],
        top_n: int = RERANK_TOP_N,
    ) -> List[Dict]:
        """
        Rerank candidate chunks using cross-encoder model.

        Args:
            query: User question
            candidate_indices: (doc_index, score) from hybrid search
            top_n: Number of final chunks to return

        Returns:
            List of dicts with 'text', 'source', 'score', 'relevant' keys.
        """
        if not candidate_indices:
            return []

        # Build query-document pairs for cross-encoder
        pairs = []
        doc_indices = []
        for doc_idx, _ in candidate_indices:
            pairs.append((query, self.texts[doc_idx]))
            doc_indices.append(doc_idx)

        # Score all pairs with cross-encoder
        ce_scores = self.cross_encoder.predict(pairs)

        # Sort by cross-encoder score descending
        scored = sorted(
            zip(doc_indices, ce_scores),
            key=lambda x: x[1],
            reverse=True,
        )

        # Take top-n and build result dicts
        results = []
        for doc_idx, score in scored[:top_n]:
            chunk_id = self.idx_to_chunk_id[doc_idx]
            chunk_meta = self.metadata_store[chunk_id]
            
            # The prompt format requested string:
            # {
            #   "text": "retrieved chunk text",
            #   "source": "Baggage Faqs",
            #   "url": "https://www.delta.com/...",
            #   "relevance_score": 1.85,
            #   "chunk_id": "baggage_faqs_chunk_7",
            #   "relevant": True/False (for internal pipeline use)
            # }

            results.append({
                "text": chunk_meta["text"],
                "source": chunk_meta["source"],
                "url": chunk_meta["url"],
                "relevance_score": float(score),
                "chunk_id": chunk_meta["chunk_id"],
                "score": float(score), # keep backwards compatibility with previous apps
                "relevant": float(score) >= RELEVANCE_THRESHOLD,
            })

        return results

    # ------------------------------------------------------------------
    # Full retrieval pipeline
    # ------------------------------------------------------------------

    def retrieve(self, query: str) -> List[Dict]:
        """
        Full retrieval pipeline:
          1. Hybrid search (BM25 + FAISS via RRF)
          2. Cross-encoder reranking
          3. Return top-N chunks with relevance flags

        Args:
            query: User question

        Returns:
            List of result dicts with 'text', 'source', 'score', 'relevant'.
            If no chunks pass the relevance threshold, returns results
            with all 'relevant' flags set to False (caller should use
            the no_answer_response from prompts.yaml).
        """
        # Step 1: Hybrid search
        candidates = self.hybrid_search(query, HYBRID_TOP_K)

        # Step 2: Cross-encoder reranking
        results = self.rerank(query, candidates, RERANK_TOP_N)

        return results

    def has_relevant_results(self, results: List[Dict]) -> bool:
        """Check if any retrieved chunks passed the relevance threshold."""
        return any(r["relevant"] for r in results)


if __name__ == "__main__":
    # Quick smoke test
    retriever = HybridRetriever()
    test_query = "What is the carry-on bag size limit?"
    print(f"\nQuery: {test_query}")

    results = retriever.retrieve(test_query)
    for i, r in enumerate(results, 1):
        print(f"\n--- Result {i} (score={r['score']:.4f}, relevant={r['relevant']}) ---")
        print(f"Source: {r['source']}")
        print(f"Text: {r['text'][:200]}...")
