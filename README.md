# Delta Airlines Policy Assistant

> Production-grade Retrieval-Augmented Generation system for Delta Airlines policy Q&A — featuring hybrid BM25+FAISS retrieval, cross-encoder reranking, citation enforcement, and CI-gated evaluation.

---

## Architecture

```
Delta Airlines Policy Pages (6 distinct categories)
    ↓ (BeautifulSoup scraping)
Cleaned text chunks (77 chunks)
    ↓ (sentence-transformers encoding)
384-dim embeddings
    ↓
    ├── FAISS IndexFlatIP (dense vector search)
    └── BM25Okapi (sparse keyword search)
            ↓
    Reciprocal Rank Fusion (RRF)
            ↓
    Top-10 hybrid candidates
            ↓ (cross-encoder/ms-marco-MiniLM-L-6-v2)
    Reranked top-3 chunks
            ↓ (citation enforcement check)
    Ollama llama3.2 + versioned prompt
            ↓
    Grounded answer with source citations
```


## Evaluation Results

| Metric | Score | Threshold |
|--------|-------|-----------|
| **Faithfulness** | 0.87 | ≥ 0.80 (Pass) |
| **Context Precision** | 0.82 | — |
| **Context Recall** | 0.79 | — |

- **Golden eval set:** 50 Q&A pairs verified against Delta Airlines policy pages
- **CI gate:** Build fails automatically if faithfulness drops below 0.80
- **Evaluation framework:** RAGAS

---

## Technologies

| Component | Technology |
|-----------|-----------|
| Web scraping | `requests` + browser headers |
| Text extraction | `BeautifulSoup4` + sentence-boundary chunking |
| Embeddings | `sentence-transformers` (all-MiniLM-L6-v2, 384-dim) |
| Dense retrieval | `FAISS` IndexFlatIP (cosine similarity) |
| Sparse retrieval | `rank_bm25` BM25Okapi |
| Fusion | Reciprocal Rank Fusion (RRF, k=60) |
| Reranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Generation | `Ollama` with llama3.2:3b (local inference) |
| Prompt management | `PyYAML` versioned config |
| Evaluation | `RAGAS` (faithfulness, context precision/recall) |
| CI/CD | GitHub Actions evaluation pipeline |
| Frontend | `Streamlit` web UI + terminal CLI |

---

## Project Structure

```
airlines_rag_system/
├── src/
│   ├── retriever.py           # Hybrid BM25+FAISS + cross-encoder reranker
│   ├── rag_pipeline.py        # Ollama LLM client with citation enforcement
│   ├── app_streamlit.py       # Streamlit web interface
│   ├── app_cli.py             # Terminal CLI interface
│   ├── build_index.py         # FAISS + BM25 index builder
│   ├── parse_delta_policies.py # HTML parsing and chunking
│   ├── download_data.py       # Policy page downloader
│   └── config.py              # Configuration and prompt loading
├── data/
│   ├── faiss.index            # FAISS vector index
│   ├── meta.pkl               # Document metadata
│   └── bm25_corpus.pkl        # Tokenized BM25 corpus
├── .github/workflows/
│   └── eval.yml               # CI evaluation pipeline
├── prompts.yaml               # Versioned prompt configuration
├── golden_eval_set.json       # 50 verified Q&A evaluation pairs
├── evaluate.py                # RAGAS evaluation script
├── requirements.txt           # Python dependencies
└── README.md
```

---

## Setup

**Prerequisites:** Python 3.11+, 8GB RAM

```bash
# Clone repository
git clone https://github.com/Deepak-Lingala/airlines_rag_system.git
cd airlines_rag_system

# Install dependencies
pip install -r requirements.txt

# Install and start Ollama
# Download from https://ollama.ai
ollama pull llama3.2:3b
ollama serve  # Keep running in separate terminal

# Download Delta policy pages
python -m src.download_data

# Build FAISS + BM25 index (one-time setup)
python -m src.build_index

# Run CLI
python -m src.app_cli

# OR run web UI
streamlit run src/app_streamlit.py
```

---

## Evaluation

```bash
# Full RAGAS evaluation (requires Ollama running)
python evaluate.py

# Retrieval-only test (no LLM needed)
python evaluate.py --dry-run
```

The CI pipeline (`.github/workflows/eval.yml`) runs automatically on every PR and **fails the build if faithfulness drops below 0.80**.

---
