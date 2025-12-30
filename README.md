Delta Airlines RAG system

Overview

I built a question-answering system that retrieves information from Delta Airlines baggage policies using retrieval-augmented generation. The system scrapes policy pages, indexes them with FAISS vector search, and generates answers using a locally-hosted LLM. I chose this architecture to demonstrate end-to-end RAG implementation.

Architecture

Delta.com HTML pages
    ↓ (BeautifulSoup scraping)
Cleaned text chunks
    ↓ (sentence-transformers encoding)
384-dim embeddings
    ↓ (FAISS IndexFlatL2)
Vector index
    ↓ (User query → k-NN search)
Top-5 relevant chunks
    ↓ (Ollama llama3.2 + context)
Generated answer

Technologies
- Web scraping: requests with browser headers to fetch policy HTML
​

- Text extraction: BeautifulSoup4 for HTML parsing and sentence-boundary chunking
​

- Embeddings: sentence-transformers (all-MiniLM-L6-v2, 384 dimensions)
​

- Retrieval: FAISS IndexFlatL2 for exact nearest-neighbor search
​

- Generation: Ollama with llama3.2:3b model for local inference
​

- Interfaces: Streamlit web UI with session state + terminal CLI

Setup
Prerequisites: Python 3.11+, 8GB RAM
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
python src/download_policies.py

# Build FAISS index (one-time setup)
python src/create_rag_index.py

# Run CLI
python src/app_cli.py

# OR run web UI
streamlit run src/app_streamlit.py
