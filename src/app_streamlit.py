"""
Streamlit web interface for Delta Airlines RAG chatbot.
Hybrid BM25+FAISS retrieval with cross-encoder reranking.
"""
import sys
from pathlib import Path

# Add project root to path so 'src' module can be imported
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import streamlit as st
import subprocess
import time
import requests

from src.retriever import HybridRetriever
from src.rag_pipeline import OllamaClient


# -- Page config ----------------------------------------------------------
st.set_page_config(
    page_title="Delta RAG Assistant",
    page_icon="./data/delta_logo.png" if Path("./data/delta_logo.png").exists() else None,
    layout="wide",
)

# -- Custom CSS for professional styling ----------------------------------
st.markdown("""
<style>
    /* Import professional font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Global styling */
    .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #003366 0%, #00264d 100%);
        color: white;
        padding: 1.5rem 2rem;
        border-radius: 8px;
        margin-bottom: 1.5rem;
    }
    .main-header h1 {
        margin: 0;
        font-size: 1.6rem;
        font-weight: 600;
        letter-spacing: -0.02em;
    }
    .main-header p {
        margin: 0.3rem 0 0 0;
        font-size: 0.85rem;
        opacity: 0.85;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: #fafbfc;
        border-right: 1px solid #e1e4e8;
    }
    [data-testid="stSidebar"] .stMarkdown h2 {
        font-size: 0.9rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #586069;
        margin-bottom: 0.75rem;
    }

    /* Metric card styling */
    [data-testid="stMetric"] {
        background: white;
        border: 1px solid #e1e4e8;
        border-radius: 6px;
        padding: 0.75rem;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.75rem;
        font-weight: 500;
        color: #586069;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.2rem;
        font-weight: 600;
        color: #24292e;
    }

    /* Status indicators */
    .status-item {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.4rem 0;
        font-size: 0.82rem;
        color: #24292e;
    }
    .status-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        display: inline-block;
    }
    .status-dot.green { background: #28a745; }
    .status-dot.blue { background: #0366d6; }
    .status-dot.gray { background: #959da5; }

    /* Chat message styling */
    [data-testid="stChatMessage"] {
        border: 1px solid #e1e4e8;
        border-radius: 8px;
        margin-bottom: 0.5rem;
    }

    /* Source expander styling */
    .streamlit-expanderHeader {
        font-size: 0.85rem;
        font-weight: 500;
    }

    /* Footer */
    .footer-text {
        text-align: center;
        font-size: 0.75rem;
        color: #959da5;
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid #e1e4e8;
    }
</style>
""", unsafe_allow_html=True)


# -- Cached resource loaders ----------------------------------------------

@st.cache_resource
def load_retriever():
    """Load and cache the hybrid retriever (BM25 + FAISS + cross-encoder)."""
    return HybridRetriever()


@st.cache_resource
def load_llm():
    """Load and cache the Ollama LLM client."""
    return OllamaClient()


@st.cache_resource
def start_ollama():
    """Start Ollama server if not already running."""
    subprocess.Popen(
        ["ollama", "serve"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(5)

    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=3)
        return response.status_code == 200
    except requests.RequestException:
        return False


# -- Header ---------------------------------------------------------------
st.markdown("""
<div class="main-header">
    <h1>Delta Airlines Policy Assistant</h1>
    <p>Hybrid BM25+FAISS retrieval with cross-encoder reranking and citation enforcement</p>
</div>
""", unsafe_allow_html=True)


# -- Sidebar --------------------------------------------------------------
with st.sidebar:
    st.markdown("## Metrics")

    if "total_queries" not in st.session_state:
        st.session_state.total_queries = 0
        st.session_state.citation_coverage_count = 0
        st.session_state.hallucination_flags = 0
        st.session_state.sum_relevance_scores = 0.0

    tq = st.session_state.total_queries
    cov_rate = (st.session_state.citation_coverage_count / tq * 100) if tq > 0 else 0
    avg_rel = (st.session_state.sum_relevance_scores / tq) if tq > 0 else 0.0

    col1, col2 = st.columns(2)
    col1.metric("Queries", tq)
    col2.metric("Coverage", f"{cov_rate:.0f}%")

    col3, col4 = st.columns(2)
    col3.metric("Avg Relevance", f"{avg_rel:.2f}")
    col4.metric("Hallucinations", st.session_state.hallucination_flags)

    st.divider()
    st.markdown("## System Status")

    ollama_status = start_ollama()

    if ollama_status:
        st.markdown('<div class="status-item"><span class="status-dot green"></span> Ollama: Connected</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-item"><span class="status-dot gray"></span> Ollama: Offline</div>', unsafe_allow_html=True)

    st.markdown('<div class="status-item"><span class="status-dot blue"></span> Retrieval: Hybrid BM25 + FAISS</div>', unsafe_allow_html=True)
    st.markdown('<div class="status-item"><span class="status-dot blue"></span> Reranker: Cross-Encoder ms-marco</div>', unsafe_allow_html=True)
    st.markdown('<div class="status-item"><span class="status-dot blue"></span> LLM: llama3.2</div>', unsafe_allow_html=True)


# -- Load components ------------------------------------------------------
retriever = load_retriever()
llm = load_llm()

st.sidebar.markdown(f'<div class="status-item"><span class="status-dot green"></span> Index: {len(retriever.texts)} chunks</div>', unsafe_allow_html=True)


# -- Chat interface -------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("citations"):
            with st.expander("View Sources"):
                for idx, cite in enumerate(message["citations"]):
                    st.markdown(f"**Source {idx+1}: {cite['source']}** (Relevance: {cite['relevance_score']:.2f})")
                    if cite.get('url'):
                        st.markdown(f"[View on Delta.com]({cite['url']})")
                    st.info(cite["excerpt"])
        elif message.get("show_warning"):
            st.warning("No matching policy found. Please visit delta.com for accurate information.")


# User input
if user_input := st.chat_input("Ask a question about Delta Airlines policies..."):
    # Display user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Retrieve and generate
    with st.chat_message("assistant"):
        with st.spinner("Searching policies..."):
            # Hybrid retrieval + reranking
            results = retriever.retrieve(user_input)

            # Generate with citation enforcement (returns structured dict)
            response_obj = llm.generate(
                prompt=user_input,
                retrieved_results=results,
            )

            answer_text = response_obj["answer"]
            st.markdown(answer_text)

            citations = response_obj.get("citations", [])
            citation_coverage = response_obj.get("citation_coverage", False)

            # Update metrics
            st.session_state.total_queries += 1
            if citation_coverage:
                st.session_state.citation_coverage_count += 1
                avg_score = sum(c["relevance_score"] for c in citations) / len(citations) if citations else 0
                st.session_state.sum_relevance_scores += avg_score
            else:
                st.session_state.hallucination_flags += 1

            # Show sources
            if citations:
                sources_names = set(c["source"] for c in citations)
                st.caption(f"**Sources:** {', '.join(sorted(sources_names))}")

                with st.expander("View Source Excerpts"):
                    for idx, cite in enumerate(citations):
                        st.markdown(f"**Source {idx+1}: {cite['source']}**")
                        if cite.get('url'):
                            st.markdown(f"[View on Delta.com]({cite['url']})")
                        st.caption(f"Relevance: {cite['relevance_score']:.2f}")
                        st.info(cite["excerpt"])

            if not citation_coverage:
                st.warning("No matching policy found. Please visit delta.com for accurate information.")

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer_text,
            "citations": citations,
            "show_warning": not citation_coverage
        }
    )


# -- Example questions & footer -------------------------------------------
with st.expander("Example Questions"):
    st.markdown("""
    - What is the carry-on bag size limit?
    - What are checked bag fees for domestic flights?
    - Can I bring a personal item on Delta?
    - Will I get a refund if I cancel my flight?
    - What is the 24-hour cancellation policy?
    """)

st.markdown('<div class="footer-text">Powered by Hybrid BM25+FAISS retrieval, cross-encoder reranking, and Ollama LLM inference</div>', unsafe_allow_html=True)
