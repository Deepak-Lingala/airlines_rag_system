"""
Streamlit web interface for Delta Airlines RAG chatbot.
"""
import streamlit as st
import faiss
import pickle
from sentence_transformers import SentenceTransformer
import ollama
import subprocess
import time
import requests


st.set_page_config(page_title="Delta RAG Assistant", layout="wide")


@st.cache_resource
def load_rag():
    """Load and cache embedding model, FAISS index, and metadata."""
    model = SentenceTransformer('all-MiniLM-L6-v2')
    index = faiss.read_index("data/delta_index.faiss")
    
    with open("data/delta_metadata.pkl", "rb") as f:
        metadata = pickle.load(f)
    
    return model, index, metadata


@st.cache_resource
def start_ollama():
    """Start Ollama server if not already running."""
    subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(5)
    
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=3)
        return response.status_code == 200
    except requests.RequestException:
        return False


def search(query, model, index, metadata, k=3):
    """Retrieve top-k relevant document chunks."""
    embedding = model.encode([query])
    distances, indices = index.search(embedding, k)
    
    return [
        metadata[i]
        for i in indices[0]
        if i < len(metadata)
    ]


def generate(query, context):
    """Generate response using Ollama with retrieved context."""
    prompt = f"""Delta Airlines Policy:
{context}

Q: {query}
A:"""
    
    try:
        stream = ollama.chat(
            model="llama3.2:1b",
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            options={"num_predict": 300, "temperature": 0.1}
        )
        
        response = ""
        for chunk in stream:
            response += chunk['message']['content']
            if len(response) > 800:
                break
        
        return response.strip()
    
    except Exception as e:
        return f"Error: Ollama not responding. Ensure 'ollama serve' is running."


# Header
st.title("Delta Airlines RAG Assistant")
st.markdown("Ask about Delta baggage policies, carry-on sizes, fees, and more.")


# Sidebar status
with st.sidebar:
    st.header("System Status")
    
    ollama_status = start_ollama()
    if ollama_status:
        st.success("Ollama: Ready")
    else:
        st.error("Ollama: Offline")
    
    st.info("Model: llama3.2:1b")


# Load RAG components
model, index, metadata = load_rag()
st.sidebar.info(f"Documents: {len(metadata)} chunks")


# Chat interface
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if user_input := st.chat_input("What is carry-on bag size?"):
    # Display user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Generate and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Searching policies..."):
            docs = search(user_input, model, index, metadata)
            context = "\n".join([d['page_content'] for d in docs])
            response = generate(user_input, context)
            st.markdown(response)
    
    st.session_state.messages.append({"role": "assistant", "content": response})


# Example questions
with st.expander("Example Questions"):
    st.markdown("""
    - What is carry-on bag size?
    - What are checked bag fees?
    - Can I bring a personal item?
    - What is the international baggage policy?
    """)
