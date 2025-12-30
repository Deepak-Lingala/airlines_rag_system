"""
Delta Airlines RAG chatbot using FAISS, sentence-transformers, and Ollama.
"""
import os
import sys
import time
import subprocess
import requests
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import warnings

warnings.filterwarnings("ignore")


class ProductionRAG:
    def __init__(self):
        self.model_name = "all-MiniLM-L6-v2"
        self.ollama_model = "llama3.2:1b"
        self.port = 11434
        self.start_ollama()
        self.setup_rag()
    
    def start_ollama(self):
        """Start Ollama server process and wait for readiness."""
        print("Starting Ollama server...")
        
        # Kill any existing Ollama processes
        try:
            subprocess.run(
                ["taskkill", "/F", "/IM", "ollama.exe"],
                capture_output=True,
                timeout=5
            )
        except Exception:
            pass
        
        # Start new Ollama server
        self.ollama_process = subprocess.Popen(
            ["ollama", "serve"],
            cwd=os.getcwd()
        )
        
        # Wait for server to be ready
        for _ in range(15):
            time.sleep(2)
            if self.ollama_health_check():
                print(f"Ollama ready: {self.ollama_model}")
                return
        
        raise RuntimeError("Ollama server failed to start within timeout")
    
    def ollama_health_check(self):
        """Check if Ollama is running and model is available."""
        try:
            response = requests.get(
                f"http://localhost:{self.port}/api/tags",
                timeout=3
            )
            models = response.json().get('models', [])
            
            # Check if model exists
            if any(self.ollama_model in m['name'] for m in models):
                return True
            
            # Auto-pull model if missing
            subprocess.run(
                ["ollama", "pull", self.ollama_model],
                capture_output=True,
                timeout=60
            )
            return True
        except Exception:
            return False
    
    def setup_rag(self):
        """Load embedding model, FAISS index, and metadata."""
        print("Loading RAG components...")
        
        self.embedding_model = SentenceTransformer(self.model_name)
        self.index = faiss.read_index("data/delta_index.faiss")
        
        with open("data/delta_metadata.pkl", "rb") as f:
            self.metadata = pickle.load(f)
        
        print(f"RAG ready with {len(self.metadata)} document chunks")
    
    def search(self, query, k=3):
        """Retrieve top-k relevant document chunks for query."""
        query_embedding = self.embedding_model.encode([query])
        distances, indices = self.index.search(query_embedding, k)
        
        return [
            self.metadata[i]
            for i in indices[0]
            if i < len(self.metadata)
        ]
    
    def generate(self, query, context):
        """Generate answer using Ollama with retrieved context."""
        prompt = f"""Delta Airlines Policy:
{context}

Q: {query}
A:"""
        
        try:
            import ollama
            
            stream = ollama.chat(
                model=self.ollama_model,
                messages=[{"role": "user", "content": prompt}],
                stream=True,
                options={"num_predict": 300, "temperature": 0.1}
            )
            
            response = ""
            for chunk in stream:
                response += chunk['message']['content']
                if len(response) > 800:  # Prevent runaway generation
                    break
            
            return response.strip()
        
        except Exception as e:
            return f"Error: {str(e)[:100]}"
    
    def chat(self):
        """Interactive chat loop."""
        print("\nDelta Airlines RAG Assistant (type 'quit' to exit)")
        
        while True:
            query = input("\nYou: ").strip()
            
            if query.lower() in ['quit', 'exit', 'bye']:
                self.cleanup()
                break
            
            if not query:
                continue
            
            docs = self.search(query)
            context = "\n".join([d['page_content'] for d in docs])
            response = self.generate(query, context)
            
            print(f"Assistant: {response}")
    
    def cleanup(self):
        """Terminate Ollama process on exit."""
        if hasattr(self, 'ollama_process'):
            self.ollama_process.terminate()
            print("Ollama server stopped")


if __name__ == "__main__":
    try:
        rag = ProductionRAG()
        rag.chat()
    except KeyboardInterrupt:
        print("\nGoodbye!")
    except Exception as e:
        print(f"Error: {e}")
        print("Ensure FAISS index exists: python src/create_rag_index.py")
