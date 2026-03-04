"""
Delta Airlines RAG chatbot — CLI interface.
Hybrid BM25+FAISS retrieval with cross-encoder reranking.
"""
import os
import sys
from pathlib import Path

# Add project root to path so 'src' module can be imported
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import time
import subprocess
import requests
import warnings

from src.retriever import HybridRetriever
from src.rag_pipeline import OllamaClient

warnings.filterwarnings("ignore")


class ProductionRAG:
    def __init__(self):
        self.ollama_model = "llama3.2:3b"
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
                timeout=5,
            )
        except Exception:
            pass

        # Start new Ollama server
        self.ollama_process = subprocess.Popen(
            ["ollama", "serve"],
            cwd=os.getcwd(),
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
                timeout=3,
            )
            models = response.json().get("models", [])

            # Check if model exists
            if any(self.ollama_model in m["name"] for m in models):
                return True

            # Auto-pull model if missing
            subprocess.run(
                ["ollama", "pull", self.ollama_model],
                capture_output=True,
                timeout=60,
            )
            return True
        except Exception:
            return False

    def setup_rag(self):
        """Load hybrid retriever and LLM client."""
        print("Loading RAG components...")
        self.retriever = HybridRetriever()
        self.llm = OllamaClient(model=self.ollama_model)
        print("RAG system ready (Hybrid BM25+FAISS + Cross-Encoder Reranking)")

    def chat(self):
        """Interactive chat loop with citation display."""
        print("\nDelta Airlines RAG Assistant (type 'quit' to exit)")
        print("Retrieval: Hybrid BM25+FAISS | Reranker: Cross-Encoder\n")

        while True:
            query = input("You: ").strip()

            if query.lower() in ["quit", "exit", "bye"]:
                self.cleanup()
                break

            if not query:
                continue

            # Hybrid retrieval + reranking
            results = self.retriever.retrieve(query)

            # Generate with citation enforcement
            response_obj = self.llm.generate(
                prompt=query, 
                retrieved_results=results
            )

            print(f"\nAssistant: {response_obj['answer']}")

            # Show sources
            citations = response_obj.get("citations", [])
            if citations:
                print("\n  Sources:")
                for i, cite in enumerate(citations, 1):
                    print(f"    [{i}] {cite['source']} (Score: {cite['relevance_score']:.2f})")
                    if cite.get('url'):
                        print(f"        Link: {cite['url']}")
            elif not response_obj.get("citation_coverage"):
                print("\n  Warning: No matching policy found.")
            print()

    def cleanup(self):
        """Terminate Ollama process on exit."""
        if hasattr(self, "ollama_process"):
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
        print("Ensure FAISS index exists: python -m src.build_index")
