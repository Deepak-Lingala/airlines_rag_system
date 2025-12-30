"""
Ollama LLM client with health checking and streaming generation.
"""
import ollama
import requests
from typing import Optional


class OllamaClient:
    def __init__(self, model="llama3.2:1b", timeout=120):
        """
        Initialize Ollama client with health check.
        
        Args:
            model: Ollama model name (e.g., 'llama3.2:1b')
            timeout: Request timeout in seconds (unused in streaming mode)
        """
        self.model = model
        self.timeout = timeout
        self.base_url = "http://localhost:11434"
        self.health_check()
    
    def health_check(self):
        """Verify Ollama server is running and accessible."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            response.raise_for_status()
            models = response.json().get('models', [])
            print(f"Ollama connected: {len(models)} models available")
        except requests.RequestException as e:
            raise ConnectionError(
                f"Ollama server not running. Start with: ollama serve\nError: {e}"
            )
    
    def generate(self, prompt: str, context: str) -> str:
        """
        Generate response using retrieved context.
        
        Args:
            prompt: User's question
            context: Retrieved document chunks concatenated
            
        Returns:
            Generated response text or error message
        """
        full_prompt = f"""CONTEXT: {context}

QUESTION: {prompt}

ANSWER:"""
        
        # Pre-generation health check
        try:
            requests.get(f"{self.base_url}/api/tags", timeout=3)
        except requests.RequestException:
            return "Error: Ollama server not responding. Restart with: ollama serve"
        
        # Streaming generation
        try:
            stream = ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": full_prompt}],
                stream=True,
                options={
                    "temperature": 0.1,
                    "top_p": 0.9,
                    "num_predict": 512
                }
            )
            
            response = ""
            for chunk in stream:
                if 'message' in chunk and 'content' in chunk['message']:
                    response += chunk['message']['content']
                    
                    # Prevent runaway generation
                    if len(response) > 1000:
                        break
            
            return response.strip()
        
        except Exception as e:
            error_msg = str(e).lower()
            
            if "timeout" in error_msg:
                return "Error: Generation timeout. Consider using a smaller model."
            
            return f"Error: {str(e)[:100]}"
