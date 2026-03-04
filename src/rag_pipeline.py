"""
Ollama LLM client with citation enforcement and prompt versioning.
"""
import ollama
import requests
from typing import Optional, List, Dict
from src.config import load_prompts, OLLAMA_MODEL


class OllamaClient:
    def __init__(self, model=OLLAMA_MODEL, timeout=120):
        """
        Initialize Ollama client with health check and versioned prompts.
        
        Args:
            model: Ollama model name (e.g., 'llama3.2:3b')
            timeout: Request timeout in seconds (unused in streaming mode)
        """
        self.model = model
        self.timeout = timeout
        self.base_url = "http://localhost:11434"
        self.prompts = load_prompts()
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
    
    def generate(self, prompt: str, retrieved_results: List[Dict]) -> Dict:
        """
        Generate response and build a structured citation object.
        """
        import json
        import re
        import os
        from pathlib import Path

        # Citation enforcement: if retriever flagged all results as irrelevant,
        # return the no-answer response instead of letting the LLM hallucinate
        has_relevant = any(r.get("relevant", False) for r in retrieved_results)
        
        if not has_relevant:
            response_text = self.prompts.get(
                "no_answer_response",
                "I cannot find this in the policy documents."
            )
            structured_response = {
                "answer": response_text,
                "citations": [],
                "citation_coverage": False,
                "chunks_retrieved": len(retrieved_results),
                "chunks_used": 0
            }
            self._log_response(structured_response)
            return structured_response

        # Build context string
        context_parts = []
        for i, r in enumerate(retrieved_results):
            score = r.get("relevance_score", r.get("score", 0.0))
            context_parts.append(
                f"[CHUNK {i+1}] Source: {r['source']} | Score: {score:.2f}\n{r['text']}"
            )
        context = "\n\n".join(context_parts)

        # Build prompt from versioned template
        system_prompt = self.prompts.get("system_prompt", "")
        qa_template = self.prompts.get("qa_prompt", "CONTEXT:\n{context}\n\nQUESTION: {question}\n\nANSWER:")
        user_prompt = qa_template.format(context=context, question=prompt)
        
        # Pre-generation health check
        try:
            requests.get(f"{self.base_url}/api/tags", timeout=3)
        except requests.RequestException:
            return {"answer": "Error: Ollama server not responding.", "citations": [], "citation_coverage": False, "chunks_retrieved": len(retrieved_results), "chunks_used": 0}
        
        # Generation
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": user_prompt})
            
            response = ollama.chat(
                model=self.model,
                messages=messages,
                stream=False,
                options={"temperature": 0.1, "top_p": 0.9, "num_predict": 512}
            )
            
            answer_text = response.get('message', {}).get('content', '').strip()
            
            # Parse [Source: X] citations
            citations = []
            used_chunks = set()
            
            # Look for patterns like [Source: 1], [Source: CHUNK 1], etc.
            matches = re.findall(r'\[Source:([^\]]+)\]', answer_text, re.IGNORECASE)
            for match in matches:
                # Extract all numbers from the match
                indices = [int(idx) for idx in re.findall(r'\d+', match)]
                for idx in indices:
                    # Chunks are 1-indexed in the prompt
                    chunk_idx = idx - 1
                    if 0 <= chunk_idx < len(retrieved_results) and chunk_idx not in used_chunks:
                        used_chunks.add(chunk_idx)
                        r = retrieved_results[chunk_idx]
                        score = r.get("relevance_score", r.get("score", 0.0))
                        
                        citations.append({
                            "source": r["source"],
                            "url": r.get("url", ""),
                            "relevance_score": float(score),
                            "excerpt": r["text"][:150] + "..."  # store snippet
                        })
            
            structured_response = {
                "answer": answer_text,
                "citations": citations,
                "citation_coverage": len(citations) > 0,
                "chunks_retrieved": len(retrieved_results),
                "chunks_used": len(used_chunks)
            }
            
            self._log_response(structured_response)
            return structured_response
            
        except Exception as e:
            return {
                "answer": f"Generation Error: {str(e)[:100]}",
                "citations": [],
                "citation_coverage": False,
                "chunks_retrieved": len(retrieved_results),
                "chunks_used": 0
            }

    def _log_response(self, response_dict: Dict):
        """Append response to JSONL log."""
        import json
        from pathlib import Path
        log_path = Path(__file__).parent.parent / "responses_log.jsonl"
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(response_dict) + "\n")
