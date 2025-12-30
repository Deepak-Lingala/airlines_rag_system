"""
Parse Delta Airlines policy HTML files into text chunks for indexing.
"""
import re
from pathlib import Path
from bs4 import BeautifulSoup
from src.config import DATA_DIR


def load_delta_policies():
    """Extract and chunk text from downloaded HTML policy files."""
    all_chunks = []
    html_files = list(DATA_DIR.glob("delta_*.html"))
    
    if not html_files:
        print("Warning: No HTML files found. Using demo data.")
        return ["[DEMO] Sample Delta policy text for testing."]
    
    print(f"Found {len(html_files)} HTML files")
    
    for path in html_files:
        print(f"Processing {path.name}...")
        
        # Load and parse HTML
        with open(path, "r", encoding="utf-8", errors='ignore') as f:
            html = f.read()
        
        soup = BeautifulSoup(html, "html.parser")
        
        # Remove non-content elements
        for tag in soup(["script", "style", "nav", "header", "footer"]):
            tag.decompose()
        
        text = soup.get_text(separator="\n").strip()
        chunks = _chunk_text(text)
        
        # Fallback to paragraph splitting if sentence chunking fails
        if len(chunks) < 3:
            paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 80]
            chunks = paragraphs[:12]
        
        # Add source labels and limit chunk count
        source_name = path.stem.replace("delta_", "").replace("_", " ").title()
        labeled_chunks = [
            f"[Source: {source_name}]\n\n{chunk[:1200]}"
            for chunk in chunks
            if len(chunk) > 100
        ]
        
        all_chunks.extend(labeled_chunks[:15])  # Max 15 chunks per file
        print(f"  Extracted {len(labeled_chunks)} chunks")
    
    print(f"Total chunks: {len(all_chunks)}")
    return all_chunks if all_chunks else ["[DEMO] Sample Delta policy text for testing."]


def _chunk_text(text, max_chunk_size=600, min_sentence_length=30, min_chunk_size=100):
    """Split text into semantic chunks using sentence boundaries."""
    sentences = re.split(r'[.!?;]+', text)
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        sentence = sentence.strip()
        
        if len(sentence) < min_sentence_length:
            continue
        
        # Start new chunk if adding sentence exceeds max size
        if len(current_chunk) + len(sentence) > max_chunk_size:
            if len(current_chunk) > min_chunk_size:
                chunks.append(current_chunk.strip())
            current_chunk = sentence
        else:
            current_chunk += ". " + sentence if current_chunk else sentence
    
    # Add final chunk if sufficient size
    if len(current_chunk) > min_chunk_size:
        chunks.append(current_chunk.strip())
    
    return chunks


if __name__ == "__main__":
    chunks = load_delta_policies()
    print(f"\nSample chunk:\n{chunks[0][:300]}...")
