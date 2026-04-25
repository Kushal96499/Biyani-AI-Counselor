import re

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200):
    """
    Splits text into chunks with overlap for better RAG context.
    """
    if not text:
        return []
    
    # Simple character-based chunking
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += (chunk_size - overlap)
        
    return chunks

def clean_text(text: str):
    """
    Removes extra whitespaces and junk from extracted text.
    """
    text = re.sub(r'\s+', ' ', text)
    return text.strip()
