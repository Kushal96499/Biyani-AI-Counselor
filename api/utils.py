# api/utils.py
import re

def chunk_text(text, chunk_size=1000):
    text = re.sub(r'\s+', ' ', text)
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
