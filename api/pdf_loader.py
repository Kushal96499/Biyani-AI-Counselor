# api/pdf_loader.py
import os
import time
import requests
from pypdf import PdfReader
from io import BytesIO
from api.logger import logger
from api.config import settings

def load_pdfs_from_links(pdf_links: list):
    data = []
    for url in pdf_links:
        if not url.strip(): continue
        try:
            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                pdf_bytes = response.content
                reader = PdfReader(BytesIO(pdf_bytes))
                full_text = ""
                for page in reader.pages:
                    txt = page.extract_text()
                    if txt: full_text += txt + "\n"
                
                if full_text.strip():
                    data.append({"text": full_text, "source": url})
            time.sleep(0.5)
        except Exception as e:
            logger.error(f"Error loading PDF {url}: {e}")
    return data
