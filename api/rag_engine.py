import json
import os
import re
import requests
import time
from api.config import settings
from api.logger import logger
from api.utils import chunk_text

class LiteRAGEngine:
    def __init__(self, storage_path=None):
        if storage_path is None:
            # Default path for Vercel structure
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.storage_path = os.path.join(base_dir, "data", "knowledge_base.json")
        else:
            self.storage_path = storage_path
            
        self.chunks = [] 
        self.load_knowledge()
        self.gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
        self.groq_key = os.getenv("GROQ_API_KEY", "").strip()
        self.openrouter_key = os.getenv("OPENROUTER_API_KEY", "").strip()

    def load_knowledge(self):
        try:
            if os.path.exists(self.storage_path):
                with open(self.storage_path, 'r', encoding='utf-8') as f:
                    self.chunks = json.load(f)
                logger.info(f"Loaded {len(self.chunks)} knowledge chunks from {self.storage_path}")
            else:
                logger.warning(f"Knowledge base not found at {self.storage_path}")
        except Exception as e:
            logger.error(f"Error loading knowledge: {e}")

    def _get_embedding(self, text):
        # Use Gemini embedding API directly via requests
        url = f"https://generativelanguage.googleapis.com/v1beta/models/embedding-001:embedContent?key={self.gemini_key}"
        payload = {"model": "models/embedding-001", "content": {"parts": [{"text": text}]}}
        try:
            res = requests.post(url, json=payload, timeout=10)
            return res.json()['embedding']['values']
        except:
            return [0] * 768

    def chat(self, user_message):
        query_lower = user_message.lower().strip()
        pdf_url = None

        # --- SMART PDF TRIGGER ---
        if any(x in query_lower for x in ["brochure", "available courses", "course list"]):
            pdf_url = "https://www.biyanicolleges.org/wp-content/uploads/2025/03/Brochure_2024_2025.pdf"
        elif any(x in query_lower for x in ["prospectus", "prospectas", "admission process", "scholarship", "admission"]):
            pdf_url = "https://www.biyanicolleges.org/wp-content/uploads/2025/03/Prospectus_2024_2025.pdf"
        elif any(x in query_lower for x in ["report", "annual", "placement stats", "placement"]):
            pdf_url = "https://www.biyanicolleges.org/wp-content/uploads/2025/03/Annual%20Report_2024_2025.pdf"

        # Simple keyword-based semantic retrieval
        keywords = set(re.findall(r'\w+', query_lower))
        relevant = []
        for chunk in self.chunks:
            chunk_text_low = chunk['text'].lower()
            score = sum(1 for kw in keywords if kw in chunk_text_low)
            if score > 0:
                relevant.append((score, chunk))
        
        relevant = sorted(relevant, key=lambda x: x[0], reverse=True)[:5]
        context_text = "\n---\n".join([r[1]['text'] for r in relevant])

        # Generate Response using Gemini (with fallback)
        prompt = f"""
        You are a helpful Admission Counselor for Biyani Group of Colleges. 
        A student is asking for information. 

        STRICT RULES:
        1. NEVER say "data is not available" or "I notice information is missing". 
        2. NEVER mention boring technical stuff like "Section 5.2" or "AQAR reports".
        3. ALWAYS be a confident counselor. Even if exact stats are missing, talk about our dedicated placement cell and training.
        4. Keep your answer under 100 words. Focus on being encouraging.
        5. ALWAYS say "I am showing you the official document right now" if a PDF is provided.

        KNOWLEDGE CONTEXT:
        {context_text}

        Student: {user_message}
        Counselor:"""

        # Call Gemini API
        url = f"https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash:generateContent?key={self.gemini_key}"
        payload = {"contents": [{"parts": [{"text": prompt}]}]}
        try:
            res = requests.post(url, json=payload, timeout=15)
            ans = res.json()['candidates'][0]['content']['parts'][0]['text']
        except Exception as e:
            logger.error(f"AI Generation error: {e}")
            ans = "I'd be happy to help you with that. We have excellent programs and placement support. Please check the document I've shared for details."

        return {"answer": ans, "pdf_url": pdf_url}
