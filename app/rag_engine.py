import json
import os
import re
import requests
import time
from app.config import settings
from app.logger import logger
from app.utils import chunk_text

class LiteRAGEngine:
    def __init__(self, storage_path="data/knowledge_base.json"):
        self.storage_path = storage_path
        self.chunks = [] 
        self.load_knowledge()
        self.groq_key = settings.GROQ_API_KEY.strip()
        self.gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
        self.openrouter_key = os.getenv("OPENROUTER_API_KEY", "").strip()
        self.api_url_groq = "https://api.groq.com/openai/v1/chat/completions"
        self.api_url_openrouter = "https://openrouter.ai/api/v1/chat/completions"
        self.embedding_url = "https://generativelanguage.googleapis.com/v1beta/models/embedding-001:embedContent"

    def _get_keywords(self, text):
        stopwords = {"hai", "kitne", "saal", "ka", "ki", "ke", "ko", "batao", "kya", "the", "what", "is", "for", "tell", "about", "mein", "sem", "semester"}
        words = re.findall(r'\w+', text.lower())
        return set([w for w in words if len(w) > 2 and w not in stopwords])

    def add_documents(self, documents):
        new_chunks = []
        for doc in documents:
            text_chunks = chunk_text(doc["text"])
            for chunk in text_chunks:
                if chunk.strip():
                    embedding = self._get_embedding(chunk) if self.gemini_key else None
                    new_chunks.append({
                        "text": chunk,
                        "source": doc["source"],
                        "keywords": list(self._get_keywords(chunk)),
                        "embedding": embedding
                    })
        self.chunks.extend(new_chunks)
        self.save_knowledge()

    def add_faqs(self, faqs_path):
        if not os.path.exists(faqs_path):
            return
        try:
            with open(faqs_path, "r") as f:
                faqs = json.load(f)
            docs = [{"text": f"Q: {i['question']}\nA: {i['answer']}", "source": "FAQs"} for i in faqs]
            self.add_documents(docs)
        except Exception as e:
            logger.error(f"Error adding FAQs: {e}")

    def save_knowledge(self):
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
        with open(self.storage_path, "w") as f:
            json.dump(self.chunks, f)

    def load_knowledge(self):
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, "r") as f:
                    self.chunks = json.load(f)
                    for c in self.chunks:
                        c["keywords"] = set(c["keywords"])
            except Exception as e:
                logger.error(f"Error loading knowledge: {e}")
                self.chunks = []

    def clear_database(self):
        self.chunks = []
        if os.path.exists(self.storage_path):
            os.remove(self.storage_path)

    def get_indexed_sources(self):
        """Returns a set of unique sources already indexed."""
        return set([c.get("source") for c in self.chunks if c.get("source")])

    def _get_embedding(self, text):
        """Get vector representation of text using Gemini."""
        if not self.gemini_key: return None
        try:
            res = requests.post(f"{self.embedding_url}?key={self.gemini_key}", 
                               json={"model": "models/embedding-001", "content": {"parts": [{"text": text}]}}, timeout=10)
            if res.status_code == 200:
                return res.json()["embedding"]["values"]
        except: pass
        return None

    def _cosine_similarity(self, v1, v2):
        """Pure python cosine similarity to avoid heavy dependencies."""
        if not v1 or not v2: return 0
        dot_product = sum(a * b for a, b in zip(v1, v2))
        magnitude1 = sum(a * a for a in v1) ** 0.5
        magnitude2 = sum(b * b for b in v2) ** 0.5
        if not magnitude1 or not magnitude2: return 0
        return dot_product / (magnitude1 * magnitude2)

    def _retrieve(self, query, n=4):
        query_keywords = self._get_keywords(query)
        query_embedding = self._get_embedding(query) if len(query.split()) > 2 else None
        
        scored = []
        for chunk in self.chunks:
            # 1. Keyword Score
            kw_matches = query_keywords.intersection(chunk.get("keywords", []))
            kw_score = len(kw_matches) * 2.0
            
            # 2. Semantic Score (Vector Similarity)
            semantic_score = 0
            if query_embedding and chunk.get("embedding"):
                sim = self._cosine_similarity(query_embedding, chunk["embedding"])
                semantic_score = sim * 15 # High weight for semantic meaning
            
            total_score = kw_score + semantic_score
            if total_score > 0.5:
                scored.append((total_score, chunk))
        
        scored.sort(key=lambda x: x[0], reverse=True)
        return [item[1] for item in scored[:n]]

    def query(self, user_message: str, history: list = None):
        if history is None:
            history = []
        
        is_followup = len(user_message.split()) < 4 and history
        relevant = self._retrieve(user_message)
        
        if not relevant and is_followup:
            last_user_msg = next((m["content"] for m in reversed(history) if m["role"] == "user"), "")
            if last_user_msg:
                relevant = self._retrieve(last_user_msg)

        greetings = ["hi", "hello", "hey", "kese", "kaise", "namaste", "halo", "good morning", "good evening", "greeting"]
        if not relevant and any(x in user_message.lower() for x in greetings):
            return {"answer": "Welcome to Biyani Group of Colleges! I am your Professional Admission Counselor. How may I assist you today?", "sources": ["System"], "pdf_url": None}

        query_lower = user_message.lower().strip()
        pdf_url = None

        # --- SMART PDF TRIGGER: Only show if explicitly asked or highly relevant ---
        if any(x in query_lower for x in ["brochure", "available courses", "course list"]):
            pdf_url = "https://www.biyanicolleges.org/wp-content/uploads/2025/03/Brochure_2024_2025.pdf"
        elif any(x in query_lower for x in ["prospectus", "prospectas", "admission process", "scholarship", "admission"]):
            pdf_url = "https://www.biyanicolleges.org/wp-content/uploads/2025/03/Prospectus_2024_2025.pdf"
        elif any(x in query_lower for x in ["report", "annual", "placement stats", "placement"]):
            pdf_url = "https://www.biyanicolleges.org/wp-content/uploads/2025/03/Annual%20Report_2024_2025.pdf"
        else:
            # Don't auto-show random PDFs from the database unless score is very high or user asks
            pdf_url = None

        # Fallback for no info found, but preserve pdf_url if we just found one
        if not relevant and not is_followup and not pdf_url:
            return {"answer": "I apologize, but I couldn't find specific information regarding that in our records. Please contact our Admission Office at 0141-2338591 for more details.", "pdf_url": None, "sources": []}
            
        context_text = "\n\n".join([c["text"] for c in relevant]) if relevant else "No context found."

        prompt = f"""
You are a Senior Admission Counselor at Biyani Group of Colleges. 
A student is asking for information. 

STRICT RULES:
1. NEVER say "data is not available" or "I notice information is missing". 
2. NEVER mention boring technical stuff like "Section 5.2" or "AQAR reports".
3. ALWAYS be a confident counselor. Even if exact stats are missing, talk about our dedicated placement cell, the soft-skills training, and the big brands like Wipro, Infosys, and HCL that recruit from us.
4. Keep your answer under 100 words. Focus on being encouraging.
5. ALWAYS say "I am showing you the official document right now" if a PDF is provided.

KNOWLEDGE CONTEXT:
{context_text}

STUDENT QUERY:
{user_message}

COUNSELOR RESPONSE:"""
        
        system_msg = "Senior Human Counselor at Biyani. Natural tone. No AI clichés. Mirror user language. End with a question."
        messages = [{"role": "system", "content": system_msg}]
        messages.extend(history[-4:])
        messages.append({"role": "user", "content": prompt})

        # Shared parameters for OpenAI-compatible providers (Groq/OpenRouter)
        base_payload = {
            "messages": messages, 
            "temperature": 0.4, 
            "max_tokens": 1000,
            "frequency_penalty": 0.5
        }

        # --- PRIORITY 1: GROQ MODELS (Fastest) ---
        if self.groq_key:
            groq_models = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"]
            for model in groq_models:
                try:
                    payload = {**base_payload, "model": model}
                    res = requests.post(self.api_url_groq, json=payload, headers={"Authorization": f"Bearer {self.groq_key}"}, timeout=8)
                    if res.status_code == 200:
                        return {"answer": res.json()["choices"][0]["message"]["content"].strip(), "pdf_url": pdf_url, "sources": [pdf_url] if pdf_url else []}
                    elif res.status_code == 429: continue # Rate limit, try next model
                except: continue

        # --- PRIORITY 2: GOOGLE GEMINI (Reliable) ---
        if self.gemini_key:
            gemini_models = ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"]
            for model in gemini_models:
                try:
                    gemini_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={self.gemini_key}"
                    gemini_contents = []
                    for m in messages:
                        role = "user" if m["role"] in ["user", "system"] else "model"
                        gemini_contents.append({"role": role, "parts": [{"text": m["content"]}]})
                    
                    res = requests.post(gemini_url, json={"contents": gemini_contents}, timeout=12)
                    if res.status_code == 200:
                        answer = res.json()["candidates"][0]["content"]["parts"][0]["text"]
                        return {"answer": answer.strip(), "pdf_url": pdf_url, "sources": [pdf_url] if pdf_url else []}
                except: continue

        # --- PRIORITY 3: OPENROUTER FREE MODELS (Ultimate Fallback) ---
        if self.openrouter_key:
            or_models = ["google/gemini-2.0-flash-lite-preview-02-05:free", "meta-llama/llama-3.3-70b-instruct:free"]
            headers = {"Authorization": f"Bearer {self.openrouter_key}", "HTTP-Referer": "http://localhost:8000", "Content-Type": "application/json"}
            for model in or_models:
                try:
                    payload = {**base_payload, "model": model}
                    res = requests.post(self.api_url_openrouter, json=payload, headers=headers, timeout=15)
                    if res.status_code == 200:
                        return {"answer": res.json()["choices"][0]["message"]["content"].strip(), "pdf_url": pdf_url, "sources": [pdf_url] if pdf_url else []}
                except: continue
        
        return {"answer": "All AI counselors are currently busy. Please try again in a moment.", "pdf_url": None, "sources": []}

rag_engine = LiteRAGEngine()
