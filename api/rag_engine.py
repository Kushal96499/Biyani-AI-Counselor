import json
import os
import re
import requests
from api.config import settings
from api.logger import logger
from api.utils import chunk_text


# ── Language Detection ──────────────────────────────────────────────────────
HINGLISH_MARKERS = {
    "kya", "hai", "ka", "ki", "ke", "ko", "kaise", "karo", "bata", "batao",
    "mujhe", "mera", "meri", "mere", "aur", "ya", "nahi", "nhi", "hoga",
    "hogi", "wala", "wali", "bhai", "yaar", "chahiye", "kaisa", "kaisi",
    "liye", "krna", "karna", "hua", "hue", "mil", "de", "dede", "toh",
    "kab", "kahan", "kitna", "kitne", "sab", "acha", "theek", "zyada",
    "thoda", "bahut", "bohot", "abhi", "jaldi", "lagta", "lagti",
}

def _detect_hinglish(text: str) -> bool:
    words = set(re.findall(r'\w+', text.lower()))
    return len(words & HINGLISH_MARKERS) >= 1


# ── LiteRAGEngine ───────────────────────────────────────────────────────────
class LiteRAGEngine:
    STOPWORDS = {
        "hai", "ka", "ki", "ke", "ko", "the", "what", "is", "for", "tell",
        "about", "mein", "sem", "semester", "and", "or", "in", "of", "a",
        "an", "to", "that", "this", "it", "are", "was", "be", "will", "can",
    }

    def __init__(self, storage_path=None):
        if storage_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.storage_path = os.path.join(base_dir, "data", "knowledge_base.json")
        else:
            self.storage_path = storage_path

        self.chunks = []
        self.load_knowledge()

        self.nvidia_key      = os.getenv("NVIDIA_API_KEY", "").strip()
        self.gemini_key      = os.getenv("GEMINI_API_KEY", "").strip()
        self.groq_key        = os.getenv("GROQ_API_KEY", "").strip()
        self.openrouter_key  = os.getenv("OPENROUTER_API_KEY", "").strip()

        self.api_url_nvidia     = "https://integrate.api.nvidia.com/v1/chat/completions"
        self.api_url_groq       = "https://api.groq.com/openai/v1/chat/completions"
        self.api_url_openrouter = "https://openrouter.ai/api/v1/chat/completions"
        self.embedding_url      = "https://generativelanguage.googleapis.com/v1beta/models/embedding-001:embedContent"

    # ── Knowledge helpers ──────────────────────────────────────────────────
    def _get_keywords(self, text):
        words = re.findall(r'\w+', text.lower())
        return set(w for w in words if len(w) > 2 and w not in self.STOPWORDS)

    def load_knowledge(self):
        try:
            if os.path.exists(self.storage_path):
                with open(self.storage_path, 'r', encoding='utf-8') as f:
                    self.chunks = json.load(f)
                for c in self.chunks:
                    c["keywords"] = set(c.get("keywords", []))
                logger.info(f"Loaded {len(self.chunks)} chunks from {self.storage_path}")
            else:
                logger.warning(f"Knowledge base not found at {self.storage_path}")
        except Exception as e:
            logger.error(f"Error loading knowledge: {e}")

    # ── Embedding + Retrieval ──────────────────────────────────────────────
    def _get_embedding(self, text):
        if not self.gemini_key:
            return [0] * 768
        url     = f"{self.embedding_url}?key={self.gemini_key}"
        payload = {"model": "models/embedding-001", "content": {"parts": [{"text": text}]}}
        try:
            res = requests.post(url, json=payload, timeout=10)
            return res.json()['embedding']['values']
        except Exception:
            return [0] * 768

    def _cosine_similarity(self, v1, v2):
        if not v1 or not v2:
            return 0
        dot = sum(a * b for a, b in zip(v1, v2))
        m1  = sum(a * a for a in v1) ** 0.5
        m2  = sum(b * b for b in v2) ** 0.5
        return dot / (m1 * m2) if m1 and m2 else 0

    def _retrieve(self, query, n=4):
        query_keywords  = self._get_keywords(query)
        query_embedding = self._get_embedding(query) if len(query.split()) > 2 else None

        scored = []
        for chunk in self.chunks:
            kw_score  = len(query_keywords.intersection(chunk.get("keywords", []))) * 2.0
            sem_score = 0
            if query_embedding and chunk.get("embedding"):
                sem_score = self._cosine_similarity(query_embedding, chunk["embedding"]) * 15
            total = kw_score + sem_score
            if total > 0.5:
                scored.append((total, chunk))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [item[1] for item in scored[:n]]

    # ── Main Chat ──────────────────────────────────────────────────────────
    def chat(self, user_message: str, history: list = None):
        if history is None:
            history = []

        query_lower = user_message.lower().strip()
        is_hinglish = _detect_hinglish(user_message)  # detect FIRST

        # PDF trigger
        pdf_url = None
        if any(x in query_lower for x in ["brochure", "available courses", "course list"]):
            pdf_url = "https://www.biyanicolleges.org/wp-content/uploads/2025/03/Brochure_2024_2025.pdf"
        elif any(x in query_lower for x in ["prospectus", "prospectas", "admission process", "scholarship", "admission"]):
            pdf_url = "https://www.biyanicolleges.org/wp-content/uploads/2025/03/Prospectus_2024_2025.pdf"
        elif any(x in query_lower for x in ["report", "annual", "placement stats", "placement"]):
            pdf_url = "https://www.biyanicolleges.org/wp-content/uploads/2025/03/Annual%20Report_2024_2025.pdf"

        # Retrieval
        relevant     = self._retrieve(user_message)
        context_text = "\n---\n".join(c["text"] for c in relevant) if relevant else "No specific context found."

        # Prompt
        if is_hinglish:
            lang_rule = (
                "User Hinglish mein baat kar raha hai. Hinglish mein jawab do "
                "(Roman script, natural aur conversational). Koi cliche words mat use karo."
            )
        else:
            lang_rule = (
                "User is speaking English. Reply in polished, professional English. "
                "No AI clichés."
            )

        system_msg = (
            f"You are a Senior Admission Counselor at Biyani Group of Colleges. "
            f"{lang_rule} "
            f"Keep answer under 90 words. End with ONE short follow-up question. "
            f"Never say 'data unavailable' — pivot confidently. "
            f"If PDF is shared, say 'I am showing you the official document right now.'"
        )

        prompt = (
            f"KNOWLEDGE:\n{context_text}\n\n"
            f"QUERY: {user_message}\n\n"
            f"COUNSELOR RESPONSE:"
        )

        messages = [{"role": "system", "content": system_msg}]
        messages.extend(history[-4:])
        messages.append({"role": "user", "content": prompt})

        base_payload = {
            "messages":          messages,
            "temperature":       0.35,
            "max_tokens":        400,
            "frequency_penalty": 0.4,
        }

        # ── PRIORITY 1: NVIDIA NIM ─────────────────────────────────────────
        if self.nvidia_key:
            nvidia_models = [
                "nvidia/nemotron-mini-4b-instruct",
                "deepseek-ai/deepseek-v3-1.5b",
                "google/gemma-3n-e4b-it",
            ]
            headers = {"Authorization": f"Bearer {self.nvidia_key}", "Content-Type": "application/json"}
            for model in nvidia_models:
                try:
                    res = requests.post(self.api_url_nvidia, json={**base_payload, "model": model}, headers=headers, timeout=12)
                    if res.status_code == 200:
                        logger.info(f"NVIDIA ({model}) success")
                        return {"answer": res.json()["choices"][0]["message"]["content"].strip(), "pdf_url": pdf_url}
                    logger.warning(f"NVIDIA ({model}) failed: {res.status_code}")
                except Exception as e:
                    logger.error(f"NVIDIA ({model}) error: {e}")

        # ── PRIORITY 2: GROQ ───────────────────────────────────────────────
        if self.groq_key:
            for model in ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]:
                try:
                    res = requests.post(
                        self.api_url_groq,
                        json={**base_payload, "model": model},
                        headers={"Authorization": f"Bearer {self.groq_key}"},
                        timeout=8,
                    )
                    if res.status_code == 200:
                        logger.info(f"Groq ({model}) success")
                        return {"answer": res.json()["choices"][0]["message"]["content"].strip(), "pdf_url": pdf_url}
                    logger.warning(f"Groq ({model}) failed: {res.status_code}")
                except Exception as e:
                    logger.error(f"Groq ({model}) error: {e}")

        # ── PRIORITY 3: GEMINI + GEMMA ─────────────────────────────────────
        if self.gemini_key:
            gemini_models = ["gemini-2.0-flash", "gemini-1.5-flash", "gemma-3-27b-it", "gemma-3-12b-it"]
            for model in gemini_models:
                try:
                    gemini_url = (
                        f"https://generativelanguage.googleapis.com/v1beta/models/"
                        f"{model}:generateContent?key={self.gemini_key}"
                    )
                    gemini_contents = []
                    for m in messages:
                        role = "user" if m["role"] in ["user", "system"] else "model"
                        gemini_contents.append({"role": role, "parts": [{"text": m["content"]}]})
                    res = requests.post(
                        gemini_url,
                        json={"contents": gemini_contents, "generationConfig": {"maxOutputTokens": 400, "temperature": 0.35}},
                        timeout=14,
                    )
                    if res.status_code == 200:
                        ans = res.json()["candidates"][0]["content"]["parts"][0]["text"]
                        logger.info(f"Gemini ({model}) success")
                        return {"answer": ans.strip(), "pdf_url": pdf_url}
                    logger.warning(f"Gemini ({model}) failed: {res.status_code}")
                except Exception as e:
                    logger.error(f"Gemini ({model}) error: {e}")

        # ── PRIORITY 4: OPENROUTER ─────────────────────────────────────────
        if self.openrouter_key:
            or_models = [
                "google/gemini-2.0-flash-lite-preview-02-05:free",
                "meta-llama/llama-3.3-70b-instruct:free",
                "deepseek/deepseek-v3-base:free",
            ]
            or_headers = {
                "Authorization": f"Bearer {self.openrouter_key}",
                "HTTP-Referer":  "https://biyani-ai-counselor.vercel.app",
                "Content-Type":  "application/json",
            }
            for model in or_models:
                try:
                    res = requests.post(self.api_url_openrouter, json={**base_payload, "model": model}, headers=or_headers, timeout=15)
                    if res.status_code == 200:
                        logger.info(f"OpenRouter ({model}) success")
                        return {"answer": res.json()["choices"][0]["message"]["content"].strip(), "pdf_url": pdf_url}
                    logger.warning(f"OpenRouter ({model}) failed: {res.status_code}")
                except Exception as e:
                    logger.error(f"OpenRouter ({model}) error: {e}")

        logger.error("All AI providers failed.")
        busy_msg = (
            "Abhi sabhi counselors busy hain. Thodi der baad try karein ya 0141-2338591 par call karein."
            if is_hinglish else
            "All AI counselors are currently busy. Please try again in a moment or call 0141-2338591."
        )
        return {"answer": busy_msg, "pdf_url": None}
