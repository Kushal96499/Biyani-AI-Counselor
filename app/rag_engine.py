import json
import os
import re
import requests
from app.config import settings
from app.logger import logger
from app.utils import chunk_text


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
    def __init__(self, storage_path=settings.KNOWLEDGE_BASE_PATH):
        self.storage_path = storage_path
        self.chunks = []
        self.load_knowledge()

        self.nvidia_key      = os.getenv("NVIDIA_API_KEY", "").strip()
        self.groq_key        = settings.GROQ_API_KEY.strip()
        self.gemini_key      = os.getenv("GEMINI_API_KEY", "").strip()
        self.openrouter_key  = os.getenv("OPENROUTER_API_KEY", "").strip()

        self.api_url_nvidia     = "https://integrate.api.nvidia.com/v1/chat/completions"
        self.api_url_groq       = "https://api.groq.com/openai/v1/chat/completions"
        self.api_url_openrouter = "https://openrouter.ai/api/v1/chat/completions"
        self.embedding_url      = "https://generativelanguage.googleapis.com/v1beta/models/embedding-001:embedContent"

    # ── Knowledge helpers ──────────────────────────────────────────────────
    STOPWORDS = {
        "hai", "ka", "ki", "ke", "ko", "the", "what", "is", "for", "tell",
        "about", "mein", "sem", "semester", "and", "or", "in", "of", "a",
        "an", "to", "that", "this", "it", "are", "was", "be", "will", "can",
    }

    def _get_keywords(self, text):
        words = re.findall(r'\w+', text.lower())
        return set(w for w in words if len(w) > 2 and w not in self.STOPWORDS)

    def add_documents(self, documents):
        new_chunks = []
        for doc in documents:
            for chunk in chunk_text(doc["text"]):
                if chunk.strip():
                    embedding = self._get_embedding(chunk) if self.gemini_key else None
                    new_chunks.append({
                        "text": chunk,
                        "source": doc["source"],
                        "keywords": list(self._get_keywords(chunk)),
                        "embedding": embedding,
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
                    c["keywords"] = set(c.get("keywords", []))
            except Exception as e:
                logger.error(f"Error loading knowledge: {e}")
                self.chunks = []

    def clear_database(self):
        self.chunks = []
        if os.path.exists(self.storage_path):
            os.remove(self.storage_path)

    def get_indexed_sources(self):
        return set(c.get("source") for c in self.chunks if c.get("source"))

    # ── Embedding + Retrieval ──────────────────────────────────────────────
    def _get_embedding(self, text):
        if not self.gemini_key:
            return None
        try:
            res = requests.post(
                f"{self.embedding_url}?key={self.gemini_key}",
                json={"model": "models/embedding-001", "content": {"parts": [{"text": text}]}},
                timeout=10,
            )
            if res.status_code == 200:
                return res.json()["embedding"]["values"]
        except Exception:
            pass
        return None

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

    # ── Main Query ─────────────────────────────────────────────────────────
    def query(self, user_message: str, history: list = None):
        if history is None:
            history = []

        # Detect language FIRST so all messages respect user's language
        is_hinglish = _detect_hinglish(user_message)

        is_followup = len(user_message.split()) < 4 and history
        relevant    = self._retrieve(user_message)

        if not relevant and is_followup:
            last_user_msg = next(
                (m["content"] for m in reversed(history) if m["role"] == "user"), ""
            )
            if last_user_msg:
                relevant = self._retrieve(last_user_msg)

        greetings = {"hi", "hello", "hey", "kese", "kaise", "namaste", "halo"}
        if not relevant and any(x in user_message.lower() for x in greetings):
            greeting_msg = (
                "Namaste! Main aapka Biyani Group of Colleges ka Admission Counselor hoon. Kaise madad kar sakta hoon? 😊"
                if is_hinglish else
                "Welcome to Biyani Group of Colleges! I'm your Admission Counselor. How can I assist you today? 😊"
            )
            return {"answer": greeting_msg, "sources": ["System"], "pdf_url": None}

        query_lower = user_message.lower().strip()
        pdf_url = None
        if any(x in query_lower for x in ["brochure", "available courses", "course list"]):
            pdf_url = "https://www.biyanicolleges.org/wp-content/uploads/2025/03/Brochure_2024_2025.pdf"
        elif any(x in query_lower for x in ["prospectus", "prospectas", "admission process", "scholarship", "admission"]):
            pdf_url = "https://www.biyanicolleges.org/wp-content/uploads/2025/03/Prospectus_2024_2025.pdf"
        elif any(x in query_lower for x in ["report", "annual", "placement stats", "placement"]):
            pdf_url = "https://www.biyanicolleges.org/wp-content/uploads/2025/03/Annual%20Report_2024_2025.pdf"

        if not relevant and not is_followup and not pdf_url:
            no_info_msg = (
                "Iske baare mein specific details abhi available nahi hain. "
                "Aap hamare Admission Office se 0141-2338591 par contact kar sakte hain!"
                if is_hinglish else
                "I couldn't find specific information on that in our records. "
                "Please contact our Admission Office at 0141-2338591 for more details."
            )
            return {"answer": no_info_msg, "pdf_url": None, "sources": []}

        context_text = "\n\n".join(c["text"] for c in relevant) if relevant else "No context."

        # ── Tight, token-efficient system prompt ───────────────────────────
        if is_hinglish:
            lang_rule = (
                "User Hinglish mein baat kar raha hai. Hinglish mein jawab do "
                "(Roman script, natural aur conversational). Koi cliche words mat use karo."
            )
        else:
            lang_rule = (
                "User is speaking English. Reply in polished, professional English. "
                "No AI clichés (no 'delve', 'seamless', 'embark')."
            )

        system_msg = (
            f"You are a Senior Admission Counselor at Biyani Group of Colleges. "
            f"{lang_rule} "
            f"Keep answer under 90 words. End with ONE short follow-up question. "
            f"Never say 'data unavailable' — pivot confidently to what you know. "
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
            "messages":         messages,
            "temperature":      0.35,
            "max_tokens":       400,       # tight = fast + cheap
            "frequency_penalty": 0.4,
        }

        # ── PRIORITY 1: NVIDIA NIM ─────────────────────────────────────────
        # Best models for chat+RAG from NVIDIA free tier
        if self.nvidia_key:
            nvidia_models = [
                "nvidia/nemotron-mini-4b-instruct",     # Fine-tuned for RAG & function calling
                "deepseek-ai/deepseek-v3-1.5b",         # Lightweight reasoning
                "google/gemma-3n-e4b-it",               # Google Gemma via NVIDIA NIM
            ]
            headers = {
                "Authorization": f"Bearer {self.nvidia_key}",
                "Content-Type":  "application/json",
            }
            for model in nvidia_models:
                try:
                    payload = {**base_payload, "model": model}
                    res = requests.post(self.api_url_nvidia, json=payload, headers=headers, timeout=12)
                    if res.status_code == 200:
                        answer = res.json()["choices"][0]["message"]["content"].strip()
                        logger.info(f"NVIDIA ({model}) success")
                        return {"answer": answer, "pdf_url": pdf_url, "sources": [pdf_url] if pdf_url else []}
                    else:
                        logger.warning(f"NVIDIA ({model}) failed: {res.status_code} - {res.text[:200]}")
                except Exception as e:
                    logger.error(f"NVIDIA ({model}) error: {e}")

        # ── PRIORITY 2: GROQ ───────────────────────────────────────────────
        if self.groq_key:
            groq_models = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]
            for model in groq_models:
                try:
                    payload = {**base_payload, "model": model}
                    res = requests.post(
                        self.api_url_groq, json=payload,
                        headers={"Authorization": f"Bearer {self.groq_key}"},
                        timeout=8,
                    )
                    if res.status_code == 200:
                        answer = res.json()["choices"][0]["message"]["content"].strip()
                        logger.info(f"Groq ({model}) success")
                        return {"answer": answer, "pdf_url": pdf_url, "sources": [pdf_url] if pdf_url else []}
                    else:
                        logger.warning(f"Groq ({model}) failed: {res.status_code} - {res.text[:200]}")
                except Exception as e:
                    logger.error(f"Groq ({model}) error: {e}")

        # ── PRIORITY 3: GOOGLE GEMINI + GEMMA ─────────────────────────────
        if self.gemini_key:
            gemini_models = [
                "gemini-2.0-flash",
                "gemini-1.5-flash",
                "gemma-3-27b-it",      # Gemma 3 27B via Gemini API
                "gemma-3-12b-it",      # Gemma 3 12B
            ]
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

                    gen_config = {"maxOutputTokens": 400, "temperature": 0.35}
                    res = requests.post(
                        gemini_url,
                        json={"contents": gemini_contents, "generationConfig": gen_config},
                        timeout=14,
                    )
                    if res.status_code == 200:
                        answer = res.json()["candidates"][0]["content"]["parts"][0]["text"]
                        logger.info(f"Gemini ({model}) success")
                        return {"answer": answer.strip(), "pdf_url": pdf_url, "sources": [pdf_url] if pdf_url else []}
                    else:
                        logger.warning(f"Gemini ({model}) failed: {res.status_code} - {res.text[:200]}")
                except Exception as e:
                    logger.error(f"Gemini ({model}) error: {e}")

        # ── PRIORITY 4: OPENROUTER FREE MODELS ────────────────────────────
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
                    payload = {**base_payload, "model": model}
                    res = requests.post(self.api_url_openrouter, json=payload, headers=or_headers, timeout=15)
                    if res.status_code == 200:
                        answer = res.json()["choices"][0]["message"]["content"].strip()
                        logger.info(f"OpenRouter ({model}) success")
                        return {"answer": answer, "pdf_url": pdf_url, "sources": [pdf_url] if pdf_url else []}
                    else:
                        logger.warning(f"OpenRouter ({model}) failed: {res.status_code} - {res.text[:200]}")
                except Exception as e:
                    logger.error(f"OpenRouter ({model}) error: {e}")

        logger.error("All AI providers failed.")
        busy_msg = (
            "Abhi sabhi counselors busy hain. Thodi der baad try karein ya 0141-2338591 par call karein."
            if is_hinglish else
            "All AI counselors are currently busy. Please try again in a moment or call 0141-2338591."
        )
        return {"answer": busy_msg, "pdf_url": None, "sources": []}


rag_engine = LiteRAGEngine()
