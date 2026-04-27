"""
app/rag_engine.py — High-Precision Production RAG Engine
────────────────────────────────────────────────────────
Persona: Senior Biyani Academic Counselor
Optimizations: Greeting Handling, High-Precision Retrieval, Correct Source Attribution
"""

import os
os.environ['TRANSFORMERS_CACHE'] = '/tmp'
os.environ['SENTENCE_TRANSFORMERS_HOME'] = '/tmp'

import re
import time
import logging
import requests
from pathlib import Path
from dotenv import load_dotenv
from fastembed import TextEmbedding
from qdrant_client import QdrantClient

# ── Environment & Config ──────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

QDRANT_URL     = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION     = os.getenv("QDRANT_COLLECTION", "biyani_ai_clean_v2")

GROQ_KEY    = os.getenv("GROQ_API_KEY", "")
GEMINI_KEY  = os.getenv("GEMINI_API_KEY", "")
NVIDIA_KEY  = os.getenv("NVIDIA_API_KEY", "")
OR_KEY      = os.getenv("OPENROUTER_API_KEY", "")

# ── Search Settings ───────────────────────────────────────────────────────────
EMBED_MODEL      = "BAAI/bge-small-en-v1.5"
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
RETRIEVAL_LIMIT  = 10
SCORE_THRESHOLD  = 0.35  # Lowered for better discovery of related info

logger = logging.getLogger("rag_engine")

# ── Helpers ───────────────────────────────────────────────────────────────────
def _detect_language(text: str) -> str:
    markers = {"kya", "hai", "hain", "ka", "ki", "ke", "ko", "kaise", "karo", "bata", "batao", "mujhe", "nhi", "nahi", "kitni", "kab", "kaun"}
    words = set(re.findall(r'\b\w+\b', text.lower()))
    if len(words & markers) >= 1 or bool(re.search(r'[\u0900-\u097F]', text)):
        return "Hinglish"
    return "English"

def _is_greeting(text: str) -> bool:
    greetings = {"hi", "hello", "hey", "hola", "namaste", "good morning", "good evening", "gm", "gn"}
    clean = re.sub(r'[^\w\s]', '', text.lower()).strip()
    return clean in greetings

PDF_TRIGGERS = {
    "brochure": "https://www.biyanicolleges.org/wp-content/uploads/2025/03/Brochure_2024_2025.pdf",
    "placement": "https://www.biyanicolleges.org/wp-content/uploads/2024/03/Placement_Brochure.pdf",
    "prospectus": "https://www.biyanicolleges.org/wp-content/uploads/2025/03/Prospectus_2024_2025.pdf",
}

def _get_pdf_url(query: str) -> str | None:
    ql = query.lower()
    for k, v in PDF_TRIGGERS.items():
        if k in ql: return v
    return None

# ── LLM Core ──────────────────────────────────────────────────────────────────
def _call_llm(messages: list[dict], query: str = "", num_chunks: int = 0) -> str | None:
    """
    Smart Model Selector following the Final Recommended Model Stack:
    1. GROQ (Primary - Fast) -> llama-3.3-70b (Complex) or llama-3.1-8b (Simple)
    2. GEMINI (Secondary - Smart) -> gemini-2.0-flash (Long/Complex)
    3. NVIDIA (Backup) -> llama-3.1-70b or 8b
    4. OPENROUTER (Final Fallback - Free)
    """
    payload = {"messages": messages, "temperature": 0.3, "max_tokens": 1000}
    
    # ── Logic Helpers ──
    is_complex = any(word in query.lower() for word in ["fees", "admission", "eligibility", "scholarship", "process", "placement", "hostel", "structure"]) or len(query.split()) > 15
    is_long = num_chunks > 2 or len(str(messages)) > 3000

    # 1. GROQ (Primary - Fastest)
    if GROQ_KEY:
        try:
            model = "llama-3.3-70b-versatile" if is_complex else "llama-3.1-8b-instant"
            r = requests.post("https://api.groq.com/openai/v1/chat/completions",
                            json={**payload, "model": model},
                            headers={"Authorization": f"Bearer {GROQ_KEY}"}, timeout=8)
            if r.status_code == 200: return r.json()["choices"][0]["message"]["content"].strip()
        except: pass

    # 2. GEMINI (Secondary - Smart Reasoning)
    if GEMINI_KEY:
        # Verified working models for this key
        for model in ["gemini-2.5-flash", "gemini-2.5-flash-lite"]:
            for ver in ["v1", "v1beta"]:
                try:
                    contents = [{"role": "model" if m["role"] == "assistant" else "user", "parts": [{"text": m["content"]}]} for m in messages if m["role"] != "system"]
                    sys_instr = next((m["content"] for m in messages if m["role"] == "system"), "")
                    r = requests.post(f"https://generativelanguage.googleapis.com/{ver}/models/{model}:generateContent?key={GEMINI_KEY}",
                                    json={"contents": contents, "system_instruction": {"parts": [{"text": sys_instr}]}}, timeout=12)
                    if r.status_code == 200: return r.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
                except: continue

    # 3. NVIDIA (Third - Accuracy)
    if NVIDIA_KEY:
        try:
            # Using the ultra-powerful Mistral Large 3 if complex, else stable Mixtral
            model = "mistralai/mistral-large-3-675b-instruct-2512" if is_complex else "mistralai/mixtral-8x7b-instruct-v0.1"
            r = requests.post("https://integrate.api.nvidia.com/v1/chat/completions",
                            json={**payload, "model": model},
                            headers={"Authorization": f"Bearer {NVIDIA_KEY}"}, timeout=12)
            if r.status_code == 200: return r.json()["choices"][0]["message"]["content"].strip()
            
            # Secondary backup for NVIDIA
            if is_complex:
                r = requests.post("https://integrate.api.nvidia.com/v1/chat/completions",
                                json={**payload, "model": "abacusai/dracarys-llama-3.1-70b-instruct"},
                                headers={"Authorization": f"Bearer {NVIDIA_KEY}"}, timeout=10)
                if r.status_code == 200: return r.json()["choices"][0]["message"]["content"].strip()
        except: pass

    # 4. OPENROUTER (Fourth - Fallback)
    if OR_KEY:
        # Use free high-capacity model as backup
        fallback_models = [
            "nvidia/nemotron-3-super-120b-a12b:free", 
            "nvidia/nemotron-3-nano-30b-a3b:free", 
            "google/gemma-4-31b-it:free"
        ]
        for model in fallback_models:
            try:
                r = requests.post("https://openrouter.ai/api/v1/chat/completions",
                                json={**payload, "model": model},
                                headers={"Authorization": f"Bearer {OR_KEY}"}, timeout=15)
                if r.status_code == 200: return r.json()["choices"][0]["message"]["content"].strip()
            except: continue

    return None

# ── RAG Engine ────────────────────────────────────────────────────────────────
class QdrantRAGEngine:
    def __init__(self):
        try:
            self._qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=25)
            # FastEmbed is lightweight (100MB) vs SentenceTransformers (4GB+)
            self._embedder = TextEmbedding(model_name=EMBED_MODEL, cache_dir="/tmp")
            logger.info("FastEmbed Engine Online.")
        except Exception as e:
            logger.error(f"Init Error: {e}")

    def _retrieve(self, query: str) -> list[dict]:
        if not self._qdrant or not self._embedder: return []
        try:
            # FastEmbed uses a generator for efficiency
            vec = list(self._embedder.embed([BGE_QUERY_PREFIX + query]))[0].tolist()
            
            # Multi-method safe search
            hits = []
            if hasattr(self._qdrant, "search"):
                hits = self._qdrant.search(collection_name=COLLECTION, query_vector=vec, limit=RETRIEVAL_LIMIT, score_threshold=SCORE_THRESHOLD, with_payload=True)
            elif hasattr(self._qdrant, "query_points"):
                hits = self._qdrant.query_points(collection_name=COLLECTION, query=vec, limit=RETRIEVAL_LIMIT, score_threshold=SCORE_THRESHOLD, with_payload=True).points
            
            seen = set()
            unique = []
            for h in hits:
                txt = h.payload.get("text", "")[:50]
                if txt not in seen:
                    unique.append(h.payload)
                    seen.add(txt)
            return unique[:3]
        except Exception as e:
            logger.error(f"Search Error: {e}")
            return []

    # Compatibility stubs
    def clear_database(self): pass
    def get_indexed_sources(self): return set()
    def add_documents(self, docs): pass
    def add_faqs(self, path): pass

    def query(self, user_message: str, history: list[dict] | None = None) -> dict:
        if history is None: history = []
        lang = _detect_language(user_message)
        pdf = _get_pdf_url(user_message)
        
        # 1. Greeting Check
        if _is_greeting(user_message):
            msg = "Namaste! Main Biyani AI Counselor hoon. Main aapki admission, courses aur campus se judi jankari mein madad kar sakta hoon. Aap kya jaanna chahte hain? 😊" if lang == "Hinglish" else "Hello! I am your Biyani AI Counselor. I can help you with admissions, courses, and campus details. How can I assist you today?"
            return {"answer": msg, "sources": [], "pdf_url": None}

        # 2. Smart Retrieval
        search_q = user_message
        if len(user_message.split()) <= 2 and history:
            prev = next((m["content"] for m in reversed(history) if m["role"] == "user"), "")
            if prev: search_q = f"{prev} {user_message}"
        
        chunks = self._retrieve(search_q)
        
        # 3. Professional Fallback
        if not chunks and not pdf:
            if lang == "Hinglish":
                msg = "Maafi chahta hoon, filhaal mere paas iski exact jankari nahi hai. Par aap chinta na karein, aap hamare admission experts ko 0141-2338591 ya 9358890991 par call kar sakte hain. Wo aapki puri sahayata karenge! 🙏"
            else:
                msg = "I apologize, but I don't have specific details on this at the moment. Please contact our admission helpdesk at 0141-2338591 or 9358890991. They will be happy to provide you with the latest information."
            return {"answer": msg, "sources": [], "pdf_url": None}

        context = "\n---\n".join([c.get("text", "") for c in chunks])
        sources = list(dict.fromkeys([c.get("url", "") for c in chunks if c.get("url")]))

        # 4. GenIUS Prompting (Intelligence v3)
        tone_instruction = (
            "Use warm Hinglish (e.g., 'Dekhiye...', 'Main aapko guide kar deta hoon...')." 
            if lang == "Hinglish" else 
            "Use a professional, helpful, and sophisticated English tone (e.g., 'Certainly...', 'Let me guide you through the details...')."
        )
        
        system = (
            f"You are the Senior Academic Counselor at Biyani Group of Colleges. Reply in {'natural Hinglish' if lang == 'Hinglish' else 'professional English'}.\n"
            "INTELLIGENCE RULES:\n"
            "1. BE DIRECT: Do NOT repeat or rephrase the user's question. Just answer it.\n"
            "2. SMART BRIDGING: If the exact info (like BCA fees) is missing, but related info (BBA fees) is available, offer it naturally while being honest about the gap.\n"
            f"3. PERSONA: Sound like a helpful human, not a bot. {tone_instruction}\n"
            "4. CALL TO ACTION: Always provide the helpline 0141-2338591 for specific details.\n"
            "5. NO HALLUCINATION: Answer strictly from context."
        )

        messages = [{"role": "system", "content": system}]
        messages.extend(history[-4:])
        messages.append({"role": "user", "content": f"CONTEXT FROM UNIVERSITY DATABASE:\n{context}\n\nUSER QUESTION: {user_message}"})

        answer = _call_llm(messages, query=user_message, num_chunks=len(chunks)) or "Maafi chahta hoon, connection issue hai. Please 0141-2338591 par call karein."
        return {"answer": answer, "sources": sources[:2], "pdf_url": pdf}

rag_engine = QdrantRAGEngine()
