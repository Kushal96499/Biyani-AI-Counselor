"""
app/rag_engine.py — Production RAG Engine
──────────────────────────────────────────
Embedding Strategy:
  - Vercel/Production: HuggingFace Inference API (no download, instant)
  - Local Fallback:    FastEmbed (local ONNX model)
Vector DB: Qdrant Cloud
LLM Stack: Groq → Gemini → NVIDIA → OpenRouter
"""

import os
import re
import time
import logging
import requests
from pathlib import Path
from cachetools import TTLCache
from dotenv import load_dotenv
from google import genai
from qdrant_client import QdrantClient

# FastEmbed is optional — used only as local fallback
try:
    _FASTEMBED_AVAILABLE = False
except ImportError:
    _FASTEMBED_AVAILABLE = False

# ── Load Environment ──────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

QDRANT_URL     = os.getenv("QDRANT_URL", "")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
COLLECTION     = os.getenv("QDRANT_COLLECTION", "biyani_ai_clean_v2")

GROQ_KEY   = os.getenv("GROQ_API_KEY", "")
GEMINI_KEY = os.getenv("GEMINI_API_KEY", "")
NVIDIA_KEY = os.getenv("NVIDIA_API_KEY", "")
OR_KEY     = os.getenv("OPENROUTER_API_KEY", "")

# ── Model & Search Config ─────────────────────────────────────────────────────
EMBED_MODEL      = "BAAI/bge-small-en-v1.5"
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
RETRIEVAL_LIMIT  = 8
SCORE_THRESHOLD  = 0.30

# Model cache: bundled inside project (committed to git, no download on Vercel)
CACHE_DIR = os.path.join(ROOT, "models")
os.makedirs(CACHE_DIR, exist_ok=True)

logger = logging.getLogger("rag_engine")


# ── PDF Triggers ──────────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    """Fix common PDF extraction artifacts like ligatures."""
    replacements = {
        "ﬃ": "ffi", "ﬀ": "ff", "ﬂ": "fl", "ﬁ": "fi", "ﬄ": "ffl",
        "ﬅ": "st", "ﬆ": "st", "—": "-", "–": "-", "’": "'", "“": "\"", "”": "\""
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
        
    # Global fixes for common extraction artifacts
    text = text.replace(" icer", " Officer")
    text = text.replace(" fficer", " Officer")
    text = text.replace(" cer", " Officer")
    if text.startswith("cer"): text = "Officer" + text[3:]
    if text.startswith("icer"): text = "Officer" + text[4:]
    if text.startswith("fficer"): text = "Officer" + text[6:]

    import re
    # Remove multiple spaces but preserve newlines
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()

PDF_TRIGGERS = {
    "brochure":   "https://www.biyanicolleges.org/wp-content/uploads/2024/08/Prospectus-2024-25.pdf",
    "prospectus": "https://www.biyanicolleges.org/wp-content/uploads/2024/08/Prospectus-2024-25.pdf",
    "placement":  "https://www.biyanicolleges.org/wp-content/uploads/2025/05/Biyani%20Placement%20Brochure.pdf",
}

def _get_pdf_url(query: str) -> str | None:
    ql = query.lower()
    for k, v in PDF_TRIGGERS.items():
        if k in ql:
            return v
    return None


# ── Language Detection ────────────────────────────────────────────────────────
_HINGLISH_MARKERS = {
    "kya", "hai", "hain", "ka", "ki", "ke", "ko", "kaise", "bata", "batao",
    "mujhe", "nhi", "nahi", "kitni", "kab", "kaun", "karo", "aur", "toh",
    "kr", "ho", "hoga", "chahiye", "fees", "liye", "mein", "wali"
}

def _detect_language(text: str) -> str:
    words = set(re.findall(r'\b\w+\b', text.lower()))
    if len(words & _HINGLISH_MARKERS) >= 1 or bool(re.search(r'[\u0900-\u097F]', text)):
        return "Hinglish"
    return "English"

def _is_greeting(text: str) -> bool:
    greetings = {"hi", "hello", "hey", "hola", "namaste", "good morning", "good evening", "gm", "gn"}
    clean = re.sub(r'[^\w\s]', '', text.lower()).strip()
    return clean in greetings


# ── LLM Caller (Tiered: Groq → Gemini → NVIDIA → OpenRouter) ─────────────────
def _call_llm(messages: list[dict], is_complex: bool = False) -> str | None:
    payload_base = {
        "messages":         messages,
        "temperature":      0.25,
        "max_tokens":       900,
        "presence_penalty": 0.2,
        "frequency_penalty": 0.2,
    }

    # ── 1. GROQ (Fastest) ──
    if GROQ_KEY:
        try:
            model = "llama-3.3-70b-versatile" if is_complex else "llama-3.1-8b-instant"
            r = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                json={**payload_base, "model": model},
                headers={"Authorization": f"Bearer {GROQ_KEY}"},
                timeout=9
            )
            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"].strip()
            logger.warning(f"Groq status: {r.status_code}")
        except Exception as e:
            logger.warning(f"Groq failed: {e}")

    # ── 2. GEMINI (Smart Reasoning) ──
    if GEMINI_KEY:
        for model in ["gemini-2.5-flash", "gemini-2.5-flash-lite"]:
            for ver in ["v1beta", "v1"]:
                try:
                    sys_text = next((m["content"] for m in messages if m["role"] == "system"), "")
                    contents = [
                        {"role": "model" if m["role"] == "assistant" else "user",
                         "parts": [{"text": m["content"]}]}
                        for m in messages if m["role"] != "system"
                    ]
                    r = requests.post(
                        f"https://generativelanguage.googleapis.com/{ver}/models/{model}:generateContent?key={GEMINI_KEY}",
                        json={"contents": contents,
                              "system_instruction": {"parts": [{"text": sys_text}]},
                              "generationConfig": {"temperature": 0.25, "maxOutputTokens": 1200}},
                        timeout=15
                    )
                    if r.status_code == 200:
                        return r.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
                    logger.warning(f"Gemini {model}/{ver} status: {r.status_code}")
                except Exception as e:
                    logger.warning(f"Gemini {model}/{ver} failed: {e}")
                    continue

    # ── 3. NVIDIA (Power Model) ──
    if NVIDIA_KEY:
        nvidia_models = [
            "mistralai/mistral-large-3-675b-instruct-2512",
            "mistralai/mixtral-8x7b-instruct-v0.1",
            "abacusai/dracarys-llama-3.1-70b-instruct",
        ]
        for model in nvidia_models:
            try:
                r = requests.post(
                    "https://integrate.api.nvidia.com/v1/chat/completions",
                    json={**payload_base, "model": model},
                    headers={"Authorization": f"Bearer {NVIDIA_KEY}"},
                    timeout=12
                )
                if r.status_code == 200:
                    return r.json()["choices"][0]["message"]["content"].strip()
                logger.warning(f"NVIDIA {model} status: {r.status_code}")
            except Exception as e:
                logger.warning(f"NVIDIA {model} failed: {e}")
                continue

    # ── 4. OPENROUTER (Free Fallback) ──
    if OR_KEY:
        or_models = [
            "nvidia/nemotron-3-super-120b-a12b:free",
            "nvidia/nemotron-3-nano-30b-a3b:free",
            "google/gemma-2-9b-it:free"
        ]
        for model in or_models:
            try:
                r = requests.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    json={**payload_base, "model": model},
                    headers={"Authorization": f"Bearer {OR_KEY}"},
                    timeout=15
                )
                if r.status_code == 200:
                    return r.json()["choices"][0]["message"]["content"].strip()
                logger.warning(f"OpenRouter {model} status: {r.status_code}")
            except Exception as e:
                logger.warning(f"OpenRouter {model} failed: {e}")
                continue

    logger.error("All LLM providers failed.")
    return None


# ── RAG Engine ────────────────────────────────────────────────────────────────
# ── Smart Embedding Cache ──────────────────────────────────────────────────
_embedding_cache = TTLCache(maxsize=1000, ttl=3600)  # Cache 1k embeddings for 1hr

class QdrantRAGEngine:
    def __init__(self):
        self._qdrant: QdrantClient | None = None
        self._embedder = None # Lazy load only if needed
        self.gemini_key = GEMINI_KEY
        
        try:
            self._qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=25)
            logger.info(f"Qdrant connected. Collection: {COLLECTION}")
        except Exception as e:
            logger.error(f"Qdrant init failed: {e}")

    def _normalize_query(self, text: str) -> str:
        """Clean query to increase cache hits"""
        text = text.lower().strip()
        text = re.sub(r'\s+', ' ', text)
        return text

    def _get_nvidia_vector(self, text: str) -> list[float] | None:
        """Tier 1: OpenRouter NVIDIA 2048 Dimensions (High Precision)"""
        token = os.getenv("OPENROUTER_API_KEY", "").strip()
        if not token:
            return None
            
        try:
            url = "https://openrouter.ai/api/v1/embeddings"
            headers = {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": "nvidia/llama-nemotron-embed-vl-1b-v2:free",
                "input": text
            }
            r = requests.post(url, json=payload, headers=headers, timeout=15)
            if r.status_code == 200:
                return r.json()["data"][0]["embedding"]
            logger.warning(f"NVIDIA API failed ({r.status_code}): {r.text}")
        except Exception as e:
            logger.warning(f"NVIDIA API exception: {e}")
        return None

    def _get_gemini_vector(self, text: str) -> list[float] | None:
        """Tier 2: Google Gemini Fallback (384 Dim)"""
        if not self.gemini_key:
            return None
        try:
            from google import genai
            client = genai.Client(api_key=self.gemini_key)
            result = client.models.embed_content(
                model="gemini-embedding-2",
                contents=text,
                config={"output_dimensionality": 384}
            )
            return result.embeddings[0].values
        except Exception as e:
            logger.warning(f"Gemini Fallback failed: {e}")
        return None
    def _get_hf_vector(self, text: str) -> list[float] | None:
        """Tier 2: HuggingFace API Fallback"""
        try:
            # Correct HF Inference API endpoint for embeddings
            url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{EMBED_MODEL}"
            payload = {"inputs": [BGE_QUERY_PREFIX + text], "options": {"wait_for_model": True}}
            headers = {"Content-Type": "application/json"}
            
            # Use HF token if available
            token = os.getenv("HUGGINGFACE_API_KEY", "").strip()
            if token:
                headers["Authorization"] = f"Bearer {token}"
            
            r = requests.post(url, json=payload, headers=headers, timeout=15)
            if r.status_code == 200:
                res = r.json()
                return res[0] if isinstance(res[0], list) else res
            
            logger.warning(f"HF API failed ({r.status_code}): {r.text}")
        except Exception as e:
            logger.warning(f"HF API fallback Exception: {e}")
        return None

    def _get_vector(self, text: str) -> list[float] | None:
        """
        Smart Embedding Dispatcher with Caching & Fallbacks.
        Optimized for Vercel Serverless.
        """
        query = self._normalize_query(text)
        
        # 1. Check Cache
        if query in _embedding_cache:
            return _embedding_cache[query]

        # 2. Try NVIDIA (Elite - 2048 Dim)
        vec = self._get_nvidia_vector(query)
        
        # 3. Try Gemini (Fallback - 384 Dim)
        if not vec:
            vec = self._get_gemini_vector(query)
        
        # 4. Try HF (Fallback - 384 Dim)
        if not vec:
            vec = self._get_hf_vector(query)
            

        # Save to Cache if successful
        if vec:
            _embedding_cache[query] = vec
            
        return vec

    def _retrieve(self, query: str) -> list[dict]:
        if not self._qdrant:
            logger.error("Qdrant not initialized.")
            return []

        vec = self._get_vector(query)
        if not vec:
            logger.error("Embedding failed — skipping retrieval.")
            return []

        try:
            # NVIDIA Elite Collection (2048 Dimensions)
            TARGET_COLLECTION = "biyani_ai_nvidia_v2"
            
            # 1. Search for Top 15 candidates (Better context for Reranker)
            response = self._qdrant.query_points(
                collection_name=TARGET_COLLECTION,
                query=vec,
                limit=15,
                score_threshold=0.05,
                with_payload=True
            )
            hits = response.points
            
            # 2. Rerank if hits found, else return empty
            if hits:
                return self._rerank(query, hits)
            return []
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            return []
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            return []

    def _rerank(self, query: str, hits: list) -> list[dict]:
        """Tier 1: NVIDIA Mistral-4b Reranker for Absolute Precision"""
        token = os.getenv("OPENROUTER_API_KEY", "").strip()
        if not token or not hits:
            return [h.payload for h in hits[:5]]

        try:
            url = "https://openrouter.ai/api/v1/chat/completions"
            headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
            
            # Prepare context for Mistral to rank
            context_text = "\n".join([f"[{i}] {h.payload.get('text', '')[:500]}" for i, h in enumerate(hits)])
            prompt = f"User Question: {query}\n\nSearch Results:\n{context_text}\n\nTask: Rank the results by relevance. Output ONLY the index [0-9] of the absolute best match. If no result is relevant, output 'NONE'."
            
            payload = {
                "model": "nvidia/rerank-qa-mistral-4b",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 5,
                "temperature": 0
            }
            
            r = requests.post(url, json=payload, headers=headers, timeout=10)
            if r.status_code == 200:
                best_id = r.json()["choices"][0]["message"]["content"].strip().upper()
                if "NONE" not in best_id:
                    # Find the corresponding hit
                    for i, h in enumerate(hits):
                        if str(i) in best_id:
                            return [h.payload] + [other.payload for j, other in enumerate(hits) if i != j][:4]
        except Exception as e:
            logger.warning(f"Reranking skipped: {e}")
            
        return [h.payload for h in hits[:5]]



    # ── Compatibility stubs (for admin routes) ────────────────────────────────
    def clear_database(self): pass
    def get_indexed_sources(self): return set()
    def add_documents(self, docs): pass
    def add_faqs(self, path): pass

    # ── Main Query Handler ────────────────────────────────────────────────────
    def query(self, user_message: str, history: list[dict] | None = None) -> dict:
        if history is None:
            history = []

        lang = _detect_language(user_message)
        pdf  = _get_pdf_url(user_message)

        # Greeting short-circuit
        if _is_greeting(user_message):
            msg = (
                "Namaste! 😊 Main Biyani AI Counselor hoon. Admissions, courses ya kisi bhi academic query ke liye main yahan hoon!"
                if lang == "Hinglish" else
                "Hello! I am your Biyani AI Counselor. I can help you with admissions, courses, fees, and campus details. How can I assist you today?"
            )
            return {"answer": msg, "sources": [], "pdf_url": None}

        # Retrieve context
        logger.info(f"[Query] '{user_message}' | Lang: {lang}")
        t0 = time.time()
        chunks = self._retrieve(user_message)
        logger.info(f"[Retrieval] {len(chunks)} chunks in {time.time()-t0:.2f}s")

        # Fallback if no context
        if not chunks and not pdf:
            msg = (
                "Dekhiye, is baare mein abhi mujhe exact jankari nahi hai. Aap hamare counselors se seedha 0141-2338591 par baat kar sakte hain — woh aapki poori help karenge! 🙏"
                if lang == "Hinglish" else
                "I don't have specific details on this right now. Please reach out to our admission helpdesk at **0141-2338591** or **9358890991** for accurate information."
            )
            return {"answer": msg, "sources": [], "pdf_url": pdf}

        # Build context & sources
        context = clean_text("\n---\n".join(c.get("text", "") for c in chunks[:4]))
        sources  = list(dict.fromkeys(c.get("url", "") for c in chunks if c.get("url")))

        # Determine complexity
        complex_keywords = {"fees", "admission", "eligibility", "scholarship", "process", "placement", "hostel", "structure", "course", "syllabus"}
        is_complex = bool(set(user_message.lower().split()) & complex_keywords) or len(user_message.split()) > 12

        # Language-specific disclaimers
        disclaimer = (
            "Note: Please note that fees and statistics are subject to change. For final confirmation, kindly visit the college admission cell."
            if lang == "English" else
            "Kripya dhyan dein ki fees aur stats mein badlav ho sakte hain. Final confirmation ke liye college admission cell se zarur milein."
        )

        # Language-specific CTA
        cta = (
            "You can visit the campus today or call us at: 0141-2338591 / 9358890991."
            if lang == "English" else
            "Aap aaj hi campus visit kar sakte hain ya humein call karein: 0141-2338591 / 9358890991."
        )

        # Build system prompt
        if lang == "Hinglish":
            tone_guidance = (
                "Role: Elite Academic Counselor. Tone: Warm, Persuasive, Natural Hinglish.\n"
                "Style: Rich & Engaging. Don't just give data; tell a story. Use 'Hamare yahan...', 'Aapke career ke liye...'.\n"
                "Formatting: Use **bold** for key benefits. Use bullet points for USPs (Unique Selling Points)."
            )
        else:
            tone_guidance = (
                "Role: Elite Academic Counselor. Tone: Professional, Persuasive, Visionary English.\n"
                "Style: Rich & Structured. Highlight Biyani's legacy, placement records, and campus life.\n"
                "Formatting: Use **bold** for metrics. Use bullet points for key reasons to join."
            )

        system = (
            f"{tone_guidance}\n\n"
            "STRICT GROUNDING RULES:\n"
            "1. ONLY LST DATA FROM CONTEXT: Do NOT mention courses like B.Tech/M.Tech unless you see them in the provided context. If a course is not mentioned, it does not exist for this conversation.\n"
            "2. NO HALLUCINATIONS: Do not guess specializations or durations. Only use what is written.\n"
            "3. BALANCED LENGTH: Provide 2-3 structured paragraphs. Highlight 4-5 key USPs with bullet points.\n"
            "4. DYNAMIC LISTS: If listing courses, end with: 'Apart from these, many other courses are also offered. For the full list, please contact the Admission Cell.'\n"
            "5. LANGUAGE HARMONY: Keep everything (answer, disclaimer, CTA) in the same language.\n"
            "6. CONTEXTUAL DISCLAIMER: IF fees/stats are mentioned, include this at the end:\n"
            f"   '{disclaimer}'\n"
            f"7. CALL TO ACTION: Always end with: '{cta}'\n\n"
            f"BIYANI DATABASE CONTEXT:\n{context}"
        )

        # Build message list (with history)
        messages = [{"role": "system", "content": system}]
        messages.extend(history[-4:])
        messages.append({"role": "user", "content": user_message})

        # Call LLM
        t1 = time.time()
        answer = _call_llm(messages, is_complex=is_complex)
        logger.info(f"[LLM] Responded in {time.time()-t1:.2f}s")

        if not answer:
            answer = (
                "Abhi connection mein thodi problem hai. Kripya 0141-2338591 par call karein. 🙏"
                if lang == "Hinglish" else
                "I'm experiencing a connection issue. Please call **0141-2338591** for immediate assistance."
            )

        return {"answer": answer, "sources": sources[:3], "pdf_url": pdf}


rag_engine = QdrantRAGEngine()
