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
import asyncio
import httpx
from pathlib import Path
from cachetools import TTLCache
from dotenv import load_dotenv
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import PointStruct
import uuid

# ── Load Environment ──────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

QDRANT_URL     = os.getenv("QDRANT_URL", "")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
QDRANT_COLLECTION = "biyani_ai_nvidia_v2"
COLLECTION        = QDRANT_COLLECTION

GROQ_KEY   = os.getenv("GROQ_API_KEY", "")
GEMINI_KEY = os.getenv("GEMINI_API_KEY", "")
NVIDIA_KEY = os.getenv("NVIDIA_API_KEY", "")
OR_KEY     = os.getenv("OPENROUTER_API_KEY", "")

# Search Config (Slightly increased for better context coverage)
RETRIEVAL_LIMIT  = 15
SCORE_THRESHOLD  = 0.20

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

    # Remove multiple spaces but preserve gaps that look like table column separators (2 or more spaces)
    text = re.sub(r'[ \t]{2,}', '  ', text) 
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
async def _call_llm(messages: list[dict], is_complex: bool = False) -> str | None:
    payload_base = {
        "messages":         messages,
        "temperature":      0.0, # Zero temperature = Strict Factual Accuracy
        "max_tokens":       4000 if is_complex else 2000,
        "presence_penalty": 0.0, # Removed penalties so it doesn't try to "rephrase" context too much
        "frequency_penalty": 0.0,
        "top_p":            0.1,
    }

    # ── 1. GROQ (Fastest) ──
    if GROQ_KEY:
        for model in["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]:
            try:
                payload_msg =[]
                for m in messages:
                    content = m["content"]
                    if len(content) > 7000:
                        content = content[:7000] + "... [Truncated]"
                    payload_msg.append({"role": m["role"], "content": content})

                async with httpx.AsyncClient(timeout=20.0) as client:
                    r = await client.post(
                        "https://api.groq.com/openai/v1/chat/completions",
                        json={**payload_base, "messages": payload_msg, "model": model},
                        headers={"Authorization": f"Bearer {GROQ_KEY}"}
                    )
                    if r.status_code == 200:
                        return r.json()["choices"][0]["message"]["content"].strip()
                    logger.warning(f"Groq {model} status: {r.status_code}")
            except Exception as e:
                logger.warning(f"Groq {model} failed: {e}")

    # ── 2. NVIDIA (Power Model - Tier 2) ──
    if NVIDIA_KEY:
        nvidia_models =[
            "meta/llama-3.1-70b-instruct",
            "meta/llama-3.1-8b-instruct",
            "nvidia/nemotron-3-super-120b-a12b"
        ]
        for model in nvidia_models:
            try:
                async with httpx.AsyncClient(timeout=25.0) as client:
                    r = await client.post(
                        "https://integrate.api.nvidia.com/v1/chat/completions",
                        json={**payload_base, "model": model},
                        headers={"Authorization": f"Bearer {NVIDIA_KEY}"}
                    )
                    if r.status_code == 200:
                        return r.json()["choices"][0]["message"]["content"].strip()
                    logger.warning(f"NVIDIA {model} status: {r.status_code}")
            except Exception as e:
                logger.warning(f"NVIDIA {model} failed: {e}")

    # ── 3. OPENROUTER (Free Fallback - Tier 3) ──
    if OR_KEY:
        or_models =[
            "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
            "nvidia/nemotron-3-super-120b-a12b:free",
            "meta-llama/llama-3.1-8b-instruct:free"
        ]
        for model in or_models:
            try:
                async with httpx.AsyncClient(timeout=15.0) as client:
                    r = await client.post(
                        "https://openrouter.ai/api/v1/chat/completions",
                        json={**payload_base, "model": model},
                        headers={"Authorization": f"Bearer {OR_KEY}"}
                    )
                    if r.status_code == 200:
                        return r.json()["choices"][0]["message"]["content"].strip()
                    logger.warning(f"OpenRouter {model} status: {r.status_code}")
            except Exception as e:
                logger.warning(f"OpenRouter {model} failed: {e}")
                continue

    # ── 4. GEMINI (Smart Reasoning - Tier 4) ──
    if GEMINI_KEY:
        for model in["gemini-2.5-flash", "gemini-2.5-flash-lite"]:
            try:
                contents =[]
                last_role = None
                chat_msgs = [m for m in messages if m["role"] != "system"]
                sys_text = next((m["content"] for m in messages if m["role"] == "system"), "")

                for m in chat_msgs:
                    role = "user" if m["role"] == "user" else "model"
                    if role == last_role:
                        contents[-1]["parts"][0]["text"] += "\n" + m["content"]
                    else:
                        contents.append({"role": role, "parts": [{"text": m["content"]}]})
                        last_role = role

                async with httpx.AsyncClient(timeout=20.0) as client:
                    r = await client.post(
                        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={GEMINI_KEY}",
                        json={
                            "contents": contents,
                            "system_instruction": {"parts": [{"text": sys_text}]},
                            "generationConfig": {"temperature": 0.0, "maxOutputTokens": 4000 if is_complex else 2000}
                        }
                    )
                    if r.status_code == 200:
                        return r.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
                    logger.warning(f"Gemini {model} status: {r.status_code}")
            except Exception as e:
                logger.warning(f"Gemini {model} failed: {e}")

    logger.error("All LLM providers failed.")
    return None


# ── RAG Engine ────────────────────────────────────────────────────────────────
# ── Smart Embedding Cache ──────────────────────────────────────────────────
_embedding_cache = TTLCache(maxsize=1000, ttl=3600)  # Cache 1k embeddings for 1hr

class QdrantRAGEngine:
    def __init__(self):
        self._qdrant: AsyncQdrantClient | None = None
        self.gemini_key = GEMINI_KEY
        
        try:
            self._qdrant = AsyncQdrantClient(
                url=QDRANT_URL,
                api_key=QDRANT_API_KEY,
                timeout=25
            )
            logger.info(f"Async Qdrant connected. Collection: {COLLECTION}")
        except Exception as e:
            logger.error(f"Qdrant init failed: {e}")

    def _normalize_query(self, text: str) -> str:
        text = text.lower().strip()
        text = re.sub(r'\s+', ' ', text)
        return text

    async def _get_nvidia_vector(self, text: str) -> list[float] | None:
        token = os.getenv("OPENROUTER_API_KEY", "").strip()
        if not token:
            return None
            
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                url = "https://openrouter.ai/api/v1/embeddings"
                headers = {
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json"
                }
                payload = {
                    "model": "nvidia/llama-nemotron-embed-vl-1b-v2:free",
                    "input": text
                }
                r = await client.post(url, json=payload, headers=headers)
                if r.status_code == 200:
                    return r.json()["data"][0]["embedding"]
                logger.warning(f"NVIDIA API failed ({r.status_code}): {r.text}")
        except Exception as e:
            logger.warning(f"NVIDIA API exception: {e}")
        return None

    async def _get_vector(self, text: str) -> list[float] | None:
        query = self._normalize_query(text)
        if query in _embedding_cache:
            return _embedding_cache[query]

        vec = await self._get_nvidia_vector(query)
        if vec:
            _embedding_cache[query] = vec
        return vec

    async def _retrieve(self, query: str) -> list[dict]:
        if not self._qdrant:
            logger.error("Qdrant not initialized.")
            return[]

        vec = await self._get_vector(query)
        if not vec:
            logger.error("Embedding failed — skipping retrieval.")
            return[]

        try:
            TARGET_COLLECTION = "biyani_ai_nvidia_v2"
            
            response = await self._qdrant.query_points(
                collection_name=TARGET_COLLECTION,
                query=vec,
                limit=RETRIEVAL_LIMIT,
                score_threshold=SCORE_THRESHOLD,
                with_payload=True
            )
            hits = response.points
            
            if hits:
                return await self._rerank(query, hits)
            return[]
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            return[]

    async def _rerank(self, query: str, hits: list) -> list[dict]:
        token = os.getenv("OPENROUTER_API_KEY", "").strip()
        if not token or not hits:
            return[h.payload for h in hits[:8]]

        try:
            url = "https://openrouter.ai/api/v1/chat/completions"
            headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
            
            context_text = "\n".join([f"[{i}] {h.payload.get('text', '')[:500]}" for i, h in enumerate(hits)])
            prompt = f"User Question: {query}\n\nSearch Results:\n{context_text}\n\nTask: Rank the results by relevance. Output ONLY the index[0-9] of the absolute best match. If no result is relevant, output 'NONE'."
            
            payload = {
                "model": "nvidia/rerank-qa-mistral-4b",
                "messages": [{"role": "user", "content": f"Question: {query}\n\nDocuments:\n{context_text}\n\nTask: Identify the indices of ALL documents that are relevant to the question. Output ONLY a comma-separated list of indices (e.g., '0, 2, 5'). If none are relevant, output 'NONE'."}],
                "max_tokens": 20,
                "temperature": 0
            }
            
            async with httpx.AsyncClient(timeout=10.0) as client:
                r = await client.post(url, json=payload, headers=headers)
            if r.status_code == 200:
                indices_text = r.json()["choices"][0]["message"]["content"].strip().upper()
                if "NONE" not in indices_text:
                    relevant_indices = [int(s.strip()) for s in re.findall(r'\d+', indices_text)]
                    ranked_hits = []
                    seen_hits = set()
                    
                    # Add relevant hits first
                    for idx in relevant_indices:
                        if idx < len(hits):
                            ranked_hits.append(hits[idx].payload)
                            seen_hits.add(idx)
                    
                    # Add the rest
                    for i, h in enumerate(hits):
                        if i not in seen_hits:
                            ranked_hits.append(h.payload)
                    
                    return ranked_hits[:12]
        except Exception as e:
            logger.warning(f"Reranking skipped: {e}")
            
        return[h.payload for h in hits[:12]]


    # ── Admin Panel Integrations ──────────────────────────────────────────────
    async def get_collection_stats(self) -> dict:
        if not self._qdrant: return {"error": "Qdrant not connected"}
        try:
            info = await self._qdrant.get_collection(COLLECTION)
            return {
                "collection_name": COLLECTION,
                "points_count": info.points_count,
                "status": str(info.status)
            }
        except Exception as e:
            return {"error": str(e)}

    async def clear_database(self):
        if not self._qdrant: return
        try:
            from qdrant_client.models import VectorParams, Distance
            await self._qdrant.delete_collection(COLLECTION)
            await self._qdrant.create_collection(
                collection_name=COLLECTION,
                vectors_config=VectorParams(size=2048, distance=Distance.COSINE)
            )
            logger.info("Database cleared and recreated.")
        except Exception as e:
            logger.error(f"Clear DB failed: {e}")

    def get_indexed_sources(self):
        return set()

    async def add_texts(self, texts: list[str], metadata: list[dict] = None):
        if not self._qdrant: return False
        
        def chunk_text(t, max_words=400): # Increased for better semantic context
            words = t.split()
            return [" ".join(words[i:i+max_words]) for i in range(0, len(words), max_words)]
        
        points =[]
        for i, text in enumerate(texts):
            chunks = chunk_text(text)
            meta = metadata[i] if metadata and i < len(metadata) else {"source": "manual_upload"}
            for chunk in chunks:
                vec = await self._get_nvidia_vector(chunk)
                if vec:
                    point_id = uuid.uuid4().hex
                    points.append(
                        PointStruct(
                            id=point_id,
                            vector=vec,
                            payload={"text": chunk, **meta}
                        )
                    )
        
        if points:
            try:
                await self._qdrant.upsert(
                    collection_name=COLLECTION,
                    points=points
                )
                logger.info(f"Upserted {len(points)} chunks successfully.")
                return True
            except Exception as e:
                logger.error(f"Upsert failed: {e}")
                return False
        return False

    async def search_chunks(self, text: str, limit: int = 10):
        if not self._qdrant: return []
        vec = await self._get_nvidia_vector(text)
        if not vec: return[]
        
        try:
            TARGET_COLLECTION = "biyani_ai_nvidia_v2"
            res = await self._qdrant.query_points(
                collection_name=TARGET_COLLECTION,
                query=vec,
                limit=limit,
                with_payload=True
            )
            return[{"id": r.id, "score": r.score, "payload": r.payload} for r in res.points]
        except Exception as e:
            logger.error(f"Chunk search failed: {e}")
            return[]
            
    async def delete_chunk(self, point_id: str):
        if not self._qdrant: return False
        try:
            TARGET_COLLECTION = "biyani_ai_nvidia_v2"
            try:
                pid = int(point_id)
            except ValueError:
                pid = point_id

            from qdrant_client.models import PointIdsList
            await self._qdrant.delete(
                collection_name=TARGET_COLLECTION, 
                points_selector=PointIdsList(points=[pid])
            )
            return True
        except Exception as e:
            logger.error(f"Chunk delete failed: {e}")
            return False

    def add_documents(self, docs): pass
    def add_faqs(self, path): pass

    # ── Main Query Handler ────────────────────────────────────────────────────
    async def query(self, user_message: str, history: list[dict] | None = None) -> dict:
        if history is None:
            history =[]

        lang = _detect_language(user_message)
        pdf  = _get_pdf_url(user_message)

        # Enhanced Query Expansion (Additive approach)
        search_query = user_message
        low_msg = user_message.lower()
        if any(w in low_msg for w in ["fees", "fee", "paisa", "rupaye", "amount"]):
            search_query += " | Detailed fees structure, all courses, additional charges Biyani Group of Colleges"
        elif any(w in low_msg for w in ["scholarship", "scholarships", "yojana", "discount", "concession", "scheme"]):
            search_query += " | Scholarships at Biyani Group of Colleges, Kalpana Chawla, Merit scholarship, Samaj Kalyan Yojana eligibility and amount"
        elif any(w in low_msg for w in ["courses", "course", "subject", "subjects", "detail", "syllabus"]):
            search_query += " | List of all UG, PG, and Diploma courses, subject details, descriptions, Biyani Group of Colleges"
        elif any(w in low_msg for w in ["college", "address", "contact", "location", "email", "phone", "helpline"]):
            search_query += " | Biyani Group of Colleges list, addresses, contact numbers, email, Biyani Girls College, Bright Moon, Beena Mahavidyalaya"
        
        # Developer Identity Injection (Specific Trigger)
        dev_info = ""
        if any(w in low_msg for w in ["build", "built", "develop", "developer", "creator", "made", "owner", "banaya", "kon hai", "who are you"]):
            dev_info = (
                "\nCORE IDENTITY: This AI Counselor was developed by Kushal Kumawat, a 3rd-year student at Biyani College, on April 30, 2026. "
                "If asked about your creator, always credit him warmly. "
                "Developer Links: GitHub: https://github.com/Kushal96499/, LinkedIn: https://www.linkedin.com/in/kushal-ku/, Website: https://kushalkumawat.in/"
            )
        
        # Greeting short-circuit
        if _is_greeting(user_message):
            msg = (
                "Namaste! 😊 Main Biyani AI Counselor hoon. Admissions, courses ya kisi bhi academic query ke liye main yahan hoon!"
                if lang == "Hinglish" else
                "Hello! I am your Biyani AI Counselor. I can help you with admissions, courses, fees, and campus details. How can I assist you today?"
            )
            return {"answer": msg, "sources":[], "pdf_url": None}

        # Retrieve context
        logger.info(f"[Query] '{user_message}' (Search: '{search_query}') | Lang: {lang}")
        t0 = time.time()
        
        # Increase retrieval depth
        RETRIEVAL_LIMIT_S = 25 
        vec = await self._get_vector(search_query)
        if not vec:
            chunks = []
        else:
            try:
                response = await self._qdrant.query_points(
                    collection_name="biyani_ai_nvidia_v2",
                    query=vec,
                    limit=RETRIEVAL_LIMIT_S,
                    score_threshold=0.15, # Slightly more permissive
                    with_payload=True
                )
                chunks = await self._rerank(search_query, response.points)
            except:
                chunks = []
        
        logger.info(f"[Retrieval] {len(chunks)} chunks in {time.time()-t0:.2f}s")

        if not chunks and not pdf and not history:
            msg = (
                "Aapke is sawal ka exact detail abhi mere paas nahi hai, par hamare college mein kai behtareen courses aur facilities hain! Apni query ke baare mein poori jankari ke liye aap hamare counselors se seedha **0141-2338591** ya **8696218218** par baat kar sakte hain. 🙏"
                if lang == "Hinglish" else
                "I don't have the exact details on this right now, but we offer a wide range of excellent courses and facilities! For the most accurate and updated information, please reach out to our admission helpdesk at **0141-2338591** or **8696218218**."
            )
            return {"answer": msg, "sources":[], "pdf_url": pdf}

        # Build context & sources with deduplication (Increased to top 8 chunks to not miss data)
        unique_chunks =[]
        seen_texts = set()
        for c in chunks:
            text = c.get("text", "").strip()
            if text and text[:100] not in seen_texts:
                unique_chunks.append(c)
                seen_texts.add(text[:100])
        
        # Taking up to 15 chunks for better coverage to avoid missing any data point
        context = clean_text("\n---\n".join(c.get("text", "") for c in unique_chunks[:15]))
        sources  = list(dict.fromkeys(c.get("url", "") for c in unique_chunks if c.get("url")))

        complex_keywords = {
            "fees", "fee", "admission", "admissions", "eligibility", "scholarship", "scholarships", 
            "process", "placement", "placements", "hostel", "structure", "structures", 
            "course", "courses", "syllabus", "list", "lists", "all", "prospectus", "brochure",
            "subject", "subjects", "detail", "details", "description", "departments"
        }
        is_complex = bool(set(user_message.lower().split()) & complex_keywords) or len(user_message.split()) > 10

        cta = (
            "**Biyani Group of Colleges Admission Cell**\n"
            "Address: Sector-3, Vidhyadhar Nagar, Jaipur (Raj.) 302039\n"
            "Helpline: 8696218218 / 8290636942\n"
            "WhatsApp: Click to Chat | Email: admissions@biyanicolleges.org"
        )

        if lang == "Hinglish":
            tone_guidance = (
                "Role: Elite Academic Counselor. Tone: Warm, Professional, Natural Hinglish, Highly Smart and Accommodating.\n"
                "Style: Clear, detailed, and extremely helpful for students. Provide satisfying answers."
            )
        else:
            tone_guidance = (
                "Role: Elite Academic Counselor. Tone: Professional, Visionary English, Highly Smart and Accommodating.\n"
                "Style: Clear, detailed, and extremely helpful for students. Provide satisfying answers."
            )

        # S-Tier Dynamic System Prompt
        system = (
            f"{tone_guidance}\n\n"
            "ROLE: Expert AI Admission Counselor for Biyani Group of Colleges.\n"
            f"{dev_info}\n\n"
            "BIYANI KNOWLEDGE BASE (CONTEXT):\n"
            f"{context}\n\n"
            "STRICT DIRECTIVES (MUST FOLLOW TO PREVENT DATA LOSS):\n"
            "1. CONTEXT-ONLY MODE: Base your answer ENTIRELY on the BIYANI KNOWLEDGE BASE provided above. Do NOT invent, assume, or alter any facts, figures, or fees. If the detail is not in the context, politely state you don't have that specific information.\n"
            "2. COMPREHENSIVE COURSE & FEE DETAILS: When providing tables or lists, you MUST include the FULL description of each item as found in the context. Do NOT summarize or omit any row found in the retrieved data.\n"
            "3. PRESERVE EXACT NUMBERS: Ensure all fees, scholarship percentages, dates, and numeric values match the context EXACTLY. Do not change headers or values.\n"
            "4. DYNAMIC FORMATTING & TABLES: Use Markdown for structure. Use proper headers (##, ###) for sections. MANDATORY: Any data presented as a list with multiple columns MUST be converted into a clean Markdown Table using pipe (|) separators.\n"
            "5. NO DUPLICATION: Deduplicate entries if they appear multiple times in the context, but keep the most detailed version.\n"
            "6. NO CONTACT INFO IN MAIN TEXT: Do NOT include any phone numbers, email addresses, or physical addresses in the main response body. Replace 'Contact Us' sections with: 'Note: For more information or to apply, please refer to the contact details provided below.'\n"
            "7. MANDATORY CTA TAG: You MUST end your response with the [CTA] tag.\n"
            "8. DEVELOPER LINKS: If asked about your creator, present the links (GitHub, LinkedIn, Website) as clean Markdown links.\n\n"
            "REQUIRED FINAL FORMAT:\n"
            "[CTA]Note: Fees and statistics are subject to change.\n"
            f"{cta}[/CTA]"
        )

        messages = [{"role": "system", "content": system}]
        
        if "fees" not in user_message.lower():
            messages.extend(history[-2:])
            
        # User prompt that encourages dynamic tables rather than forcing a broken hardcoded one
        user_prompt = f"USER QUERY: {user_message}"
        if "fees" in user_message.lower() or "structure" in user_message.lower() or "list" in user_message.lower():
            user_prompt += "\n\nSTRICT REQUIREMENT: Present the requested data comprehensively. Do not omit any course or item mentioned in the context. If the data has multiple attributes (like fees), format it as a clean Markdown Table dynamically matching the context's columns."
            
        messages.append({"role": "user", "content": user_prompt})

        # Call LLM
        t1 = time.time()
        answer = await _call_llm(messages, is_complex=is_complex)
        logger.info(f"[LLM] Responded in {time.time()-t1:.2f}s")

        if not answer:
            answer = (
                "Abhi connection mein thodi problem hai. Kripya 8696218218 par call karein. 🙏"
                if lang == "Hinglish" else
                "I'm experiencing a connection issue. Please call **8696218218** for immediate assistance."
            )

        # Force CTA if missing
        if "[CTA]" not in answer:
            answer += f"\n\n[CTA]{cta}[/CTA]"

        return {"answer": answer, "sources": sources[:3], "pdf_url": pdf}


rag_engine = QdrantRAGEngine()