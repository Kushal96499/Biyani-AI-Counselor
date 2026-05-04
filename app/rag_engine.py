"""
app/rag_engine.py — Production RAG Engine
──────────────────────────────────────────
Embedding Strategy:
  - Vercel/Production: HuggingFace Inference API (no download, instant)
  - Local Fallback:    FastEmbed (local ONNX model)
Vector DB: Qdrant Cloud
LLM Stack: Groq → NVIDIA → OpenRouter
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
from app.cache_manager import cache_manager

# ── Load Environment ──────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

QDRANT_URL     = os.getenv("QDRANT_URL", "")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "biyani_clean_v2")
COLLECTION        = QDRANT_COLLECTION

GROQ_KEY   = os.getenv("GROQ_API_KEY", "")
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




# ── Rule-Based Boost ────────────────────────────────────────────────────────
RULE_BASED_KNOWLEDGE = {
    "fees": "Biyani Group of Colleges offers competitive fee structures for various courses like BCA, BBA, MBA, MCA, B.Com, and more. Fees vary by course and college (Biyani Girls College, Biyani Institute of Science & Management, etc.). Scholarships are available based on merit and category. For exact current year fees, students should refer to the Admission Cell.",
    "admission": "Admission at Biyani Group of Colleges is based on both merit and entrance exams (for specific courses like MBA/MCA). The process involves filling an online/offline application form, document verification, and a personal interview in some cases. Direct admission is available for many UG courses.",
    "contact": "You can contact the Biyani Admission Cell at 8696218218 or 0141-2338591. Location: Sector-3, Vidhyadhar Nagar, Jaipur, Rajasthan 302039. Email: admissions@biyanicolleges.org",
    "courses": "Biyani Group offers courses in IT (BCA, MCA, MSc IT), Management (BBA, MBA), Commerce (B.Com, M.Com), Science (B.Sc, M.Sc), Arts (BA, MA), and Law. We have specialized colleges for Girls and Co-ed institutions as well.",
    "campus": (
        "Biyani Group of Colleges has THREE campuses in Jaipur, Rajasthan:\n\n"
        "1. MAIN CAMPUS — Vaidehi 'वैदेही' (Campus No. 1)\n"
        "   Sector-3, Vidhyadhar Nagar, Jaipur (Raj.) 302039\n"
        "   Phone: +91-8696218218, +91-8290636942\n"
        "   Email: director@biyanicolleges.org | admissions@biyanicolleges.org\n\n"
        "2. Maitreyi 'मैत्रेयी' Campus\n"
        "   Behind Bhaitariya Bhairoo Temple, Kalwar-Jobner Road, Kalwar, Jaipur (Raj.) 303706\n"
        "   Tel: +91-141-2589951-52, +91-8696218218\n\n"
        "3. Dhanvantari 'धन्वन्तरि' Campus\n"
        "   Champapura, Kalwar Road, Jaipur (Raj.) 303706\n"
        "   Tel: +91-141-2860175, +91-7568650965\n\n"
        "Overseas Office: Biyani Bio Seeds, 1-1 Asahi-dai, Nomi, Ishikawa 923-1292 Japan\n"
        "Email: biyani@jaist.ac.jp | Phone: +81-761-51-1591"
    )
}

def _get_rule_boost(query: str) -> str:
    ql = query.lower()
    matches = []
    if "fee" in ql or "paisa" in ql: matches.append(RULE_BASED_KNOWLEDGE["fees"])
    if "admission" in ql or "apply" in ql or "process" in ql: matches.append(RULE_BASED_KNOWLEDGE["admission"])
    if "contact" in ql or "call" in ql or "number" in ql or "address" in ql: matches.append(RULE_BASED_KNOWLEDGE["contact"])
    if "course" in ql or "subject" in ql or "degree" in ql: matches.append(RULE_BASED_KNOWLEDGE["courses"])
    if any(w in ql for w in ["campus", "campuses", "location", "vaidehi", "maitreyi", "dhanvantari", "kalwar"]):
        matches.append(RULE_BASED_KNOWLEDGE["campus"])
    return "\n".join(matches) if matches else ""


# ── RAG Intelligence Engine ───────────────────────────────────────────────────
# Semantic matching now handled by Redis

class QdrantRAGEngine:
    def __init__(self):
        self._qdrant: AsyncQdrantClient | None = None
        self._client = httpx.AsyncClient(timeout=30.0, follow_redirects=True)
        
        try:
            self._qdrant = AsyncQdrantClient(
                url=QDRANT_URL,
                api_key=QDRANT_API_KEY,
                timeout=25
            )
            logger.info(f"Async Engine initialized. Qdrant: {COLLECTION}")
        except Exception as e:
            logger.error(f"Qdrant init failed: {e}")

    async def close(self):
        await self._client.aclose()
        await cache_manager.close()
        if self._qdrant:
            await self._qdrant.close()

    async def _call_llm(self, messages: list[dict], is_complex: bool = False) -> str | None:
        payload_base = {
            "messages":         messages,
            "temperature":      0.0,
            "top_p":            1,
            "max_tokens":       2000 if is_complex else 1000,
        }

        # Tier 1: GROQ (Primary - Fastest & Smartest)
        if GROQ_KEY:
            for model in ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "qwen/qwen3-32b"]:
                try:
                    headers = {"Authorization": f"Bearer {GROQ_KEY}", "Content-Type": "application/json"}
                    payload = {**payload_base, "messages": messages, "model": model}
                    r = await self._client.post("https://api.groq.com/openai/v1/chat/completions", json=payload, headers=headers, timeout=8.0)
                    if r.status_code == 200:
                        return r.json()["choices"][0]["message"]["content"].strip()
                    elif r.status_code == 429:
                        err_msg = r.text.lower()
                        if "tpd" in err_msg or "daily" in err_msg:
                            logger.warning(f"Groq Daily Limit Reached. Switching provider...")
                            break # Daily limit is account-wide, switch tier
                        else:
                            logger.warning(f"Groq Minute Limit. Trying next model...")
                            continue # Minute limit might be model-specific
                except Exception as e:
                    logger.debug(f"Groq {model} failed: {e}")
                    continue

        # Tier 2: NVIDIA Chat NIM (Secondary - High Speed)
        if NVIDIA_KEY:
            for model in ["upstage/solar-10.7b-instruct", "mistralai/mistral-large-3-675b-instruct-2512"]:
                try:
                    url = "https://integrate.api.nvidia.com/v1/chat/completions"
                    headers = {"Authorization": f"Bearer {NVIDIA_KEY}", "Content-Type": "application/json"}
                    
                    # Merge system into first user message for maximum compatibility
                    combined_messages = []
                    if len(messages) > 0 and messages[0]["role"] == "system":
                        sys_content = messages[0]["content"]
                        first_user_content = next((m["content"] for m in messages if m["role"] == "user"), "")
                        combined_messages.append({"role": "user", "content": f"INSTRUCTIONS: {sys_content}\n\nUSER QUERY: {first_user_content}"})
                        # Add rest of the history
                        combined_messages.extend([m for m in messages if m != messages[0] and m["content"] != first_user_content])
                    else:
                        combined_messages = messages

                    payload = {"messages": combined_messages, "model": model, "max_tokens": 1024}
                    r = await self._client.post(url, json=payload, headers=headers, timeout=7.0)
                    if r.status_code == 200:
                        return r.json()["choices"][0]["message"]["content"].strip()
                except Exception as e:
                    logger.warning(f"NVIDIA NIM {model} failed: {e}")

        # Tier 3: OpenRouter (Fallback - Emergency)
        if OR_KEY:
            # Use confirmed working IDs from test_api.py
            for model in ["meta-llama/llama-3.1-70b-instruct", "nvidia/nemotron-3-super-120b-a12b:free"]:
                try:
                    url = "https://openrouter.ai/api/v1/chat/completions"
                    headers = {"Authorization": f"Bearer {OR_KEY}", "Content-Type": "application/json"}
                    payload = {**payload_base, "messages": messages, "model": model}
                    r = await self._client.post(url, json=payload, headers=headers, timeout=8.0)
                    if r.status_code == 200:
                        return r.json()["choices"][0]["message"]["content"].strip()
                except:
                    continue
        
        return None

    def _normalize_query(self, text: str) -> str:
        text = text.lower().strip()
        text = re.sub(r'\s+', ' ', text)
        return text

    async def _get_nvidia_nim_vector(self, text: str) -> list[float] | None:
        token = os.getenv("NVIDIA_API_KEY", "").strip()
        if not token: return None
        try:
            url = "https://integrate.api.nvidia.com/v1/embeddings"
            headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
            payload = {
                "model": "nvidia/nv-embed-v1",
                "input": text,
                "input_type": "query",
                "encoding_format": "float"
            }
            r = await self._client.post(url, json=payload, headers=headers)
            if r.status_code == 200: return r.json()["data"][0]["embedding"]
        except: pass
        return None

    async def _get_nvidia_vector(self, text: str) -> list[float] | None:
        token = os.getenv("OPENROUTER_API_KEY", "").strip()
        if not token: return None
        try:
            url = "https://openrouter.ai/api/v1/embeddings"
            headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
            payload = {"model": "nvidia/llama-nemotron-embed-vl-1b-v2:free", "input": text}
            r = await self._client.post(url, json=payload, headers=headers)
            if r.status_code == 200: return r.json()["data"][0]["embedding"]
        except: pass
        return None

    async def _get_vector(self, text: str) -> list[float] | None:
        query = self._normalize_query(text)
        # Use Nemotron (MATCHING THE INDEXED DATA)
        vec = await self._get_nvidia_vector(query)
        return vec

    async def _retrieve(self, query: str, vector: list = None) -> list[dict]:
        if not self._qdrant: return []
        
        # Use provided vector or compute new one only if missing
        vec = vector if vector else await self._get_vector(query)
        if not vec: return []

        try:
            # Optimized Retrieval Limits (Balanced for Details & Tokens)
            limit = 3
            threshold = 0.35

            response = await self._qdrant.query_points(
                collection_name=COLLECTION,
                query=vec,
                limit=limit,
                score_threshold=threshold,
                with_payload=True
            )
            hits = response.points
            
            if hits:
                # Conditional Reranking Logic (Optimization)
                keywords = ["fee", "admission", "course", "placement", "hostel", "scholarship", "eligibility"]
                is_complex = len(query.split()) > 7 or any(k in query.lower() for k in keywords)
                
                if is_complex:
                    return await self._rerank(query, hits[:6]) # Rerank only top 6
                else:
                    return [h.payload for h in hits[:3]] # Instant return for simple queries
            return []
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            return[]

    async def _rerank(self, query: str, hits: list) -> list[dict]:
        """Official NVIDIA Reranker NIM (rerank-qa-mistral-4b) integration."""
        nvidia_key = os.getenv("NVIDIA_API_KEY", "").strip()
        if not nvidia_key or not hits:
            return [h.payload for h in hits[:8]]

        try:
            url = "https://ai.api.nvidia.com/v1/retrieval/nvidia/reranking"
            headers = {
                "Authorization": f"Bearer {nvidia_key}",
                "Content-Type": "application/json",
                "Accept": "application/json"
            }
            
            # Prepare passages from hits
            passages = [{"text": h.payload.get('text', '')} for h in hits[:20]]
            
            payload = {
                "model": "nv-rerank-qa-mistral-4b:1",
                "query": {"text": query},
                "passages": passages,
                "truncate": "END"
            }
            
            r = await self._client.post(url, json=payload, headers=headers)
            if r.status_code == 200:
                data = r.json()
                # NVIDIA returns a list of rankings with index and logit
                rankings = data.get("rankings", [])
                
                scored_hits = []
                for item in rankings:
                    idx = item["index"]
                    score = item["logit"] # Use logit/score for sorting
                    if idx < len(hits):
                        scored_hits.append((score, hits[idx].payload))
                
                # Sort by logit score descending
                scored_hits.sort(key=lambda x: x[0], reverse=True)
                return [item[1] for item in scored_hits[:6]]
            else:
                logger.warning(f"NVIDIA Rerank API Error {r.status_code}: {r.text}")
                return [h.payload for h in hits[:6]]
        except Exception as e:
            logger.error(f"NVIDIA Rerank failed: {e}")
            return [h.payload for h in hits[:6]]

    # ── Admin Panel Integrations ──────────────────────────────────────────────


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
        
        def chunk_text(t, max_chars=1500): 
            # Smart chunking: try to preserve lines and tables
            if "|" in t and "---" in t:
                # If it looks like a table, return as is (tables shouldn't be split)
                return [t]
            
            chunks = []
            current_chunk = []
            current_length = 0
            
            for line in t.split("\n"):
                if current_length + len(line) > max_chars and current_chunk:
                    chunks.append("\n".join(current_chunk))
                    current_chunk = []
                    current_length = 0
                current_chunk.append(line)
                current_length += len(line) + 1
                
            if current_chunk:
                chunks.append("\n".join(current_chunk))
            return chunks
        
        points =[]
        for i, text in enumerate(texts):
            chunks = chunk_text(text)
            meta = metadata[i] if metadata and i < len(metadata) else {"source": "manual_upload"}
            for chunk in chunks:
                vec = await self._get_nvidia_vector(chunk)
                if vec:
                    point_id = str(uuid.uuid4()) # Standard UUID with dashes
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
            TARGET_COLLECTION = COLLECTION
            res = await self._qdrant.query_points(
                collection_name=TARGET_COLLECTION,
                query=vec,
                limit=limit,
                with_payload=True
            )
            return[{"id": str(r.id), "score": r.score, "payload": r.payload, "priority": r.payload.get("priority", 1.0)} for r in res.points]
        except Exception as e:
            logger.error(f"Chunk search failed: {e}")
            return[]
            
    async def delete_chunk(self, point_id: str):
        if not self._qdrant: return False
        try:
            TARGET_COLLECTION = COLLECTION
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

    async def boost_chunk(self, point_id: str):
        if not self._qdrant: return False
        try:
            # Robust ID handling (convert to int if numeric)
            try:
                pid = int(point_id)
            except ValueError:
                pid = point_id

            # 1. Fetch current point to get existing payload
            res = await self._qdrant.retrieve(
                collection_name=COLLECTION,
                ids=[pid],
                with_payload=True
            )
            if not res: return False
            
            payload = res[0].payload or {}
            current_priority = payload.get("priority", 1.0)
            
            # Convert categorical priority to numeric if needed
            if isinstance(current_priority, str):
                priority_map = {"low": 1.0, "medium": 1.5, "high": 2.0, "boosted": 2.5}
                current_priority = priority_map.get(current_priority.lower(), 1.0)
                
            new_priority = float(current_priority) + 0.5 # Safe numeric boost
            
            # 2. Update only the priority field
            await self._qdrant.set_payload(
                collection_name=COLLECTION,
                payload={"priority": new_priority},
                points=[pid]
            )
            return True
        except Exception as e:
            logger.error(f"Boost failed: {e}")
            return False

    def add_documents(self, docs): pass
    def add_faqs(self, path): pass

    # ── Main Query Handler ────────────────────────────────────────────────────
    async def query(self, user_message: str, history: list[dict] | None = None) -> dict:
        """Core RAG pipeline with Redis caching."""
        if history is None:
            history =[]

        lang = _detect_language(user_message)
        pdf  = _get_pdf_url(user_message)
        
        # 1. Normalize and Check Cache (Exact + Semantic)
        search_query = self._normalize_query(user_message)
        
        # Get query embedding early for semantic cache check
        query_vec = await self._get_vector(search_query)
        
        # Check Redis Cache
        cached_res = await cache_manager.get_cache(user_message, query_vec)
        if cached_res:
            return {
                "answer": cached_res["answer"],
                "sources": ["Verified Institutional Data"],
                "pdf_url": cached_res.get("pdf_url"),
                "is_cached": True
            }

        # 2. Query Expansion (Additive approach)
        low_msg = user_message.lower()
        if any(w in low_msg for w in ["fees", "fee", "paisa", "rupaye", "amount"]):
            search_query += " | Detailed fees structure, all courses, additional charges Biyani Group of Colleges"
        elif any(w in low_msg for w in ["scholarship", "scholarships", "yojana", "discount", "concession", "scheme"]):
            search_query += " | Scholarships at Biyani Group of Colleges, Kalpana Chawla, Merit scholarship, Samaj Kalyan Yojana eligibility and amount"
        elif any(w in low_msg for w in ["courses", "course", "subject", "subjects", "detail", "syllabus"]):
            search_query += " | List of all UG, PG, and Diploma courses, fee structure, subject details, descriptions, Biyani Group of Colleges"
            if "fee" not in low_msg:
                search_query += " (Fetch names only if possible)"
        elif any(w in low_msg for w in ["college", "address", "contact", "location", "email", "phone", "helpline"]):
            search_query += " | Biyani Group of Colleges list, addresses, contact numbers, email, Biyani Girls College, Bright Moon, Beena Mahavidyalaya"
        elif any(w in low_msg for w in ["workshop", "workshops", "linux", "rhcsa", "red hat"]):
            search_query += " | Linux Red Hat workshop (RHCSA), 12-day workshop, hands-on training, system administration, shell scripting, virtualization, upcoming workshops"
        elif any(w in low_msg for w in ["event", "events", "festival", "fest", "celebration", "annual", "cultural"]):
            search_query += " | Biyani Group events, cultural festivals, Biyani Mela, Annual Function, Spectrum, seminars, guest lectures, college celebrations"
        elif any(w in low_msg for w in ["niaa", "nia", "assistant", "chatbot", "help"]):
            search_query += " | Biyani AI Assistant Niaa, assistant features, chatbot questions, student helper, automated support"
        elif any(w in low_msg for w in ["campus", "campuses", "location", "vaidehi", "maitreyi", "dhanvantari", "kalwar", "vidhyadhar"]):
            search_query += " | Biyani Group campus locations, Vaidehi main campus Sector-3 Vidhyadhar Nagar, Maitreyi campus Kalwar, Dhanvantari campus, all campus addresses contact numbers"
        
        # Developer Identity Injection (Specific Trigger)
        dev_info = ""
        if any(w in low_msg for w in ["build", "built", "develop", "developer", "creator", "made", "owner", "banaya", "kon hai", "who are you"]):
            dev_info = (
                "\nCORE IDENTITY: This AI Counselor was developed by Kushal Kumawat, a 3rd-year student at Biyani College (Batch 2023-2026), created on April 30, 2026. "
                "If asked about your creator, always credit him warmly and mention his links. "
                "Developer Links: [GitHub](https://github.com/Kushal96499/), [LinkedIn](https://www.linkedin.com/in/kushal-ku/), [Website](https://kushalkumawat.in/)"
            )
        
        # Special Workshop Knowledge injection if relevant
        workshop_knowledge = ""
        if any(w in low_msg for w in ["linux", "workshop", "rhcsa"]):
            workshop_knowledge = (
                "\nLINUX WORKSHOP INFO: Biyani Group organized a 12-day Linux Red Hat workshop (RHCSA) in collaboration with RHEL. "
                "Highlights: Hands-on training, real-world apps, expert faculty, certificates. "
                "Concepts: Basic commands, SysAdmin, Shell scripting, Virtualization. "
                "Upcoming: Another workshop is planned soon. Contact admissions@biyanicolleges.edu.in for details."
            )

        # Campus format override — suppress table, use readable list
        campus_note = ""
        is_campus_query = any(w in low_msg for w in ["campus", "campuses", "location", "vaidehi", "maitreyi", "dhanvantari", "kalwar", "vidhyadhar", "address"])
        if is_campus_query:
            campus_note = (
                "\nFORMAT OVERRIDE FOR THIS QUERY: The user is asking about campus locations. "
                "DO NOT use a table. Present each campus as a clearly numbered entry with emoji icons. "
                "Format each campus like:\n"
                "📍 **Campus Name** \n Address, City, PIN \n 📞 Phone | ✉️ Email\n"
                "IMPORTANT: After listing all 3 Indian campuses, ALSO include the Overseas Office (Biyani Bio Seeds, Japan) "
                "using a 🌏 icon. Do NOT skip the overseas office — it is part of the official information.\n"
                "Keep it warm, clean, and easy to read. No table columns. No GFM table syntax."
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
        
        # --- RETRIEVAL OPTIMIZATION (Point 2) ---
        RETRIEVAL_LIMIT_S = 10 
        SCORE_THRESHOLD_S = 0.35 
        
        # Reuse existing query_vec (Point 1)
        if not query_vec:
            chunks = []
        else:
            try:
                # Add strict timeout (Point 6)
                # Remove hard threshold from query to allow manual boosting
                response = await asyncio.wait_for(
                    self._qdrant.query_points(
                        collection_name=COLLECTION,
                        query=query_vec,
                        limit=RETRIEVAL_LIMIT_S,
                        with_payload=True
                    ),
                    timeout=5.0
                )
                
                # --- CONDITIONAL RERANKING LOGIC ---
                keywords = ["fee", "admission", "course", "placement", "hostel", "scholarship", "eligibility"]
                is_complex_query = len(search_query.split()) > 7 or any(k in search_query.lower() for k in keywords)
                
                # --- MANUAL PRIORITY & THRESHOLDING ---
                MANUAL_THRESHOLD = 0.28
                scored_points = []
                for p in response.points:
                    try:
                        # --- ROBUST PRIORITY MAPPING ---
                        priority = p.payload.get("priority", 1.0)
                        
                        # Handle categorical strings found in existing database
                        if isinstance(priority, str):
                            priority_map = {"low": 1.0, "medium": 1.5, "high": 2.0, "boosted": 2.5}
                            priority = priority_map.get(priority.lower(), 1.0)
                        
                        # Handle lists if any
                        if isinstance(priority, list) and len(priority) > 0:
                            priority = priority[0]
                        
                        boost_factor = float(priority) if priority is not None else 1.0
                        p_score = float(p.score) * boost_factor
                        
                        if p_score >= MANUAL_THRESHOLD:
                            scored_points.append((p_score, p))
                    except Exception as e:
                        logger.warning(f"Skipping point due to scoring error: {e}")
                        continue
                
                # Sort by adjusted score
                scored_points.sort(key=lambda x: x[0], reverse=True)
                
                if is_complex_query and scored_points:
                    # Limit reranker input
                    top_points = [x[1] for x in scored_points[:6]]
                    try:
                        chunks = await self._rerank(search_query, top_points)
                    except:
                        chunks = [p.payload for p in top_points[:4]]
                else:
                    # Instant fallback (Top 4 boosted)
                    chunks = [x[1].payload for x in scored_points[:4]]
            except Exception as e:
                logger.warning(f"Retrieval/Rerank failed: {e}")
                chunks = []
        
        logger.info(f"[Retrieval] {len(chunks)} chunks in {time.time()-t0:.2f}s")

        if not chunks and not pdf and not history:
            msg = (
                "Aapke is sawal ka exact detail abhi mere paas nahi hai, par hamare college mein kai behtareen courses aur facilities hain! Apni query ke baare mein poori jankari ke liye aap hamare counselors se seedha **0141-2338591** ya **8696218218** par baat kar sakte hain. 🙏"
                if lang == "Hinglish" else
                "I don't have the exact details on this right now, but we offer a wide range of excellent courses and facilities! For the most accurate and updated information, please reach out to our admission helpdesk at **0141-2338591** or **8696218218**."
            )
            return {"answer": msg, "sources":[], "pdf_url": pdf}

        # --- SMART FILTERING & BEST CONTEXT SELECTION ---
        filtered_chunks = []
        seen_texts = set()
        RELEVANT_KEYWORDS = ["biyani", "college", "fees", "admission", "course", "placement", "hostel", "scholarship", "eligibility"]
        
        for c in chunks:
            text = c.get("text", "").strip()
            word_count = len(text.split())
            
            # 1. Meaningful length (>120 words)
            # 2. Keyword relevance
            has_keywords = any(kw in text.lower() for kw in RELEVANT_KEYWORDS)
            
            if word_count > 120 and has_keywords:
                if text[:100] not in seen_texts:
                    filtered_chunks.append(c)
                    seen_texts.add(text[:100])
        
        # Fallback if filtering is too aggressive
        if not filtered_chunks and chunks:
            filtered_chunks = chunks[:5]
            
        # Select top 3 highly relevant chunks (Point 4)
        final_chunks = filtered_chunks[:3]
        
        # Structured Context Format with Source Attribution
        context_parts = []
        for i, c in enumerate(final_chunks):
            content = c.get("text", "").strip()
            source = c.get("title", "Unknown Source")
            context_parts.append(f"[Section {i+1} | Source: {source}]\n{content}")
            
        context = clean_text("\n\n".join(context_parts))
        
        # Rule-based boost injection
        rule_boost = _get_rule_boost(user_message)
        if rule_boost:
            context = f"[Core Policy & General Info]\n{rule_boost}\n\n" + context
        sources  = list(dict.fromkeys(c.get("url", "") for c in final_chunks if c.get("url")))

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

        # Token-Optimized System Prompt
        system = (
            f"Role: Senior Admission Counselor at Biyani Group of Colleges. Tone: Warm, Elite, Professional.\n"
            f"Language: {'Hinglish (Natural)' if lang == 'Hinglish' else 'Professional English'}.\n"
            f"{dev_info}\n{workshop_knowledge}{campus_note}\n\n"
            "CONTEXT:\n"
            f"{context}\n\n"
            "PERSONA: You are 'Biyani AI Counselor', a dedicated institutional assistant for Biyani Group of Colleges. Your tone is professional, warm, and student-centric.\n\n"
            "DIRECTIVES:\n"
            "1. CONVERSATIONAL WRAP: NEVER start with just a table. Begin with a helpful sentence as a counselor. End with a supportive closing note or a call-to-action.\n"
            "2. LANGUAGE MATCH: Use the user's style (Hinglish/English) naturally.\n"
            "3. INSTITUTIONAL FOCUS: Only provide information about Biyani Group of Colleges. IGNORE any technical, programming, or coding examples (like JavaScript events) that might appear in the context. Focus on campus life, festivals (Biyani Mela, Spectrum), academic seminars, and workshops.\n"
            "4. UNIVERSAL TABLE RULES: Render tabular data as a GFM Table. PRESERVE ALL COLUMNS. START with 'S.No.' column (Centered with dots). IMPORTANT: Tables MUST be wrapped in your conversational response.\n"
            "5. CONTACT POLICY: Show 'Biyani Shikshan / Samiti@grayquest.com' ONLY for EMI/Installment queries. Otherwise, stick to the general [CTA].\n"
            "6. COMPREHENSIVENESS: Provide the FULL, detailed information from the context. NEVER skip, shorten, or summarize lists, tables, or itemized details. If the context has 50 items, show all 50.\n"
            "7. GROUNDING: Use ONLY context info. No hallucinations. Maintain institutional pride.\n"
            "8. NIA ASSISTANT: 'Niaa' is a separate Biyani Virtual Assistant. If asked about Niaa, provide information based on the context. Do NOT claim to be Niaa yourself.\n"
            "9. CTA: Conclude EVERY response with the [CTA] tag after your closing note.\n"
        )

        messages = [{"role": "system", "content": system}]
        
        if "fees" not in user_message.lower():
            messages.extend(history[-2:])
            
        # User prompt
        user_prompt = f"USER QUERY: {user_message}"
        if is_campus_query:
            user_prompt += (
                "\n\nSTRICT FORMAT: Respond with a warm intro sentence, then list each campus as a numbered "
                "entry (📍 icon, bold name, address on next line, phone and email). "
                "After the 3 Indian campuses, add a separate section for the Overseas Office "
                "(🌏 Biyani Bio Seeds, Japan — include full address, email, and phone). "
                "NO TABLES. Include ALL locations from the context without skipping any."
            )
        elif "fees" in low_msg or "structure" in low_msg or "list" in low_msg:
            user_prompt += "\n\nSTRICT REQUIREMENT: Provide the complete data in the required GFM Table format. Ensure all courses/items from the context are included."
            
        messages.append({"role": "user", "content": user_prompt})

        # Call LLM
        t1 = time.time()
        answer = await self._call_llm(messages, is_complex=is_complex)
        logger.info(f"[LLM] Responded in {time.time()-t1:.2f}s")

        if not answer:
            answer = (
                "Abhi connection mein thodi problem hai. Kripya 8696218218 par call karein. 🙏"
                if lang == "Hinglish" else
                "I'm experiencing a connection issue. Please call **8696218218** for immediate assistance."
            )

        # Force CTA if missing or ensure tags exist
        if "[CTA]" not in answer:
            answer += f"\n\n[CTA]{cta}[/CTA]"
        elif "[/CTA]" not in answer:
            # If LLM started [CTA] but forgot [/CTA]
            answer += "[/CTA]"

        final_result = {"answer": answer, "sources": sources[:3], "pdf_url": pdf}
        if answer and query_vec and "sorry" not in answer.lower():
            asyncio.create_task(cache_manager.set_cache(user_message, answer, query_vec, pdf))
        return final_result


rag_engine = QdrantRAGEngine()