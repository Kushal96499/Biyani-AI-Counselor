import os
import json
import hashlib
import time
import httpx
import logging
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

# Ensure env is loaded before class instantiation
ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

logger = logging.getLogger("cache_manager")

class RedisCacheManager:
    def __init__(self):
        self.url = os.getenv("UPSTASH_REDIS_REST_URL", "").strip('"')
        self.token = os.getenv("UPSTASH_REDIS_REST_TOKEN", "").strip('"')
        self.version = "v1"
        self.client = httpx.AsyncClient(timeout=5.0)
        self.is_active = bool(self.url and self.token)
        
        if not self.is_active:
            logger.warning("Redis Cache is INACTIVE: Missing URL or Token in .env")

    def _normalize_query(self, query: str) -> str:
        """Standardizes query for exact matching."""
        import re
        q = query.lower().strip()
        q = re.sub(r'[^\w\s]', '', q) # Remove punctuation
        q = re.sub(r'\s+', ' ', q)    # Normalize spaces
        return q

    def _get_exact_key(self, normalized_query: str) -> str:
        h = hashlib.md5(normalized_query.encode()).hexdigest()
        return f"chat:{self.version}:exact:{h}"

    def _get_semantic_key_list(self) -> str:
        return f"chat:{self.version}:semantic_list"

    async def _redis_cmd(self, command: list):
        """Execute a command via Upstash REST API."""
        if not self.is_active: return None
        try:
            headers = {"Authorization": f"Bearer {self.token}"}
            r = await self.client.post(self.url, json=command, headers=headers)
            if r.status_code == 200:
                return r.json().get("result")
        except Exception as e:
            logger.error(f"Redis Command Error: {e}")
        return None

    def _calculate_cosine_similarity(self, v1, v2):
        v1 = np.array(v1)
        v2 = np.array(v2)
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    async def get_cache(self, query: str, query_embedding: list = None) -> dict | None:
        if not self.is_active: return None
        
        normalized = self._normalize_query(query)
        exact_key = self._get_exact_key(normalized)
        
        # 1. Check Exact Match
        cached_data = await self._redis_cmd(["GET", exact_key])
        if cached_data:
            logger.info(f"Cache HIT (Exact): {normalized}")
            return json.loads(cached_data)

        # 2. Check Semantic Match (if embedding provided)
        if query_embedding:
            # We store a list of keys for semantic search
            semantic_list = await self._redis_cmd(["LRANGE", self._get_semantic_key_list(), "0", "100"])
            if semantic_list:
                for entry_json in semantic_list:
                    entry = json.loads(entry_json)
                    similarity = self._calculate_cosine_similarity(query_embedding, entry["embedding"])
                    if similarity > 0.88: # Relaxed threshold for better semantic reuse
                        logger.info(f"Cache HIT (Semantic: {similarity:.2f}): {query}")
                        # Also save as exact match to speed up next time
                        await self.set_cache(query, entry["answer"], entry["embedding"])
                        return entry
        
        return None

    async def clear_cache(self):
        """Clears all cached chat data."""
        if not self.is_active: return
        try:
            # Delete exact match keys and semantic list
            # A simpler way for Upstash is FLUSHDB if only used for this app
            await self._redis_cmd(["FLUSHDB"])
            logger.info("Redis Cache cleared successfully.")
        except Exception as e:
            logger.error(f"Failed to clear Redis Cache: {e}")

    async def set_cache(self, query: str, answer: str, embedding: list, pdf_url: str = None):
        # Point 5: Only cache high-quality, long answers
        if not self.is_active or not answer: return
        if len(answer) < 100 or "sorry" in answer.lower() or "difficulty" in answer.lower():
            return

        normalized = self._normalize_query(query)
        exact_key = self._get_exact_key(normalized)
        
        # Use a very long TTL (1 year) since we auto-clear cache on new data upload
        ttl = 31536000 

        cache_obj = {
            "query": query,
            "normalized_query": normalized,
            "answer": answer,
            "pdf_url": pdf_url,
            "embedding": embedding,
            "timestamp": time.time()
        }
        cache_str = json.dumps(cache_obj)

        # Save Exact Match
        await self._redis_cmd(["SET", exact_key, cache_str, "EX", str(ttl)])

        # Save to Semantic List (Keep last 100 entries for similarity)
        # We store minimal info in semantic list to keep it fast
        semantic_entry = {
            "embedding": embedding,
            "answer": answer,
            "pdf_url": pdf_url,
            "query": query
        }
        await self._redis_cmd(["LPUSH", self._get_semantic_key_list(), json.dumps(semantic_entry)])
        await self._redis_cmd(["LTRIM", self._get_semantic_key_list(), "0", "99"]) # Keep only 100

    async def log_chat_to_redis(self, question: str, answer: str, ip: str):
        """Saves a chat log entry to a capped list in Redis and increments counters."""
        if not self.is_active: return
        try:
            # 1. Increment Global Questions Counter
            await self._redis_cmd(["INCR", "stats:total_questions"])
            
            # 2. Increment Unique Sessions (using IP+Hour as a set)
            hour_key = f"stats:sessions:{time.strftime('%Y-%m-%d-%H')}"
            await self._redis_cmd(["SADD", hour_key, ip])
            await self._redis_cmd(["EXPIRE", hour_key, "172800"]) # Keep session sets for 48 hours
            
            # 3. Save Chat Log to a capped list (Keep last 100)
            log_entry = f"{time.strftime('%Y-%m-%d %H:%M:%S')} | IP: {ip} | Q: {question[:50]}..."
            await self._redis_cmd(["LPUSH", "stats:chat_logs", log_entry])
            await self._redis_cmd(["LTRIM", "stats:chat_logs", "0", "99"]) # Keep only top 100
        except Exception as e:
            logger.error(f"Redis Logging failed: {e}")

    async def get_redis_stats(self):
        """Retrieves persistent stats from Redis."""
        if not self.is_active: return {"total_questions": 0, "unique_sessions": 0, "redis_logs": []}
        try:
            # Get total questions
            total = await self._redis_cmd(["GET", "stats:total_questions"])
            
            # Get unique sessions for current hour
            sessions = await self._redis_cmd(["SCARD", f"stats:sessions:{time.strftime('%Y-%m-%d-%H')}"])
            
            # Get last 100 logs
            logs = await self._redis_cmd(["LRANGE", "stats:chat_logs", "0", "99"])
            
            return {
                "total_questions": int(total) if total else 0,
                "unique_sessions": int(sessions) if sessions else 0,
                "redis_logs": logs if logs else []
            }
        except Exception as e:
            logger.error(f"Failed to fetch Redis stats: {e}")
            return {"total_questions": 0, "unique_sessions": 0, "redis_logs": []}

    async def close(self):
        await self.client.aclose()

# Global Instance
cache_manager = RedisCacheManager()
