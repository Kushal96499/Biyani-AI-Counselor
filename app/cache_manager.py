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
        if not self.is_active or not answer or "sorry" in answer.lower():
            return

        normalized = self._normalize_query(query)
        exact_key = self._get_exact_key(normalized)
        
        # Determine TTL based on query content
        ttl = 3600 # 1 hour default
        lower_q = normalized.lower()
        if any(word in lower_q for word in ["fee", "admission", "seat", "intake"]):
            ttl = 1800 # 30 mins for critical data
        elif any(word in lower_q for word in ["about", "vision", "mission", "where"]):
            ttl = 86400 # 1 day for static info

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

    async def close(self):
        await self.client.aclose()

# Global Instance
cache_manager = RedisCacheManager()
