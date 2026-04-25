import requests
import json
from app.config import settings
from app.logger import logger

class GeminiClient:
    """
    A pure requests-based client for Gemini to avoid dependencies on 
    google-generativeai and numpy.
    """
    def __init__(self):
        self.api_key = settings.GEMINI_API_KEY
        self.base_url = "https://generativelanguage.googleapis.com/v1beta/models"

    def embed_content(self, texts, task_type="retrieval_document"):
        if isinstance(texts, str):
            texts = [texts]
            
        embeddings = []
        try:
            for text in texts:
                url = f"{self.base_url}/embedding-001:embedContent?key={self.api_key}"
                payload = {
                    "model": "models/embedding-001",
                    "content": {"parts": [{"text": text}]},
                    "taskType": task_type
                }
                response = requests.post(url, json=payload)
                response.raise_for_status()
                embeddings.append(response.json()["embedding"]["values"])
            return embeddings
        except Exception as e:
            logger.error(f"Gemini API Embedding Error: {e}")
            return []

    def generate_content(self, prompt):
        try:
            url = f"{self.base_url}/{settings.GEMINI_MODEL}:generateContent?key={self.api_key}"
            payload = {
                "contents": [{"parts": [{"text": prompt}]}]
            }
            response = requests.post(url, json=payload)
            response.raise_for_status()
            return response.json()["candidates"][0]["content"]["parts"][0]["text"]
        except Exception as e:
            logger.error(f"Gemini API Generation Error: {e}")
            return "Error generating response."

# Singleton instance
gemini_client = GeminiClient()
