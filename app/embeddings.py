import google.generativeai as genai
from app.config import settings
from app.logger import logger

class GeminiEmbeddings:
    def __init__(self):
        self.api_key = settings.GEMINI_API_KEY
        # Trying the stable v1 API instead of v1beta
        self.url = "https://generativelanguage.googleapis.com/v1/models/text-embedding-004:embedContent"

    def embed_documents(self, texts):
        import requests
        all_embeddings = []
        
        for text in texts:
            try:
                payload = {
                    "model": "models/text-embedding-004",
                    "content": {"parts": [{"text": text}]},
                    "taskType": "RETRIEVAL_DOCUMENT"
                }
                # Try v1 Stable
                response = requests.post(f"{self.url}?key={self.api_key}", json=payload)
                
                # If v1 fails, try v1beta with older model as a last resort
                if response.status_code == 404:
                    beta_url = "https://generativelanguage.googleapis.com/v1beta/models/embedding-001:embedContent"
                    payload["model"] = "models/embedding-001"
                    response = requests.post(f"{beta_url}?key={self.api_key}", json=payload)

                if response.status_code != 200:
                    logger.error(f"Gemini API Error {response.status_code}: {response.text}")
                    raise Exception(f"API Error {response.status_code}: {response.text}")

                all_embeddings.append(response.json()["embedding"]["values"])
            except Exception as e:
                logger.error(f"Embedding failed: {e}")
                raise Exception(f"Gemini API Error: {e}")
                
        return all_embeddings

    def embed_query(self, text):
        try:
            result = genai.embed_content(
                model=self.model,
                content=text,
                task_type="retrieval_query"
            )
            return result['embedding']
        except Exception as e:
            logger.error(f"Error generating query embedding: {str(e)}")
            return []

# Custom embedding function for ChromaDB
class ChromaGeminiEmbeddingFunction:
    def __init__(self):
        self.embedder = GeminiEmbeddings()

    def __call__(self, input):
        return self.embedder.embed_documents(input)

    def name(self):
        return "gemini_embeddings"
