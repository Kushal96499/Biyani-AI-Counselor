import math
import json
import os
from app.logger import logger

class LiteVectorStore:
    def __init__(self, storage_path="data/vector_store.json"):
        self.storage_path = storage_path
        self.documents = []  # List of {"text": str, "embedding": list, "metadata": dict}
        self.load()

    def add_documents(self, texts, embeddings, metadatas):
        for text, emb, meta in zip(texts, embeddings, metadatas):
            self.documents.append({
                "text": text,
                "embedding": emb,
                "metadata": meta
            })
        self.save()

    def save(self):
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
        try:
            with open(self.storage_path, "w") as f:
                json.dump(self.documents, f)
        except Exception as e:
            logger.error(f"Failed to save vector store: {e}")

    def load(self):
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, "r") as f:
                    self.documents = json.load(f)
            except Exception as e:
                logger.error(f"Failed to load vector store: {e}")

    def clear(self):
        self.documents = []
        if os.path.exists(self.storage_path):
            os.remove(self.storage_path)

    def _cosine_similarity(self, v1, v2):
        dot_product = sum(a * b for a, b in zip(v1, v2))
        magnitude1 = math.sqrt(sum(a * a for a in v1))
        magnitude2 = math.sqrt(sum(b * b for b in v2))
        if magnitude1 == 0 or magnitude2 == 0:
            return 0
        return dot_product / (magnitude1 * magnitude2)

    def query(self, query_embedding, n_results=5):
        if not self.documents:
            return {"documents": [[]], "metadatas": [[]]}

        # Calculate similarities
        scored_docs = []
        for doc in self.documents:
            score = self._cosine_similarity(query_embedding, doc["embedding"])
            scored_docs.append((score, doc))

        # Sort by similarity score descending
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        top_docs = scored_docs[:n_results]

        return {
            "documents": [[d[1]["text"] for d in top_docs]],
            "metadatas": [[d[1]["metadata"] for d in top_docs]]
        }
