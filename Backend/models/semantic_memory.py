# backend/models/semantic_memory.py

import os
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from fastembed import TextEmbedding

from utils.qdrant_connection import CLIENT, COLLECTION_NAME, EMBEDDING


EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


# -----------------------------
# Upload chunks to Qdrant
# -----------------------------
def upsert_chunks_to_qdrant(chunks, filename, ext):
    points = []

    for c in chunks:
        vec = list(EMBEDDING.embed([c]))[0].tolist()

        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vec,
                payload={
                    "text": c,
                    "file": filename,
                    "type": ext.upper(),
                    "domain": "Healthcare"
                },
            )
        )

    CLIENT.upsert(
        collection_name=COLLECTION_NAME,
        points=points
    )

    return len(points)


# -----------------------------
# Search helper
# -----------------------------
def search_qdrant(question, top=5):
    vec = list(EMBEDDING.embed([question]))[0].tolist()

    results = CLIENT.query_points(
        collection_name=COLLECTION_NAME,
        query=vec,
        limit=top
    )

    return results.points


# -----------------------------
# Semantic Medical Memory
# -----------------------------
class SemanticMedicalMemory:

    def __init__(self):

        self.embedder = TextEmbedding(
            model_name=EMBEDDING_MODEL
        )

        self.client = QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY"),
            timeout=60,
        )

        self.collection = COLLECTION_NAME

    # -------------------------
    # Create embedding
    # -------------------------
    def get_embedding(self, text: str):
        return list(
            self.embedder.embed([text])
        )[0].tolist()

    # -------------------------
    # Query Qdrant
    # -------------------------
    def query(
        self,
        query_text: str,
        top_k: int = 5,
        domain_filter: str = "Healthcare"
    ):

        vector = self.get_embedding(query_text)

        query_filter = {
            "must": [
                {
                    "key": "domain",
                    "match": {
                        "value": domain_filter
                    }
                }
            ]
        }

        results = self.client.search(
        collection_name=self.collection,
        query_vector=vector,
        limit=top_k,
        query_filter=query_filter
    )

        hits = []

        for hit in results:

            payload = hit.payload or {}

            hits.append({
                "score": float(hit.score),
                "text": payload.get("text", ""),
                "response": payload.get("response", ""),
                "complex_cot": payload.get("complex_cot", ""),
                "source": payload.get("source", ""),
                "domain": payload.get("domain", ""),
                "chunk_idx": payload.get("chunk_idx", "")
            })

        return hits