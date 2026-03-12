"""
Vector store using FAISS for semantic similarity search.

IndexFlatIP = exact inner product search over normalized vectors = cosine similarity.
Correct and simple for small corpora (< 10k chunks).
"""
import numpy as np
import faiss
from src.models import Chunk, RetrievedChunk
from src.rag.embedder import embed_chunks, embed_query
from src import config


class VectorStore:
    def __init__(self, chunks: list):
        self.chunks = chunks
        embeddings = embed_chunks(chunks).astype(np.float32)
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)

    def search(self, query: str, top_k: int = None) -> list:
        """Return top-k most relevant chunks for the query."""
        k = top_k or config.TOP_K
        query_vec = embed_query(query).reshape(1, -1).astype(np.float32)
        scores, indices = self.index.search(query_vec, k)
        return [
            RetrievedChunk(chunk=self.chunks[idx], score=float(score))
            for score, idx in zip(scores[0], indices[0])
            if idx >= 0
        ]
