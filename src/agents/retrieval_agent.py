"""
RetrievalAgent — retrieval with optional cross-encoder re-ranking.

Pipeline:
  1. FAISS bi-encoder search  → fast candidate recall (top RECALL_K chunks)
  2. CrossEncoder re-ranking  → precise relevance scoring on query+chunk pairs
  3. Return top TOP_K after re-ranking

WHY two stages?
  Bi-encoders (FAISS) embed query and chunk independently — fast but less precise.
  Cross-encoders see query + chunk together — slower but significantly more accurate.
  Two-stage keeps latency low while improving final precision.
"""
from sentence_transformers import CrossEncoder
from src.rag.retriever import VectorStore
from src import config

# Fetch more candidates than needed so the reranker has room to reorder.
# RECALL_K >> TOP_K gives the reranker meaningful signal.
RECALL_K = 12

# Small, fast cross-encoder fine-tuned on MS MARCO passage ranking.
# ~80 MB, no API cost, strong on short factual passages.
_reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


class RetrievalAgent:

    def __init__(self, vector_store: VectorStore):
        self.store = vector_store

    def retrieve(self, query: str) -> list:
        """
        Two-stage retrieval: FAISS recall → cross-encoder rerank → top-K.
        """
        # Stage 1: retrieve a broad candidate set via FAISS
        candidates = self.store.search(query, top_k=RECALL_K)

        # Stage 2: score each (query, chunk) pair with the cross-encoder
        pairs = [(query, rc.chunk.text) for rc in candidates]
        scores = _reranker.predict(pairs)

        # Re-sort by cross-encoder score (descending) and keep top_k
        reranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
        return [rc for _, rc in reranked[:config.TOP_K]]
