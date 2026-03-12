"""
RetrievalAgent — finds relevant contract clauses for a query.

Single responsibility: semantic search over the document corpus.
Decoupled from answer generation so retrieval can be swapped independently.
"""
from src.rag.retriever import VectorStore


class RetrievalAgent:
    def __init__(self, vector_store: VectorStore):
        self.store = vector_store

    def retrieve(self, query: str) -> list:
        """Return the most relevant contract chunks for the query."""
        return self.store.search(query)
