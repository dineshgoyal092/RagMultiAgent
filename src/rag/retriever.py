"""
Vector store — builds a FAISS index and answers semantic search queries.

FAISS IndexFlatIP:
  Flat = exact brute-force search (no approximation, always finds true nearest neighbours)
  IP   = inner product metric (equals cosine similarity when vectors are unit-length)

  Best choice for small corpora (< 10k chunks). For millions of vectors,
  IndexIVFFlat (approximate, partitioned) would be faster.
"""
import numpy as np
import faiss
from src.models import Chunk, RetrievedChunk
from src.rag.embedder import embed_chunks, embed_query
from src import config


class VectorStore:

    def __init__(self, chunks: list):
        """
        Build the FAISS index from all document chunks.
        Called once at startup — takes a few seconds for embedding.
        """
        # Keep the original Chunk objects so we can map search results back to them
        self.chunks = chunks

        # Step 1: embed all chunks into a float32 matrix of shape (N, 384)
        embeddings = embed_chunks(chunks).astype(np.float32)

        # Step 2: create an index for 384-dimensional vectors using inner product
        # embeddings.shape[1] = 384 (the vector dimension)
        self.index = faiss.IndexFlatIP(embeddings.shape[1])

        # Step 3: add all chunk vectors to the index in one batch
        # After this call, self.index holds all vectors and can be searched
        self.index.add(embeddings)

    def search(self, query: str, top_k: int = None) -> list:
        """
        Find the top-k most semantically similar chunks for a query.
        Returns a list of RetrievedChunk objects sorted by similarity score.
        """
        k = top_k or config.TOP_K  # default = 4 chunks per query

        # Embed the query into the same 384-dim space as the stored chunks
        # reshape(1, -1) because FAISS expects a 2D array (batch of queries)
        query_vec = embed_query(query).reshape(1, -1).astype(np.float32)

        # FAISS search returns two arrays of shape (1, k):
        #   scores  — cosine similarity values (higher = more relevant)
        #   indices — positions in self.chunks corresponding to each score
        scores, indices = self.index.search(query_vec, k)

        # scores[0] and indices[0]: unwrap the single-query batch dimension
        return [
            RetrievedChunk(chunk=self.chunks[idx], score=float(score))
            for score, idx in zip(scores[0], indices[0])
            if idx >= 0   # FAISS returns -1 when index has fewer than k items
        ]
