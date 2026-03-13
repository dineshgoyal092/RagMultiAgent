"""
Embedding layer — converts text into dense vectors for similarity search.

Model: all-MiniLM-L6-v2
  - Runs locally, no API cost
  - Produces 384-dimensional vectors
  - normalize_embeddings=True makes vectors unit-length so that
    dot product == cosine similarity (required by FAISS IndexFlatIP)
"""
from typing import Optional
import numpy as np
from sentence_transformers import SentenceTransformer
from src import config
from src.models import Chunk

# Module-level variable holds the model after first load.
# Avoids reloading the 80 MB model on every embed call.
_model: Optional[SentenceTransformer] = None


def _get_model() -> SentenceTransformer:
    """Lazy-load: download/load model on first call, reuse on every subsequent call."""
    global _model
    if _model is None:
        # First call triggers download (~80 MB) and loads into memory
        _model = SentenceTransformer(config.EMBEDDING_MODEL)
    return _model


def embed_chunks(chunks: list) -> np.ndarray:
    """
    Embed all chunks into a matrix. Called once at startup to build the FAISS index.
    Returns shape (N, 384) — one row per chunk.
    """
    # Extract plain text from each Chunk object
    texts = [c.text for c in chunks]

    # encode() returns a numpy array; normalize=True makes vectors unit-length
    # show_progress_bar=False keeps startup output clean
    return _get_model().encode(texts, normalize_embeddings=True, show_progress_bar=False)


def embed_query(query: str) -> np.ndarray:
    """
    Embed a single search query. Called at search time for each user question.
    Returns shape (384,) — must use same model + normalization as embed_chunks
    so that dot product gives correct cosine similarity.
    """
    # encode() expects a list; [0] unwraps the single result to a 1D array
    return _get_model().encode([query], normalize_embeddings=True)[0]
