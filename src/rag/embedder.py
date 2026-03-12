"""
Embedding layer using sentence-transformers.

Model: all-MiniLM-L6-v2
- Runs locally (no API key, no cost)
- Returns 384-dimensional vectors
- normalize_embeddings=True enables cosine similarity via dot product
"""
from typing import Optional
import numpy as np
from sentence_transformers import SentenceTransformer
from src import config
from src.models import Chunk

_model: Optional[SentenceTransformer] = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(config.EMBEDDING_MODEL)
    return _model


def embed_chunks(chunks: list) -> np.ndarray:
    """Embed a list of Chunks. Returns (N x 384) float32 array."""
    texts = [c.text for c in chunks]
    return _get_model().encode(texts, normalize_embeddings=True, show_progress_bar=False)


def embed_query(query: str) -> np.ndarray:
    """Embed a single query string. Returns (384,) float32 array."""
    return _get_model().encode([query], normalize_embeddings=True)[0]
