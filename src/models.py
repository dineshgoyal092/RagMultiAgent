"""Shared data models used across agents and RAG components."""
from dataclasses import dataclass
from typing import List


@dataclass
class Chunk:
    """One clause-level section from a contract document."""
    text: str       # full clause text
    doc_name: str   # e.g. "nda_acme_vendor"
    section: str    # e.g. "3. Term and Termination"
    chunk_id: str   # e.g. "nda_acme_vendor#3"


@dataclass
class RetrievedChunk:
    """A chunk paired with its similarity score from vector search."""
    chunk: Chunk
    score: float    # cosine similarity (0–1)


@dataclass
class Answer:
    """Final answer returned to the user."""
    response: str
    sources: List[RetrievedChunk]   # chunks used to generate the answer
    risk_flags: List[str]           # lines containing [RISK]
    is_out_of_scope: bool = False   # True if query was refused
