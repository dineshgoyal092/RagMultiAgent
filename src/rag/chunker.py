"""
Document chunker — splits contracts into clause-level chunks.

WHY clause-level (not sliding-window)?
  Legal contracts are structured as numbered obligations: "1. Definitions",
  "2. Confidentiality", etc. Each clause is semantically self-contained.
  Splitting on clause boundaries preserves the full obligation in one chunk,
  so the retriever always returns complete, coherent context to the LLM.
"""
import re
import os
from src.models import Chunk


def load_and_chunk(data_dir: str) -> list:
    """Load every .txt file in data_dir and return a flat list of Chunks."""

    chunks = []

    # os.listdir returns filenames only, sorted() gives consistent ordering
    for filename in sorted(os.listdir(data_dir)):

        # Skip non-text files (.json, .docx, etc.)
        if not filename.endswith(".txt"):
            continue

        path = os.path.join(data_dir, filename)

        # doc_name is used as the citation key, e.g. "nda_acme_vendor"
        doc_name = filename.replace(".txt", "")

        with open(path, encoding="utf-8") as f:
            text = f.read()

        # _chunk_document returns a list of Chunks; extend flattens into one list
        chunks.extend(_chunk_document(text, doc_name))

    return chunks


def _chunk_document(text: str, doc_name: str) -> list:
    """Split one document into numbered-section chunks."""

    # re.split with a lookahead keeps the delimiter at the START of each part.
    # Pattern (?=\n\d+\.) matches just before lines like "\n1." "\n2." "\n10."
    # Without lookahead, the section numbers would be consumed and lost.
    parts = re.split(r"(?=\n\d+\.)", text.strip())

    chunks = []

    for i, part in enumerate(parts):
        part = part.strip()

        # Skip empty strings that appear when the doc starts with whitespace
        if not part:
            continue

        # First line is always the section heading, e.g. "3. Term and Termination"
        # This becomes the section label shown in citations: [doc §section]
        section = part.split("\n")[0].strip()

        chunks.append(Chunk(
            text=part,               # full clause text sent to the LLM as context
            doc_name=doc_name,       # which contract this came from
            section=section,         # which section/clause within that contract
            chunk_id=f"{doc_name}#{i}",  # unique ID, used for deduplication in display
        ))

    return chunks
