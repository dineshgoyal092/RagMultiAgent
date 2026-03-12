"""
Document chunker — splits contracts into clause-level chunks.

Strategy: split on numbered sections (1., 2., 3. ...).
Each clause is a self-contained obligation, so clause-level chunks
preserve meaning better than fixed-size sliding windows.
"""
import re
import os
from src.models import Chunk


def load_and_chunk(data_dir: str) -> list:
    """Load every .txt file in data_dir and return a flat list of Chunks."""
    chunks = []
    for filename in sorted(os.listdir(data_dir)):
        if filename.endswith(".txt"):
            path = os.path.join(data_dir, filename)
            doc_name = filename.replace(".txt", "")
            with open(path, encoding="utf-8") as f:
                text = f.read()
            chunks.extend(_chunk_document(text, doc_name))
    return chunks


def _chunk_document(text: str, doc_name: str) -> list:
    """Split one document into numbered-section chunks."""
    # Split before any line that starts with a digit and period (e.g. "1.")
    parts = re.split(r"(?=\n\d+\.)", text.strip())

    chunks = []
    for i, part in enumerate(parts):
        part = part.strip()
        if not part:
            continue
        section = part.split("\n")[0].strip()   # first line = section heading
        chunks.append(Chunk(
            text=part,
            doc_name=doc_name,
            section=section,
            chunk_id=f"{doc_name}#{i}",
        ))
    return chunks
