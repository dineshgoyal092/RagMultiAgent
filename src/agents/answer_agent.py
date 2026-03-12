"""
AnswerAgent — generates a grounded answer from retrieved contract clauses.

Single responsibility: given a query + retrieved chunks + conversation history,
produce a precise answer with citations and risk flags.
"""
from src.models import RetrievedChunk, Answer
from src.llm_client import chat

SYSTEM_PROMPT = """\
You are a precise legal contract analyst.

Rules:
1. Answer ONLY from the provided contract excerpts.
2. Cite every claim as [DocName §Section] e.g. [nda_acme_vendor §3].
3. If the information is not in the excerpts, say: "Not specified in the provided documents."
4. Prefix any legal risk on its own line with [RISK].
5. Be concise. Do not give legal advice beyond what the documents state.
"""


def _format_context(chunks: list) -> str:
    """Format retrieved chunks into a readable context block."""
    return "\n\n---\n\n".join(
        f"[{rc.chunk.doc_name} | {rc.chunk.section}]\n{rc.chunk.text}"
        for rc in chunks
    )


class AnswerAgent:
    def answer(self, query: str, chunks: list, history: list) -> Answer:
        """Generate an answer grounded in the retrieved chunks."""
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(history[-4:])       # last 2 conversation turns for context
        messages.append({
            "role": "user",
            "content": (
                f"Contract excerpts:\n\n{_format_context(chunks)}\n\n"
                f"Question: {query}"
            ),
        })

        response = chat(messages, temperature=0.1, max_tokens=600)
        risk_flags = [line.strip() for line in response.splitlines() if "[RISK]" in line]
        return Answer(response=response, sources=chunks, risk_flags=risk_flags)
