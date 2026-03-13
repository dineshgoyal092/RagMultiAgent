"""
AnswerAgent — generates a grounded answer from retrieved contract clauses.

Single responsibility: given (query + retrieved chunks + conversation history),
produce a precise answer with source citations and risk flags.
"""
from src.models import RetrievedChunk, Answer
from src.llm_client import chat

# System prompt that controls the model's behaviour for every answer.
# Three key constraints:
#   Rule 1 — prevents hallucination (must only use what was retrieved)
#   Rule 2 — enforces citations so answers are verifiable
#   Rule 4 — makes risks machine-parseable via [RISK] prefix
SYSTEM_PROMPT = """\
You are a precise legal contract analyst.

Rules:
1. Answer ONLY from the provided contract excerpts.
2. Cite every claim as [DocName §Section] e.g. [nda_acme_vendor §3].
3. If information is absent, say: "Not specified in the provided documents."
4. Flag legal risks on their own line with [RISK] (e.g. missing liability cap, strict deadlines,
   conflicting governing laws, unauthorized data sharing).
5. Quote exact values verbatim — numbers, dates, jurisdictions, key phrases.
6. Be concise. No legal advice beyond what the documents state.
"""


def _format_context(chunks: list) -> str:
    """
    Format retrieved chunks into a labelled context block for the LLM.

    Each chunk is prefixed with [doc_name | section] so the model knows
    exactly where each piece of text comes from and can produce accurate citations.
    Chunks are separated by "---" to visually delimit clause boundaries.
    """
    return "\n\n---\n\n".join(
        f"[{rc.chunk.doc_name} | {rc.chunk.section}]\n{rc.chunk.text}"
        for rc in chunks
    )


class AnswerAgent:

    def answer(self, query: str, chunks: list, history: list) -> Answer:
        """Generate a grounded answer using the retrieved contract chunks."""

        # Start with the system prompt that enforces grounding + citation rules
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        # Inject the last 4 history messages (= 2 full turns: user + assistant each).
        # This gives the model enough context for follow-up questions like
        # "what about the termination clause?" after asking about confidentiality.
        # Capped at 4 to avoid bloating the prompt with the entire session.
        messages.extend(history[-4:])

        # Final user message combines:
        #   - The retrieved contract clauses (as labelled excerpts)
        #   - The actual question being asked
        messages.append({
            "role": "user",
            "content": (
                f"Contract excerpts:\n\n{_format_context(chunks)}\n\n"
                f"Question: {query}"
            ),
        })

        # temperature=0.1 — near-deterministic; slight variation is acceptable for answers
        # max_tokens=600  — enough for a thorough multi-clause answer with citations
        response = chat(messages, temperature=0.1, max_tokens=600)

        # Extract lines that contain [RISK] for structured risk reporting.
        # These are surfaced separately in the Answer object so callers can
        # display them prominently or trigger alerts.
        risk_flags = [line.strip() for line in response.splitlines() if "[RISK]" in line]

        return Answer(response=response, sources=chunks, risk_flags=risk_flags)
