"""
OrchestratorAgent — the brain of the multi-agent system.

Responsibilities:
  1. Classify query intent (ANSWERABLE vs OUT_OF_SCOPE) before doing any work
  2. Route answerable queries: RetrievalAgent → AnswerAgent
  3. Maintain conversation history for multi-turn follow-up questions
"""
from src.models import Answer
from src.agents.retrieval_agent import RetrievalAgent
from src.agents.answer_agent import AnswerAgent
from src.llm_client import chat

# Prompt for the intent router.
# Two categories only keeps the classification simple and unambiguous.
# temperature=0 + max_tokens=5 makes this call cheap and deterministic.
ROUTER_PROMPT = """\
Classify this legal query as ANSWERABLE or OUT_OF_SCOPE.

ANSWERABLE: questions about contract facts, terms, risks, comparisons, consequences, or
            summaries of contract content (e.g. "summarize all risks", "what happens if X").
OUT_OF_SCOPE: requests to draft/write/compose content, give legal strategy, or predict outcomes.

Reply with exactly one word: ANSWERABLE or OUT_OF_SCOPE
"""

# Standard refusal shown for out-of-scope queries.
# Hard-coded string (not LLM-generated) so it is always safe and consistent.
REFUSAL = (
    "I can analyse existing contracts and identify risks, but drafting documents "
    "or providing legal strategy is outside my scope. Please consult a licensed attorney."
)


class OrchestratorAgent:

    def __init__(self, retrieval: RetrievalAgent, answer: AnswerAgent):
        self.retrieval = retrieval
        self.answer = answer

        # Stores alternating user/assistant messages for multi-turn context.
        # Passed to AnswerAgent so follow-up questions ("what about termination?")
        # are answered correctly without repeating context.
        self.history = []

    def process(self, query: str) -> Answer:
        """
        Full pipeline for one user query:
          1. Route  — is this query answerable from the contracts?
          2. Retrieve — find the most relevant contract clauses
          3. Answer  — generate a grounded response with citations
        """

        # ── Step 1: Intent classification ─────────────────────────────────
        # Check before retrieval so we don't waste tokens on out-of-scope requests
        if self._is_out_of_scope(query):
            # Save to history so the conversation remains coherent
            self._save(query, REFUSAL)
            # is_out_of_scope=True lets the evaluator distinguish refusals from answers
            return Answer(response=REFUSAL, sources=[], risk_flags=[], is_out_of_scope=True)

        # ── Step 2: Semantic retrieval ─────────────────────────────────────
        # Delegates to RetrievalAgent → VectorStore.search() → FAISS
        # Returns top-4 RetrievedChunk objects ranked by cosine similarity
        chunks = self.retrieval.retrieve(query)

        # ── Step 3: Grounded answer generation ────────────────────────────
        # AnswerAgent receives chunks + history so it can cite clauses and
        # handle follow-up questions from earlier in the conversation
        ans = self.answer.answer(query, chunks, self.history)

        # Save this turn so future queries can reference it
        self._save(query, ans.response)

        return ans

    def _is_out_of_scope(self, query: str) -> bool:
        """
        Ask the LLM to classify the query as ANSWERABLE or OUT_OF_SCOPE.
        temperature=0  → same answer every time for the same input
        max_tokens=5   → we only need one word ("ANSWERABLE" or "OUT_OF_SCOPE")
        """
        result = chat(
            messages=[
                {"role": "system", "content": ROUTER_PROMPT},
                {"role": "user",   "content": query},
            ],
            temperature=0,
            max_tokens=5,
        )

        # .upper() makes the check case-insensitive in case model varies casing
        return "OUT_OF_SCOPE" in result.upper()

    def _save(self, query: str, response: str) -> None:
        """Append one full conversation turn (user + assistant) to history."""
        self.history.append({"role": "user",      "content": query})
        self.history.append({"role": "assistant",  "content": response})
