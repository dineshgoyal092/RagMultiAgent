"""
OrchestratorAgent — the brain of the system.

Responsibilities:
  1. Classify query intent: ANSWERABLE or OUT_OF_SCOPE
  2. Route to RetrievalAgent + AnswerAgent for answerable queries
  3. Maintain multi-turn conversation history
"""
from src.models import Answer
from src.agents.retrieval_agent import RetrievalAgent
from src.agents.answer_agent import AnswerAgent
from src.llm_client import chat

ROUTER_PROMPT = """\
Classify this legal query into one of two categories:

ANSWERABLE   — asks what a contract says, compares clauses, or identifies risks.
OUT_OF_SCOPE — asks to draft/rewrite documents, give legal strategy, or predict outcomes.

Reply with exactly one word: ANSWERABLE or OUT_OF_SCOPE
"""

REFUSAL = (
    "I can analyse existing contracts and identify risks, but drafting documents "
    "or providing legal strategy is outside my scope. Please consult a licensed attorney."
)


class OrchestratorAgent:
    def __init__(self, retrieval: RetrievalAgent, answer: AnswerAgent):
        self.retrieval = retrieval
        self.answer = answer
        self.history = []   # conversation history [{"role": ..., "content": ...}]

    def process(self, query: str) -> Answer:
        """Process one user query and return an Answer."""

        # Step 1: classify intent
        if self._is_out_of_scope(query):
            self._save(query, REFUSAL)
            return Answer(response=REFUSAL, sources=[], risk_flags=[], is_out_of_scope=True)

        # Step 2: retrieve relevant clauses
        chunks = self.retrieval.retrieve(query)

        # Step 3: generate grounded answer
        ans = self.answer.answer(query, chunks, self.history)
        self._save(query, ans.response)
        return ans

    def _is_out_of_scope(self, query: str) -> bool:
        result = chat(
            messages=[
                {"role": "system", "content": ROUTER_PROMPT},
                {"role": "user",   "content": query},
            ],
            temperature=0,
            max_tokens=5,
        )
        return "OUT_OF_SCOPE" in result.upper()

    def _save(self, query: str, response: str) -> None:
        self.history.append({"role": "user",      "content": query})
        self.history.append({"role": "assistant",  "content": response})
