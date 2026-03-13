"""
Legal Contract Analysis — Multi-Agent RAG System

Usage:
    python main.py            # interactive chat
    python main.py --eval     # LLM-as-judge evaluation
    python main.py --labeled  # ground-truth labeled evaluation
"""
import sys
from src import config
from src.rag.chunker import load_and_chunk
from src.rag.retriever import VectorStore
from src.agents.retrieval_agent import RetrievalAgent
from src.agents.answer_agent import AnswerAgent
from src.agents.orchestrator import OrchestratorAgent

EVAL_QUERIES = [
    "What is the notice period for terminating the NDA?",
    "What is the uptime commitment in the SLA?",
    "Which law governs the Vendor Services Agreement?",
    "Do confidentiality obligations survive termination of the NDA?",
    "Is liability capped for breach of confidentiality?",
    "What remedies are available if the SLA uptime is not met?",
    "Is Vendor XYZ's liability capped for data breaches?",
    "Which agreement governs data breach notification timelines?",
    "Are there conflicting governing laws across agreements?",
    "Are there any legal risks related to liability exposure?",
    "Identify any clauses that could pose financial risk to Acme Corp.",
    "Is there any unlimited liability in these agreements?",
    "Can Vendor XYZ share Acme's confidential data with subcontractors?",
    "What happens if Vendor delays breach notification beyond 72 hours?",
    "Summarize all risks for Acme Corp in one paragraph.",
    "Can you draft a better NDA for me?",
    "What legal strategy should Acme take against Vendor XYZ?",
]


def build_system() -> OrchestratorAgent:
    """Load documents, build vector index, wire up agents."""
    print("Loading documents...", end=" ", flush=True)
    chunks = load_and_chunk(config.DATA_DIR)
    print(f"{len(chunks)} chunks loaded")

    print("Building vector index...", end=" ", flush=True)
    store = VectorStore(chunks)
    print("ready\n")

    return OrchestratorAgent(
        retrieval=RetrievalAgent(store),
        answer=AnswerAgent(),
    )


def display(answer) -> None:
    """Print the answer, sources, and any risk flags."""
    print(f"\n{answer.response}")

    if answer.sources:
        seen = set()
        lines = []
        for rc in answer.sources:
            key = f"{rc.chunk.doc_name} | {rc.chunk.section}"
            if key not in seen:
                lines.append(key)
                seen.add(key)
        print("\nSources: " + "  /  ".join(lines))


def main() -> None:
    print("=" * 55)
    print("  Legal Contract Analysis  -  Multi-Agent RAG")
    print("=" * 55)

    orch = build_system()

    if "--eval" in sys.argv:
        from src.evaluation.evaluator import run_eval
        run_eval(orch, EVAL_QUERIES)
        return

    if "--labeled" in sys.argv:
        from src.evaluation.labeled_eval import run_labeled_eval
        run_labeled_eval(orch)
        return

    print("Ask questions about the contracts. Type 'quit' to exit.\n")

    while True:
        try:
            query = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye.")
            break

        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            print("Goodbye.")
            break

        display(orch.process(query))
        print()


if __name__ == "__main__":
    main()
