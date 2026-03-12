"""
LLM-as-judge evaluation.

Scores each answer on two criteria (0-3):
  Relevance     — does the answer address the question?
  Groundedness  — is the answer supported by document citations?
"""
from src.llm_client import chat

JUDGE_PROMPT = """\
Rate this legal Q&A on two criteria (0-3 each):

Relevance:    Does the answer directly address the question?
Groundedness: Is the answer supported by document citations or quoted clauses?

Question: {question}
Answer:   {answer}

Reply in this exact format:
Relevance: <0-3>
Groundedness: <0-3>
Reason: <one sentence>
"""


def evaluate(question: str, answer_text: str) -> dict:
    """Score one answer. Returns {relevance, groundedness, reason}."""
    response = chat(
        messages=[{"role": "user", "content": JUDGE_PROMPT.format(
            question=question, answer=answer_text
        )}],
        temperature=0,
        max_tokens=80,
    )
    scores = {"relevance": 0, "groundedness": 0, "reason": ""}
    for line in response.splitlines():
        if line.startswith("Relevance:"):
            try: scores["relevance"] = int(line.split(":")[1].strip()[0])
            except: pass
        elif line.startswith("Groundedness:"):
            try: scores["groundedness"] = int(line.split(":")[1].strip()[0])
            except: pass
        elif line.startswith("Reason:"):
            scores["reason"] = line.split(":", 1)[1].strip()
    return scores


def run_eval(orchestrator, queries: list) -> None:
    """Run evaluation over a list of queries and print results."""
    print("\n" + "=" * 60)
    print("  EVALUATION RESULTS")
    print("=" * 60)

    total_r, total_g, count = 0, 0, 0

    for query in queries:
        orchestrator.history = []
        ans = orchestrator.process(query)

        if ans.is_out_of_scope:
            print(f"\nQ: {query[:65]}")
            print("   -> Correct out-of-scope refusal")
            continue

        s = evaluate(query, ans.response)
        total_r += s["relevance"]
        total_g += s["groundedness"]
        count += 1

        print(f"\nQ: {query[:65]}")
        print(f"   Relevance: {s['relevance']}/3  |  Groundedness: {s['groundedness']}/3")
        print(f"   {s['reason']}")

    if count:
        print("\n" + "-" * 60)
        print(f"  Avg Relevance:     {total_r/count:.1f}/3")
        print(f"  Avg Groundedness:  {total_g/count:.1f}/3")
    print("=" * 60 + "\n")
