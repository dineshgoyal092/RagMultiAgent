"""
LLM-as-judge evaluation.

Scores each answer on two criteria (0-3):
  Relevance     — does the answer address the question?
  Groundedness  — is the answer supported by document citations?

WHY LLM-as-judge?
  Writing exact ground-truth for open-ended Q&A is expensive.
  A second LLM call can assess quality at scale without human labellers.
  Trade-off: scores can vary slightly between runs — complement with labeled_eval.py
  for deterministic checks.
"""
from src.llm_client import chat

# Structured prompt forces the judge to return parseable scores.
# Using {question} and {answer} placeholders filled at call time.
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
    """
    Score one Q&A pair using the LLM judge.
    Returns dict: {relevance: int, groundedness: int, reason: str}
    """

    # temperature=0 → deterministic scores, same result on repeated runs
    # max_tokens=80 → judge only needs 3 lines, no need for a long response
    response = chat(
        messages=[{"role": "user", "content": JUDGE_PROMPT.format(
            question=question, answer=answer_text
        )}],
        temperature=0,
        max_tokens=80,
    )

    # Default scores in case the model doesn't follow the format exactly
    scores = {"relevance": 0, "groundedness": 0, "reason": ""}

    # Parse each line of the structured response
    for line in response.splitlines():
        if line.startswith("Relevance:"):
            # e.g. "Relevance: 3" → take the first character after the colon
            try: scores["relevance"] = int(line.split(":")[1].strip()[0])
            except: pass  # leave default 0 if parsing fails

        elif line.startswith("Groundedness:"):
            try: scores["groundedness"] = int(line.split(":")[1].strip()[0])
            except: pass

        elif line.startswith("Reason:"):
            # split(":", 1) keeps colons in the reason text intact
            scores["reason"] = line.split(":", 1)[1].strip()

    return scores


def run_eval(orchestrator, queries: list) -> None:
    """Run LLM-as-judge evaluation over a list of queries and print results."""

    print("\n" + "=" * 60)
    print("  EVALUATION RESULTS")
    print("=" * 60)

    total_r, total_g, count = 0, 0, 0

    for query in queries:
        # Clear history between queries so each is evaluated independently
        orchestrator.history = []
        ans = orchestrator.process(query)

        # Out-of-scope refusals are correct behaviour — report but don't score them
        if ans.is_out_of_scope:
            print(f"\nQ: {query[:65]}")
            print("   -> Correct out-of-scope refusal")
            continue

        # Score the answer with the LLM judge
        s = evaluate(query, ans.response)
        total_r += s["relevance"]
        total_g += s["groundedness"]
        count += 1

        print(f"\nQ: {query[:65]}")
        print(f"   Relevance: {s['relevance']}/3  |  Groundedness: {s['groundedness']}/3")
        print(f"   {s['reason']}")

    # Print averages only if at least one answerable query was scored
    if count:
        print("\n" + "-" * 60)
        print(f"  Avg Relevance:     {total_r/count:.1f}/3")
        print(f"  Avg Groundedness:  {total_g/count:.1f}/3")
    print("=" * 60 + "\n")
