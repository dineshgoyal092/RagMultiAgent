"""
Labeled Dataset Evaluator

Evaluates system against ground-truth labeled queries on 4 metrics:

  Answer Match     — expected keywords present in response
  Citation Match   — correct document(s) cited
  Risk Flag Match  — [RISK] present/absent as expected
  Refusal Match    — system refused when it should (or answered when it should)

Unlike LLM-as-judge, these are deterministic checks against human-labeled ground truth.
"""
import json
import os
from dataclasses import dataclass, field
from typing import Optional
from src.agents.orchestrator import OrchestratorAgent
from src.models import Answer


@dataclass
class LabeledCase:
    id: str
    category: str
    query: str
    expected_answer_contains: list
    expected_citations: list             # doc names that must appear in sources
    expected_sections: list              # section numbers that should appear in answer
    expect_risk: bool
    expect_refusal: bool
    ground_truth: str


@dataclass
class CaseResult:
    case: LabeledCase
    answer: Answer
    answer_match: bool
    citation_match: bool
    risk_match: bool
    refusal_match: bool

    @property
    def fully_passed(self) -> bool:
        return self.answer_match and self.citation_match and self.risk_match and self.refusal_match

    @property
    def score(self) -> float:
        """Partial credit: fraction of 4 checks that passed."""
        return sum([self.answer_match, self.citation_match,
                    self.risk_match, self.refusal_match]) / 4.0


def _load_dataset(path: str) -> list[LabeledCase]:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return [LabeledCase(**r) for r in raw]


def _check_answer_match(answer: Answer, case: LabeledCase) -> bool:
    """All expected keywords must appear (case-insensitive) in the response."""
    if not case.expected_answer_contains:
        return True
    text = answer.response.lower()
    return all(kw.lower() in text for kw in case.expected_answer_contains)


def _check_citation_match(answer: Answer, case: LabeledCase) -> bool:
    """All expected doc names must appear in the retrieved sources."""
    if not case.expected_citations:
        return True  # no citation expected (hallucination / out-of-scope cases)
    cited_docs = {rc.chunk.doc_name.lower() for rc in answer.sources}
    return all(doc.lower() in cited_docs for doc in case.expected_citations)


def _check_risk_match(answer: Answer, case: LabeledCase) -> bool:
    """[RISK] must be present if expected, absent if not expected."""
    has_risk = "[RISK]" in answer.response or bool(answer.risk_flags)
    return has_risk == case.expect_risk


def _check_refusal_match(answer: Answer, case: LabeledCase) -> bool:
    """Refusal/out-of-scope handling matches expectation."""
    refused = answer.is_out_of_scope or any(
        phrase in answer.response.lower()
        for phrase in ["not specified in the provided", "outside my scope",
                       "not able", "cannot", "outside", "not found in"]
    )
    return refused == case.expect_refusal


def evaluate_case(orch: OrchestratorAgent, case: LabeledCase) -> CaseResult:
    orch.history = []  # fresh context per case
    answer = orch.process(case.query)
    return CaseResult(
        case=case,
        answer=answer,
        answer_match=_check_answer_match(answer, case),
        citation_match=_check_citation_match(answer, case),
        risk_match=_check_risk_match(answer, case),
        refusal_match=_check_refusal_match(answer, case),
    )


def run_labeled_eval(orch: OrchestratorAgent, dataset_path: "Optional[str]" = None) -> None:
    if dataset_path is None:
        dataset_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "data", "labeled_dataset.json"
        )

    cases = _load_dataset(dataset_path)

    print("\n" + "=" * 70)
    print("  LABELED DATASET EVALUATION")
    print(f"  {len(cases)} cases  |  4 metrics per case: Answer / Citation / Risk / Refusal")
    print("=" * 70)

    results: list[CaseResult] = []
    by_category: dict[str, list[CaseResult]] = {}

    for case in cases:
        r = evaluate_case(orch, case)
        results.append(r)
        by_category.setdefault(case.category, []).append(r)

        a = "A" if r.answer_match   else "."
        c = "C" if r.citation_match else "."
        k = "R" if r.risk_match     else "."
        f = "F" if r.refusal_match  else "."
        status = "PASS" if r.fully_passed else "FAIL"

        print(f"\n[{status}] {case.id:<6} [{case.category:<12}] [{a}{c}{k}{f}]  {case.query[:60]}{'...' if len(case.query)>60 else ''}")
        if not r.fully_passed:
            if not r.answer_match:
                missing = [kw for kw in case.expected_answer_contains
                           if kw.lower() not in r.answer.response.lower()]
                print(f"         !! Answer missing: {missing}")
            if not r.citation_match:
                cited = [rc.chunk.doc_name for rc in r.answer.sources]
                print(f"         !! Citation expected {case.expected_citations}, got {cited}")
            if not r.risk_match:
                print(f"         !! Risk expected={case.expect_risk}, got={bool(r.answer.risk_flags)}")
            if not r.refusal_match:
                print(f"         !! Refusal expected={case.expect_refusal}")

    # ── Per-category summary ───────────────────────────────────────────────
    print("\n" + "-" * 70)
    print(f"  {'Category':<14} {'Cases':>5} {'Pass':>5} {'AnswerMatch':>12} {'CitationMatch':>14} {'RiskMatch':>10} {'RefusalMatch':>12}")
    print("-" * 70)

    total_pass = 0
    for cat, cat_results in by_category.items():
        n = len(cat_results)
        passed = sum(1 for r in cat_results if r.fully_passed)
        total_pass += passed
        am = sum(1 for r in cat_results if r.answer_match)
        cm = sum(1 for r in cat_results if r.citation_match)
        rm = sum(1 for r in cat_results if r.risk_match)
        fm = sum(1 for r in cat_results if r.refusal_match)
        print(f"  {cat:<14} {n:>5} {passed:>5}    {am}/{n}{'':>8} {cm}/{n}{'':>10} {rm}/{n}{'':>6} {fm}/{n}")

    print("-" * 70)
    n_total = len(results)
    print(f"  {'TOTAL':<14} {n_total:>5} {total_pass:>5}    "
          f"{sum(r.answer_match for r in results)}/{n_total}{'':>8} "
          f"{sum(r.citation_match for r in results)}/{n_total}{'':>10} "
          f"{sum(r.risk_match for r in results)}/{n_total}{'':>6} "
          f"{sum(r.refusal_match for r in results)}/{n_total}")

    avg_score = sum(r.score for r in results) / len(results)
    print(f"\n  Overall Pass Rate : {total_pass}/{n_total}  ({100*total_pass/n_total:.0f}%)")
    print(f"  Avg Partial Score : {avg_score:.2f}/1.0")

    # ── Legend ─────────────────────────────────────────────────────────────
    print("\n  Legend: A=AnswerMatch  C=CitationMatch  R=RiskMatch  F=RefusalMatch")
    print("=" * 70 + "\n")
