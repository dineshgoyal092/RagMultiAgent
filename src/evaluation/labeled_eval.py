"""
Labeled evaluation — checks system answers against 28 ground-truth cases.

4 checks per case: AnswerMatch, CitationMatch, RiskMatch, RefusalMatch.
"""
import json
import os
from dataclasses import dataclass
from typing import Optional
from src.agents.orchestrator import OrchestratorAgent
from src.models import Answer


@dataclass
class LabeledCase:
    id: str
    category: str
    query: str
    expected_answer_contains: list
    expected_citations: list
    expected_sections: list
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
    def fully_passed(self):
        return self.answer_match and self.citation_match and self.risk_match and self.refusal_match

    @property
    def score(self):
        return sum([self.answer_match, self.citation_match,
                    self.risk_match, self.refusal_match]) / 4.0


def _load_dataset(path):
    with open(path, encoding="utf-8") as f:
        return [LabeledCase(**r) for r in json.load(f)]


def _check_answer_match(answer, case):
    if not case.expected_answer_contains:
        return True
    text = answer.response.lower()
    return all(kw.lower() in text for kw in case.expected_answer_contains)


def _check_citation_match(answer, case):
    if not case.expected_citations:
        return True
    cited = {rc.chunk.doc_name.lower() for rc in answer.sources}
    return all(doc.lower() in cited for doc in case.expected_citations)


def _check_risk_match(answer, case):
    has_risk = "[RISK]" in answer.response or bool(answer.risk_flags)
    return has_risk == case.expect_risk


def _check_refusal_match(answer, case):
    refused = answer.is_out_of_scope or any(
        phrase in answer.response.lower()
        for phrase in [
            "not specified in the provided",
            "outside my scope",
            "please consult a licensed",
        ]
    )
    return refused == case.expect_refusal


def evaluate_case(orch, case):
    orch.history = []
    answer = orch.process(case.query)
    return CaseResult(
        case=case, answer=answer,
        answer_match=_check_answer_match(answer, case),
        citation_match=_check_citation_match(answer, case),
        risk_match=_check_risk_match(answer, case),
        refusal_match=_check_refusal_match(answer, case),
    )


def run_labeled_eval(orch, dataset_path=None):
    if dataset_path is None:
        dataset_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "data", "labeled_dataset.json"
        )

    cases = _load_dataset(dataset_path)
    results = [evaluate_case(orch, case) for case in cases]

    for r in results:
        status = "PASS" if r.fully_passed else "FAIL"
        print(f"[{status}] {r.case.id} — {r.case.query[:55]}")

    passed = sum(1 for r in results if r.fully_passed)
    avg = sum(r.score for r in results) / len(results)
    print(f"\nPass: {passed}/{len(results)} | Avg Score: {avg:.2f}")
