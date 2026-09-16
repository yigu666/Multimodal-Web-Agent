from __future__ import annotations

from .evidence_support import support


def image_marginal_evidence_gain(
    accepted_answers,
    information_before: str,
    information_after: str,
    *,
    question: str = "",
) -> float:
    before = support(accepted_answers, information_before, question=question)
    after = support(accepted_answers, information_after, question=question)
    return min(max(after - before, 0.0), 1.0)
