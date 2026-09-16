from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

from multimodal_web_agent.evaluation.unified_agent.audit_v1_2.answer_equivalence import (
    evaluate_answer_v2,
)

from .config import AnswerWeights


@dataclass(frozen=True)
class AnswerScoreBreakdown:
    em_v1: int
    token_f1_v1: float
    deterministic_equivalence_v2: int
    answer_score: float
    matched_alias: str | None
    equivalence_reasons: tuple[str, ...]

    def to_dict(self) -> dict:
        return asdict(self)


def score_answer(
    prediction: str | None,
    accepted_answers: Iterable[str],
    *,
    question: str,
    weights: AnswerWeights,
) -> AnswerScoreBreakdown:
    result = evaluate_answer_v2(
        prediction,
        accepted_answers,
        question=question,
        count_semantic_as_strict=False,
    )
    deterministic = int(result.em_v2_strict)
    score = (
        weights.em_v1_weight * result.em_v1
        + weights.token_f1_v1_weight * result.token_f1_v1
        + weights.deterministic_equivalence_v2_weight * deterministic
    )
    return AnswerScoreBreakdown(
        em_v1=int(result.em_v1),
        token_f1_v1=float(result.token_f1_v1),
        deterministic_equivalence_v2=deterministic,
        answer_score=min(max(float(score), 0.0), 1.0),
        matched_alias=result.matched_alias,
        equivalence_reasons=tuple(result.equivalence_reasons),
    )
