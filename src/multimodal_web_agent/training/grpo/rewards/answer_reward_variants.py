from __future__ import annotations

from dataclasses import asdict, dataclass
import math


LEGACY_V2 = "hierarchical_grounded_search_v2"
ANSWER_DOMINANT_POSITIVE = "answer_dominant_positive"


@dataclass(frozen=True)
class VariantReward:
    mode: str
    answer_quality: float
    answer_dominance_score: float
    grounded_quality: float
    positive_grounding_bonus: float
    negative_shaping_total: float
    terminal_reward: float

    def to_dict(self) -> dict[str, float | str]:
        return asdict(self)


def answer_quality(*, em_v1: float, token_f1_v1: float) -> float:
    return 0.70 * float(em_v1) + 0.30 * float(token_f1_v1)


def score_variant_terminal(
    *, mode: str, em_v1: float, token_f1_v1: float,
    gold_support: float, prediction_support: float,
    legacy_answer_score: float = 0.0, legacy_evidence_use: float = 0.0,
    legacy_missed_evidence: float = 0.0,
    legacy_wrong_span_penalty: float = 0.0,
) -> VariantReward:
    values = (
        em_v1, token_f1_v1, gold_support, prediction_support,
        legacy_answer_score, legacy_evidence_use,
        legacy_missed_evidence, legacy_wrong_span_penalty,
    )
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("non-finite Answer Reward input")
    em = min(max(float(em_v1), 0.0), 1.0)
    f1 = min(max(float(token_f1_v1), 0.0), 1.0)
    gold = min(max(float(gold_support), 0.0), 1.0)
    prediction = min(max(float(prediction_support), 0.0), 1.0)
    quality = answer_quality(em_v1=em, token_f1_v1=f1)
    grounded = quality * gold * prediction
    dominance = 4.0 * em + f1

    if mode == LEGACY_V2:
        positive = 0.25 * float(legacy_evidence_use)
        negative = (
            0.30 * float(legacy_missed_evidence)
            + float(legacy_wrong_span_penalty)
        )
        terminal = float(legacy_answer_score) + positive - negative
        terminal = min(max(terminal, -1.0), 1.0)
    elif mode == ANSWER_DOMINANT_POSITIVE:
        positive = 0.15 * (1.0 - quality) * grounded
        negative = 0.0
        terminal = min(max(quality + positive, 0.0), 1.0)
    else:
        raise ValueError(f"unsupported Answer Reward mode: {mode}")
    return VariantReward(
        mode=mode,
        answer_quality=float(quality),
        answer_dominance_score=float(dominance),
        grounded_quality=float(grounded),
        positive_grounding_bonus=float(positive),
        negative_shaping_total=float(negative),
        terminal_reward=float(terminal),
    )


def effective_text_query_advantage(
    raw_advantage: float, *, enabled: bool, positive_only: bool,
) -> float:
    value = float(raw_advantage)
    if not math.isfinite(value):
        raise ValueError("Text Query Advantage is non-finite")
    if not enabled:
        return 0.0
    return max(value, 0.0) if positive_only else value


def combined_text_advantage(
    *, terminal_advantage: float, query_advantage: float,
    terminal_weight: float, query_weight: float,
) -> float:
    value = (
        float(terminal_weight) * float(terminal_advantage)
        + float(query_weight) * float(query_advantage)
    )
    if not math.isfinite(value):
        raise ValueError("combined Text Search advantage is non-finite")
    return value

