from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Any, Iterable, Mapping, Sequence

from .config import TextQueryAdvantageWeights, TextQueryWeights
from .evidence_support import SupportMatch, support_match


_RANKED_LINE_RE = re.compile(r"^\s*(\d+)\.\s+(.*?)\s*$")
_DOCUMENT_SUFFIX_RE = re.compile(r"\s*\[cache_document=[^\]]+\]\s*$")


@dataclass(frozen=True)
class RankUtility:
    value: float
    best_rank: int | None
    best_support: SupportMatch
    ranked_supports: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class TextQueryUtility:
    coverage_mask: float
    query_utility_masked: bool
    tool_execution_failure: bool
    actual_rank_utility: float
    question_baseline_rank_utility: float
    query_improvement: float
    text_query_utility: float
    text_query_advantage: float
    actual_rank_details: RankUtility

    def to_dict(self) -> dict:
        return asdict(self)


def extract_ranked_result_texts(information_text: str) -> list[str]:
    rows: list[tuple[int, str]] = []
    for line in str(information_text or "").splitlines():
        match = _RANKED_LINE_RE.match(line)
        if not match:
            continue
        value = _DOCUMENT_SUFFIX_RE.sub("", match.group(2)).strip()
        rows.append((int(match.group(1)), value))
    rows.sort(key=lambda item: item[0])
    return [text for _, text in rows]


def rank_sensitive_utility(
    accepted_answers: Iterable[str],
    returned_results: Sequence[str],
    *,
    question: str = "",
) -> RankUtility:
    best_value = 0.0
    best_rank = None
    best_support = SupportMatch(None, "none", None, 0.0)
    rows = []
    for rank, text in enumerate(returned_results, start=1):
        match = support_match(accepted_answers, text, question=question)
        utility = match.support_score / rank
        rows.append({"rank": rank, "support": match.to_dict(), "rank_utility": utility})
        if utility > best_value:
            best_value = utility
            best_rank = rank
            best_support = match
    return RankUtility(float(best_value), best_rank, best_support, tuple(rows))


def score_text_query_utility(
    *,
    accepted_answers: Iterable[str],
    actual_results: Sequence[str] | None = None,
    actual_information: str = "",
    coverage_mask: float,
    question_baseline_rank_utility: float,
    tool_execution_failure: bool,
    question: str,
    weights: TextQueryWeights,
    advantage_weights: TextQueryAdvantageWeights,
) -> TextQueryUtility:
    results = list(actual_results or extract_ranked_result_texts(actual_information))
    # A failed tool call has no usable retrieval result, even if a backend
    # happened to attach stale/partial rows to the failure payload.  The
    # counterfactual may still be negative through the frozen question-only
    # baseline, exactly as specified by the Reward v2 contract.
    actual = rank_sensitive_utility(
        accepted_answers,
        [] if tool_execution_failure else results,
        question=question,
    )
    baseline = min(max(float(question_baseline_rank_utility), 0.0), 1.0)
    improvement = actual.value - baseline
    masked = float(coverage_mask) <= 0.0
    if tool_execution_failure or masked:
        utility = 0.0
    else:
        utility = (
            weights.actual_rank_weight * actual.value
            + weights.improvement_weight * improvement
        )
        utility = min(max(utility, weights.utility_min), weights.utility_max)
    if masked:
        advantage = 0.0
    else:
        advantage = (
            advantage_weights.improvement_weight * improvement
            + advantage_weights.actual_rank_weight * actual.value
        )
        advantage = min(
            max(advantage, advantage_weights.min_advantage),
            advantage_weights.max_advantage,
        )
    return TextQueryUtility(
        coverage_mask=0.0 if masked else 1.0,
        query_utility_masked=masked,
        tool_execution_failure=bool(tool_execution_failure),
        actual_rank_utility=actual.value,
        question_baseline_rank_utility=baseline,
        query_improvement=float(improvement),
        text_query_utility=float(utility),
        text_query_advantage=float(advantage),
        actual_rank_details=actual,
    )


def cache_row(cache: Mapping[str, Mapping[str, Any]], prompt_id: str) -> Mapping[str, Any]:
    try:
        return cache[str(prompt_id)]
    except KeyError as exc:
        raise KeyError(f"Reward v2 cache has no prompt_id={prompt_id}") from exc
