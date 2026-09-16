from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

from .answer_normalizer import normalize_answer
from .cache_reader import CacheEntry
from .entity_extractor import clean_search_title
from .verifier import information_supports_answer


@dataclass(frozen=True)
class ImageEvidenceScore:
    top1_exact_alias: bool
    top3_exact_alias_count: int
    top1_normalized_contains_alias: bool
    shortest_support_rank: Optional[int]
    supporting_title_count: int
    support_ranks: Tuple[int, ...]
    total_score: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _exact_alias_match(title: str, accepted_answers: Sequence[str]) -> bool:
    normalized_title = normalize_answer(clean_search_title(title))
    aliases = {
        normalize_answer(answer)
        for answer in accepted_answers
        if normalize_answer(answer)
    }
    return bool(normalized_title) and normalized_title in aliases


def score_image_evidence(
    entry: CacheEntry,
    accepted_answers: Sequence[str],
    *,
    top_k: int = 3,
) -> ImageEvidenceScore:
    """Score answer-bearing image-result titles without changing cache order."""

    if top_k < 1:
        raise ValueError("top_k must be positive")
    selected = entry.usable_image_results[:top_k]
    support_ranks = tuple(
        rank
        for rank, (_index, title, _descriptor) in enumerate(selected, start=1)
        if information_supports_answer(title, accepted_answers)
    )
    exact_count = sum(
        _exact_alias_match(title, accepted_answers)
        for _index, title, _descriptor in selected
    )
    top1_title = selected[0][1] if selected else ""
    top1_contains = bool(
        top1_title
        and information_supports_answer(top1_title, accepted_answers)
    )
    top1_exact = bool(
        top1_title and _exact_alias_match(top1_title, accepted_answers)
    )
    total_score = 0
    if 1 in support_ranks:
        total_score += 4
    total_score += 2 * sum(rank in {2, 3} for rank in support_ranks)
    if len(support_ranks) > 1:
        total_score += 1
    return ImageEvidenceScore(
        top1_exact_alias=top1_exact,
        top3_exact_alias_count=exact_count,
        top1_normalized_contains_alias=top1_contains,
        shortest_support_rank=min(support_ranks) if support_ranks else None,
        supporting_title_count=len(support_ranks),
        support_ranks=support_ranks,
        total_score=total_score,
    )
