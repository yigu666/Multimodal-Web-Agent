from __future__ import annotations

from collections import Counter, defaultdict
from enum import Enum
from typing import Any, Dict, Iterable, Mapping, Optional


class RejectionReason(str, Enum):
    DUPLICATE_SOURCE = "duplicate_source"
    SPLIT_CONFLICT = "split_conflict"
    ROUTE_INELIGIBLE = "route_ineligible"
    CACHE_MISS = "cache_miss"
    EMPTY_INFORMATION = "empty_information"
    EMPTY_ANSWER = "empty_answer"
    SOURCE_ALREADY_USED = "source_already_used"
    IMAGE_ENTITY_NOT_FOUND = "image_entity_not_found"
    IMAGE_ENTITY_TOO_GENERIC = "image_entity_too_generic"
    RELATION_NOT_FOUND = "relation_not_found"
    ANSWER_ALREADY_IN_IMAGE_INFORMATION = "answer_already_in_image_information"
    EVIDENCE_MISS = "evidence_miss"
    EMPTY_QUERY = "empty_query"
    QUERY_TOO_SHORT = "query_too_short"
    QUERY_TOO_LONG = "query_too_long"
    ANSWER_LEAK = "answer_leak"
    UNAVAILABLE_CONTEXT_LEAK = "unavailable_context_leak"
    TEXT_EVIDENCE_MISS = "text_evidence_miss"
    TEXT_EVIDENCE_ONLY_IMAGE_CONTEXT = "text_evidence_only_image_context"
    PARSER_INVALID = "parser_invalid"
    FORBIDDEN_VISUAL_PLACEHOLDER = "forbidden_visual_placeholder"
    PAIR_INCONSISTENT = "pair_inconsistent"
    CHAIN_INCONSISTENT = "chain_inconsistent"
    SEQUENCE_TOO_LONG = "sequence_too_long"
    TARGET_TRUNCATED = "target_truncated"


def make_rejection(
    source_data_id: str,
    attempted_route: str,
    reason: RejectionReason,
    details: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "source_data_id": str(source_data_id),
        "attempted_route": attempted_route,
        "reason": reason.value,
        "details": dict(details or {}),
    }


def validate_rejection(record: Mapping[str, Any]) -> None:
    if not str(record.get("source_data_id", "")).strip():
        raise ValueError("rejection has no source_data_id")
    if not str(record.get("attempted_route", "")).strip():
        raise ValueError("rejection has no attempted_route")
    RejectionReason(str(record.get("reason", "")))
    if not isinstance(record.get("details"), Mapping):
        raise ValueError("rejection details must be a mapping")


def rejection_distributions(
    records: Iterable[Mapping[str, Any]],
) -> Dict[str, Any]:
    reason_counts: Counter[str] = Counter()
    route_counts: Dict[str, Counter[str]] = defaultdict(Counter)
    total = 0
    for record in records:
        validate_rejection(record)
        total += 1
        reason = str(record["reason"])
        route = str(record["attempted_route"])
        reason_counts[reason] += 1
        route_counts[route][reason] += 1
    return {
        "rejected_attempt_count": total,
        "rejection_reason_distribution": dict(sorted(reason_counts.items())),
        "rejection_reason_distribution_by_route": {
            route: dict(sorted(counts.items()))
            for route, counts in sorted(route_counts.items())
        },
    }
