from __future__ import annotations

from enum import Enum
from typing import Any, Dict, Iterable, Mapping


class SharedRejectionReason(str, Enum):
    NO_VALID_ACTION = "no_valid_action"
    UNRESOLVABLE_VISUAL_REFERENCE = "unresolvable_visual_reference"
    MISSING_VISIBLE_ENTITY = "missing_visible_entity"
    GENERIC_QUERY = "generic_query"
    QUESTION_COPY_QUERY = "question_copy_query"
    CONFLICTING_ROUTE_LABEL = "conflicting_route_label"
    AMBIGUOUS_CANONICAL_ACTION = "ambiguous_canonical_action"
    UNREACHABLE_EVIDENCE = "unreachable_evidence"
    IMAGE_CACHE_MISS = "image_cache_miss"
    EMPTY_INFORMATION = "empty_information"
    ANSWER_UNSUPPORTED = "answer_unsupported"
    ANSWER_ALREADY_VISIBLE = "answer_already_visible"
    QUERY_ANSWER_LEAK = "query_answer_leak"
    ENTITY_NOT_VISIBLE = "entity_not_visible"
    SAME_CONTEXT_EVIDENCE_ONLY = "same_context_evidence_only"
    DUPLICATE_ENTITY_GROUP = "duplicate_entity_group"
    NEAR_DUPLICATE_GROUP = "near_duplicate_group"
    SPLIT_LEAK_RISK = "split_leak_risk"
    STRICT_PARSER_INVALID = "strict_parser_invalid"


def reason_values(reasons: Iterable[object]) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                reason.value if isinstance(reason, Enum) else str(reason)
                for reason in reasons
                if str(reason)
            }
        )
    )


def rejection_record(
    *,
    source_data_id: str,
    attempted_route: str,
    state_type: str,
    reasons: Iterable[object],
    question: str,
    visible_text_context: str = "",
    target_query: str = "",
    valid_action_set: Iterable[str] = (),
    metadata: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    return {
        "source_data_id": source_data_id,
        "stage": "shared_executability_gate",
        "attempted_route": attempted_route,
        "state_type": state_type,
        "decision": "reject",
        "rejection_reasons": list(reason_values(reasons)),
        "question": question,
        "visible_text_context": visible_text_context,
        "target_query": target_query,
        "valid_action_set": sorted(set(valid_action_set)),
        "repair_mode": "reject_only",
        "metadata": dict(metadata or {}),
    }
