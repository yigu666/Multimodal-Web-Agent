from __future__ import annotations

from collections import Counter
import re
from typing import Any, Mapping, Sequence


LABEL_MAPPING_RULE_VERSION = "unified-agent-eval-v1-1-release-selection"
_KNOWLEDGE_QUESTION = re.compile(
    r"\b(when|what year|which year|histor|recent|latest|current|"
    r"how many|how much|number of|collaborat|organization|organises?|"
    r"organizes?|closed|opened|started|finished|damaged|component|"
    r"technical advisor|valuation|paper|figure|purpose|known for|"
    r"diet|eat|located|country|continent)\b",
    re.IGNORECASE,
)


def source_split(example: Mapping[str, Any]) -> str:
    metadata = example.get("source_metadata") or {}
    if metadata.get("source_split"):
        return str(metadata["source_split"])
    if example.get("source_dataset") == "fvqa_test":
        return "historical_fvqa_test"
    if example.get("source_dataset") == "mmsearch":
        return "mmsearch_source_unspecified"
    return "unknown"


def task_label_provenance(
    examples: Sequence[Mapping[str, Any]],
    assignments: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    by_id = {str(row["eval_id"]): row for row in assignments}
    records = []
    suspects = []
    for example in examples:
        episode_id = str(example["eval_id"])
        assignment = by_id.get(episode_id)
        if assignment is None:
            raise RuntimeError("Dev example is missing selection assignment")
        if assignment.get("task_type") != example.get("task_type"):
            raise RuntimeError("selection assignment task type mismatch")
        metadata = example.get("source_metadata") or {}
        declared = list(metadata.get("declared_eligible_task_types") or [])
        raw_label = metadata.get("raw_task_label")
        reasons = []
        if example["task_type"] == "search_free" and _KNOWLEDGE_QUESTION.search(
            str(example["question"])
        ):
            reasons.append("search_free_question_has_external_knowledge_relation")
        if example["task_type"] == "search_free" and declared:
            reasons.append("source_declares_search_eligible_task_type")
        if metadata.get("task_type_requires_manual_review") is True:
            reasons.append("source_metadata_marks_task_type_for_manual_review")
        suspect = bool(reasons)
        recommendation = "keep"
        if suspect:
            if any(reason in reasons for reason in (
                "search_free_question_has_external_knowledge_relation",
                "source_declares_search_eligible_task_type",
            )):
                recommendation = "exclude_from_route_metric"
            elif metadata.get("task_type_requires_manual_review") is True:
                recommendation = "manual_review_required"
            else:
                recommendation = "derive_with_new_rule_later"
        record = {
            "episode_id": episode_id,
            "source_dataset": example["source_dataset"],
            "source_split": source_split(example),
            "raw_task_label": raw_label if raw_label is not None else "unknown",
            "declared_eligible_task_types": declared,
            "mapped_task_type": example["task_type"],
            "search_required": bool(example["search_required"]),
            "mapping_rule": (
                "search task eligibility is derived from answer reachability in "
                "truncated frozen evidence; exact maximum-flow assigns search "
                "roles; unassigned candidates form the search-free pool; "
                "search_required is mapped_task_type != search_free"
            ),
            "mapping_rule_version": LABEL_MAPPING_RULE_VERSION,
            "label_confidence": "derived",
            "original_source_truth_available": raw_label is not None,
            "selection_seed": assignment.get("selection_seed"),
            "assignment_source": (
                "reconstructed_from_frozen_dev_and_audited_v1_1_release_code"
                if assignment.get("assignment_reconstructed_from_frozen_dev")
                else "selection_assignment_record"
            ),
            "suspect": suspect,
            "suspect_reasons": reasons,
            "suggested_handling": recommendation,
        }
        records.append(record)
        if suspect:
            suspects.append({
                **record,
                "question": example["question"],
                "accepted_answers": list(example["answer_aliases"]),
                "why_suspicious": "; ".join(reasons),
            })
    summary = {
        "schema_version": "unified-agent-eval-v1-2-task-label-audit-v1",
        "episode_count": len(records),
        "provenance": records,
        "label_confidence_counts": dict(Counter(
            row["label_confidence"] for row in records
        )),
        "suspect_case_count": len(suspects),
        "suspect_by_task_type": dict(Counter(
            row["mapped_task_type"] for row in suspects
        )),
        "suggested_handling_counts": dict(Counter(
            row["suggested_handling"] for row in suspects
        )),
        "formal_labels_modified": False,
        "route_metrics_should_treat_labels_as_source_truth": False,
    }
    return summary, suspects
