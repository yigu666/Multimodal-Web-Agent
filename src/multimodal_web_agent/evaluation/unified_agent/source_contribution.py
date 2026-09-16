from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .capacity_assignment import SEARCH_TASK_TYPES, solve_joint_capacity
from .source_adapters.base import SourceCandidate, SourceScan


def _overlaps(
    candidate_ids: set[str],
    eligibility: Mapping[str, Sequence[str]],
) -> dict[str, int]:
    patterns = {
        "visual_only_count": {"visual_search_required"},
        "text_only_count": {"text_search_required"},
        "mixed_only_count": {"mixed_search_required"},
        "visual_text_overlap_count": {
            "visual_search_required", "text_search_required"
        },
        "visual_mixed_overlap_count": {
            "visual_search_required", "mixed_search_required"
        },
        "text_mixed_overlap_count": {
            "text_search_required", "mixed_search_required"
        },
        "all_three_overlap_count": set(SEARCH_TASK_TYPES),
    }
    result = {key: 0 for key in patterns}
    for candidate_id in candidate_ids:
        values = set(eligibility.get(candidate_id, ()))
        for field, pattern in patterns.items():
            if values == pattern:
                result[field] += 1
                break
    return result


def build_source_contribution(
    scans: Sequence[SourceScan],
    *,
    post_dedup_candidates: Sequence[SourceCandidate],
    eligibility: Mapping[str, Sequence[str]],
    target_per_type: int,
    expansion_source_names: set[str],
    leakage_rejected_ids: set[str] | None = None,
    duplicate_rejected_ids: set[str] | None = None,
    image_near_duplicate_rejected_ids: set[str] | None = None,
) -> dict[str, Any]:
    surviving = {item.candidate_key for item in post_dedup_candidates}
    leakage_rejected_ids = leakage_rejected_ids or set()
    duplicate_rejected_ids = duplicate_rejected_ids or set()
    image_near_duplicate_rejected_ids = (
        image_near_duplicate_rejected_ids or set()
    )
    groups = []
    for scan in scans:
        all_source_ids = {
            item.candidate_key for item in scan.candidates
        }
        source_ids = {
            item.candidate_key for item in scan.candidates
            if item.candidate_key in surviving
        }
        eligible_counts = {
            task_type: sum(
                task_type in eligibility.get(candidate_id, ())
                for candidate_id in source_ids
            )
            for task_type in SEARCH_TASK_TYPES
        }
        acquisition = scan.inventory.get("acquisition") or {}
        groups.append({
            "source_name": scan.source_name,
            "source_status": scan.inventory.get(
                "source_status",
                scan.inventory.get("status", "available"),
            ),
            "is_expansion_source": (
                scan.source_name in expansion_source_names
            ),
            "candidate_ids": source_ids,
            "raw_downloaded_records": (
                acquisition.get("raw_records")
                if acquisition else scan.raw_record_count
                if scan.raw_record_count is not None
                else scan.candidate_count
            ),
            "schema_parsed": (
                acquisition.get(
                    "schema_parsed", acquisition.get("raw_records")
                )
                if acquisition else scan.raw_record_count
                if scan.raw_record_count is not None
                else scan.candidate_count
            ),
            "normalized_records": (
                acquisition.get("accepted")
                if acquisition else scan.candidate_count
            ),
            "accepted_before_unified_filtering": (
                acquisition.get("accepted")
                if acquisition else scan.candidate_count
            ),
            "hard_rejected": (
                acquisition.get("rejected")
                if acquisition else len(scan.hard_rejected_rows)
            ),
            "leakage_rejected": len(
                all_source_ids & leakage_rejected_ids
            ),
            "duplicate_rejected": len(
                all_source_ids & duplicate_rejected_ids
            ),
            "image_near_duplicates_rejected": len(
                all_source_ids & image_near_duplicate_rejected_ids
            ),
            "quarantined": int(
                acquisition.get("quarantined", 0)
                if acquisition else scan.inventory.get("quarantined", 0)
            ),
            "accepted": len(source_ids),
            "visual_eligible": eligible_counts[
                "visual_search_required"
            ],
            "text_eligible": eligible_counts[
                "text_search_required"
            ],
            "mixed_eligible": eligible_counts[
                "mixed_search_required"
            ],
        })
    return build_source_contribution_from_groups(
        groups,
        eligibility=eligibility,
        target_per_type=target_per_type,
    )


def build_source_contribution_from_groups(
    groups: Sequence[Mapping[str, Any]],
    *,
    eligibility: Mapping[str, Sequence[str]],
    target_per_type: int,
) -> dict[str, Any]:
    cumulative: dict[str, Sequence[str]] = {}
    rows = []
    for group in groups:
        source_ids = set(group.get("candidate_ids", ()))
        before = solve_joint_capacity(
            cumulative, target_per_type=target_per_type
        )
        cumulative.update({
            candidate_id: eligibility[candidate_id]
            for candidate_id in sorted(source_ids)
            if candidate_id in eligibility
        })
        after = solve_joint_capacity(
            cumulative, target_per_type=target_per_type
        )
        source_search_ids = source_ids & set(eligibility)
        row = {
            "source": group["source_name"],
            "source_name": group["source_name"],
            "is_expansion_source": bool(
                group.get("is_expansion_source")
            ),
            "source_status": group.get(
                "source_status", "available"
            ),
            "new_unique_candidates": len(source_ids),
            "new_search_union_candidates": len(source_search_ids),
            "raw_downloaded_records": int(
                group.get("raw_downloaded_records", len(source_ids))
            ),
            "schema_parsed": int(
                group.get(
                    "schema_parsed",
                    group.get("raw_downloaded_records", len(source_ids)),
                )
            ),
            "normalized_records": int(
                group.get("normalized_records", len(source_ids))
            ),
            "hard_rejected": int(group.get("hard_rejected", 0)),
            "leakage_rejected": int(
                group.get("leakage_rejected", 0)
            ),
            "duplicate_rejected": int(
                group.get("duplicate_rejected", 0)
            ),
            "image_near_duplicates_rejected": int(
                group.get("image_near_duplicates_rejected", 0)
            ),
            "quarantined": int(group.get("quarantined", 0)),
            "accepted": int(group.get("accepted", len(source_ids))),
            "visual_eligible": int(
                group.get("visual_eligible", 0)
            ),
            "text_eligible": int(group.get("text_eligible", 0)),
            "mixed_eligible": int(group.get("mixed_eligible", 0)),
            **_overlaps(source_search_ids, eligibility),
            "joint_capacity_before": before.total_assignable,
            "joint_capacity_after": after.total_assignable,
            "joint_capacity_before_source": before.total_assignable,
            "joint_capacity_after_source": after.total_assignable,
            "joint_capacity_gain": (
                after.total_assignable - before.total_assignable
            ),
            "balanced_min_before_source": (
                before.minimum_assigned_type_count
            ),
            "balanced_min_before": (
                before.minimum_assigned_type_count
            ),
            "balanced_min_after_source": (
                after.minimum_assigned_type_count
            ),
            "balanced_min_after": (
                after.minimum_assigned_type_count
            ),
            "balanced_min_gain": (
                after.minimum_assigned_type_count
                - before.minimum_assigned_type_count
            ),
        }
        row["acquisition"] = {
            "raw_records": row["raw_downloaded_records"],
            "schema_parsed": row["schema_parsed"],
            "accepted_before_unified_filtering": int(
                group.get(
                    "accepted_before_unified_filtering",
                    row["normalized_records"],
                )
            ),
            "quarantined": row["quarantined"],
            "rejected": row["hard_rejected"],
        }
        row["unified_filtering"] = {
            "training_leakage_rejected": row["leakage_rejected"],
            "internal_duplicates_rejected": row["duplicate_rejected"],
            "image_near_duplicates_rejected": (
                row["image_near_duplicates_rejected"]
            ),
            "net_new_candidates": row["new_unique_candidates"],
        }
        row["eligibility"] = {
            "visual_eligible": row["visual_eligible"],
            "text_eligible": row["text_eligible"],
            "mixed_eligible": row["mixed_eligible"],
        }
        row["capacity"] = {
            "joint_capacity_before": row["joint_capacity_before"],
            "joint_capacity_after": row["joint_capacity_after"],
            "joint_capacity_gain": row["joint_capacity_gain"],
            "balanced_min_before": row["balanced_min_before"],
            "balanced_min_after": row["balanced_min_after"],
            "balanced_min_gain": row["balanced_min_gain"],
        }
        rows.append(row)
    return {
        "schema_version": "unified-agent-eval-v1-source-contribution-v1",
        "sources": rows,
        "evaluation_fields": [
            "joint_capacity_gain",
            "balanced_min_gain",
        ],
    }


def contribution_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Unified Agent Eval v1 Source Contribution",
        "",
        "| Source | Status | Unique | Search union | Joint before | "
        "Joint after | Joint gain | Balanced-min gain |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["sources"]:
        lines.append(
            "| {source_name} | {source_status} | {new_unique_candidates} | "
            "{new_search_union_candidates} | "
            "{joint_capacity_before_source} | "
            "{joint_capacity_after_source} | {joint_capacity_gain} | "
            "{balanced_min_gain} |".format(**row)
        )
    lines += [
        "",
        "Source value is measured by `joint_capacity_gain` and "
        "`balanced_min_gain`, not by raw record count.",
        "",
    ]
    return "\n".join(lines)


def write_source_contribution(
    root: Path,
    report: Mapping[str, Any],
) -> None:
    root = Path(root)
    (root / "source_contribution.json").write_text(
        json.dumps(
            report, ensure_ascii=False, indent=2, sort_keys=True
        ) + "\n",
        encoding="utf-8",
    )
    (root / "source_contribution.md").write_text(
        contribution_markdown(report),
        encoding="utf-8",
    )
