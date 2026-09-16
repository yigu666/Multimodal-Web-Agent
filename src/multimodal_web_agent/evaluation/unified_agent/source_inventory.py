from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .source_adapters.base import SourceScan


LIFECYCLE_FIELDS = (
    "raw_source_records",
    "parsed_candidates",
    "hard_rejected_candidates",
    "post_hard_reject_candidates",
    "training_leak_rejected_candidates",
    "post_leakage_candidates",
    "internal_duplicate_rejected_candidates",
    "post_dedup_candidates",
    "search_eligible_candidates",
    "manual_review_candidates",
    "automatic_quality_candidates",
)


def build_source_inventory(
    scans: Sequence[SourceScan],
    *,
    leakage_rejected_ids: set[str],
    duplicate_rejected_ids: set[str],
    search_eligible_ids: set[str],
    manual_review_ids: set[str],
    eligibility_by_id: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, Any]:
    eligibility_by_id = dict(eligibility_by_id or {})
    rows = []
    for scan in scans:
        candidate_ids = {item.candidate_key for item in scan.candidates}
        raw = (
            scan.raw_record_count
            if scan.raw_record_count is not None
            else scan.candidate_count + len(scan.hard_rejected_rows)
        )
        adapter_hard = len(scan.hard_rejected_rows)
        leakage = len(candidate_ids & leakage_rejected_ids)
        post_leakage_ids = candidate_ids - leakage_rejected_ids
        duplicates = len(post_leakage_ids & duplicate_rejected_ids)
        post_dedup_ids = post_leakage_ids - duplicate_rejected_ids
        search_eligible = len(post_dedup_ids & search_eligible_ids)
        if raw != adapter_hard + scan.candidate_count:
            raise ValueError(
                "%s raw/parsed/hard-reject count is not conserved"
                % scan.source_name
            )
        if scan.candidate_count != leakage + duplicates + len(post_dedup_ids):
            raise ValueError(
                "%s parsed lifecycle count is not conserved"
                % scan.source_name
            )
        inventory = dict(scan.inventory)
        status = inventory.get(
            "source_status", inventory.get("status")
        )
        if status not in {
            "ready", "available", "partial", "unavailable", "invalid"
        }:
            status = (
                "available" if scan.available and not adapter_hard
                else "partial" if scan.available
                else "unavailable" if raw == 0
                else "invalid"
            )
        row = {
            "source_name": scan.source_name,
            "configured": bool(inventory.get("configured", True)),
            "enabled": bool(inventory.get("enabled", True)),
            "root_exists": bool(inventory.get("root_exists", scan.available)),
            "manifest_exists": bool(
                inventory.get("manifest_exists", False)
            ),
            "license_verified": bool(
                inventory.get("license_verified", False)
            ),
            "annotation_files_found": int(
                inventory.get("annotation_files_found", int(scan.available))
            ),
            "image_files_found": int(
                inventory.get("image_files_found", scan.candidate_count)
            ),
            "evidence_files_found": int(
                inventory.get("evidence_files_found", 0)
            ),
            "hashes_verified": bool(
                inventory.get("hashes_verified", False)
            ),
            "raw_record_count": raw,
            "parsed_record_count": scan.candidate_count,
            "hard_rejected_count": adapter_hard,
            "post_leakage_count": len(post_leakage_ids),
            "post_dedup_count": len(post_dedup_ids),
            "eligible_search_union_count": search_eligible,
            "visual_eligible_count": sum(
                "visual_search_required"
                in eligibility_by_id.get(candidate_id, ())
                for candidate_id in post_dedup_ids
            ),
            "text_eligible_count": sum(
                "text_search_required"
                in eligibility_by_id.get(candidate_id, ())
                for candidate_id in post_dedup_ids
            ),
            "mixed_eligible_count": sum(
                "mixed_search_required"
                in eligibility_by_id.get(candidate_id, ())
                for candidate_id in post_dedup_ids
            ),
            "source_status": status,
            "status": status,
            "blocking_reasons": list(
                inventory.get("blocking_reasons", ())
            ),
            "message": scan.message,
            "details": inventory,
            "input_files_sha256": dict(scan.input_files_sha256),
            "skipped_candidate_counts": dict(
                scan.skipped_candidate_counts
            ),
            "lifecycle": {
                "raw_source_records": raw,
                "parsed_candidates": scan.candidate_count,
                "hard_rejected_candidates": adapter_hard,
                "post_hard_reject_candidates": scan.candidate_count,
                "training_leak_rejected_candidates": leakage,
                "post_leakage_candidates": len(post_leakage_ids),
                "internal_duplicate_rejected_candidates": duplicates,
                "post_dedup_candidates": len(post_dedup_ids),
                "search_eligible_candidates": search_eligible,
                "manual_review_candidates": len(
                    post_dedup_ids & manual_review_ids
                ),
                "automatic_quality_candidates": len(
                    post_dedup_ids & manual_review_ids
                ),
            },
        }
        rows.append(row)
    global_lifecycle = {
        field: sum(row["lifecycle"][field] for row in rows)
        for field in LIFECYCLE_FIELDS
    }
    if global_lifecycle["raw_source_records"] != (
        global_lifecycle["hard_rejected_candidates"]
        + global_lifecycle["parsed_candidates"]
    ):
        raise ValueError("global raw lifecycle count is not conserved")
    if global_lifecycle["parsed_candidates"] != (
        global_lifecycle["training_leak_rejected_candidates"]
        + global_lifecycle["internal_duplicate_rejected_candidates"]
        + global_lifecycle["post_dedup_candidates"]
    ):
        raise ValueError("global parsed lifecycle count is not conserved")
    return {
        "schema_version": "unified-agent-eval-v1-source-inventory",
        "sources": rows,
        "global_lifecycle": global_lifecycle,
        "lifecycle_conservation_passed": True,
        "manual_review_candidates_deprecated": True,
        "manual_review_candidates_required": False,
    }


def inventory_markdown(inventory: Mapping[str, Any]) -> str:
    headers = (
        "Source",
        "Status",
        "Raw",
        "Parsed",
        "Hard Reject",
        "Leak Reject",
        "Duplicate Reject",
        "Post-dedup",
        "Search Eligible",
    )
    lines = [
        "# Unified Agent Eval v1 Source Inventory",
        "",
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" if index < 2 else "---:" for index in range(
            len(headers)
        )) + "|",
    ]
    for row in inventory["sources"]:
        life = row["lifecycle"]
        lines.append("| " + " | ".join(map(str, (
            row["source_name"],
            row["status"],
            life["raw_source_records"],
            life["parsed_candidates"],
            life["hard_rejected_candidates"],
            life["training_leak_rejected_candidates"],
            life["internal_duplicate_rejected_candidates"],
            life["post_dedup_candidates"],
            life["search_eligible_candidates"],
        ))) + " |")
    lines += [
        "",
        "Lifecycle conservation passed: `true`.",
        "",
    ]
    return "\n".join(lines)


def write_inventory(
    root: Path,
    inventory: Mapping[str, Any],
) -> None:
    Path(root).mkdir(parents=True, exist_ok=True)
    (Path(root) / "source_inventory.json").write_text(
        json.dumps(
            inventory, ensure_ascii=False, indent=2, sort_keys=True
        ) + "\n",
        encoding="utf-8",
    )
    (Path(root) / "source_inventory.md").write_text(
        inventory_markdown(inventory),
        encoding="utf-8",
    )


def merge_readiness_into_inventory(
    inventory: Mapping[str, Any],
    readiness_reports: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Add readiness visibility without importing or parsing source rows."""
    result = json.loads(json.dumps(inventory))
    by_name = {
        row["source_name"]: row for row in result["sources"]
    }
    for row in result["sources"]:
        status = row.get(
            "source_status",
            row.get("status", "available"),
        )
        row["source_status"] = status
        row["status"] = status
        row.setdefault("configured", True)
        row.setdefault("enabled", True)
        row.setdefault("root_exists", True)
        row.setdefault("manifest_exists", False)
        row.setdefault("license_verified", False)
        row.setdefault("hashes_verified", False)
        row.setdefault("blocking_reasons", [])
        row.setdefault("visual_eligible_count", 0)
        row.setdefault("text_eligible_count", 0)
        row.setdefault("mixed_eligible_count", 0)
    for readiness in readiness_reports:
        name = str(readiness["source_name"])
        row = by_name.get(name)
        if row is None:
            lifecycle = {field: 0 for field in LIFECYCLE_FIELDS}
            row = {
                "source_name": name,
                "message": "%s source is %s" % (
                    name, readiness["source_status"]
                ),
                "details": {},
                "input_files_sha256": {},
                "skipped_candidate_counts": {},
                "raw_record_count": 0,
                "parsed_record_count": 0,
                "hard_rejected_count": 0,
                "post_leakage_count": 0,
                "post_dedup_count": 0,
                "eligible_search_union_count": 0,
                "visual_eligible_count": 0,
                "text_eligible_count": 0,
                "mixed_eligible_count": 0,
                "lifecycle": lifecycle,
            }
            result["sources"].append(row)
            by_name[name] = row
        for field in (
            "configured",
            "enabled",
            "root_exists",
            "manifest_exists",
            "license_verified",
            "annotation_files_found",
            "image_files_found",
            "evidence_files_found",
            "hashes_verified",
            "blocking_reasons",
        ):
            row[field] = readiness[field]
        row["source_status"] = readiness["source_status"]
        row["status"] = readiness["source_status"]
        row["details"] = {
            **dict(row.get("details", {})),
            "readiness": dict(readiness),
        }
    result["sources"] = sorted(
        result["sources"], key=lambda row: row["source_name"]
    )
    result["readiness_merged"] = True
    return result
