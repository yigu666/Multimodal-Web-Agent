from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from .answer_metrics import answer_reachable
from .capacity_assignment import (
    SEARCH_TASK_TYPES,
    solve_joint_capacity,
)
from .data_builder import load_history_references, scan_sources
from .environment import (
    IMAGE_RECORD_MAX_CHARS,
    TEXT_RECORD_MAX_CHARS,
    truncate_record,
)
from .fingerprints import sha256_file
from .leakage import audit_internal_duplicates, audit_leakage
from .review_package import (
    create_automatic_quality_output_if_capacity_passes,
)
from .source_contribution import (
    build_source_contribution,
    write_source_contribution,
)
from .source_adapters import (
    GenericHeldoutSourceAdapter,
    InfoSeekAdapter,
    MMSearchHeldOutAdapter,
)
from .source_adapters.base import SourceCandidate, SourceScan
from .source_inventory import build_source_inventory, write_inventory
from .source_readiness import (
    adapter_config_from_readiness,
    check_source_readiness,
)


class SourceExpansionInvalid(RuntimeError):
    pass


def _json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            value, ensure_ascii=False, indent=2, sort_keys=True, default=str
        ) + "\n",
        encoding="utf-8",
    )


def _jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(
                dict(row),
                ensure_ascii=False,
                sort_keys=True,
                default=str,
            ) + "\n")


def _candidate_eligibility(
    candidate: SourceCandidate,
) -> tuple[tuple[str, ...], dict[str, list[str]]]:
    image_records = [
        truncate_record(value, IMAGE_RECORD_MAX_CHARS)
        for value in candidate.image_search_records[:5]
    ]
    text_records = [
        truncate_record(value, TEXT_RECORD_MAX_CHARS)
        for value in candidate.text_corpus_records
    ]
    records = {
        "visual_search_required": [
            value for value in image_records if value
        ],
        "text_search_required": [
            value for value in text_records if value
        ],
        "mixed_search_required": [
            value for value in image_records + text_records if value
        ],
    }
    computed = {
        task_type
        for task_type in SEARCH_TASK_TYPES
        if answer_reachable(candidate.answer_aliases, records[task_type])
    }
    declared = set(
        candidate.source_metadata.get("declared_eligible_task_types") or ()
    )
    if declared:
        computed &= declared
    eligible = tuple(
        task_type for task_type in SEARCH_TASK_TYPES
        if task_type in computed
    )
    return eligible, {
        task_type: records[task_type] if task_type in computed else []
        for task_type in SEARCH_TASK_TYPES
    }


def _candidate_row(
    candidate: SourceCandidate,
    *,
    eligible_task_types: Sequence[str] = (),
    eligibility_evidence: Mapping[str, Sequence[str]] | None = None,
    rejection_reasons: Sequence[str] = (),
) -> dict[str, Any]:
    return {
        "candidate_id": candidate.candidate_key,
        "candidate_sha256": candidate.candidate_sha256,
        "source_dataset": candidate.source_dataset,
        "source_data_id": candidate.source_data_id,
        "question": candidate.question,
        "query_image_sha256": candidate.image_sha256,
        "query_image_extension": candidate.image_extension,
        "query_image_path": candidate.source_metadata.get(
            "query_image_path"
        ),
        "query_image_role_verified": bool(
            candidate.source_metadata.get("query_image_role_verified")
        ),
        "query_image_role_source": candidate.source_metadata.get(
            "query_image_role_source"
        ),
        "retrieval_result_images_excluded_from_input": bool(
            candidate.source_metadata.get(
                "retrieval_result_images_excluded_from_input"
            )
        ),
        "answer_aliases": list(candidate.answer_aliases),
        "offline_evidence_records": list(
            candidate.image_search_records
            + candidate.text_corpus_records
        ),
        "source_url": candidate.source_metadata.get("source_url"),
        "license_name": candidate.source_metadata.get("license"),
        "suggested_task_type": candidate.suggested_task_type,
        "eligible_task_types": list(eligible_task_types),
        "eligibility_evidence": {
            key: list(values)
            for key, values in (eligibility_evidence or {}).items()
        },
        "image_search_record_count": len(
            candidate.image_search_records
        ),
        "text_search_record_count": len(candidate.text_corpus_records),
        "source_metadata": candidate.source_metadata,
        "rejection_reasons": list(rejection_reasons),
    }


def _evidence_rows(
    candidates: Sequence[SourceCandidate],
) -> list[dict[str, Any]]:
    rows = []
    for candidate in candidates:
        for evidence_source, records in (
            ("image_search", candidate.image_search_records),
            ("text_search", candidate.text_corpus_records),
        ):
            for index, raw_text in enumerate(records):
                text = str(raw_text).strip()
                if not text:
                    continue
                rows.append({
                    "candidate_id": candidate.candidate_key,
                    "evidence_source": evidence_source,
                    "evidence_record_id": "%s:%s:%d" % (
                        candidate.candidate_key,
                        evidence_source,
                        index,
                    ),
                    "evidence_text": text,
                    "evidence_sha256": hashlib.sha256(
                        text.encode("utf-8")
                    ).hexdigest(),
                    "answer_reachable": answer_reachable(
                        candidate.answer_aliases, [text]
                    ),
                })
    return rows


def _scan_configured_sources(
    root: Path,
    config: Mapping[str, Any],
) -> tuple[list[SourceScan], list[dict[str, Any]], set[str]]:
    scans = []
    readiness_reports = []
    expansion_source_names = set()
    sources = config["sources"]
    existing = sources.get("existing_candidates", {})
    if existing.get("enabled") is True:
        data_config_path = Path(existing["data_config"])
        if not data_config_path.is_absolute():
            data_config_path = root / data_config_path
        data_config = yaml.safe_load(
            data_config_path.read_text(encoding="utf-8")
        )
        try:
            scans.extend(scan_sources(root, data_config))
        except Exception as exc:
            raise SourceExpansionInvalid(
                "existing candidate source is invalid: %r" % (exc,)
            ) from exc
    for name, source in sources.items():
        if name == "existing_candidates":
            continue
        expansion_source_names.add(name)
        readiness = check_source_readiness(root, name, source)
        readiness_reports.append(readiness)
        status = readiness["source_status"]
        if status == "invalid":
            raise SourceExpansionInvalid(
                "%s source package is invalid: %s"
                % (
                    name,
                    ", ".join(readiness["blocking_reasons"]),
                )
            )
        if status != "ready":
            scans.append(SourceScan(
                source_name=name,
                available=False,
                candidate_count=0,
                message="%s source is %s" % (name, status),
                inventory=dict(readiness),
            ))
            continue
        source = adapter_config_from_readiness(source, readiness)
        adapter_name = source.get("adapter")
        if adapter_name == "infoseek":
            adapter = InfoSeekAdapter(source, project_root=root)
        elif adapter_name == "mmsearch":
            adapter = MMSearchHeldOutAdapter(
                source, project_root=root
            )
        elif adapter_name == "generic_heldout":
            adapter = GenericHeldoutSourceAdapter(
                source,
                project_root=root,
                source_name=name,
            )
        else:
            raise SourceExpansionInvalid(
                "unsupported Held-out Adapter: %s" % adapter_name
            )
        try:
            scan = adapter.scan()
            merged_inventory = {
                **dict(readiness),
                **dict(scan.inventory),
                "source_status": "ready",
                "blocking_reasons": [],
            }
            scans.append(SourceScan(
                source_name=scan.source_name,
                available=scan.available,
                candidate_count=scan.candidate_count,
                message=scan.message,
                candidates=scan.candidates,
                input_files_sha256=scan.input_files_sha256,
                skipped_candidate_counts=scan.skipped_candidate_counts,
                raw_record_count=scan.raw_record_count,
                hard_rejected_rows=scan.hard_rejected_rows,
                inventory=merged_inventory,
            ))
        except Exception as exc:
            raise SourceExpansionInvalid(
                "%s source is invalid: %r" % (name, exc)
            ) from exc
    return scans, readiness_reports, expansion_source_names


def _capacity_report(
    candidates: Sequence[SourceCandidate],
    eligibility: Mapping[str, Sequence[str]],
    target_per_type: int,
) -> tuple[dict[str, Any], dict[str, str]]:
    joint = solve_joint_capacity(
        eligibility, target_per_type=target_per_type
    )
    search_free_eligible = len(candidates)
    search_union = len(eligibility)
    target_search_total = target_per_type * 3
    search_free_independent = max(
        0, len(candidates) - len(joint.assignments)
    )
    maximum_equal_per_type = min(
        joint.maximum_equal_per_type,
        len(candidates) // 4,
    )
    passed = (
        search_free_eligible >= target_per_type
        and len(candidates) >= target_per_type * 4
        and joint.full_quota_satisfied
        and joint.total_assignable >= target_search_total
        and search_free_independent >= target_per_type
    )
    balanced_visual_gap = max(
        0, target_per_type - maximum_equal_per_type
    )
    raw_counts = joint.raw_max_flow_counts
    result = {
        "schema_version": "unified-agent-eval-v1-joint-capacity-v2",
        "solver": "exact_max_flow",
        "targets": {
            "search_free": target_per_type,
            "visual_search_required": target_per_type,
            "text_search_required": target_per_type,
            "mixed_search_required": target_per_type,
        },
        "candidate_capacity": {
            "search_free_eligible_count": search_free_eligible,
            "visual_eligible_count": sum(
                "visual_search_required" in values
                for values in eligibility.values()
            ),
            "text_eligible_count": sum(
                "text_search_required" in values
                for values in eligibility.values()
            ),
            "mixed_eligible_count": sum(
                "mixed_search_required" in values
                for values in eligibility.values()
            ),
            "search_eligible_union_count": search_union,
        },
        "joint_assignment": {
            "objective": "max_total_then_max_min_then_min_spread",
            "maximum_joint_search_assignment": joint.total_assignable,
            "visual_assigned": joint.visual_assigned,
            "text_assigned": joint.text_assigned,
            "mixed_assigned": joint.mixed_assigned,
            "minimum_assigned_type_count": (
                joint.minimum_assigned_type_count
            ),
            "assignment_spread": joint.assignment_spread,
            "full_quota_satisfied": joint.full_quota_satisfied,
        },
        "raw_max_flow_assignment": {
            "diagnostic_only": True,
            "distribution_interpretable": False,
            "visual_assigned": raw_counts["visual_search_required"],
            "text_assigned": raw_counts["text_search_required"],
            "mixed_assigned": raw_counts["mixed_search_required"],
        },
        "shortfall": {
            "minimum_net_new_search_candidates_required": max(
                0, target_search_total - joint.total_assignable
            ),
            "balanced_baseline_visual_gap": balanced_visual_gap,
            "balanced_baseline_text_gap": balanced_visual_gap,
            "balanced_baseline_mixed_gap": balanced_visual_gap,
            "joint_total_shortfall": max(
                0, target_search_total - joint.total_assignable
            ),
            "balanced_baseline_is_not_independent_type_shortfall": True,
        },
        "balanced_capacity": {
            "maximum_equal_per_search_type": maximum_equal_per_type,
            "maximum_equal_four_way_size": (
                maximum_equal_per_type * 4
            ),
        },
        "non_balanced_capacity": {
            "max_total_with_search_free_quota_250": (
                target_per_type + search_union
            ),
            "is_balanced_capacity": False,
        },
        "search_free_eligible_count": search_free_eligible,
        "visual_search_eligible_count": sum(
            "visual_search_required" in values
            for values in eligibility.values()
        ),
        "text_search_eligible_count": sum(
            "text_search_required" in values
            for values in eligibility.values()
        ),
        "mixed_search_eligible_count": sum(
            "mixed_search_required" in values
            for values in eligibility.values()
        ),
        "search_eligible_union_count": search_union,
        "target_search_free_count": target_per_type,
        "target_visual_search_count": target_per_type,
        "target_text_search_count": target_per_type,
        "target_mixed_search_count": target_per_type,
        "target_search_total_count": target_search_total,
        "minimum_new_search_candidates_required": max(
            0, target_search_total - search_union
        ),
        "max_total_with_search_free_quota_250": (
            target_per_type + search_union
        ),
        "simple_equal_four_way_upper_bound": (
            (search_union // 3) * 4
        ),
        "maximum_joint_search_assignment": joint.total_assignable,
        "maximum_equal_per_type_assignment": (
            maximum_equal_per_type
        ),
        "maximum_equal_four_way_size_by_assignment": (
            maximum_equal_per_type * 4
        ),
        "joint_visual_assigned_count": joint.visual_assigned,
        "joint_text_assigned_count": joint.text_assigned,
        "joint_mixed_assigned_count": joint.mixed_assigned,
        "visual_remaining_shortfall": balanced_visual_gap,
        "text_remaining_shortfall": balanced_visual_gap,
        "mixed_remaining_shortfall": balanced_visual_gap,
        "joint_total_remaining_shortfall": max(
            0, target_search_total - joint.total_assignable
        ),
        "search_free_independent_capacity_after_assignment": (
            search_free_independent
        ),
        "full_quota_satisfied": joint.full_quota_satisfied,
        "capacity_gate_passed": passed,
        "balanced_dataset_max_theoretical_size": None,
        "balanced_dataset_max_theoretical_size_deprecated": True,
        "deprecated_reason": (
            "The previous value represented a non-balanced total, not a "
            "balanced assignment capacity."
        ),
        "bottleneck_summary": joint.bottleneck_summary,
    }
    return result, joint.assignments


def _capacity_markdown(report: Mapping[str, Any]) -> str:
    fields = (
        "search_free_eligible_count",
        "visual_search_eligible_count",
        "text_search_eligible_count",
        "mixed_search_eligible_count",
        "search_eligible_union_count",
        "maximum_joint_search_assignment",
        "maximum_equal_per_type_assignment",
        "maximum_equal_four_way_size_by_assignment",
        "joint_visual_assigned_count",
        "joint_text_assigned_count",
        "joint_mixed_assigned_count",
        "joint_total_remaining_shortfall",
        "capacity_gate_passed",
    )
    lines = [
        "# Unified Agent Eval v1 Exact Joint Capacity",
        "",
        "| Field | Value |",
        "|---|---:|",
    ]
    lines.extend(
        "| %s | %s |" % (field, report[field])
        for field in fields
    )
    lines += [
        "",
        "`max_total_with_search_free_quota_250` is explicitly non-balanced.",
        "",
        "The raw maximum-flow distribution is diagnostic-only. The formal "
        "distribution uses max-total, max-min, minimum-spread assignment.",
        "",
    ]
    return "\n".join(lines)


def _shortage_report(capacity: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "unified-agent-eval-v1-shortage-report",
        "capacity_gate_passed": capacity["capacity_gate_passed"],
        "minimum_net_new_search_candidates": capacity["shortfall"][
            "minimum_net_new_search_candidates_required"
        ],
        "minimum_new_search_candidates_required_by_union": capacity[
            "minimum_new_search_candidates_required"
        ],
        "raw_source_expansion_target_recommended": 800,
        "post_hard_reject_target_recommended": 550,
        "post_review_net_target_required": 400,
        "planning_fields_are_not_success_gates": True,
        "formal_data_published": False,
        "review_package_created": False,
        "review_package_deprecated": True,
        "automatic_quality_created": capacity["capacity_gate_passed"],
        "frozen_environment_created": False,
        "raw_sft_eval_run": False,
        "test_embargo_opened": False,
    }


def build_source_expansion(
    root: Path,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    root = Path(root).resolve()
    boundaries = config["boundaries"]
    if any(boundaries.values()):
        raise SourceExpansionInvalid(
            "source expansion boundaries must all remain false"
        )
    capacity_config = config["capacity"]
    if (
        capacity_config.get("solver") != "exact_max_flow"
        or capacity_config.get("require_full_joint_quota") is not True
        or capacity_config.get(
            "generate_review_package_only_if_capacity_passes"
        ) is not True
    ):
        raise SourceExpansionInvalid(
            "source expansion exact-capacity contract is not enabled"
        )
    output = Path(config["output"]["staging_dir"])
    if not output.is_absolute():
        output = root / output
    if output.exists():
        raise SourceExpansionInvalid(
            "refusing to overwrite existing source-expansion staging"
        )
    temporary = output.with_name(
        ".unified_agent_eval_v1_expansion.tmp-%d" % os.getpid()
    )
    if temporary.exists():
        raise SourceExpansionInvalid(
            "source-expansion temporary directory already exists"
        )
    scans, readiness_reports, expansion_source_names = (
        _scan_configured_sources(root, config)
    )
    candidates = [
        candidate for scan in scans for candidate in scan.candidates
    ]
    if len({item.candidate_key for item in candidates}) != len(candidates):
        raise SourceExpansionInvalid(
            "Held-out sources emitted duplicate candidate IDs"
        )
    history_path = Path(config["history_data_config"])
    if not history_path.is_absolute():
        history_path = root / history_path
    history_config = yaml.safe_load(
        history_path.read_text(encoding="utf-8")
    )
    try:
        references, history_status = load_history_references(
            root, history_config
        )
    except Exception as exc:
        raise SourceExpansionInvalid(
            "leakage history is invalid: %r" % (exc,)
        ) from exc
    missing_history = [
        row["label"] for row in history_status
        if not row["available"] and not row["future_source"]
    ]
    if missing_history:
        raise SourceExpansionInvalid(
            "required leakage history is unavailable: %s"
            % ", ".join(missing_history)
        )
    leakage = audit_leakage(candidates, references)
    leakage_rejected_ids = set(
        leakage["hard_reject_candidates"]
    )
    post_leakage = [
        candidate for candidate in candidates
        if candidate.candidate_key not in leakage_rejected_ids
    ]
    duplicates = audit_internal_duplicates(post_leakage)
    duplicate_rejected_ids = set(
        duplicates["hard_reject_candidates"]
    )
    image_near_duplicate_ids = {
        key
        for pair in duplicates.get("near_duplicate_image_pairs", ())
        for key in (pair["left"], pair["right"])
        if key in duplicate_rejected_ids
    }
    post_dedup = [
        candidate for candidate in post_leakage
        if candidate.candidate_key not in duplicate_rejected_ids
    ]
    eligibility = {}
    eligibility_evidence = {}
    for candidate in post_dedup:
        task_types, evidence = _candidate_eligibility(candidate)
        if task_types:
            eligibility[candidate.candidate_key] = task_types
        eligibility_evidence[candidate.candidate_key] = evidence
    target_values = {
        int(config["targets"][name])
        for name in (
            "search_free",
            "visual_search_required",
            "text_search_required",
            "mixed_search_required",
        )
    }
    if len(target_values) != 1:
        raise SourceExpansionInvalid(
            "exact joint solver requires equal four-way targets"
        )
    target_per_type = target_values.pop()
    capacity, assignments = _capacity_report(
        post_dedup, eligibility, target_per_type
    )
    review_created = bool(capacity["capacity_gate_passed"])
    proposed_assignments = dict(assignments) if review_created else {}
    if review_created:
        search_free_candidates = [
            candidate.candidate_key for candidate in post_dedup
            if candidate.candidate_key not in assignments
        ][:target_per_type]
        proposed_assignments.update({
            candidate_id: "search_free"
            for candidate_id in search_free_candidates
        })
        if len(proposed_assignments) != target_per_type * 4:
            raise SourceExpansionInvalid(
                "full capacity passed but review proposal is incomplete"
            )
    inventory = build_source_inventory(
        scans,
        leakage_rejected_ids=leakage_rejected_ids,
        duplicate_rejected_ids=duplicate_rejected_ids,
        search_eligible_ids=set(eligibility),
        manual_review_ids=set(proposed_assignments),
        eligibility_by_id=eligibility,
    )
    contribution = build_source_contribution(
        scans,
        post_dedup_candidates=post_dedup,
        eligibility=eligibility,
        target_per_type=target_per_type,
        expansion_source_names=expansion_source_names,
        leakage_rejected_ids=leakage_rejected_ids,
        duplicate_rejected_ids=duplicate_rejected_ids,
        image_near_duplicate_rejected_ids=image_near_duplicate_ids,
    )
    source_data_required = bool(expansion_source_names) and not any(
        row["source_status"] == "ready"
        for row in readiness_reports
    )
    temporary.mkdir(parents=True)
    try:
        candidate_dir = temporary / "candidates"
        all_rows = [_candidate_row(item) for item in candidates]
        post_dedup_rows = [
            _candidate_row(
                item,
                eligible_task_types=eligibility.get(
                    item.candidate_key, ()
                ),
                eligibility_evidence=eligibility_evidence[
                    item.candidate_key
                ],
            )
            for item in post_dedup
        ]
        _jsonl(
            candidate_dir / "all_parsed_candidates.jsonl", all_rows
        )
        _jsonl(
            candidate_dir / "hard_rejected_candidates.jsonl",
            [
                row for scan in scans
                for row in scan.hard_rejected_rows
            ],
        )
        _jsonl(
            candidate_dir / "leakage_rejected_candidates.jsonl",
            [
                _candidate_row(
                    item,
                    rejection_reasons=leakage[
                        "hard_reject_candidates"
                    ][item.candidate_key],
                )
                for item in candidates
                if item.candidate_key in leakage_rejected_ids
            ],
        )
        _jsonl(
            candidate_dir / "duplicate_rejected_candidates.jsonl",
            [
                _candidate_row(
                    item,
                    rejection_reasons=("internal_duplicate",),
                )
                for item in post_leakage
                if item.candidate_key in duplicate_rejected_ids
            ],
        )
        _jsonl(
            candidate_dir / "post_dedup_candidates.jsonl",
            post_dedup_rows,
        )
        _jsonl(
            candidate_dir / "search_eligible_candidates.jsonl",
            [
                row for row in post_dedup_rows
                if row["candidate_id"] in eligibility
            ],
        )
        _jsonl(
            temporary / "evidence/offline_evidence_staging.jsonl",
            _evidence_rows(post_dedup),
        )
        write_inventory(temporary, inventory)
        write_source_contribution(temporary, contribution)
        _json(
            temporary / "source_readiness.json",
            {
                "schema_version": (
                    "unified-eval-source-readiness-report-v1"
                ),
                "sources": readiness_reports,
                "source_data_required": source_data_required,
            },
        )
        _json(temporary / "joint_capacity.json", capacity)
        (temporary / "joint_capacity.md").write_text(
            _capacity_markdown(capacity), encoding="utf-8"
        )
        shortage = _shortage_report(capacity)
        _json(temporary / "shortage_report.json", shortage)
        (temporary / "shortage_report.md").write_text(
            "# Unified Agent Eval v1 Shortage\n\n"
            "- Minimum joint shortfall: `%d`.\n"
            "- Recommended raw records to add: `800`.\n"
            "- Recommended post-hard-reject candidates: `550`.\n"
            "- Recommended post-review net candidates: `400`.\n"
            % capacity["joint_total_remaining_shortfall"],
            encoding="utf-8",
        )
        _json(temporary / "leakage_audit.json", leakage)
        _json(temporary / "internal_duplicate_audit.json", duplicates)
        _json(temporary / "history_sources.json", history_status)
        if review_created:
            create_automatic_quality_output_if_capacity_passes(
                temporary,
                capacity_gate_passed=True,
                candidate_rows=post_dedup_rows,
                assignments=proposed_assignments,
            )
        manifest = {
            "schema_version": (
                "unified-agent-eval-v1-source-expansion-staging"
            ),
            "capacity_gate_passed": review_created,
            "review_package_status": (
                "DEPRECATED_NOT_REQUIRED"
            ),
            "automatic_quality_status": (
                "READY" if review_created else "NOT_GENERATED"
            ),
            "manual_review_required": False,
            "automatic_acquisition": bool(
                config.get("acquisition_mode")
            ),
            "formal_data_published": False,
            "processed_dataset_touched": False,
            "frozen_environment_created": False,
            "raw_sft_eval_run": False,
            "test_embargo_opened": False,
            "source_count": len(scans),
            "source_data_required": source_data_required,
            "parsed_candidate_count": len(candidates),
            "post_dedup_candidate_count": len(post_dedup),
        }
        _json(temporary / "staging_manifest.json", manifest)
        files = sorted(
            path for path in temporary.rglob("*")
            if path.is_file() and path.name != "files.sha256"
        )
        (temporary / "files.sha256").write_text(
            "\n".join(
                "%s  %s" % (
                    sha256_file(path),
                    path.relative_to(temporary).as_posix(),
                )
                for path in files
            ) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, output)
    except Exception:
        raise
    return {
        "capacity": capacity,
        "inventory": inventory,
        "staging_dir": output.as_posix(),
        "review_package_created": False,
        "review_package_deprecated": True,
        "automatic_quality_created": review_created,
        "source_data_required": source_data_required,
    }
