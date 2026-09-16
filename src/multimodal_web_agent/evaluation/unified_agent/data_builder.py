from __future__ import annotations

from collections import Counter
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from .answer_metrics import answer_reachable
from .capacity_assignment import SEARCH_TASK_TYPES, solve_joint_capacity
from .embargo import initialize_test_embargo
from .environment import (
    IMAGE_RECORD_MAX_CHARS,
    TEXT_RECORD_MAX_CHARS,
    build_frozen_environment,
    truncate_record,
)
from .fingerprints import initialize_registry, sha256_file
from .leakage import (
    LeakageReference,
    audit_internal_duplicates,
    audit_leakage,
    references_from_fvqa_parquet,
    references_from_jsonl,
)
from .schema import SCHEMA_VERSION, TASK_TYPES, UnifiedEvalExample
from .source_adapters import (
    FVQATestAdapter,
    InfoSeekAdapter,
    MMSearchAdapter,
    SimpleVQAAdapter,
)
from .source_adapters.base import SourceCandidate, SourceScan
from .task_types import balanced_split

# Kept only to reproduce the pre-acquisition experiment.  The automatic
# acquisition pipeline never imports or invokes this legacy approval path.
LEGACY_MANUAL_REVIEW = {
    "deprecated": True,
    "required": False,
    "replacement": "automatic held-out acquisition quality decisions",
}


class ReviewRequired(RuntimeError):
    pass


class DataShortfall(RuntimeError):
    pass


def _resolve(root: Path, value: str | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    return path if path.is_absolute() else root / path


def _json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(
                dict(row), ensure_ascii=False, sort_keys=True
            ) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _candidate_reachable(
    candidate: SourceCandidate,
    task_type: str | None,
) -> bool:
    if task_type == "search_free":
        return True
    if task_type == "visual_search_required":
        records = [
            truncate_record(value, IMAGE_RECORD_MAX_CHARS)
            for value in candidate.image_search_records[:5]
        ]
    elif task_type == "text_search_required":
        records = [
            truncate_record(value, TEXT_RECORD_MAX_CHARS)
            for value in candidate.text_corpus_records
        ]
    else:
        records = [
            truncate_record(value, IMAGE_RECORD_MAX_CHARS)
            for value in candidate.image_search_records[:5]
        ] + [
            truncate_record(value, TEXT_RECORD_MAX_CHARS)
            for value in candidate.text_corpus_records
        ]
    return answer_reachable(candidate.answer_aliases, records)


def scan_sources(root: Path, config: Mapping[str, Any]) -> list[SourceScan]:
    sources = config["sources"]
    adapters = [
        FVQATestAdapter(
            _resolve(root, sources["fvqa_test"].get("parquet")),
            _resolve(root, sources["fvqa_test"].get("image_search_cache")),
        ),
        MMSearchAdapter(
            _resolve(root, sources["mmsearch"].get("end2end")),
            _resolve(root, sources["mmsearch"].get("rerank")),
            _resolve(root, sources["mmsearch"].get("summarization")),
        ),
        InfoSeekAdapter(_resolve(root, sources["infoseek"].get("path"))),
        SimpleVQAAdapter(_resolve(root, sources["simplevqa"].get("path"))),
    ]
    return [adapter.scan() for adapter in adapters]


def load_history_references(
    root: Path,
    config: Mapping[str, Any],
) -> tuple[list[LeakageReference], list[dict[str, Any]]]:
    references = []
    statuses = []
    for source in config.get("history_sources", []):
        path = _resolve(root, source.get("path"))
        available = path is not None and path.is_file()
        statuses.append({
            "label": source["label"],
            "path": source.get("path"),
            "available": available,
            "future_source": bool(source.get("future_source", False)),
            "sha256": sha256_file(path) if available else None,
        })
        if not available:
            continue
        if source["format"] == "fvqa_parquet":
            references.extend(references_from_fvqa_parquet(
                path, source["label"]
            ))
        elif source["format"] == "jsonl":
            references.extend(references_from_jsonl(path, source["label"]))
        else:
            raise ValueError("unsupported history source format")
    return references, statuses


def _draft_rows(
    candidates: Sequence[SourceCandidate],
    hard_reject: Mapping[str, Sequence[str]],
    high_similarity_keys: set[str],
) -> list[dict[str, Any]]:
    rows = []
    for candidate in sorted(candidates, key=lambda item: item.candidate_key):
        reasons = list(hard_reject.get(candidate.candidate_key, ()))
        reachability_by_task_type = {
            task_type: _candidate_reachable(candidate, task_type)
            for task_type in TASK_TYPES
        }
        rows.append({
            "candidate_key": candidate.candidate_key,
            "candidate_sha256": candidate.candidate_sha256,
            "source_dataset": candidate.source_dataset,
            "source_data_id": candidate.source_data_id,
            "question": candidate.question,
            "image_sha256": candidate.image_sha256,
            "answer_aliases": list(candidate.answer_aliases),
            "suggested_task_type": candidate.suggested_task_type,
            "task_type": candidate.suggested_task_type,
            "search_required": (
                candidate.suggested_task_type != "search_free"
                if candidate.suggested_task_type else None
            ),
            "image_question_match": False,
            "answer_aliases_correct": False,
            "fixed_environment_answer_reachable": (
                reachability_by_task_type[candidate.suggested_task_type]
                if candidate.suggested_task_type else None
            ),
            "reachability_by_task_type": reachability_by_task_type,
            "no_training_leakage": not reasons,
            "high_similarity_review_required": (
                candidate.candidate_key in high_similarity_keys
            ),
            "high_similarity_reviewed": False,
            "reviewed": False,
            "approved": False,
            "primary_reviewer": "",
            "secondary_reviewer": "",
            "legacy_manual_review_deprecated": True,
            "legacy_manual_review_required": False,
            "adjudication_status": "pending",
            "hard_rejection_reasons": reasons,
            "reviewer_notes": "",
        })
    return rows


def _pre_review_capacity(
    candidates: Sequence[SourceCandidate],
    hard_reject: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    eligible = [
        candidate
        for candidate in candidates
        if not hard_reject.get(candidate.candidate_key)
    ]
    reachability = {
        task_type: sum(
            _candidate_reachable(candidate, task_type)
            for candidate in eligible
        )
        for task_type in TASK_TYPES
    }
    joint_eligibility = {
        candidate.candidate_key: tuple(
            task_type for task_type in SEARCH_TASK_TYPES
            if _candidate_reachable(candidate, task_type)
        )
        for candidate in eligible
    }
    joint_eligibility = {
        key: values for key, values in joint_eligibility.items() if values
    }
    joint = solve_joint_capacity(joint_eligibility, target_per_type=250)
    required_per_type = 250
    required_search_total = required_per_type * 3
    # Mixed reachability is the union of frozen Image and Text evidence.
    # Every candidate eligible for any of the three search-required tasks
    # therefore belongs to this same union and can occupy only one split row.
    search_union_capacity = reachability["mixed_search_required"]
    search_union_shortfall = max(
        0, required_search_total - search_union_capacity
    )
    total_capacity_shortfall = max(0, 1000 - len(eligible))
    minimum_shortfall = max(
        joint.total_shortfall,
        search_union_shortfall,
        total_capacity_shortfall,
    )
    return {
        "eligible_after_hard_reject_count": len(eligible),
        "required_total": 1000,
        "required_per_task_type": required_per_type,
        "required_search_required_total": required_search_total,
        "reachability_upper_bound_by_task_type": reachability,
        "search_required_union_capacity": search_union_capacity,
        "search_required_union_shortfall": search_union_shortfall,
        "total_capacity_shortfall": total_capacity_shortfall,
        "minimum_total_shortfall_lower_bound": minimum_shortfall,
        "max_total_with_search_free_quota_250": (
            required_per_type + search_union_capacity
        ),
        "simple_equal_four_way_upper_bound": (
            (search_union_capacity // 3) * 4
        ),
        "maximum_joint_search_assignment": joint.total_assignable,
        "maximum_equal_per_type_assignment": (
            joint.maximum_equal_per_type
        ),
        "maximum_equal_four_way_size_by_assignment": (
            joint.maximum_equal_four_way_size
        ),
        "joint_visual_assigned_count": joint.visual_assigned,
        "joint_text_assigned_count": joint.text_assigned,
        "joint_mixed_assigned_count": joint.mixed_assigned,
        "visual_remaining_shortfall": joint.visual_shortfall,
        "text_remaining_shortfall": joint.text_shortfall,
        "mixed_remaining_shortfall": joint.mixed_shortfall,
        "joint_total_remaining_shortfall": joint.total_shortfall,
        "balanced_dataset_max_theoretical_size": None,
        "balanced_dataset_max_theoretical_size_deprecated": True,
        "deprecated_reason": (
            "The previous value represented a non-balanced total, not a "
            "balanced assignment capacity."
        ),
        "passed": (
            minimum_shortfall == 0
            and len(eligible) >= 1000
            and joint.full_quota_satisfied
        ),
    }


def _approval_map(
    path: Path,
    candidates: Sequence[SourceCandidate],
    high_similarity_keys: set[str],
    hard_reject: Mapping[str, Sequence[str]],
) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        raise ReviewRequired("manual approval file is missing")
    candidate_map = {item.candidate_key: item for item in candidates}
    approvals = {}
    for row in _read_jsonl(path):
        key = str(row.get("candidate_key", ""))
        candidate = candidate_map.get(key)
        if candidate is None:
            raise ValueError("approval references an unknown candidate")
        if row.get("candidate_sha256") != candidate.candidate_sha256:
            raise ValueError("approval candidate fingerprint is stale")
        if key in approvals:
            raise ValueError("duplicate manual approval candidate")
        if row.get("approved") is True:
            required_true = (
                "reviewed",
                "image_question_match",
                "answer_aliases_correct",
                "no_training_leakage",
            )
            if not all(row.get(name) is True for name in required_true):
                raise ValueError("approved candidate has incomplete checks")
            task_type = str(row.get("task_type", ""))
            if task_type not in TASK_TYPES:
                raise ValueError("approved candidate has invalid task_type")
            if bool(row.get("search_required")) != (
                task_type != "search_free"
            ):
                raise ValueError("approved search_required is inconsistent")
            actual_reachable = _candidate_reachable(candidate, task_type)
            if row.get("fixed_environment_answer_reachable") is not (
                actual_reachable
            ):
                raise ValueError(
                    "approved reachability field differs from frozen evidence"
                )
            if task_type != "search_free" and not actual_reachable:
                raise ValueError("approved search-required evidence is unreachable")
            if key in high_similarity_keys and row.get(
                "high_similarity_reviewed"
            ) is not True:
                raise ValueError("high-similarity candidate lacks review")
            if hard_reject.get(key):
                raise ValueError("hard-leak candidate cannot be approved")
            if not str(row.get("primary_reviewer", "")).strip():
                raise ValueError("approved candidate lacks primary reviewer")
        approvals[key] = row
    return approvals


def _example(
    candidate: SourceCandidate,
    approval: Mapping[str, Any],
    *,
    split: str,
    index: int,
) -> UnifiedEvalExample:
    extension = candidate.image_extension.lower()
    if extension not in {".jpg", ".jpeg", ".png", ".webp"}:
        extension = ".jpg"
    value = UnifiedEvalExample(
        eval_id="%s:%s:%06d" % (SCHEMA_VERSION, split, index),
        source_dataset=candidate.source_dataset,
        source_data_id=candidate.source_data_id,
        question=candidate.question,
        image_path="images/%s%s" % (candidate.image_sha256, extension),
        image_sha256=candidate.image_sha256,
        answer_aliases=candidate.answer_aliases,
        task_type=str(approval["task_type"]),
        search_required=bool(approval["search_required"]),
        source_metadata={
            **candidate.source_metadata,
            "candidate_sha256": candidate.candidate_sha256,
            "primary_reviewer": approval["primary_reviewer"],
            "secondary_reviewer": approval.get("secondary_reviewer", ""),
            "adjudication_status": approval.get(
                "adjudication_status", "approved"
            ),
        },
    )
    value.validate()
    return value


def _hash_manifest(paths: Sequence[Path], root: Path) -> str:
    lines = [
        "%s  %s" % (sha256_file(path), path.relative_to(root).as_posix())
        for path in sorted(paths, key=lambda item: item.as_posix())
    ]
    return "\n".join(lines) + "\n"


def build_dataset(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    root = Path(root).resolve()
    output = _resolve(root, config["output"]["directory"])
    manifest_path = _resolve(root, config["output"]["manifest"])
    audit_path = _resolve(root, config["output"]["audit"])
    hashes_path = _resolve(root, config["output"]["hashes"])
    draft_path = _resolve(root, config["manual_review"]["draft_file"])
    approval_path = _resolve(root, config["manual_review"]["approval_file"])
    build_report_path = _resolve(root, config["manual_review"]["build_report"])
    embargo_path = _resolve(root, config["output"]["test_embargo"])
    registry_path = _resolve(root, config["output"]["model_registry"])
    assert all(path is not None for path in (
        output, manifest_path, audit_path, hashes_path, draft_path,
        approval_path, build_report_path, embargo_path, registry_path,
    ))
    for path in (output, manifest_path, audit_path, hashes_path):
        if path.exists():
            raise FileExistsError("refusing to overwrite formal Eval data: %s" % path)

    scans = scan_sources(root, config)
    candidates = [
        item for scan in scans for item in scan.candidates
    ]
    if len({item.candidate_key for item in candidates}) != len(candidates):
        raise ValueError("source adapters emitted duplicate candidate keys")
    references, history_statuses = load_history_references(root, config)
    leakage = audit_leakage(candidates, references)
    internal = audit_internal_duplicates(
        candidates,
        pre_rejected_keys=set(leakage["hard_reject_candidates"]),
    )
    hard_reject = {
        key: list(reasons)
        for key, reasons in leakage["hard_reject_candidates"].items()
    }
    for key in internal["hard_reject_candidates"]:
        hard_reject.setdefault(key, []).append("internal_duplicate")
    high_similarity_keys = {
        row["candidate_key"]
        for row in leakage["high_similarity_manual_review"]
    }
    draft = _draft_rows(
        candidates, hard_reject, high_similarity_keys
    )
    _jsonl(draft_path, draft)
    template = draft_path.with_name("unified_agent_eval_v1_manual_audit_template.md")
    template.write_text(
        "# Unified Agent Eval v1 Manual Audit\n\n"
        "Review every candidate independently. Set reviewed/approved and all "
        "checks explicitly; assign one of the four frozen task types. "
        "Do not approve hard leakage or unreachable search-required samples.\n",
        encoding="utf-8",
    )
    report = {
        "schema_version": "unified-agent-eval-v1-build-report",
        "source_scans": [scan.summary() for scan in scans],
        "candidate_count": len(candidates),
        "history_sources": history_statuses,
        "history_reference_count": len(references),
        "leakage": leakage,
        "internal_duplicates": internal,
        "manual_audit_draft": str(draft_path.relative_to(root)).replace("\\", "/"),
        "manual_approval_file": str(approval_path.relative_to(root)).replace("\\", "/"),
        "formal_data_published": False,
    }
    pre_review_capacity = _pre_review_capacity(candidates, hard_reject)
    report["pre_review_capacity"] = pre_review_capacity
    _json(build_report_path, report)
    unavailable_required_history = [
        row["label"] for row in history_statuses
        if not row["future_source"] and not row["available"]
    ]
    if unavailable_required_history:
        raise ReviewRequired(
            "required leakage sources unavailable: %s"
            % ", ".join(unavailable_required_history)
        )
    if not pre_review_capacity["passed"]:
        raise DataShortfall(json.dumps(
            {
                "stage": "pre_review_joint_capacity",
                **pre_review_capacity,
            },
            sort_keys=True,
        ))
    approvals = _approval_map(
        approval_path, candidates, high_similarity_keys, hard_reject
    )
    dev_pairs, test_pairs, shortfall = balanced_split(
        candidates, approvals, seed=int(config["split"]["seed"])
    )
    report["shortfall_by_task_type"] = shortfall
    if any(shortfall.values()):
        _json(build_report_path, report)
        raise DataShortfall(json.dumps(shortfall, sort_keys=True))
    selected_pairs = dev_pairs + test_pairs
    selected_candidates = [item[0] for item in selected_pairs]
    selected_leakage = audit_leakage(selected_candidates, references)
    selected_internal = audit_internal_duplicates(selected_candidates)
    if not selected_leakage["hard_gate_passed"] or not selected_internal["passed"]:
        raise ValueError("selected Eval data failed leakage/duplicate hard gates")
    run_id = "%d" % os.getpid()
    temporary = output.with_name(".%s.tmp-%s" % (output.name, run_id))
    if temporary.exists():
        raise FileExistsError(temporary)
    temporary.mkdir(parents=True)
    try:
        images = temporary / "images"
        images.mkdir()
        for candidate in selected_candidates:
            extension = candidate.image_extension.lower()
            if extension not in {".jpg", ".jpeg", ".png", ".webp"}:
                extension = ".jpg"
            path = images / (candidate.image_sha256 + extension)
            if path.exists() and sha256_file(path) != candidate.image_sha256:
                raise RuntimeError("image filename/hash collision")
            if not path.exists():
                path.write_bytes(candidate.image_bytes)
        dev_examples = [
            _example(candidate, approval, split="dev", index=index)
            for index, (candidate, approval) in enumerate(dev_pairs, 1)
        ]
        test_examples = [
            _example(candidate, approval, split="test", index=index)
            for index, (candidate, approval) in enumerate(test_pairs, 1)
        ]
        _jsonl(temporary / "dev.jsonl", [item.to_dict() for item in dev_examples])
        _jsonl(temporary / "test.jsonl", [item.to_dict() for item in test_examples])
        _jsonl(
            temporary / "manual_audit.jsonl",
            [dict(approval) for _, approval in selected_pairs],
        )
        (temporary / "manual_audit_template.md").write_text(
            template.read_text(encoding="utf-8"), encoding="utf-8"
        )
        environment_manifest = build_frozen_environment(
            temporary / "environment",
            selected_candidates,
            image_search_top_k=5,
            text_search_top_k=5,
        )
        unreachable = [
            candidate.candidate_key
            for candidate, approval in selected_pairs
            if not _candidate_reachable(candidate, approval["task_type"])
        ]
        if unreachable:
            raise ValueError("selected search-required answers are unreachable")
        preview = [
            "# Unified Agent Eval v1 Dev Preview",
            "",
        ]
        for item in dev_examples[:20]:
            preview += [
                "## `%s`" % item.eval_id,
                "",
                "- Task type: `%s`" % item.task_type,
                "- Question: %s" % item.question,
                "- Answer aliases: `%s`" % ", ".join(item.answer_aliases),
                "",
            ]
        (temporary / "sample_preview_dev.md").write_text(
            "\n".join(preview), encoding="utf-8"
        )
        counts = {
            "dev": len(dev_examples),
            "test": len(test_examples),
            "total": len(dev_examples) + len(test_examples),
            "dev_by_task_type": dict(Counter(
                item.task_type for item in dev_examples
            )),
            "test_by_task_type": dict(Counter(
                item.task_type for item in test_examples
            )),
        }
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "counts": counts,
            "source_scans": [scan.summary() for scan in scans],
            "history_sources": history_statuses,
            "split_seed": int(config["split"]["seed"]),
            "test_preview_generated": False,
            "dynamic_internet_accessed": False,
            "manual_review_complete": True,
        }
        audit = {
            "schema_version": "unified-agent-eval-v1-audit",
            "passed": True,
            "counts": counts,
            "manual_approved_count": len(selected_pairs),
            "leakage": selected_leakage,
            "internal_duplicates": selected_internal,
            "search_required_unreachable_count": 0,
            "test_preview_present": False,
        }
        _json(temporary / "manifest.json", manifest)
        _json(temporary / "audit.json", audit)
        _json(temporary / "audit_summary.json", {
            "schema_version": "unified-agent-eval-v1-audit-summary",
            "passed": True,
            "dev_count": 200,
            "test_count": 800,
            "manual_approved_count": 1000,
            "hard_leak_count": 0,
            "unreachable_count": 0,
        })
        environment_files = [
            path for path in (temporary / "environment").rglob("*")
            if path.is_file() and path.name != "environment_files.sha256"
        ]
        (temporary / "environment/environment_files.sha256").write_text(
            _hash_manifest(environment_files, temporary), encoding="utf-8"
        )
        os.replace(temporary, output)
        _json(manifest_path, manifest)
        _json(audit_path, audit)
        formal_files = [
            path for path in output.rglob("*")
            if path.is_file()
        ] + [manifest_path, audit_path]
        hashes_path.write_text(
            _hash_manifest(formal_files, root), encoding="utf-8"
        )
        initialize_test_embargo(embargo_path)
        initialize_registry(registry_path)
        report.update({
            "formal_data_published": True,
            "shortfall_by_task_type": shortfall,
            "counts": counts,
        })
        _json(build_report_path, report)
        return {"manifest": manifest, "audit": audit, "report": report}
    except Exception:
        # Preserve the temporary directory for failure audit; never publish it.
        raise
