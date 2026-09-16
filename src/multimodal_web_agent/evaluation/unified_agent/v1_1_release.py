from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any, Iterable, Mapping, Sequence

import yaml

from .data_builder import load_history_references
from .fingerprints import sha256_file
from .leakage import audit_internal_duplicates, audit_leakage
from .release_selection import (
    RELEASE_TASK_TYPES,
    deterministic_stratified_split,
    select_balanced_release,
)
from .source_adapters.base import SourceCandidate
from .source_expansion import (
    _candidate_eligibility,
    _scan_configured_sources,
)


SCHEMA_VERSION = "unified-agent-eval-v1-1"
DISPLAY_NAME = "Unified Agent Eval v1.1 Balanced 900"
ACQUISITION_REVISION = "visual-infoseek-acquisition-v2-tar-membership"
KNOWN_LIMITATION = "wiki6m_long_paragraph_false_negative_quarantine"


class V11ReleaseError(RuntimeError):
    pass


@dataclass(frozen=True)
class ReleasePaths:
    output: Path
    manifest: Path
    audit: Path
    hashes: Path
    reserved_ids: Path
    reserved_images: Path
    embargo: Path
    registry: Path


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            value, ensure_ascii=False, indent=2, sort_keys=True, default=str
        ) + "\n",
        encoding="utf-8",
    )


def _jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(
                dict(row), ensure_ascii=False, sort_keys=True, default=str
            ) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _tree_hash_entries(root: Path) -> list[tuple[str, str]]:
    return [
        (sha256_file(path), path.relative_to(root).as_posix())
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "files.sha256"
    ]


def _write_tree_hashes(root: Path) -> None:
    entries = _tree_hash_entries(root)
    (root / "files.sha256").write_text(
        "".join("%s  %s\n" % row for row in entries),
        encoding="utf-8",
    )


def _paths(root: Path, config: Mapping[str, Any]) -> ReleasePaths:
    output = config["output"]
    return ReleasePaths(
        output=_resolve(root, output["directory"]),
        manifest=_resolve(root, output["manifest"]),
        audit=_resolve(root, output["audit"]),
        hashes=_resolve(root, output["hashes"]),
        reserved_ids=_resolve(root, output["reserved_ids"]),
        reserved_images=_resolve(root, output["reserved_images"]),
        embargo=_resolve(root, output["test_embargo"]),
        registry=_resolve(root, output["model_registry"]),
    )


def _v1_files(staging: Path) -> list[Path]:
    names = (
        "joint_capacity.json",
        "shortage_report.json",
        "source_contribution.json",
        "automatic_quarantine.jsonl",
        "staging_manifest.json",
        "files.sha256",
    )
    return [staging / name for name in names if (staging / name).is_file()]


def _v1_snapshot(staging: Path) -> dict[str, str]:
    required = (
        staging / "joint_capacity.json",
        staging / "shortage_report.json",
        staging / "staging_manifest.json",
        staging / "files.sha256",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise V11ReleaseError(
            "original v1 audit is incomplete: %s" % ", ".join(missing)
        )
    capacity = json.loads(
        (staging / "joint_capacity.json").read_text(encoding="utf-8")
    )
    if capacity.get("capacity_gate_passed") is not False:
        raise V11ReleaseError(
            "original v1 must remain DATA_SHORTFALL"
        )
    targets = capacity.get("targets") or {}
    if set(int(value) for value in targets.values()) != {250}:
        raise V11ReleaseError("original v1 target is not 250 per type")
    assignment = capacity.get("joint_assignment") or {}
    achieved = (
        int(assignment.get("visual_assigned", -1)),
        int(assignment.get("text_assigned", -1)),
        int(assignment.get("mixed_assigned", -1)),
    )
    if achieved != (235, 235, 234):
        raise V11ReleaseError(
            "original v1 achieved capacity changed: %r" % (achieved,)
        )
    if int(capacity.get("joint_total_remaining_shortfall", -1)) != 46:
        raise V11ReleaseError("original v1 shortfall is not 46")
    return {
        path.relative_to(staging).as_posix(): sha256_file(path)
        for path in _v1_files(staging)
    }


def _source_scan_config(
    history_data_config: str,
    visual_root: str,
) -> dict[str, Any]:
    return {
        "sources": {
            "existing_candidates": {
                "enabled": True,
                "adapter": "existing_unified_eval_candidates",
                "data_config": history_data_config,
            },
            "visual_infoseek_2023": {
                "enabled": True,
                "adapter": "generic_heldout",
                "root_dir": visual_root,
                "manifest_file": "source_manifest.json",
                "evidence_mode": "text",
                "field_map": {
                    "source_id": "source_data_id",
                    "question": "question",
                    "answer_aliases": "answer_aliases",
                    "query_image": "query_image_path",
                    "image_evidence": "image_search_records",
                    "text_evidence": "text_corpus_records",
                    "evidence": "offline_evidence_records",
                    "eligible_task_types": "eligible_task_types",
                    "source_split": "source_split",
                    "entity_id": "entity_id",
                    "wikipedia_title": "wikipedia_title",
                },
            },
        }
    }


def _load_approved_candidates(
    root: Path,
    config: Mapping[str, Any],
) -> tuple[list[SourceCandidate], dict[str, tuple[str, ...]], dict[str, Any]]:
    staging = _resolve(root, config["v1_staging_dir"])
    accepted_rows = _read_jsonl(
        staging / "candidates/post_dedup_candidates.jsonl"
    )
    accepted_by_id = {row["candidate_id"]: row for row in accepted_rows}
    if len(accepted_by_id) != len(accepted_rows):
        raise V11ReleaseError("v1 staging contains duplicate candidate IDs")
    scans, readiness, _ = _scan_configured_sources(
        root,
        _source_scan_config(
            config["history_data_config"],
            config["visual_infoseek_root"],
        ),
    )
    scanned = {
        candidate.candidate_key: candidate
        for scan in scans for candidate in scan.candidates
    }
    missing = sorted(set(accepted_by_id) - set(scanned))
    if missing:
        raise V11ReleaseError(
            "approved staging candidates cannot be reconstructed: %s"
            % ", ".join(missing[:20])
        )
    candidates = []
    eligibility = {}
    fingerprint_mismatch = []
    for candidate_id, staging_row in accepted_by_id.items():
        candidate = scanned[candidate_id]
        if staging_row.get("candidate_sha256") != candidate.candidate_sha256:
            fingerprint_mismatch.append(candidate_id)
            continue
        computed, _ = _candidate_eligibility(candidate)
        declared = tuple(staging_row.get("eligible_task_types") or ())
        allowed = tuple(value for value in computed if value in declared)
        candidates.append(candidate)
        if allowed:
            eligibility[candidate_id] = allowed
    if fingerprint_mismatch:
        raise V11ReleaseError(
            "candidate fingerprint changed: %s"
            % ", ".join(fingerprint_mismatch[:20])
        )
    return candidates, eligibility, {
        "source_scans": [scan.summary() for scan in scans],
        "source_readiness": readiness,
        "staging_candidate_count": len(accepted_rows),
    }


def _candidate_evidence(candidate: SourceCandidate) -> dict[str, Any]:
    image_records = list(candidate.image_search_records)
    text_records = list(candidate.text_corpus_records)
    payload = {
        "schema_version": "unified-agent-eval-v1-1-evidence",
        "candidate_id": candidate.candidate_key,
        "image_search_records": image_records,
        "text_corpus_records": text_records,
    }
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    payload["content_sha256"] = hashlib.sha256(encoded).hexdigest()
    return payload


def _materialize_image(
    candidate: SourceCandidate,
    image_dir: Path,
) -> str:
    suffix = candidate.image_extension.casefold()
    if not suffix.startswith("."):
        suffix = "." + suffix
    if suffix not in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
        suffix = ".jpg"
    name = candidate.image_sha256 + suffix
    path = image_dir / name
    if path.is_file():
        if sha256_file(path) != candidate.image_sha256:
            raise V11ReleaseError("image Hash collision: %s" % name)
    else:
        path.write_bytes(candidate.image_bytes)
    return "images/" + name


def _example(
    candidate: SourceCandidate,
    *,
    task_type: str,
    split: str,
    index: int,
    image_path: str,
    evidence_path: str,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "benchmark_version": "unified_agent_eval_v1_1",
        "eval_id": "unified-agent-eval-v1-1:%s:%06d" % (split, index),
        "source_dataset": candidate.source_dataset,
        "source_data_id": candidate.source_data_id,
        "question": candidate.question,
        "image_path": image_path,
        "image_sha256": candidate.image_sha256,
        "answer_aliases": list(candidate.answer_aliases),
        "task_type": task_type,
        "search_required": task_type != "search_free",
        "maximum_agent_turns": 3,
        "maximum_tool_calls": 2,
        "maximum_image_search_calls": 1,
        "maximum_text_search_calls": 1,
        "evidence_path": evidence_path,
        "candidate_sha256": candidate.candidate_sha256,
        "source_metadata": dict(candidate.source_metadata),
    }


def _counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counter = Counter(str(row["task_type"]) for row in rows)
    return {task_type: counter[task_type] for task_type in RELEASE_TASK_TYPES}


def _write_external_hashes(root: Path, paths: ReleasePaths) -> None:
    files = sorted([
        *[path for path in paths.output.rglob("*") if path.is_file()],
        paths.manifest,
        paths.audit,
        paths.reserved_ids,
        paths.reserved_images,
    ])
    paths.hashes.parent.mkdir(parents=True, exist_ok=True)
    paths.hashes.write_text(
        "".join(
            "%s  %s\n" % (
                sha256_file(path),
                path.relative_to(root).as_posix(),
            )
            for path in files
        ),
        encoding="utf-8",
    )


def build_v1_1_release(
    root: Path,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    root = Path(root).resolve()
    paths = _paths(root, config)
    protected = (
        paths.output, paths.manifest, paths.audit, paths.hashes,
        paths.reserved_ids, paths.reserved_images, paths.embargo,
        paths.registry,
    )
    existing = [str(path) for path in protected if path.exists()]
    if existing:
        raise FileExistsError(
            "refusing to overwrite v1.1 release: %s" % ", ".join(existing)
        )
    staging = _resolve(root, config["v1_staging_dir"])
    before = _v1_snapshot(staging)
    visual_config = yaml.safe_load(
        _resolve(root, config["visual_infoseek_config"]).read_text(
            encoding="utf-8"
        )
    )
    if visual_config["oven_images"].get("allow_new_shard_downloads") is not False:
        raise V11ReleaseError("new OVEN shard downloads must remain disabled")
    acquisition_report = json.loads(
        _resolve(root, config["visual_infoseek_final_report"]).read_text(
            encoding="utf-8"
        )
    )
    if acquisition_report.get("implementation_revision") != (
        ACQUISITION_REVISION
    ):
        raise V11ReleaseError("unexpected Visual InfoSeek implementation")
    if int(acquisition_report.get("downloaded_bytes", -1)) != 0:
        raise V11ReleaseError(
            "v1.1 must be built from a zero-new-download acquisition run"
        )

    candidates, eligibility, scan_report = _load_approved_candidates(
        root, config
    )
    history_config = yaml.safe_load(
        _resolve(root, config["history_data_config"]).read_text(
            encoding="utf-8"
        )
    )
    references, history_status = load_history_references(root, history_config)
    missing_history = [
        row["label"] for row in history_status
        if not row["available"] and not row["future_source"]
    ]
    if missing_history:
        raise V11ReleaseError(
            "leakage history unavailable: %s" % ", ".join(missing_history)
        )
    leakage = audit_leakage(candidates, references)
    leakage_rejected = set(leakage["hard_reject_candidates"])
    post_leakage = [
        item for item in candidates
        if item.candidate_key not in leakage_rejected
    ]
    duplicates = audit_internal_duplicates(post_leakage)
    duplicate_rejected = set(duplicates["hard_reject_candidates"])
    final_candidates = [
        item for item in post_leakage
        if item.candidate_key not in duplicate_rejected
    ]
    final_ids = {item.candidate_key for item in final_candidates}
    final_eligibility = {
        key: value for key, value in eligibility.items() if key in final_ids
    }
    selection_config = config["selection"]
    selection = select_balanced_release(
        final_candidates,
        final_eligibility,
        target_per_type=int(selection_config["target_per_type"]),
        seed=int(selection_config["seed"]),
        minimum_reserve_per_type=int(
            selection_config["minimum_reserve_per_type"]
        ),
    )
    achieved = selection["capacity"]
    if (
        achieved["visual_search_required"],
        achieved["text_search_required"],
        achieved["mixed_search_required"],
    ) != (235, 235, 234):
        raise V11ReleaseError(
            "current audited search capacity changed: %s" % achieved
        )
    target = int(selection_config["target_per_type"])
    temporary = paths.output.with_name(
        ".unified_agent_eval_v1_1.tmp-%d" % os.getpid()
    )
    if temporary.exists():
        raise FileExistsError(temporary)
    temporary.mkdir(parents=True)
    try:
        image_dir = temporary / "images"
        evidence_dir = temporary / "evidence"
        image_dir.mkdir()
        evidence_dir.mkdir()
        dev_records = []
        test_records = []
        assignment_records = []
        reserve_records = []
        counters = {"dev": 0, "test": 0}
        for type_index, task_type in enumerate(RELEASE_TASK_TYPES):
            selected = selection["selected"][task_type]
            dev, test = deterministic_stratified_split(
                selected,
                dev_count=int(selection_config["dev_per_type"]),
                seed=int(selection_config["seed"]) + type_index,
            )
            for split, rows, target_rows in (
                ("dev", dev, dev_records),
                ("test", test, test_records),
            ):
                for candidate in rows:
                    counters[split] += 1
                    image_path = _materialize_image(candidate, image_dir)
                    evidence_name = (
                        candidate.candidate_sha256 + ".json"
                    )
                    _json(
                        evidence_dir / evidence_name,
                        _candidate_evidence(candidate),
                    )
                    row = _example(
                        candidate,
                        task_type=task_type,
                        split=split,
                        index=counters[split],
                        image_path=image_path,
                        evidence_path="evidence/" + evidence_name,
                    )
                    target_rows.append(row)
                    assignment_records.append({
                        "candidate_id": candidate.candidate_key,
                        "candidate_sha256": candidate.candidate_sha256,
                        "eval_id": row["eval_id"],
                        "task_type": task_type,
                        "split": split,
                        "selection_seed": int(selection_config["seed"]),
                    })
            for candidate in selection["reserves"][task_type]:
                image_path = _materialize_image(candidate, image_dir)
                evidence_name = candidate.candidate_sha256 + ".json"
                _json(
                    evidence_dir / evidence_name,
                    _candidate_evidence(candidate),
                )
                reserve_records.append({
                    **_example(
                        candidate,
                        task_type=task_type,
                        split="reserve",
                        index=len(reserve_records) + 1,
                        image_path=image_path,
                        evidence_path="evidence/" + evidence_name,
                    ),
                    "reserve_only": True,
                    "replacement_allowed_before_freeze": True,
                    "replacement_allowed_after_freeze": False,
                })
        if _counts(dev_records) != {
            task_type: 50 for task_type in RELEASE_TASK_TYPES
        }:
            raise V11ReleaseError("Dev split is not 50 per type")
        if _counts(test_records) != {
            task_type: 175 for task_type in RELEASE_TASK_TYPES
        }:
            raise V11ReleaseError("Test split is not 175 per type")
        reserve_counts = _counts(reserve_records)
        if min(reserve_counts.values()) < int(
            selection_config["minimum_reserve_per_type"]
        ):
            raise V11ReleaseError("release reserve is below five per type")

        _jsonl(temporary / "dev.jsonl", dev_records)
        _jsonl(temporary / "test.jsonl", test_records)
        _jsonl(
            temporary / "reserve_candidates.jsonl", reserve_records
        )
        _jsonl(
            temporary / "selection_assignments.jsonl",
            assignment_records,
        )
        limitation = {
            "schema_version": "unified-agent-eval-v1-1-limitations",
            "implementation_revision": ACQUISITION_REVISION,
            "known_limitations": [{
                "id": KNOWN_LIMITATION,
                "impact": "false_negative_candidate_exclusion_only",
                "accepted": True,
                "regeneration_after_freeze": False,
            }],
            "evidence_window_fix_implemented": False,
            "quarantine_reprocessed": False,
            "new_oven_shards_downloaded": 0,
        }
        _json(temporary / "known_limitations.json", limitation)
        selection_report = {
            "schema_version": "unified-agent-eval-v1-1-selection-v1",
            "selection_seed": int(selection_config["seed"]),
            "selection_order": (
                "global exact search assignment, diversity ranking, "
                "balanced selection, deterministic stratified split"
            ),
            "selected_counts": {
                "total": len(dev_records) + len(test_records),
                "by_task_type": _counts(dev_records + test_records),
                "dev": len(dev_records),
                "test": len(test_records),
            },
            "reserve_counts": reserve_counts,
            "achieved_preselection_capacity": achieved,
            "manual_selection": False,
            "model_score_selection": False,
            "test_result_selection": False,
        }
        _json(temporary / "selection_report.json", selection_report)
        selected_candidates = [
            candidate
            for rows in selection["selected"].values()
            for candidate in rows
        ]
        final_leakage = audit_leakage(selected_candidates, references)
        final_duplicates = audit_internal_duplicates(selected_candidates)
        if not final_leakage["hard_gate_passed"]:
            raise V11ReleaseError("selected release failed leakage audit")
        if final_duplicates["hard_reject_candidates"]:
            raise V11ReleaseError("selected release contains duplicates")
        audit = {
            "schema_version": "unified-agent-eval-v1-1-audit",
            "passed": True,
            "original_v1_status": "DATA_SHORTFALL",
            "original_design_target": {
                "per_type": 250,
                "total": 1000,
                "status": "FAILED_BY_CAPACITY",
            },
            "maximum_achieved_capacity": achieved,
            "remaining_original_search_shortfall": 46,
            "release_rebaseline": {
                "per_type": 225,
                "total": 900,
                "dev": 200,
                "test": 700,
            },
            "decision": (
                "accept false-negative evidence limitation; do not recover "
                "quarantine; do not download new OVEN shards"
            ),
            "known_limitations": [KNOWN_LIMITATION],
            "explicit_rebaseline_not_v1_gate_pass": True,
            "original_v1_files_preserved": True,
            "counts": {
                "total": 900,
                "dev": 200,
                "test": 700,
                "selected_by_task_type": _counts(
                    dev_records + test_records
                ),
                "dev_by_task_type": _counts(dev_records),
                "test_by_task_type": _counts(test_records),
                "reserve_by_task_type": reserve_counts,
            },
            "leakage": final_leakage,
            "internal_duplicates": final_duplicates,
            "quarantine_selected_count": 0,
            "new_oven_shards_downloaded": 0,
            "downloaded_bytes": 0,
            "manual_data_preparation_required": False,
            "manual_review_required": False,
            "test_accessed": False,
            "environment_frozen": False,
        }
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "display_name": DISPLAY_NAME,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "original_design_target": {
                "per_type": 250,
                "total": 1000,
                "status": "failed_by_capacity",
            },
            "original_capacity_gate": "FAILED_BY_CAPACITY",
            "remaining_original_search_shortfall": 46,
            "release_target": {
                "per_type": target,
                "total": 900,
                "dev": 200,
                "test": 700,
            },
            "achieved_preselection_capacity": achieved,
            "known_limitations": [KNOWN_LIMITATION],
            "acquisition_implementation_revision": ACQUISITION_REVISION,
            "new_oven_shards_downloaded": 0,
            "downloaded_bytes": 0,
            "manual_data_preparation_required": False,
            "manual_review_required": False,
            "selection_seed": int(selection_config["seed"]),
            "environment_frozen": False,
            "test_embargo_opened": False,
            "source_scan": scan_report,
            "history_sources": history_status,
            "explicit_rebaseline_not_v1_gate_pass": True,
        }
        _json(temporary / "audit.json", audit)
        _json(temporary / "manifest.json", manifest)
        _write_tree_hashes(temporary)
        if _v1_snapshot(staging) != before:
            raise V11ReleaseError("original v1 audit changed during build")
        os.replace(temporary, paths.output)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise

    external_values = (
        (paths.manifest, manifest),
        (paths.audit, audit),
        (paths.reserved_ids, {
            "schema_version": "unified-agent-eval-v1-1-reserved-ids",
            "reserved_source_ids": sorted({
                row["source_data_id"]
                for row in dev_records + test_records
            }),
        }),
    )
    for path, value in external_values:
        _json(path, value)
    paths.reserved_images.parent.mkdir(parents=True, exist_ok=True)
    paths.reserved_images.write_text(
        "".join(
            "%s  %s:%s\n" % (
                row["image_sha256"],
                row["source_dataset"],
                row["source_data_id"],
            )
            for row in sorted(
                dev_records + test_records,
                key=lambda item: (
                    item["image_sha256"], item["source_data_id"]
                ),
            )
        ),
        encoding="utf-8",
    )
    _json(paths.embargo, {
        "schema_version": "unified-agent-eval-v1-1-test-embargo",
        "opened": False,
        "evaluation_count": 0,
        "allowed_model_count": 4,
        "registered_models": [],
    })
    _json(paths.registry, {
        "schema_version": "unified-agent-eval-v1-1-model-registry",
        "registered_models": {},
    })
    _write_external_hashes(root, paths)
    return {
        "manifest": manifest,
        "audit": audit,
        "selection_report": selection_report,
        "paths": paths,
    }


def verify_v1_1_release(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    root = Path(root).resolve()
    paths = _paths(root, config)
    manifest = json.loads(paths.manifest.read_text(encoding="utf-8"))
    audit = json.loads(paths.audit.read_text(encoding="utf-8"))
    dev = _read_jsonl(paths.output / "dev.jsonl")
    test = _read_jsonl(paths.output / "test.jsonl")
    reserves = _read_jsonl(paths.output / "reserve_candidates.jsonl")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise V11ReleaseError("v1.1 manifest schema mismatch")
    if len(dev) != 200 or len(test) != 700:
        raise V11ReleaseError("v1.1 split count mismatch")
    if _counts(dev) != {
        task_type: 50 for task_type in RELEASE_TASK_TYPES
    }:
        raise V11ReleaseError("v1.1 Dev balance mismatch")
    if _counts(test) != {
        task_type: 175 for task_type in RELEASE_TASK_TYPES
    }:
        raise V11ReleaseError("v1.1 Test balance mismatch")
    if min(_counts(reserves).values()) < 5:
        raise V11ReleaseError("v1.1 reserve balance mismatch")
    if audit.get("passed") is not True:
        raise V11ReleaseError("v1.1 audit did not pass")
    return {
        "schema_version": SCHEMA_VERSION,
        "dev_count": len(dev),
        "test_count": len(test),
        "reserve_count": len(reserves),
        "dev_by_task_type": _counts(dev),
        "test_by_task_type": _counts(test),
        "reserve_by_task_type": _counts(reserves),
        "passed": True,
    }
