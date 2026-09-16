from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence

from multimodal_web_agent.data.protocol_sft.builder import (
    BuildConfig,
    _direct_trajectory,
    _image_trajectory,
)
from multimodal_web_agent.data.protocol_sft.cache_reader import (
    ImageSearchCache,
    sha256_file,
)
from multimodal_web_agent.data.protocol_sft.fvqa_reader import (
    FVQARecord,
    read_fvqa_records,
)
from multimodal_web_agent.data.protocol_sft.information_formatter import (
    format_image_information,
)
from multimodal_web_agent.data.protocol_sft.schema import Trajectory

from .duplicate_grouper import build_image_fingerprint_index
from .gate import SharedExecutabilityGate
from .image_search_support import validate_image_search_action
from .query_executability import infer_query_anchor_provenance
from .pool_manifest import write_json, write_jsonl
from .rejection import rejection_record


MASTER_POOL_SCHEMA = "validated-master-pool-v0.1"
MASTER_POOL_V0_2_SCHEMA = "validated-master-pool-v0.2"


@dataclass(frozen=True)
class MasterPoolBuildConfig:
    source_label: str
    cache_label: str
    seed: int = 20260727
    image_top_k: int = 3
    previous_trajectory_path: str = ""
    maximum_question_copy_ratio_without_entity: float = 0.85
    require_evidence_reachability: bool = True
    repair_mode: str = "reject_only"
    schema_version: str = MASTER_POOL_SCHEMA
    gate_schema_version: str = SharedExecutabilityGate.schema_version
    gate_config_sha256: str = ""
    automatic_route_conversion: bool = False
    automatic_query_rewrite: bool = False
    protocol_sft_schema: str | None = None

    def validate(self) -> None:
        if self.repair_mode != "reject_only":
            raise ValueError("master pool only supports reject_only")
        if self.image_top_k < 1:
            raise ValueError("image_top_k must be positive")
        if self.schema_version not in {
            MASTER_POOL_SCHEMA, MASTER_POOL_V0_2_SCHEMA
        }:
            raise ValueError("unsupported master pool schema")
        if self.gate_schema_version != SharedExecutabilityGate.schema_version:
            raise ValueError(
                "gate schema mismatch: expected %s, got %s"
                % (
                    SharedExecutabilityGate.schema_version,
                    self.gate_schema_version,
                )
            )
        if self.automatic_route_conversion or self.automatic_query_rewrite:
            raise ValueError("automatic repair is forbidden")
        if (
            self.schema_version == MASTER_POOL_V0_2_SCHEMA
            and self.protocol_sft_schema != "protocol-sft-v0.5"
        ):
            raise ValueError("Master Pool v0.2 must bind Protocol-SFT v0.5")


@dataclass(frozen=True)
class MasterPoolBuildResult:
    accepted: list[dict[str, Any]]
    rejected: list[dict[str, Any]]
    quarantined: list[dict[str, Any]]
    action_validity: list[dict[str, Any]]
    source_groups: list[dict[str, Any]]
    manifest: dict[str, Any]


def _convert_previous_trajectory(value: Mapping[str, Any]) -> Trajectory:
    raw = json.loads(json.dumps(value))
    raw["schema_version"] = "protocol-sft-v0.4"
    old_id = str(raw["trajectory_id"])
    suffix = old_id.split(":", 1)[1] if ":" in old_id else old_id
    raw["trajectory_id"] = "protocol_sft_v0_4:%s" % suffix
    raw.setdefault("source", {})["validated_pool_origin"] = (
        "protocol_sft_v0_3_revalidated"
    )
    trajectory = Trajectory.from_dict(raw)
    trajectory.validate()
    return trajectory


def _read_previous(path: Path | None) -> list[Trajectory]:
    if path is None or not path.is_file():
        return []
    values = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            values.append(_convert_previous_trajectory(json.loads(line)))
    return values


def _raw_candidates(
    records: Sequence[FVQARecord],
    cache: ImageSearchCache,
    *,
    config: MasterPoolBuildConfig,
) -> list[Trajectory]:
    build_config = BuildConfig(
        seed=config.seed,
        schema_version="protocol-sft-v0.4",
        image_top_k=config.image_top_k,
    )
    candidates = []
    for record in records:
        if record.category == "search_free":
            candidates.append(
                _direct_trajectory(
                    record,
                    config.source_label,
                    build_config,
                    source_metadata={
                        "validated_pool_origin": "raw_search_free"
                    },
                )
            )
            continue
        entry = cache.get(record.data_id)
        support = validate_image_search_action(
            question=record.question,
            image_cache_entry=entry,
            answer_aliases=record.accepted_answers,
            require_answer_support=True,
        )
        if support.executable and entry is not None:
            candidates.append(
                _image_trajectory(
                    record,
                    format_image_information(
                        entry, top_k=config.image_top_k
                    ),
                    config.source_label,
                    build_config,
                    source_metadata={
                        "validated_pool_origin": "raw_image_cache"
                    },
                )
            )
    return candidates


def _candidate_key(trajectory: Trajectory) -> tuple[str, str]:
    return trajectory.source_data_id, trajectory.route


def _apply_source_level_reject_first(
    accepted: Sequence[dict[str, Any]],
    rejected: Sequence[dict[str, Any]],
    action_validity: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Do not silently replace a conflicting labeled route with another route."""
    text_routes = {
        "text_search",
        "text_search_answer",
        "image_text_search_answer",
    }
    non_executable_text_reasons = {
        "conflicting_route_label",
        "generic_query",
        "missing_visible_entity",
        "question_copy_query",
        "unreachable_evidence",
        "unresolvable_visual_reference",
    }
    tainted_sources = {
        str(row["source_data_id"])
        for row in rejected
        if (
            str(row.get("attempted_route", "")) in text_routes
            and non_executable_text_reasons.intersection(
                row.get("rejection_reasons", ())
            )
        )
    }
    if not tainted_sources:
        return list(accepted), list(rejected)
    kept = []
    final_rejected = list(rejected)
    for candidate in accepted:
        source_data_id = str(candidate["source_data_id"])
        if source_data_id not in tainted_sources:
            kept.append(candidate)
            continue
        trajectory = candidate["trajectory"]
        final_rejected.append(
            rejection_record(
                source_data_id=source_data_id,
                attempted_route=str(candidate["route"]),
                state_type=",".join(candidate.get("state_types", ())),
                reasons=("conflicting_route_label",),
                question=str(trajectory.get("question", "")),
                visible_text_context="",
                target_query="",
                valid_action_set=(),
                metadata={
                    "candidate_id": candidate["candidate_id"],
                    "source_level_reject_first": True,
                    "reason": (
                        "another candidate from the same source has a "
                        "conflicting labeled route; automatic route "
                        "conversion is forbidden"
                    ),
                },
            )
        )
    for row in action_validity:
        if str(row["source_data_id"]) not in tainted_sources:
            continue
        row["source_level_decision"] = "reject"
        row["source_level_rejection_reasons"] = [
            "conflicting_route_label"
        ]
    final_rejected.sort(
        key=lambda row: (
            str(row["source_data_id"]),
            str(row.get("attempted_route", "")),
            str(row.get("candidate_id", "")),
        )
    )
    return kept, final_rejected


def build_validated_master_pool(
    *,
    source_path: Path,
    cache_path: Path,
    config: MasterPoolBuildConfig,
) -> MasterPoolBuildResult:
    config.validate()
    records = read_fvqa_records(source_path)
    record_by_id = {record.data_id: record for record in records}
    cache = ImageSearchCache.load(cache_path, label=config.cache_label)
    fingerprints = build_image_fingerprint_index(source_path)
    previous_path = (
        Path(config.previous_trajectory_path)
        if config.previous_trajectory_path
        else None
    )
    trajectories = _raw_candidates(records, cache, config=config)
    trajectories.extend(_read_previous(previous_path))
    unique = {}
    for trajectory in trajectories:
        unique.setdefault(_candidate_key(trajectory), trajectory)
    trajectories = [
        unique[key] for key in sorted(unique, key=lambda item: (item[0], item[1]))
    ]

    gate = SharedExecutabilityGate(
        maximum_question_copy_ratio_without_entity=(
            config.maximum_question_copy_ratio_without_entity
        ),
        require_evidence_reachability=config.require_evidence_reachability,
    )
    accepted = []
    rejected = []
    action_validity = []
    source_groups = []
    for trajectory in trajectories:
        record = record_by_id.get(trajectory.source_data_id)
        entry = cache.get(trajectory.source_data_id)
        gate_value = gate.evaluate_trajectory(
            trajectory.to_dict(),
            image_cache_entry=entry,
            fingerprint=fingerprints.get(trajectory.source_data_id),
            historically_audited_direct=(
                trajectory.route == "direct_answer"
                and record is not None
                and record.category == "search_free"
            ),
        )
        candidate = {
            "candidate_id": trajectory.trajectory_id,
            "source_data_id": trajectory.source_data_id,
            "route": trajectory.route,
            "state_types": [
                step.transition for step in trajectory.steps
            ],
            "valid_action_set_by_state": gate_value[
                "valid_action_set_by_state"
            ],
            "canonical_sft_action_by_state": gate_value[
                "canonical_sft_action_by_state"
            ],
            "visible_entity_provenance": gate_value[
                "visible_entity_provenance"
            ],
            "evidence_document_ids": gate_value[
                "evidence_document_ids"
            ],
            "entity_group_id": gate_value["entity_group_id"],
            "near_duplicate_group_id": gate_value[
                "near_duplicate_group_id"
            ],
            "source_hash": gate_value["source_hash"],
            "image_dimensions": gate_value["image_dimensions"],
            "cache_hash": cache.file_sha256,
            "repair_mode": "reject_only",
            "trajectory": trajectory.to_dict(),
            "gate_schema_version": config.gate_schema_version,
        }
        anchor_records = []
        for validation in gate_value["state_validations"]:
            if validation["state_type"] not in {
                "initial_to_text_search",
                "image_information_to_text_search",
            }:
                continue
            metadata = validation["metadata"]
            anchor = infer_query_anchor_provenance(
                query=metadata.get("target_query", ""),
                question=trajectory.question,
                information=(
                    metadata.get("visible_text_context", "")
                    if validation["state_type"]
                    == "image_information_to_text_search"
                    else ""
                ),
            )
            if anchor is not None:
                anchor_records.append(
                    {"state_type": validation["state_type"], **anchor}
                )
        candidate["query_anchor_records"] = anchor_records
        if anchor_records:
            primary_anchor = anchor_records[0]
            candidate["query_anchor"] = primary_anchor["query_anchor"]
            candidate["query_anchor_type"] = primary_anchor[
                "query_anchor_type"
            ]
            candidate["query_anchor_provenance"] = primary_anchor[
                "query_anchor_provenance"
            ]
        if (
            trajectory.route in {
                "text_search_answer", "image_text_search_answer"
            }
            and not candidate["query_anchor_records"]
        ):
            gate_value["decision"] = "reject"
            gate_value["rejection_reasons"] = sorted(
                set(gate_value["rejection_reasons"])
                | {"missing_visible_entity", "generic_query"}
            )
        action_validity.append(
            {
                "candidate_id": trajectory.trajectory_id,
                **gate_value,
            }
        )
        source_groups.append(
            {
                "candidate_id": trajectory.trajectory_id,
                "source_data_id": trajectory.source_data_id,
                "entity_group_id": gate_value["entity_group_id"],
                "near_duplicate_group_id": gate_value[
                    "near_duplicate_group_id"
                ],
                "source_hash": gate_value["source_hash"],
                "image_dhash": gate_value["image_dhash"],
                "image_dimensions": gate_value["image_dimensions"],
            }
        )
        if gate_value["decision"] == "accept":
            accepted.append(candidate)
        else:
            first = gate_value["state_validations"][0]
            rejected.append(
                rejection_record(
                    source_data_id=trajectory.source_data_id,
                    attempted_route=trajectory.route,
                    state_type=",".join(candidate["state_types"]),
                    reasons=gate_value["rejection_reasons"],
                    question=trajectory.question,
                    visible_text_context=first["metadata"].get(
                        "visible_text_context", ""
                    ),
                    target_query=first["metadata"].get(
                        "target_query", ""
                    ),
                    valid_action_set=first.get("valid_actions", ()),
                    metadata={
                        "candidate_id": trajectory.trajectory_id,
                        "state_validations": gate_value[
                            "state_validations"
                        ],
                    },
                )
            )

    accepted, rejected = _apply_source_level_reject_first(
        accepted, rejected, action_validity
    )
    reason_distribution = Counter(
        reason
        for row in rejected
        for reason in row["rejection_reasons"]
    )
    route_distribution = Counter(row["route"] for row in accepted)
    manifest = {
        "schema_version": config.schema_version,
        "master_pool_schema": config.schema_version,
        "protocol_sft_schema": config.protocol_sft_schema,
        "gate_schema": config.gate_schema_version,
        "gate_schema_version": config.gate_schema_version,
        "shared_executability_gate_schema": config.gate_schema_version,
        "policy": "reject_first",
        "repair_mode": "reject_only",
        "automatic_route_conversion": False,
        "automatic_query_rewrite": False,
        "source_kind": "FVQA Train",
        "source_label": config.source_label,
        "source_sha256": sha256_file(source_path),
        "cache_label": config.cache_label,
        "cache_sha256": cache.file_sha256,
        "gate_config_sha256": config.gate_config_sha256,
        "code_commit": _git_value("rev-parse", "HEAD"),
        "dirty_status": (
            "dirty" if _git_value("status", "--porcelain") not in {
                "", "unknown"
            } else _git_value("status", "--porcelain")
        ) or "clean",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "counts": {
            "raw_source_records": len(records),
            "candidate_trajectories": len(trajectories),
            "accepted_candidates": len(accepted),
            "rejected_candidates": len(rejected),
            "quarantined_candidates": 0,
        },
        "accepted_route_distribution": dict(sorted(route_distribution.items())),
        "rejection_reason_distribution": dict(
            sorted(reason_distribution.items())
        ),
        "accepted_pool_quality": {
            "no_valid_action_count": 0,
            "unresolvable_visual_reference_count": 0,
            "missing_visible_entity_count": 0,
        },
        "eval_sources_included": False,
        "fvqa_test_included": False,
        "passed": bool(accepted),
    }
    return MasterPoolBuildResult(
        accepted=accepted,
        rejected=rejected,
        quarantined=[],
        action_validity=action_validity,
        source_groups=source_groups,
        manifest=manifest,
    )


def _git_value(*args: str) -> str:
    try:
        value = subprocess.run(
            ["git", *args],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return "unknown"
    return value.stdout.strip() if value.returncode == 0 else "unknown"


def write_master_pool(
    result: MasterPoolBuildResult,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "accepted_candidates.jsonl", result.accepted)
    write_jsonl(output_dir / "rejected_candidates.jsonl", result.rejected)
    write_jsonl(
        output_dir / "quarantined_candidates.jsonl", result.quarantined
    )
    write_jsonl(output_dir / "source_groups.jsonl", result.source_groups)
    write_jsonl(
        output_dir / "action_validity.jsonl", result.action_validity
    )
    write_json(output_dir / "manifest.json", result.manifest)
