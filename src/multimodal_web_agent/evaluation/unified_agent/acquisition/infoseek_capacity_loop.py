from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

from ..source_expansion import build_source_expansion


BASELINE_JOINT_CAPACITY = 408
BASELINE_BALANCED_MIN = 136


def validate_baseline_capacity(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    joint = value["joint_assignment"]
    if (
        int(joint["maximum_joint_search_assignment"])
        != BASELINE_JOINT_CAPACITY
        or int(joint["visual_assigned"]) != BASELINE_BALANCED_MIN
        or int(joint["text_assigned"]) != BASELINE_BALANCED_MIN
        or int(joint["mixed_assigned"]) != BASELINE_BALANCED_MIN
    ):
        raise ValueError(
            "Visual InfoSeek baseline must be 408 and 136/136/136"
        )
    return value


def expansion_config(
    *,
    history_data_config: str,
    normalized_root: Path,
    output_root: Path,
    target_per_type: int,
) -> dict[str, Any]:
    return {
        "schema_version": "unified-agent-eval-v1-source-expansion-v2",
        "targets": {
            "search_free": target_per_type,
            "visual_search_required": target_per_type,
            "text_search_required": target_per_type,
            "mixed_search_required": target_per_type,
        },
        "history_data_config": history_data_config,
        "sources": {
            "existing_candidates": {
                "enabled": True,
                "adapter": "existing_unified_eval_candidates",
                "data_config": history_data_config,
            },
            "visual_infoseek_2023": {
                "enabled": True,
                "adapter": "generic_heldout",
                "root_dir": str(normalized_root),
                "manifest_file": "source_manifest.json",
                "evidence_mode": "text",
                "field_map": {
                    "source_id": "source_data_id",
                    "question": "question",
                    "answers": "answer_aliases",
                    "query_image": "query_image_path",
                    "image_evidence": "image_search_records",
                    "text_evidence": "text_corpus_records",
                    "evidence": "offline_evidence_records",
                    "eligible_task_types": "eligible_task_types",
                    "source_split": "source_split",
                },
            },
        },
        "capacity": {
            "solver": "exact_max_flow",
            "require_full_joint_quota": True,
            "generate_review_package_only_if_capacity_passes": True,
        },
        "boundaries": {
            "publish_processed_dataset": False,
            "create_formal_approval": False,
            "freeze_environment": False,
            "run_model_evaluation": False,
            "open_test_embargo": False,
        },
        "acquisition_mode": True,
        "output": {"staging_dir": str(output_root)},
    }


def evaluate_capacity(
    project_root: Path,
    *,
    history_data_config: str,
    normalized_root: Path,
    output_root: Path,
    target_per_type: int = 250,
) -> dict[str, Any]:
    config = expansion_config(
        history_data_config=history_data_config,
        normalized_root=normalized_root,
        output_root=output_root,
        target_per_type=target_per_type,
    )
    return build_source_expansion(Path(project_root), config)


def capacity_stop_status(
    capacity: Mapping[str, Any],
    *,
    formal_target_per_type: int,
    buffer_target_per_type: int,
) -> str | None:
    assignment = capacity["joint_assignment"]
    values = [
        int(assignment["visual_assigned"]),
        int(assignment["text_assigned"]),
        int(assignment["mixed_assigned"]),
    ]
    eligible = capacity.get("candidate_capacity", {})
    buffer_values = [
        int(eligible.get("visual_eligible_count", 0)),
        int(eligible.get("text_eligible_count", 0)),
        int(eligible.get("mixed_eligible_count", 0)),
    ]
    if (
        min(values) >= formal_target_per_type
        and min(buffer_values) >= buffer_target_per_type
    ):
        return "INFOSEEK_CAPACITY_BUFFER_READY"
    if min(values) >= formal_target_per_type:
        return "INFOSEEK_FORMAL_CAPACITY_READY"
    return None


def save_checkpoint(path: Path, value: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(
            dict(value), ensure_ascii=False, indent=2, sort_keys=True
        ) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def load_checkpoint(path: Path) -> dict[str, Any] | None:
    path = Path(path)
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))
