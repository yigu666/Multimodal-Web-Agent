from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from multimodal_web_agent.data.protocol_sft.schema import StateActionExample

from .config import SFTConfig, sha256_file
from .dataset import load_split


V04_SCHEMA = "protocol-sft-v0.4"
V04_SPLIT_COUNTS = {"train": 800, "dev": 100, "test": 100}
V04_ROUTE_COUNTS = {
    "direct_answer": 310,
    "image_search_answer": 290,
    "image_text_search_answer": 10,
    "text_search_answer": 40,
}
V04_TRANSITION_COUNTS = {
    "all": {
        "initial_to_direct_answer": 310,
        "initial_to_image_search": 300,
        "image_information_to_answer": 290,
        "initial_to_text_search": 40,
        "image_information_to_text_search": 10,
        "text_information_to_answer": 50,
    },
    "train": {
        "initial_to_direct_answer": 248,
        "initial_to_image_search": 240,
        "image_information_to_answer": 232,
        "initial_to_text_search": 32,
        "image_information_to_text_search": 8,
        "text_information_to_answer": 40,
    },
    "dev": {
        "initial_to_direct_answer": 31,
        "initial_to_image_search": 30,
        "image_information_to_answer": 29,
        "initial_to_text_search": 4,
        "image_information_to_text_search": 1,
        "text_information_to_answer": 5,
    },
    "test": {
        "initial_to_direct_answer": 31,
        "initial_to_image_search": 30,
        "image_information_to_answer": 29,
        "initial_to_text_search": 4,
        "image_information_to_text_search": 1,
        "text_information_to_answer": 5,
    },
}
ZERO_QUALITY_FIELDS = (
    "strict_parser_invalid_targets",
    "answer_leak_count",
    "query_answer_leak_count",
    "unavailable_context_leak_count",
    "generic_query_count",
    "missing_visible_entity_count",
    "unresolvable_visual_reference_count",
    "conflicting_route_label_count",
    "no_valid_action_count",
    "unreachable_evidence_count",
    "source_route_reuse_count",
    "source_split_leak_count",
    "trajectory_split_leak_count",
    "entity_group_split_leak_count",
    "near_duplicate_split_leak_count",
    "target_truncation_count",
)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("JSON root must be an object: %s" % path)
    return value


def transition_counts(
    examples: Sequence[StateActionExample],
) -> dict[str, int]:
    return dict(sorted(Counter(row.transition for row in examples).items()))


def assert_transition_counts(
    examples: Sequence[StateActionExample],
    split: str,
) -> dict[str, int]:
    actual = transition_counts(examples)
    expected = V04_TRANSITION_COUNTS[split]
    if actual != expected:
        raise ValueError(
            "v0.4 %s transition counts differ: actual=%r expected=%r"
            % (split, actual, expected)
        )
    return actual


def validate_v0_4_training_contract(
    config: SFTConfig,
) -> tuple[dict[str, Any], dict[str, list[StateActionExample]]]:
    if not config.is_v0_4:
        raise ValueError("v0.4 contract requires protocol-sft-v0.4")
    if config.data.manifest_file is None or config.data.audit_file is None:
        raise ValueError("v0.4 manifest and audit paths are required")
    manifest = _read_json(config.data.manifest_file)
    audit = _read_json(config.data.audit_file)
    if manifest.get("schema_version") != V04_SCHEMA:
        raise ValueError("v0.4 manifest schema mismatch")
    if audit.get("schema_version") != V04_SCHEMA + "-audit":
        raise ValueError("v0.4 audit schema mismatch")
    if audit.get("passed") is not True:
        raise ValueError("v0.4 dataset audit has not passed")
    if int(audit.get("state_action_examples", -1)) != 1000:
        raise ValueError("v0.4 audit state-action count mismatch")
    if audit.get("split_counts") != V04_SPLIT_COUNTS:
        raise ValueError("v0.4 audit split counts mismatch")
    if manifest.get("counts", {}).get("routes") != V04_ROUTE_COUNTS:
        raise ValueError("v0.4 route counts mismatch")
    if manifest.get("counts", {}).get("splits") != V04_SPLIT_COUNTS:
        raise ValueError("v0.4 manifest split counts mismatch")
    if manifest.get("new_test_embargoed") is not True:
        raise ValueError("v0.4 Test is not embargoed")
    if manifest.get("test_preview_generated") is not False:
        raise ValueError("v0.4 Test preview must not exist")
    if manifest.get("test_used_for_selection") is not False:
        raise ValueError("v0.4 Test was used for selection")
    nonzero = {
        key: int(audit.get(key, -1))
        for key in ZERO_QUALITY_FIELDS
        if int(audit.get(key, -1)) != 0
    }
    if nonzero:
        raise ValueError("v0.4 audit quality gates failed: %r" % nonzero)

    splits = {
        "train": load_split(config.data.train_file, 800),
        "dev": load_split(config.data.dev_file, 100),
    }
    counts = {
        split: assert_transition_counts(examples, split)
        for split, examples in splits.items()
    }
    metadata: dict[str, Any] = {
        "dataset_schema": V04_SCHEMA,
        "dataset_manifest_hash": sha256_file(config.data.manifest_file),
        "dataset_audit_hash": sha256_file(config.data.audit_file),
        "dataset_audit_passed": True,
        "state_action_examples": 1000,
        "split_counts": dict(V04_SPLIT_COUNTS),
        "transition_counts": counts,
        "new_test_embargoed": True,
        "test_preview_generated": False,
        "test_used_for_selection": False,
        "test_accessed": False,
    }
    if (
        config.data.master_pool_manifest_file is not None
        and config.data.master_pool_manifest_file.is_file()
    ):
        master = _read_json(config.data.master_pool_manifest_file)
        metadata["validated_master_pool_schema"] = master.get(
            "schema_version"
        )
        metadata["validated_master_pool_manifest_hash"] = sha256_file(
            config.data.master_pool_manifest_file
        )
    else:
        metadata["validated_master_pool_schema"] = None
        metadata["validated_master_pool_manifest_hash"] = None
    return metadata, splits


def experiment_data_provenance(
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "validated_master_pool_schema": contract.get(
            "validated_master_pool_schema"
        ),
        "validated_master_pool_manifest_hash": contract.get(
            "validated_master_pool_manifest_hash"
        ),
        "protocol_sft_schema": contract["dataset_schema"],
        "protocol_sft_manifest_hash": contract["dataset_manifest_hash"],
        "protocol_sft_audit_hash": contract["dataset_audit_hash"],
        "shared_executability_gate": True,
        "data_quality_policy": "reject_first",
        "repair_mode": "reject_only",
        "automatic_query_rewrite": False,
        "automatic_route_conversion": False,
        "dataset_changed_from_v0_3": True,
        "training_algorithm_changed_from_rebalanced_v0_3": False,
        "weighted_loss_enabled": False,
    }
