from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

from .fingerprints import sha256_file, sha256_tree


GRPO_PROVENANCE_SCHEMA = "unified-agent-eval-v1-1-grpo-provenance-v1"
GRPO_RECOVERY_SCHEMA = "grpo-reward-v0-full-recovery-v1"


def _resolve(project_root: Path, value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = Path(project_root) / path
    return path.resolve()


def validate_grpo_recovery(
    project_root: Path,
    configured: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate that the configured GRPO Adapter is the recovered Full output."""
    root = Path(project_root).resolve()
    if configured.get("enabled") is not True:
        raise ValueError("GRPO must be enabled for evaluation")
    if configured.get("stage") != "grpo":
        raise ValueError("GRPO evaluation stage mismatch")
    adapter = _resolve(root, str(configured["adapter_path"]))
    manifest_path = _resolve(root, str(configured["recovery_manifest"]))
    if not adapter.is_dir():
        raise FileNotFoundError("recovered GRPO Adapter is missing: %s" % adapter)
    if not (adapter / "adapter_model.safetensors").is_file():
        raise FileNotFoundError("recovered GRPO Adapter tensor file is missing")
    if not manifest_path.is_file():
        raise FileNotFoundError("GRPO recovery manifest is missing")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    required_equal = {
        "schema_version": GRPO_RECOVERY_SCHEMA,
        "stage": "full_recovered",
        "recovery_passed": True,
        "training_compute_complete": True,
        "canonical_full_contract_complete": False,
        "prompt_count": 2048,
        "rollout_count": 8192,
        "group_size": 4,
        "optimizer_steps": 512,
        "source_rollout_group_count": 2048,
        "source_checkpoint_count": 8,
        "source_dev_evaluation_count": 5,
        "test_accessed": False,
        "mmsearch_accessed": False,
        "per_update_metrics_available": False,
        "per_update_metrics_fabricated": False,
    }
    mismatches = {
        key: {"expected": expected, "actual": manifest.get(key)}
        for key, expected in required_equal.items()
        if manifest.get(key) != expected
    }
    if mismatches:
        raise RuntimeError(
            "GRPO recovery contract mismatch: %s"
            % json.dumps(mismatches, ensure_ascii=False, sort_keys=True)
        )

    engineering = manifest.get("engineering_contract", {})
    checkpoint_evidence = manifest.get("lora_checkpoint_evidence", {})
    if engineering.get("engineering_hard_gates_recovered") is not True:
        raise RuntimeError("GRPO recovered engineering gates are not passing")
    if engineering.get("original_runtime_lora_gate_was_false_negative") is not True:
        raise RuntimeError("GRPO recovery does not prove the LoRA false negative")
    if int(engineering.get("lora_parameter_change_count_lower_bound", 0)) <= 0:
        raise RuntimeError("GRPO recovery has no changed LoRA parameter evidence")
    if checkpoint_evidence.get("lora_change_proved_by_tensor_content") is not True:
        raise RuntimeError("GRPO LoRA tensor-content change was not proved")
    if int(checkpoint_evidence.get("unique_checkpoint_file_hash_count", 0)) != 8:
        raise RuntimeError("GRPO recovery does not contain eight distinct checkpoints")

    selected_name = str(manifest.get("selected_reward_v0_checkpoint", ""))
    selected_from_selection = str(
        manifest.get("checkpoint_selection", {}).get("selected_checkpoint", "")
    )
    expected_adapter = (manifest_path.parent / selected_name).resolve()
    if not selected_name or selected_name != selected_from_selection:
        raise RuntimeError("GRPO selected checkpoint identity is inconsistent")
    if expected_adapter != adapter:
        raise RuntimeError("configured GRPO Adapter is not the recovered selection")

    adapter_tree_hash = sha256_tree(adapter)
    if manifest.get("selected_checkpoint_tree_sha256") != adapter_tree_hash:
        raise RuntimeError("recovered GRPO Adapter tree fingerprint mismatch")
    unavailable = manifest_path.parent / "update_metrics.unavailable.json"
    if not unavailable.is_file():
        raise RuntimeError("GRPO update-metrics limitation artifact is missing")
    limitation = json.loads(unavailable.read_text(encoding="utf-8"))
    if (
        limitation.get("available") is not False
        or int(limitation.get("expected_row_count", -1)) != 512
        or int(limitation.get("fabricated_rows", -1)) != 0
    ):
        raise RuntimeError("GRPO update-metrics limitation artifact is invalid")
    if (manifest_path.parent / "update_metrics.jsonl").exists():
        raise RuntimeError("unexpected reconstructed GRPO update_metrics.jsonl")

    return {
        "schema_version": GRPO_PROVENANCE_SCHEMA,
        "model_id": "grpo",
        "evaluation_release": "unified-agent-eval-v1-1",
        "eligible_for_frozen_dev": True,
        "eligible_for_frozen_test_after_four_model_registration": True,
        "recovery_manifest": Path(configured["recovery_manifest"]).as_posix(),
        "recovery_manifest_sha256": sha256_file(manifest_path),
        "adapter_path": Path(configured["adapter_path"]).as_posix(),
        "adapter_tree_sha256": adapter_tree_hash,
        "training_compute_complete": True,
        "canonical_full_contract_complete": False,
        "optimizer_steps": 512,
        "prompt_count": 2048,
        "rollout_count": 8192,
        "checkpoint_count": 8,
        "frozen_grpo_dev_evaluation_count": 5,
        "lora_change_proved_by_tensor_content": True,
        "per_update_metrics_available": False,
        "per_update_metrics_fabricated": False,
        "audit_limitation": str(manifest["audit_limitation"]),
        "unified_eval_test_accessed": False,
        "grpo_training_test_accessed": False,
        "mmsearch_accessed": False,
    }


def provenance_path(
    project_root: Path,
    configured: Mapping[str, Any],
) -> Path:
    return _resolve(project_root, str(configured["evaluation_provenance"]))


def ensure_immutable_provenance(path: Path, expected: Mapping[str, Any]) -> None:
    path = Path(path)
    value = dict(expected)
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != value:
            raise RuntimeError("registered GRPO evaluation provenance is immutable")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp-%d" % os.getpid())
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)
