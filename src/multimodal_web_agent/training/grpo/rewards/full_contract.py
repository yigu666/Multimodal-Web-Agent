from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


FULL_CONTRACT_MARKER = "GRPO_REWARD_V2_FULL_CONTRACT_READY"
FULL_COMPLETE_MARKER = "GRPO_REWARD_V2_FULL_COMPLETE"
FULL_SCHEMA = "grpo-reward-v2-full-training-contract-v1"

FULL_VARIANTS = {
    "reward_v21": {
        "reward_config": "configs/grpo/reward_v2_1_answer_dominant_positive.yaml",
        "full_contract": "outputs/grpo_reward_v21_full_contract/run_manifest.json",
        "output_dir": "outputs/grpo_reward_v21_full",
        "selected_checkpoint": "selected_reward_v21_checkpoint",
    },
}


def full_contract_marker(experiment_id: str) -> str:
    return f"GRPO_{str(experiment_id).upper()}_FULL_CONTRACT_READY"


def full_complete_marker(experiment_id: str) -> str:
    return f"GRPO_{str(experiment_id).upper()}_FULL_COMPLETE"

EXPECTED_SCALE = {
    "prompt_pool": 2048,
    "group_size": 4,
    "total_rollouts": 8192,
    "prompt_groups_per_update": 4,
    "trajectories_per_update": 16,
    "total_updates": 512,
    "pool_passes": 1,
}

ENGINEERING_STOP_KEYS = (
    "nonfinite_reward_advantage_ratio_loss_gradient",
    "behavior_logprob_alignment",
    "input_identity",
    "environment_information_mask",
    "visual_frozen",
    "projector_frozen",
    "checkpoint_save_reload",
    "frozen_artifact_hashes",
    "cumulative_lora_change",
)

DIAGNOSTIC_ONLY_KEYS = (
    "text_search_count",
    "local_advantage_density",
    "average_tool_calls",
    "image_search_count",
    "short_term_reward",
    "average_trajectory_length",
    "search_budget",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tree(path: Path) -> str:
    root = Path(path)
    files = sorted(item for item in root.rglob("*") if item.is_file())
    if not files:
        raise RuntimeError(f"cannot fingerprint empty directory: {root}")
    digest = hashlib.sha256()
    for item in files:
        digest.update(item.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        with item.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def load_full_training_config(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    validate_full_training_config(value)
    return value


def validate_full_training_config(value: Mapping[str, Any]) -> None:
    if value.get("schema_version") != FULL_SCHEMA or value.get("mode") != "full":
        raise ValueError("Reward v2 Full training schema mismatch")
    if dict(value.get("scale", {})) != EXPECTED_SCALE:
        raise ValueError("Reward v2 Full scale differs from the frozen contract")
    experiment_id = str(value.get("experiment_id", "reward_v21"))
    if experiment_id not in FULL_VARIANTS:
        raise ValueError("unknown Answer Reward Full experiment")
    expected_paths = dict(FULL_VARIANTS[experiment_id])
    expected_paths.pop("selected_checkpoint")
    expected_paths["smoke_baseline"] = "outputs/grpo_reward_v2_text_exploration_smoke_128"
    for key, expected in expected_paths.items():
        if value.get(key) != expected:
            raise ValueError(f"Reward v2 Full {key} differs from the frozen path")
    if int(value.get("run_seed", -1)) != 20260730:
        raise ValueError("Answer Reward Full seed differs")
    selected = value.get(
        "selected_checkpoint_dirname",
        FULL_VARIANTS[experiment_id]["selected_checkpoint"],
    )
    if selected != FULL_VARIANTS[experiment_id]["selected_checkpoint"]:
        raise ValueError("Answer Reward selected checkpoint path differs")
    initialization = dict(value.get("initialization", {}))
    if initialization.get("source") != "frozen_sft_adapter":
        raise ValueError("Reward v2 Full must initialize from frozen SFT")
    if initialization.get("reward_v0_checkpoint_allowed") is not False:
        raise ValueError("Reward v0 checkpoint continuation is forbidden")
    if initialization.get("reward_v2_checkpoint_allowed") is not False:
        raise ValueError("Reward v2 checkpoint continuation is forbidden")
    if initialization.get("resume_checkpoint_allowed") is not False:
        raise ValueError("uncontracted checkpoint resume is forbidden")
    adapter_path = str(initialization.get("adapter_path", "")).casefold()
    if any(name in adapter_path for name in (
        "reward_v0", "reward_v2_full", "reward_v21",
    )):
        raise ValueError("Reward checkpoint initialization path is forbidden")
    checkpoint = dict(value.get("checkpoint", {}))
    if checkpoint != {
        "every_prompt_groups": 256,
        "expected_count": 8,
        "save_optimizer_state": False,
        "reload_validate_each_checkpoint": True,
        "final_selection": "prompt_2048",
    }:
        raise ValueError("Reward v2 checkpoint contract differs")
    if dict(value.get("optimizer", {})) != {
        "actor_learning_rate": 5e-7,
        "optimizer": "paged_adamw_8bit",
        "max_grad_norm": 1.0,
        "weight_decay": 0.0,
        "ppo_epochs": 1,
        "clip_ratio": 0.2,
    }:
        raise ValueError("Reward v2 optimizer contract differs")
    if dict(value.get("continuous_contract", {})) != {
        "check_every_update": True,
        "full_hash_check_every_updates": 64,
        "behavior_logprob_max_abs_error": 1e-3,
        "stop_on_nonfinite": True,
        "stop_on_input_identity_failure": True,
        "stop_on_environment_token_loss_leak": True,
        "stop_on_visual_or_projector_change": True,
        "stop_if_no_cumulative_lora_change_by_checkpoint": True,
        "stop_on_checkpoint_save_or_reload_failure": True,
        "stop_on_frozen_artifact_hash_change": True,
    }:
        raise ValueError("Reward v2 continuous engineering contract differs")
    if tuple(value.get("diagnostic_only_never_stop", ())) != DIAGNOSTIC_ONLY_KEYS:
        raise ValueError("diagnostic-only fields differ from the Full contract")
    boundaries = dict(value.get("boundaries", {}))
    if boundaries != {
        "automatic_start": False,
        "automatic_retry": False,
        "reward_v0_checkpoint_access": False,
        "unified_frozen_dev_access_during_training": False,
        "unified_frozen_test_access": False,
    }:
        raise ValueError("Reward v2 Full boundary contract differs")


class FrozenArtifactGuard:
    """Continuous metadata checks plus periodic cryptographic verification."""

    def __init__(
        self, project_root: Path,
        artifacts: Mapping[str, Mapping[str, Any]],
    ) -> None:
        self.project_root = Path(project_root).resolve()
        self.artifacts = {str(key): dict(value) for key, value in artifacts.items()}
        if not self.artifacts:
            raise RuntimeError("Full contract has no frozen artifacts")

    def _path(self, relative: str) -> Path:
        path = (self.project_root / relative).resolve()
        try:
            path.relative_to(self.project_root)
        except ValueError as exc:
            raise RuntimeError("frozen artifact escapes project root") from exc
        return path

    def verify_metadata(self) -> None:
        for relative, expected in self.artifacts.items():
            path = self._path(relative)
            if not path.is_file():
                raise RuntimeError(f"FROZEN_ARTIFACT_CHANGED: missing {relative}")
            stat = path.stat()
            if int(stat.st_size) != int(expected["size"]):
                raise RuntimeError(f"FROZEN_ARTIFACT_CHANGED: size {relative}")
            if int(stat.st_mtime_ns) != int(expected["mtime_ns"]):
                raise RuntimeError(f"FROZEN_ARTIFACT_CHANGED: mtime {relative}")

    def verify_hashes(self) -> None:
        self.verify_metadata()
        for relative, expected in self.artifacts.items():
            actual = sha256_file(self._path(relative))
            if actual != str(expected["sha256"]):
                raise RuntimeError(f"FROZEN_ARTIFACT_CHANGED: sha256 {relative}")


def validate_update_contract(update: Mapping[str, Any]) -> dict[str, bool]:
    finite_fields = (
        "loss", "gradient_norm", "clip_fraction", "policy_ratio_mean",
        "policy_ratio_max", "max_logprob_alignment_error",
        "exploration_behavior_logprob_error_max",
    )
    finite = all(
        math.isfinite(float(update.get(key, math.nan))) for key in finite_fields
    ) and bool(update.get("gradient_finite"))
    behavior = max(
        float(update.get("max_logprob_alignment_error", math.inf)),
        float(update.get("exploration_behavior_logprob_error_max", math.inf)),
    ) < 1e-3
    identity = all(
        int(update.get(key, 1)) == 0
        for key in (
            "input_identity_failure_count", "response_token_mismatch_count",
            "processor_hash_mismatch_count", "target_truncation_count",
        )
    )
    information = int(update.get("information_token_train_mask_sum", 1)) == 0
    gates = {
        "nonfinite_reward_advantage_ratio_loss_gradient": finite,
        "behavior_logprob_alignment": behavior,
        "input_identity": identity,
        "environment_information_mask": information,
    }
    failed = [key for key, passed in gates.items() if not passed]
    if failed:
        raise RuntimeError("FULL_CONTINUOUS_CONTRACT_FAILED: " + ", ".join(failed))
    return gates


def numeric_values_finite(values: Any) -> bool:
    if isinstance(values, bool) or values is None:
        return True
    if isinstance(values, (int, float)):
        return math.isfinite(float(values))
    if isinstance(values, Mapping):
        return all(numeric_values_finite(value) for value in values.values())
    if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
        return all(numeric_values_finite(value) for value in values)
    return True


def read_contract_manifest(
    path: Path, *, experiment_id: str = "reward_v21"
) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    expected_marker = full_contract_marker(experiment_id)
    if value.get("marker") != expected_marker:
        raise RuntimeError("Answer Reward Full contract marker is invalid")
    if value.get("experiment_id", experiment_id) != experiment_id:
        raise RuntimeError("Answer Reward Full contract identity differs")
    if value.get("full_execution_authorized") is not True:
        raise RuntimeError("Reward v2 Full execution is not authorized")
    if value.get("automatic_start") is not False:
        raise RuntimeError("Reward v2 Full contract permits automatic start")
    if value.get("unified_frozen_test_accessed") is not False:
        raise RuntimeError("Reward v2 Full contract accessed Frozen Test")
    return value
