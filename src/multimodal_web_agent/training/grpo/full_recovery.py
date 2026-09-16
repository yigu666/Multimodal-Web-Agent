from __future__ import annotations

from collections import Counter
import gc
import json
import math
from pathlib import Path
import re
import shutil
from typing import Any, Mapping, Sequence

import torch

from .formal_contract import (
    ALLOWED_REWARD_VALUES,
    BEHAVIOR_CONTRACT_FIELDS,
    behavior_diagnostics,
    formal_chain_manifest_fields,
)
from .schema import PromptPoolItem, RolloutRecord, validate_rollout_group
from .server_runner import (
    _compute_group_values,
    _public_rollout,
    _reward_summary,
    _validate_pool,
    read_jsonl,
    sha256_file,
    sha256_tree,
    write_json,
)
from .reward_v0_mmsearch_like import REWARD_NAME, score_reward_v0


RECOVERY_SCHEMA = "grpo-reward-v0-full-recovery-v1"
EXPECTED_ADAPTER_SHA256 = (
    "45baf20eb386804c717013303989e9d382d2b4299de6090a6b4abde7799fa2d5"
)
EXPECTED_CHECKPOINT_PROMPTS = tuple(range(256, 2049, 256))
EXPECTED_DEV_PROMPTS = (0, 512, 1024, 1536, 2048)


def _validate_formal_chain(project_root: Path) -> dict[str, Any]:
    paths = {
        "prerequisite": project_root
        / "outputs/grpo/reward_v0_top3_prerequisites/run_manifest.json",
        "reward_audit": project_root
        / "outputs/grpo/reward_v0_audit/run_manifest.json",
        "contract_smoke": project_root
        / "outputs/grpo/reward_v0_contract_smoke/run_manifest.json",
        "integration": project_root
        / "outputs/grpo/text_search_path_integration_v1/run_manifest.json",
        "smoke_v2": project_root
        / "outputs/grpo/reward_v0_smoke_v2/run_manifest.json",
    }
    if not all(path.is_file() for path in paths.values()):
        raise RuntimeError("formal GRPO prerequisite manifest set is incomplete")
    values = {
        name: json.loads(path.read_text(encoding="utf-8"))
        for name, path in paths.items()
    }
    prerequisite = values["prerequisite"]
    reward_audit = values["reward_audit"]
    contract_smoke = values["contract_smoke"]
    integration = values["integration"]
    smoke_v2 = values["smoke_v2"]
    checks = (
        prerequisite.get("passed") is True,
        prerequisite.get("contract_reuse_allowed") is True,
        prerequisite.get("reward_audit_rerun_required") is False,
        reward_audit.get("stage") == "reward_audit",
        int(reward_audit.get("rollout_count", -1)) == 1024,
        contract_smoke.get("stage") == "contract_smoke",
        int(contract_smoke.get("optimizer_steps", -1)) == 1,
        integration.get("passed") is True,
        integration.get("marker") == "GRPO_TEXT_SEARCH_PATH_INTEGRATION_PASS",
        integration.get("used_for_training") is False,
        integration.get("used_for_reward") is False,
        integration.get("used_for_advantage") is False,
        smoke_v2.get("stage") == "smoke_v2",
        int(smoke_v2.get("optimizer_steps", -1)) == 8,
        smoke_v2.get("engineering_contract", {}).get(
            "engineering_hard_gates_passed"
        )
        is True,
        smoke_v2.get("rank1_probe_used_for_training") is False,
        smoke_v2.get("rank1_probe_used_for_gate") is False,
        smoke_v2.get("rank1_probe_used_for_selection") is False,
    )
    if not all(checks):
        raise RuntimeError("formal GRPO prerequisite chain is not passing")
    if not all(value.get("test_accessed") is False for value in values.values()):
        raise RuntimeError("formal GRPO prerequisite chain accessed Test")
    expected_environment = formal_chain_manifest_fields()["formal_environment"]
    for name in ("prerequisite", "integration", "smoke_v2"):
        if values[name].get("formal_environment") != expected_environment:
            raise RuntimeError(f"{name} formal environment is not frozen top-3")
    if prerequisite.get("reward_name") != REWARD_NAME or smoke_v2.get(
        "reward_name"
    ) != REWARD_NAME:
        raise RuntimeError("formal chain Reward v0 identity mismatch")
    if prerequisite.get("adapter_sha256") != smoke_v2.get("adapter_hash"):
        raise RuntimeError("formal chain adapter fingerprint mismatch")
    return {
        "validated": True,
        "manifest_sha256": {
            name: sha256_file(path) for name, path in paths.items()
        },
    }


def parse_completed_full_lora_false_negative_log(text: str) -> dict[str, Any]:
    failures = re.findall(
        r"Full engineering contract failed:\s*([^\r\n]+)", text
    )
    if len(failures) != 1 or failures[0].strip() != "lora_change":
        raise RuntimeError(
            "console log does not prove the isolated lora_change false negative"
        )
    required = (
        "[full] optimizer_step=512/512 prompts=2048/2048",
        "[full] running Frozen Dev evaluation at prompt 2048",
    )
    missing = [marker for marker in required if marker not in text]
    if missing:
        raise RuntimeError(f"console log is missing completion markers: {missing}")
    return {
        "optimizer_steps_observed": 512,
        "prompts_observed": 2048,
        "final_frozen_dev_started": True,
        "runtime_final_gate_failure_set": ["lora_change"],
        "all_other_runtime_final_gates_passed": True,
    }


def _checkpoint_tensor_evidence(
    *, initial_adapter_file: Path, checkpoint_files: Sequence[Path]
) -> dict[str, Any]:
    from safetensors import safe_open

    if len(checkpoint_files) != 8:
        raise RuntimeError("recovery requires exactly eight adapter checkpoints")
    file_hashes = [sha256_file(path) for path in checkpoint_files]
    if len(set(file_hashes)) != len(file_hashes):
        raise RuntimeError("LoRA checkpoints are not all content-distinct")
    first_path, final_path = checkpoint_files[0], checkpoint_files[-1]
    with safe_open(first_path, framework="pt", device="cpu") as first, safe_open(
        final_path, framework="pt", device="cpu"
    ) as final:
        first_keys = set(first.keys())
        final_keys = set(final.keys())
        if first_keys != final_keys or not first_keys:
            raise RuntimeError("checkpoint tensor key set changed or is empty")
        changed = 0
        finite = True
        maximum_delta = 0.0
        for key in sorted(first_keys):
            left = first.get_tensor(key)
            right = final.get_tensor(key)
            finite = finite and bool(torch.isfinite(left).all())
            finite = finite and bool(torch.isfinite(right).all())
            if not torch.equal(left, right):
                changed += 1
                maximum_delta = max(
                    maximum_delta,
                    float((left.float() - right.float()).abs().max()),
                )
    if not finite:
        raise RuntimeError("checkpoint contains non-finite LoRA tensors")
    if changed <= 0 or maximum_delta <= 0.0:
        raise RuntimeError("checkpoint tensors do not prove LoRA change")

    with safe_open(
        initial_adapter_file, framework="pt", device="cpu"
    ) as initial, safe_open(final_path, framework="pt", device="cpu") as final:
        initial_keys = set(initial.keys())
        final_keys = set(final.keys())
        if initial_keys != final_keys or not initial_keys:
            raise RuntimeError(
                "SFT-init and final checkpoint tensor key sets differ or are empty"
            )
        init_to_final_changed = 0
        init_to_final_maximum_delta = 0.0
        for key in sorted(initial_keys):
            left = initial.get_tensor(key)
            right = final.get_tensor(key)
            if left.shape != right.shape or left.dtype != right.dtype:
                raise RuntimeError(
                    "SFT-init and final checkpoint tensor contracts differ"
                )
            if not bool(torch.isfinite(left).all()) or not bool(
                torch.isfinite(right).all()
            ):
                raise RuntimeError("adapter comparison contains non-finite tensors")
            if not torch.equal(left, right):
                init_to_final_changed += 1
                init_to_final_maximum_delta = max(
                    init_to_final_maximum_delta,
                    float((left.float() - right.float()).abs().max()),
                )
    if init_to_final_changed <= 0 or init_to_final_maximum_delta <= 0.0:
        raise RuntimeError("SFT-init and final LoRA tensor contents are identical")
    return {
        "initial_adapter_file_sha256": sha256_file(initial_adapter_file),
        "checkpoint_file_sha256": file_hashes,
        "unique_checkpoint_file_hash_count": len(set(file_hashes)),
        "first_vs_final_changed_tensor_count": changed,
        "first_vs_final_max_abs_delta": maximum_delta,
        "sft_init_vs_final_changed_tensor_count": init_to_final_changed,
        "sft_init_vs_final_max_abs_delta": init_to_final_maximum_delta,
        "all_checkpoint_tensors_finite": True,
        "lora_change_proved_by_tensor_content": True,
    }


def select_recovered_checkpoint(
    evaluations: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    if len(evaluations) != 5:
        raise RuntimeError("recovery requires SFT Init plus four Dev evaluations")
    init = evaluations[0]
    eligible = []
    for row in evaluations[1:]:
        passes = (
            float(row.get("protocol_valid_rate", 0.0))
            >= float(init.get("protocol_valid_rate", 0.0)) - 0.01
            and float(row.get("malformed_rate", 1.0))
            <= float(init.get("malformed_rate", 1.0)) + 0.01
            and float(row.get("finish_rate", 0.0))
            >= float(init.get("finish_rate", 0.0)) - 0.02
            and int(row.get("forged_information_count", 0)) == 0
        )
        if passes:
            eligible.append(row)
    if not eligible:
        raise RuntimeError("no recovered checkpoint passes protocol gates")
    eligible.sort(
        key=lambda row: (
            -float(row.get("overall_f1", 0.0)),
            -float(row.get("search_required_em", 0.0)),
            float(row.get("unnecessary_search_rate", 1.0)),
            -float(row.get("protocol_valid_rate", 0.0)),
            float(row.get("average_search_calls", 999.0)),
            int(row["prompt_count_processed"]),
        )
    )
    return eligible[0]


def _load_group(path: Path) -> list[RolloutRecord]:
    try:
        value = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        value = torch.load(path, map_location="cpu")
    if not isinstance(value, list) or not all(
        isinstance(record, RolloutRecord) for record in value
    ):
        raise RuntimeError(f"invalid tensor rollout group: {path}")
    return value


def _write_line(handle: Any, value: Mapping[str, Any]) -> None:
    handle.write(json.dumps(dict(value), ensure_ascii=False, sort_keys=True) + "\n")


def recover_completed_full(
    *,
    project_root: Path,
    failed_dir: Path,
    console_log: Path,
    output_dir: Path,
) -> dict[str, Any]:
    project_root = project_root.resolve()
    failed_dir = failed_dir.resolve()
    console_log = console_log.resolve()
    output_dir = output_dir.resolve()
    if not failed_dir.is_dir() or not console_log.is_file():
        raise FileNotFoundError("failed Full directory or console log is missing")
    if output_dir.exists():
        raise FileExistsError(output_dir)
    temporary = output_dir.with_name(output_dir.name + ".tmp.recovery")
    if temporary.exists():
        raise FileExistsError(temporary)
    temporary.mkdir(parents=True)

    log_evidence = parse_completed_full_lora_false_negative_log(
        console_log.read_text(encoding="utf-8", errors="replace")
    )
    formal_chain_evidence = _validate_formal_chain(project_root)
    pool_manifest, pool_audit = _validate_pool(project_root)
    pool_rows = read_jsonl(
        project_root / "data/processed/grpo_prompt_pool_v1/train.jsonl"
    )
    prompts = [PromptPoolItem.from_dict(row) for row in pool_rows]
    if len(prompts) != 2048:
        raise RuntimeError("current frozen GRPO Train Pool is not 2048 prompts")

    group_paths = sorted((failed_dir / "rollout_records").glob("group_*.pt"))
    expected_group_names = [f"group_{index:06d}.pt" for index in range(2048)]
    if [path.name for path in group_paths] != expected_group_names:
        raise RuntimeError("failed Full does not contain 2048 continuous groups")

    checkpoint_dirs = [
        failed_dir / "checkpoints" / f"prompt_{prompt:04d}"
        for prompt in EXPECTED_CHECKPOINT_PROMPTS
    ]
    checkpoint_files = [path / "adapter_model.safetensors" for path in checkpoint_dirs]
    if not all(path.is_file() for path in checkpoint_files):
        raise RuntimeError("failed Full checkpoint set is incomplete")
    initial_adapter_dir = (
        project_root / "outputs/protocol_format_sft_v1_full/selected_adapter"
    )
    initial_adapter_file = initial_adapter_dir / "adapter_model.safetensors"
    if not initial_adapter_file.is_file():
        raise RuntimeError("frozen SFT-init adapter tensor file is missing")
    if sha256_tree(initial_adapter_dir) != EXPECTED_ADAPTER_SHA256:
        raise RuntimeError("frozen SFT-init adapter fingerprint mismatch")
    checkpoint_evidence = _checkpoint_tensor_evidence(
        initial_adapter_file=initial_adapter_file,
        checkpoint_files=checkpoint_files,
    )

    dev_paths = [failed_dir / "dev_evaluations" / "sft_init.json"] + [
        failed_dir / "dev_evaluations" / f"prompt_{prompt:04d}.json"
        for prompt in EXPECTED_DEV_PROMPTS[1:]
    ]
    if not all(path.is_file() for path in dev_paths):
        raise RuntimeError("failed Full Frozen Dev set is incomplete")
    evaluations = [json.loads(path.read_text(encoding="utf-8")) for path in dev_paths]
    if [int(row["prompt_count_processed"]) for row in evaluations] != list(
        EXPECTED_DEV_PROMPTS
    ):
        raise RuntimeError("Frozen Dev prompt schedule mismatch")

    public_path = temporary / "rollout_records.jsonl"
    components_path = temporary / "reward_components.jsonl"
    groups_path = temporary / "group_statistics.jsonl"
    all_records: list[RolloutRecord] = []
    all_group_stats: list[dict[str, Any]] = []
    rollout_uids: set[str] = set()
    processor_hashes: set[str] = set()
    chat_template_hashes: set[str] = set()
    base_model_hashes: set[str] = set()
    adapter_hashes: set[str] = set()
    action_token_count = 0
    information_train_mask_sum = 0
    category_counts: Counter[str] = Counter()

    with public_path.open("w", encoding="utf-8", newline="\n") as public_handle, \
         components_path.open("w", encoding="utf-8", newline="\n") as component_handle, \
         groups_path.open("w", encoding="utf-8", newline="\n") as group_handle:
        for index, (path, prompt) in enumerate(zip(group_paths, prompts)):
            records = _load_group(path)
            validate_rollout_group(records, group_size=4)
            if records[0].prompt_uid != prompt.prompt_uid:
                raise RuntimeError(f"prompt order mismatch at group {index}")
            if {record.rollout_index for record in records} != {0, 1, 2, 3}:
                raise RuntimeError(f"rollout index contract failed at group {index}")
            advantages, stats = _compute_group_values(records)
            if not bool(torch.isfinite(advantages).all()):
                raise RuntimeError(f"non-finite recovered advantages at group {index}")
            if len(stats) != 1:
                raise RuntimeError(f"invalid group statistics at group {index}")
            row = stats[0]
            if not all(
                math.isfinite(float(row[key]))
                for key in (
                    "group_reward_mean",
                    "group_reward_std",
                    "advantage_mean",
                    "advantage_std",
                )
            ):
                raise RuntimeError(f"non-finite group statistics at group {index}")
            if row["zero_variance_group"] and (
                abs(float(row["advantage_mean"])) > 1e-7
                or abs(float(row["advantage_std"])) > 1e-7
            ):
                raise RuntimeError(f"zero-variance advantage failure at group {index}")
            all_group_stats.append(row)
            _write_line(group_handle, row)

            for record in records:
                if record.rollout_uid in rollout_uids:
                    raise RuntimeError("duplicate recovered rollout_uid")
                rollout_uids.add(record.rollout_uid)
                if record.category != prompt.category:
                    raise RuntimeError("recovered rollout category mismatch")
                if record.data_id != prompt.data_id:
                    raise RuntimeError("recovered rollout data identity mismatch")
                if record.image_sha256 != prompt.image_sha256:
                    raise RuntimeError("recovered rollout image identity mismatch")
                category_counts[record.category] += 1
                reward = float(record.reward_total)
                if min(abs(reward - allowed) for allowed in ALLOWED_REWARD_VALUES) > 1e-6:
                    raise RuntimeError("recovered rollout has undefined Reward v0")
                recomputed_reward = score_reward_v0(
                    record, prompt.to_dict()
                ).to_dict()
                if record.reward_components != recomputed_reward or abs(
                    reward - float(recomputed_reward["reward_total"])
                ) > 1e-9:
                    raise RuntimeError("recovered Reward v0 replay mismatch")
                full_ids = torch.as_tensor(record.full_input_ids)
                response_ids = torch.as_tensor(record.response_ids)
                policy_mask = torch.as_tensor(record.policy_action_mask)
                information_mask = torch.as_tensor(record.information_mask)
                old_log_probs = torch.as_tensor(record.old_log_probs)
                if int(full_ids.numel()) > 1536:
                    raise RuntimeError("recovered target truncation contract failed")
                if not torch.equal(full_ids[1:], response_ids):
                    raise RuntimeError("recovered response token mismatch")
                if old_log_probs.shape != response_ids.shape or not bool(
                    torch.isfinite(old_log_probs).all()
                ):
                    raise RuntimeError("recovered old log-prob contract failed")
                overlap = int((policy_mask * information_mask).sum().item())
                information_train_mask_sum += overlap
                action_token_count += int(policy_mask.sum().item())
                if overlap != 0 or int(policy_mask.sum().item()) <= 0:
                    raise RuntimeError("recovered policy/information mask failure")
                processor_hashes.add(record.processor_hash)
                chat_template_hashes.add(record.chat_template_hash)
                base_model_hashes.add(record.base_model_hash)
                adapter_hashes.add(record.adapter_hash)
                _write_line(public_handle, _public_rollout(record))
                _write_line(
                    component_handle,
                    dict(
                        record.reward_components,
                        prompt_uid=record.prompt_uid,
                        rollout_uid=record.rollout_uid,
                    ),
                )
                for field in (
                    "full_input_ids",
                    "attention_mask",
                    "position_ids",
                    "response_ids",
                    "policy_action_mask",
                    "information_mask",
                    "old_log_probs",
                    "pixel_values",
                    "image_grid_thw",
                ):
                    setattr(record, field, None)
                all_records.append(record)
            del records
            if (index + 1) % 64 == 0:
                print(
                    f"[full_recovery] groups={index + 1}/2048 "
                    f"rollouts={(index + 1) * 4}",
                    flush=True,
                )
                gc.collect()

    if len(all_records) != 8192 or len(all_group_stats) != 2048:
        raise RuntimeError("recovered Full cardinality mismatch")
    if category_counts != {"search_free": 3072, "search_required": 5120}:
        raise RuntimeError(f"recovered category counts mismatch: {category_counts}")
    if information_train_mask_sum != 0 or action_token_count <= 0:
        raise RuntimeError("recovered aggregate mask contract failed")
    for name, values in (
        ("processor", processor_hashes),
        ("chat_template", chat_template_hashes),
        ("base_model", base_model_hashes),
        ("adapter", adapter_hashes),
    ):
        if len(values) != 1 or not next(iter(values)):
            raise RuntimeError(f"recovered {name} fingerprint mismatch")
    if next(iter(adapter_hashes)) != EXPECTED_ADAPTER_SHA256:
        raise RuntimeError("recovered rollout SFT adapter fingerprint mismatch")

    reward_metrics = _reward_summary(all_records, all_group_stats)
    behavior = behavior_diagnostics(
        all_records, all_group_stats, reward_metrics=reward_metrics
    )
    write_json(temporary / "reward_distribution.json", reward_metrics)
    write_json(temporary / "behavior_diagnostics.json", behavior)
    write_json(
        temporary / "update_metrics.unavailable.json",
        {
            "available": False,
            "expected_row_count": 512,
            "reason": (
                "The original pre-fix runner retained per-update metrics only "
                "in process memory and raised at the faulty final LoRA gate."
            ),
            "fabricated_rows": 0,
            "runtime_gate_evidence": log_evidence,
        },
    )

    selected_metrics = select_recovered_checkpoint(evaluations)
    selected_prompt = int(selected_metrics["prompt_count_processed"])
    selected_source = failed_dir / "checkpoints" / f"prompt_{selected_prompt:04d}"
    selected_output = temporary / "selected_reward_v0_checkpoint"
    shutil.copytree(selected_source, selected_output)
    selection = {
        "selected_checkpoint": "selected_reward_v0_checkpoint",
        "source_failed_checkpoint": str(selected_source),
        "selected_prompt_count": selected_prompt,
        "metrics": dict(selected_metrics),
        "final_teacher_selected": False,
        "recovered_selection": True,
    }
    write_json(temporary / "checkpoint_selection.json", selection)

    recovered_engineering_contract = {
        "engineering_hard_gates_recovered": True,
        "original_runtime_all_non_lora_gates_passed": True,
        "original_runtime_lora_gate_was_false_negative": True,
        "lora_change_detection_method": (
            "sft_init_and_checkpoint_safetensors_content_comparison"
        ),
        "lora_parameter_change_count_lower_bound": checkpoint_evidence[
            "sft_init_vs_final_changed_tensor_count"
        ],
        "checkpoint_count": len(checkpoint_dirs),
        "frozen_dev_evaluation_count": len(dev_paths),
        "test_accessed": False,
    }
    write_json(
        temporary / "full_engineering_contract.recovered.json",
        recovered_engineering_contract,
    )

    manifest = {
        "schema_version": RECOVERY_SCHEMA,
        "stage": "full_recovered",
        "status": "recovered_with_explicit_update_metrics_limitation",
        "recovery_passed": True,
        "training_compute_complete": True,
        "canonical_full_contract_complete": False,
        "prompt_count": 2048,
        "rollout_count": 8192,
        "group_size": 4,
        "prompt_groups_per_update": 4,
        "optimizer_steps": 512,
        "pool_passes": 1,
        "reward_name": REWARD_NAME,
        "test_accessed": False,
        "mmsearch_accessed": False,
        "source_failed_run": str(failed_dir),
        "source_console_log": str(console_log),
        "source_console_sha256": sha256_file(console_log),
        "formal_chain_evidence": formal_chain_evidence,
        "prompt_pool_manifest_sha256": sha256_file(
            project_root / "data/processed/grpo_prompt_pool_v1/manifest.json"
        ),
        "prompt_pool_audit_sha256": sha256_file(
            project_root / "data/processed/grpo_prompt_pool_v1/audit.json"
        ),
        "prompt_pool_schema_version": pool_manifest.get("schema_version"),
        "prompt_pool_audit_passed": bool(pool_audit.get("passed")),
        "source_rollout_group_count": len(group_paths),
        "source_checkpoint_count": len(checkpoint_dirs),
        "source_dev_evaluation_count": len(dev_paths),
        "final_frozen_dev_artifact_present": True,
        "rollout_uid_count": len(rollout_uids),
        "category_rollout_counts": dict(category_counts),
        "information_token_train_mask_sum": information_train_mask_sum,
        "policy_action_token_count": action_token_count,
        "processor_hash": next(iter(processor_hashes)),
        "chat_template_hash": next(iter(chat_template_hashes)),
        "base_model_hash": next(iter(base_model_hashes)),
        "adapter_hash": next(iter(adapter_hashes)),
        "selected_reward_v0_checkpoint": "selected_reward_v0_checkpoint",
        "checkpoint_selection": selection,
        "original_runtime_gate_evidence": log_evidence,
        "engineering_contract": recovered_engineering_contract,
        "lora_checkpoint_evidence": checkpoint_evidence,
        "behavior_diagnostics": behavior,
        "per_update_metrics_available": False,
        "per_update_metrics_fabricated": False,
        "audit_limitation": (
            "Exact 512-row update_metrics were not persisted by the original "
            "pre-fix runner and cannot be reconstructed after process exit."
        ),
        "prompt_pool_hash": sha256_file(
            project_root / "data/processed/grpo_prompt_pool_v1/train.jsonl"
        ),
        "reward_config_hash": sha256_file(
            project_root / "configs/grpo/reward_v0_mmsearch_like.yaml"
        ),
        "selected_checkpoint_tree_sha256": sha256_tree(selected_output),
        **BEHAVIOR_CONTRACT_FIELDS,
        **formal_chain_manifest_fields(),
    }
    write_json(temporary / "run_manifest.json", manifest)
    (temporary / "recovery_report.md").write_text(
        "# Reward v0 Full Recovery\n\n"
        "Training compute completed, and the isolated LoRA gate failure was "
        "disproved by eight distinct checkpoints and tensor-content comparison.\n\n"
        "Exact per-update metrics were not persisted by the pre-fix runner; no "
        "replacement rows were fabricated. This is a recovered baseline, not a "
        "canonical no-limitation Full publication.\n\n"
        "GRPO_REWARD_V0_FULL_RECOVERED\n",
        encoding="utf-8",
    )
    temporary.rename(output_dir)
    return manifest
