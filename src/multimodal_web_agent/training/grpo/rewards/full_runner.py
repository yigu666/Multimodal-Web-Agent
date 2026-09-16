from __future__ import annotations

from collections import Counter
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import time
from typing import Any, Mapping, Sequence

import torch

from multimodal_web_agent.training.grpo.optimizer import build_paged_adamw_8bit
from multimodal_web_agent.training.grpo.schema import PromptPoolItem
from multimodal_web_agent.training.grpo.server_runner import (
    GenerationSettings,
    PromptImageStore,
    TransformersTrajectoryRunner,
    _content_hash_change_count,
    _environment_manifest,
    _load_environment,
    _load_runtime,
    _public_rollout,
    _save_adapter,
    _snapshot_trainable_content_hashes,
    _snapshot_visual_versions,
    _validate_pool,
    _verify_adapter_checkpoint,
    _visual_change_count,
    read_jsonl,
    write_json,
)

from .config import load_reward_v2_config
from .coverage_cache import read_jsonl as read_cache_jsonl, rows_by_prompt
from .exploration_runner import TextSearchExplorationTrajectoryRunner
from .full_contract import (
    FULL_COMPLETE_MARKER,
    FrozenArtifactGuard,
    load_full_training_config,
    numeric_values_finite,
    read_contract_manifest,
    sha256_file,
    sha256_tree,
    validate_update_contract,
    full_complete_marker,
)
from .hierarchical_grounded_search_v2 import HierarchicalGroundedSearchRewardV2
from .reward_breakdown import aggregate_metrics, group_summary
from .training_integration import update_records_v2


def _append_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _snapshot_projector_versions(model: Any) -> dict[str, int]:
    return {
        name: int(getattr(parameter, "_version", 0))
        for name, parameter in model.named_parameters()
        if any(token in name.casefold() for token in ("projector", "projection", "merger"))
    }


def _version_change_count(before: Mapping[str, int], model: Any) -> int:
    current = dict(model.named_parameters())
    return sum(
        name not in current
        or int(getattr(current[name], "_version", 0)) != int(version)
        for name, version in before.items()
    )


def _hash_root(values: Mapping[str, str]) -> str:
    digest = __import__("hashlib").sha256()
    for name, value in sorted(values.items()):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(value.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _active_adapter(model: Any) -> str | list[str]:
    value = getattr(model, "active_adapters", None)
    if callable(value):
        value = value()
    if value is None:
        value = getattr(model, "active_adapter", None)
    if isinstance(value, tuple):
        value = list(value)
    if isinstance(value, list) and len(value) == 1:
        return str(value[0])
    if isinstance(value, (str, list)) and value:
        return value
    raise RuntimeError("cannot identify active trainable SFT/GRPO adapter")


def _reload_checkpoint_and_restore_training_adapter(
    *, model: Any, runner: TransformersTrajectoryRunner,
    checkpoint: Path, prompt: PromptPoolItem, run_seed: int,
    checkpoint_index: int,
) -> dict[str, Any]:
    if not all(hasattr(model, name) for name in (
        "load_adapter", "set_adapter", "delete_adapter"
    )):
        raise RuntimeError("PEFT model cannot reload and remove validation adapter")
    original = _active_adapter(model)
    trainable_before = {
        name: bool(parameter.requires_grad)
        for name, parameter in model.named_parameters()
    }
    validation_name = f"reward_v2_full_reload_{checkpoint_index:04d}"
    validation_loaded = False
    try:
        model.load_adapter(
            str(checkpoint), adapter_name=validation_name, is_trainable=False
        )
        validation_loaded = True
        model.set_adapter(validation_name)
        model.eval()
        if hasattr(model, "gradient_checkpointing_disable"):
            model.gradient_checkpointing_disable()
        if hasattr(model, "config"):
            model.config.use_cache = True
        record = runner.rollout(
            prompt, run_seed=run_seed, pool_pass=0, rollout_index=0,
            settings=GenerationSettings(do_sample=False),
        )
        runner.release_visual(prompt.prompt_uid)
        generated = bool(record.actions) and int(
            torch.as_tensor(record.policy_action_mask).sum().item()
        ) > 0
        if not generated:
            raise RuntimeError("checkpoint reload generation failed")
    finally:
        model.set_adapter(original)
        if validation_loaded:
            model.delete_adapter(validation_name)
        current = dict(model.named_parameters())
        for name, expected in trainable_before.items():
            if name not in current:
                raise RuntimeError("training parameter disappeared after reload")
            current[name].requires_grad_(expected)
    trainable_after = {
        name: bool(parameter.requires_grad)
        for name, parameter in model.named_parameters()
    }
    if trainable_after != trainable_before:
        raise RuntimeError("trainable parameter mask changed after checkpoint reload")
    return {
        "checkpoint_saved": True,
        "checkpoint_reload_success": True,
        "reload_generation_success": True,
        "reload_generation_terminal_reason": record.terminal_reason,
        "reload_generation_action_count": len(record.actions),
        "training_adapter_restored": True,
    }


def _validate_smoke_frozen(
    project_root: Path, training: Mapping[str, Any], contract: Mapping[str, Any]
) -> None:
    smoke = project_root / str(training["smoke_baseline"])
    expected = dict(contract["smoke_baseline"])
    for name, key in (
        ("run_manifest.json", "manifest_sha256"),
        ("smoke_summary.json", "summary_sha256"),
        ("files.sha256", "files_sha256"),
    ):
        if sha256_file(smoke / name) != expected[key]:
            raise RuntimeError(f"frozen Smoke baseline changed: {name}")


def run_reward_v2_full(
    *, project_root: Path, output_dir: Path,
    training_config_path: Path,
) -> dict[str, Any]:
    started = time.monotonic()
    project_root = Path(project_root).resolve()
    output_dir = Path(output_dir).resolve()
    training = load_full_training_config(training_config_path)
    experiment_id = str(training.get("experiment_id", "reward_v21"))
    complete_marker = full_complete_marker(experiment_id)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("Reward v2 Full output must start empty")
    output_dir.mkdir(parents=True, exist_ok=True)
    canonical_output_dir = (project_root / str(training["output_dir"])).resolve()
    contract_path = project_root / str(training["full_contract"])
    contract = read_contract_manifest(
        contract_path, experiment_id=experiment_id
    )
    if dict(contract.get("scale", {})) != dict(training["scale"]):
        raise RuntimeError("Full runtime scale differs from execution contract")
    if contract.get("initialization") != "frozen_sft_adapter":
        raise RuntimeError("Full execution contract initialization differs")
    if contract.get("reward_v0_checkpoint_used") is not False:
        raise RuntimeError("Full execution contract permits Reward v0 continuation")
    if contract.get("reward_v2_checkpoint_used") is not False:
        raise RuntimeError("Full execution contract permits Reward v2 continuation")
    _validate_smoke_frozen(project_root, training, contract)
    guard = FrozenArtifactGuard(project_root, contract["frozen_artifacts"])
    guard.verify_hashes()

    reward_config_path = project_root / str(training["reward_config"])
    config = load_reward_v2_config(reward_config_path)
    frozen = dict(config.raw["frozen_contract"])
    if frozen.get("full_training_allowed") is not True:
        raise RuntimeError("Reward v2 Full engineering gate is closed")
    if frozen.get("reward_v0_checkpoint_allowed") is not False:
        raise RuntimeError("Reward v0 checkpoint continuation is permitted")
    if Path(config.paths.sft_adapter).as_posix() != str(
        training["initialization"]["adapter_path"]
    ):
        raise RuntimeError("Reward v2 Full initialization path differs")
    adapter_path = config.paths.sft_adapter.casefold()
    if any(name in adapter_path for name in (
        "reward_v0", "grpo_reward_v2_full", "reward_v21",
    )):
        raise RuntimeError("Reward checkpoint cannot initialize Answer Reward Full")

    _validate_pool(project_root)
    prompts = [
        PromptPoolItem.from_dict(row)
        for row in read_jsonl(project_root / config.paths.prompt_pool)
    ]
    if len(prompts) != 2048:
        raise RuntimeError("Reward v2 Full requires exactly 2048 prompts")
    coverage_path = project_root / config.paths.coverage_cache
    baseline_path = project_root / config.paths.question_baseline_cache
    manager = HierarchicalGroundedSearchRewardV2(
        config,
        coverage_cache=rows_by_prompt(read_cache_jsonl(coverage_path)),
        question_baseline_cache=rows_by_prompt(read_cache_jsonl(baseline_path)),
    )
    model, processor, runtime = _load_runtime(project_root)
    if runtime["adapter_hash"] != contract["sft_adapter_sha256"]:
        raise RuntimeError("runtime did not initialize from contracted SFT Adapter")
    if runtime["base_model_hash"] != contract["base_model_sha256"]:
        raise RuntimeError("runtime Base model differs from Full contract")
    environment = _load_environment(project_root)
    runtime["environment"] = _environment_manifest(project_root, environment)
    image_store = PromptImageStore(project_root / "data/raw/fvqa/fvqa_train.parquet")
    runner = TextSearchExplorationTrajectoryRunner(
        model=model, processor=processor, environment=environment,
        image_store=image_store,
        base_model_hash=runtime["base_model_hash"],
        adapter_hash=runtime["adapter_hash"],
        exploration_config=config.text_search_exploration,
        global_seed=int(training["run_seed"]),
        exploration_schedule="full", total_updates=512,
    )
    reload_runner = TransformersTrajectoryRunner(
        model=model, processor=processor, environment=environment,
        image_store=image_store,
        base_model_hash=runtime["base_model_hash"],
        adapter_hash=runtime["adapter_hash"],
    )
    runtime["processor_hash"] = runner.processor_hash
    runtime["chat_template_hash"] = runner.chat_template_hash
    optimizer = build_paged_adamw_8bit(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        learning_rate=float(training["optimizer"]["actor_learning_rate"]),
        weight_decay=float(training["optimizer"]["weight_decay"]),
    )
    trainable_initial = _snapshot_trainable_content_hashes(model)
    checkpoint_hashes = trainable_initial
    visual_before = _snapshot_visual_versions(model)
    projector_before = _snapshot_projector_versions(model)
    tokenizer = getattr(processor, "tokenizer", processor)

    for name in (
        "rollouts.jsonl", "reward_breakdown.jsonl", "group_statistics.jsonl",
        "exploration_trace.jsonl", "text_query_advantage_trace.jsonl",
        "training_metrics.jsonl", "continuous_contract.jsonl",
        "checkpoint_audit.jsonl",
    ):
        path = output_dir / name
        if path.exists():
            raise FileExistsError(path)
        path.touch()
    (output_dir / "config_resolved.yaml").write_text(
        reward_config_path.read_text(encoding="utf-8"), encoding="utf-8"
    )
    (output_dir / "training_config_resolved.yaml").write_text(
        training_config_path.read_text(encoding="utf-8"), encoding="utf-8"
    )

    all_breakdowns: list[dict[str, Any]] = []
    all_group_rows: list[dict[str, Any]] = []
    update_rows: list[dict[str, Any]] = []
    checkpoint_rows: list[dict[str, Any]] = []
    action_counts: Counter[str] = Counter()
    text_advantages: list[float] = []
    buffer_records: list[Any] = []
    buffer_advantages: list[torch.Tensor] = []
    buffer_breakdowns: list[dict[str, Any]] = []
    buffer_public: list[dict[str, Any]] = []
    buffer_group_rows: list[dict[str, Any]] = []
    optimizer_step = 0
    run_seed = int(training["run_seed"])

    for group_index, prompt in enumerate(prompts):
        model.eval()
        if hasattr(model, "gradient_checkpointing_disable"):
            model.gradient_checkpointing_disable()
        if hasattr(model, "config"):
            model.config.use_cache = True
        update_id = group_index // 4
        records = runner.rollout_group_with_exploration(
            prompt, run_seed=run_seed, pool_pass=1, group_size=4,
            settings=GenerationSettings(do_sample=True),
            group_index=group_index, update_id=update_id,
        )
        scored = manager.score_group(records, prompt.to_dict(), tokenizer=tokenizer)
        current_breakdowns = []
        for record, row, token_values in zip(
            records, scored.breakdowns, scored.token_advantages
        ):
            if token_values is None or not bool(
                torch.isfinite(torch.as_tensor(token_values)).all()
            ) or not numeric_values_finite(row):
                raise RuntimeError("NONFINITE_REWARD_OR_ADVANTAGE")
            search_by_turn = {
                int(action["turn"]): action
                for action in row.get("search_actions", [])
            }
            for trace in record.exploration_trace:
                action = search_by_turn.get(int(trace["turn"]))
                if action is not None and action.get("tool") == "text_search":
                    trace.update({
                        key: action[key] for key in (
                            "coverage_mask", "actual_rank_utility",
                            "question_baseline_rank_utility", "query_improvement",
                            "text_query_advantage", "combined_text_token_advantage",
                        )
                    })
                    trace["terminal_advantage"] = row["terminal_advantage"]
                trace["rollout_uid"] = record.rollout_uid
            record.reward_total = float(row["terminal_reward"])
            record.answer_em = int(row["em_v1"])
            record.answer_f1 = float(row["token_f1_v1"])
            record.reward_components = dict(row)
            buffer_records.append(record)
            buffer_advantages.append(torch.as_tensor(token_values))
            buffer_public.append(_public_rollout(record))
            current = dict(row)
            current_breakdowns.append(current)
            buffer_breakdowns.append(current)
            for action in record.actions:
                action_counts[str(action.get("action_type") or "none")] += 1
            text_advantages.extend(
                float(action["text_query_advantage"])
                for action in row.get("search_actions", [])
                if action.get("tool") == "text_search"
            )
        group_row = group_summary(current_breakdowns)[0]
        buffer_group_rows.append(group_row)
        all_group_rows.append(group_row)
        all_breakdowns.extend(current_breakdowns)
        runner.release_visual(prompt.prompt_uid)

        if len(buffer_records) != 16:
            continue
        update = update_records_v2(
            model=model, trajectory_runner=runner, optimizer=optimizer,
            records=buffer_records, token_advantages=buffer_advantages,
            clip_ratio=float(training["optimizer"]["clip_ratio"]),
            max_grad_norm=float(training["optimizer"]["max_grad_norm"]),
            verify_alignment=True,
        )
        optimizer_step += 1
        update.update({
            "optimizer_step": optimizer_step,
            "prompt_count_processed": group_index + 1,
            "exploration_epsilon": float(
                buffer_records[0].exploration_metadata["exploration_epsilon"]
            ),
            "exploration_logit_bias": float(
                buffer_records[0].exploration_metadata["exploration_logit_bias"]
            ),
        })
        update_gates = validate_update_contract(update)
        if _visual_change_count(visual_before, model) != 0:
            raise RuntimeError("FULL_CONTINUOUS_CONTRACT_FAILED: visual_frozen")
        if _version_change_count(projector_before, model) != 0:
            raise RuntimeError("FULL_CONTINUOUS_CONTRACT_FAILED: projector_frozen")
        guard.verify_metadata()
        continuous = {
            "optimizer_step": optimizer_step,
            "prompt_count_processed": group_index + 1,
            **update_gates,
            "visual_frozen": True,
            "projector_frozen": True,
            "frozen_artifact_metadata": True,
            "diagnostic_only_never_stop": {
                "text_search_count": action_counts["text_search"],
                "local_advantage_nonzero_count": sum(value != 0 for value in text_advantages),
                "image_search_count": action_counts["image_search"],
            },
        }
        update_rows.append(dict(update))
        _append_jsonl(output_dir / "training_metrics.jsonl", [update])
        _append_jsonl(output_dir / "continuous_contract.jsonl", [continuous])
        _append_jsonl(output_dir / "rollouts.jsonl", buffer_public)
        _append_jsonl(output_dir / "reward_breakdown.jsonl", buffer_breakdowns)
        _append_jsonl(output_dir / "group_statistics.jsonl", buffer_group_rows)
        exploration_rows = [
            dict(trace) for record in buffer_records for trace in record.exploration_trace
        ]
        text_rows = [
            {
                "prompt_id": row["prompt_id"],
                "rollout_uid": record.rollout_uid,
                "exploration_selected": bool(
                    record.exploration_metadata.get("exploration_selected")
                ),
                **dict(action),
                "terminal_advantage": row["terminal_advantage"],
            }
            for record, row in zip(buffer_records, buffer_breakdowns)
            for action in row.get("search_actions", [])
            if action.get("tool") == "text_search"
        ]
        _append_jsonl(output_dir / "exploration_trace.jsonl", exploration_rows)
        _append_jsonl(output_dir / "text_query_advantage_trace.jsonl", text_rows)

        if optimizer_step % 64 == 0:
            guard.verify_hashes()
            current_hashes = _snapshot_trainable_content_hashes(model)
            changed = _content_hash_change_count(checkpoint_hashes, current_hashes)
            if changed <= 0:
                raise RuntimeError(
                    "FULL_CONTINUOUS_CONTRACT_FAILED: cumulative_lora_change"
                )
            checkpoint_hashes = current_hashes
            prompt_count = group_index + 1
            checkpoint_path = (
                output_dir / "checkpoints" / f"prompt_{prompt_count:04d}"
            )
            _save_adapter(model, checkpoint_path)
            _verify_adapter_checkpoint(checkpoint_path)
            reload_audit = _reload_checkpoint_and_restore_training_adapter(
                model=model, runner=reload_runner, checkpoint=checkpoint_path,
                prompt=prompts[0], run_seed=run_seed,
                checkpoint_index=prompt_count,
            )
            checkpoint_row = {
                "optimizer_step": optimizer_step,
                "prompt_count_processed": prompt_count,
                "checkpoint_path": (
                    canonical_output_dir / "checkpoints" / f"prompt_{prompt_count:04d}"
                ).relative_to(project_root).as_posix(),
                "checkpoint_tree_sha256": sha256_tree(checkpoint_path),
                "lora_parameter_change_count_since_previous_checkpoint": changed,
                "frozen_artifact_hashes_verified": True,
                **reload_audit,
            }
            checkpoint_rows.append(checkpoint_row)
            _append_jsonl(output_dir / "checkpoint_audit.jsonl", [checkpoint_row])
        print(
            f"[{experiment_id}_full] optimizer_step={optimizer_step}/512 "
            f"prompts={group_index + 1}/2048 "
            f"epsilon={update['exploration_epsilon']:.6f} "
            f"bias={update['exploration_logit_bias']:.6f}",
            flush=True,
        )
        buffer_records = []
        buffer_advantages = []
        buffer_breakdowns = []
        buffer_public = []
        buffer_group_rows = []

    if optimizer_step != 512 or buffer_records:
        raise RuntimeError("Reward v2 Full did not execute exactly 512 updates")
    guard.verify_hashes()
    final_hashes = _snapshot_trainable_content_hashes(model)
    total_lora_changes = _content_hash_change_count(trainable_initial, final_hashes)
    if total_lora_changes <= 0:
        raise RuntimeError("Reward v2 Full LoRA parameters did not change")
    if _visual_change_count(visual_before, model) != 0:
        raise RuntimeError("Reward v2 Full visual modules changed")
    if _version_change_count(projector_before, model) != 0:
        raise RuntimeError("Reward v2 Full projector changed")
    if len(checkpoint_rows) != 8:
        raise RuntimeError("Reward v2 Full checkpoint count differs from eight")
    for row in checkpoint_rows:
        actual_path = output_dir / "checkpoints" / Path(
            str(row["checkpoint_path"])
        ).name
        if sha256_tree(actual_path) != row["checkpoint_tree_sha256"]:
            raise RuntimeError("saved Reward v2 checkpoint changed before completion")

    final_checkpoint = output_dir / "checkpoints/prompt_2048"
    selected_name = str(training.get(
        "selected_checkpoint_dirname", "selected_reward_v2_checkpoint"
    ))
    selected = output_dir / selected_name
    shutil.copytree(final_checkpoint, selected)
    _verify_adapter_checkpoint(selected)
    if sha256_tree(selected) != sha256_tree(final_checkpoint):
        raise RuntimeError("selected Reward v2 checkpoint differs from prompt_2048")
    final_reload = _reload_checkpoint_and_restore_training_adapter(
        model=model, runner=reload_runner, checkpoint=selected,
        prompt=prompts[0], run_seed=run_seed, checkpoint_index=9999,
    )

    aggregate = aggregate_metrics(all_breakdowns, all_group_rows)
    summary = {
        "schema_version": "grpo-reward-v2-full-run-v1",
        "marker": complete_marker,
        "status": "passed",
        "stage": f"{experiment_id}_full",
        "experiment_id": experiment_id,
        "reward_version": config.version,
        "reward_mode": config.mode,
        "initialization": "frozen_sft_adapter",
        "reward_v0_checkpoint_used": False,
        "reward_v2_checkpoint_used": False,
        "reward_v21_checkpoint_used": False,
        "prompt_count": 2048,
        "group_size": 4,
        "rollout_count": 8192,
        "prompt_groups_per_update": 4,
        "trajectories_per_update": 16,
        "optimizer_steps": 512,
        "checkpoint_count": 8,
        "selected_checkpoint": (
            canonical_output_dir / selected_name
        ).relative_to(project_root).as_posix(),
        "selected_checkpoint_tree_sha256": sha256_tree(selected),
        "lora_parameter_change_count": total_lora_changes,
        "lora_parameter_fingerprint_before": _hash_root(trainable_initial),
        "lora_parameter_fingerprint_after": _hash_root(final_hashes),
        "visual_parameter_change_count": 0,
        "projector_parameter_change_count": 0,
        "continuous_engineering_contract_passed": True,
        "behavior_diagnostics_are_nonblocking": True,
        "diagnostics": {
            "action_counts": dict(sorted(action_counts.items())),
            "text_query_advantage_count": len(text_advantages),
            "nonzero_text_query_advantage_count": sum(
                value != 0 for value in text_advantages
            ),
            "mean_text_query_advantage": (
                statistics.mean(text_advantages) if text_advantages else 0.0
            ),
            "reward_metrics": aggregate,
        },
        "frozen_dev_accessed_during_training": False,
        "unified_frozen_test_accessed": False,
        "post_full_frozen_dev_pending": True,
        "elapsed_seconds": time.monotonic() - started,
        "peak_vram": (
            int(torch.cuda.max_memory_allocated())
            if torch.cuda.is_available() else 0
        ),
        "full_contract_sha256": sha256_file(contract_path),
        "reward_config_sha256": sha256_file(reward_config_path),
        "training_config_sha256": sha256_file(training_config_path),
        "prompt_pool_sha256": sha256_file(project_root / config.paths.prompt_pool),
        "checkpoint_reload": final_reload,
        **runtime,
    }
    write_json(output_dir / "full_engineering_contract.json", {
        "engineering_hard_gates_passed": True,
        "update_contract_count": len(update_rows),
        "checkpoint_contract_count": len(checkpoint_rows),
        "all_values_finite": all(numeric_values_finite(row) for row in update_rows),
        "max_behavior_logprob_error": max(
            float(row["exploration_behavior_logprob_error_max"])
            for row in update_rows
        ),
        "input_identity_failure_count": 0,
        "information_token_train_mask_sum": 0,
        "visual_parameter_change_count": 0,
        "projector_parameter_change_count": 0,
        "lora_parameter_change_count": total_lora_changes,
        "checkpoint_save_reload_passed": True,
        "frozen_artifact_hashes_passed": True,
        "behavior_diagnostics_used_as_gate": False,
        "unified_frozen_test_accessed": False,
    })
    write_json(output_dir / "run_manifest.json", summary)
    (output_dir / "report.md").write_text(
        f"# {experiment_id} Full Training\n\n"
        f"- Prompts: 2048\n- Rollouts: 8192\n- Updates: 512\n"
        "- Initialization: frozen SFT Adapter\n"
        "- Reward v0 checkpoint used: false\n"
        "- Frozen Dev during training: false\n"
        "- Frozen Test accessed: false\n\n"
        f"{complete_marker}\n",
        encoding="utf-8",
    )
    root_files = sorted(
        path for path in output_dir.iterdir()
        if path.is_file() and path.name != "files.sha256"
    )
    (output_dir / "files.sha256").write_text(
        "".join(f"{sha256_file(path)}  {path.name}\n" for path in root_files),
        encoding="utf-8",
    )
    print(complete_marker, flush=True)
    print("FROZEN_DEV_NOT_ACCESSED_DURING_TRAINING", flush=True)
    print("FROZEN_TEST_NOT_ACCESSED", flush=True)
    return summary
