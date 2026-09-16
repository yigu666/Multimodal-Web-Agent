from __future__ import annotations

from collections import Counter
import json
import os
from pathlib import Path
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
from multimodal_web_agent.training.grpo.rewards.config import load_reward_v2_config
from multimodal_web_agent.training.grpo.rewards.coverage_cache import (
    read_jsonl as read_cache_jsonl,
    rows_by_prompt,
)
from multimodal_web_agent.training.grpo.rewards.full_contract import (
    numeric_values_finite,
    validate_update_contract,
)
from multimodal_web_agent.training.grpo.rewards.full_runner import (
    _hash_root,
    _reload_checkpoint_and_restore_training_adapter,
    _snapshot_projector_versions,
    _version_change_count,
)
from multimodal_web_agent.training.grpo.rewards.hierarchical_grounded_search_v2 import (
    HierarchicalGroundedSearchRewardV2,
)
from multimodal_web_agent.training.grpo.rewards.reward_breakdown import (
    aggregate_metrics,
    group_summary,
)
from multimodal_web_agent.training.grpo.rewards.training_integration import (
    update_records_v2,
)
from multimodal_web_agent.training.grpo.stage2_contract import (
    CONTRACT_MARKER,
    load_stage2_config,
    resolve_v21_selected_checkpoint,
    sha256_file,
    sha256_tree,
)


def _append_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _load_stage2_runtime(
    project_root: Path, checkpoint: Path,
) -> tuple[Any, Any, dict[str, Any]]:
    from multimodal_web_agent.training.sft.config import load_config
    from multimodal_web_agent.training.sft.model_factory import load_qwen_lora

    config = load_config(
        project_root / "configs/protocol_sft/train_full_format_v1.yaml",
        project_root=project_root,
    )
    expected = sha256_tree(checkpoint)
    print(f"[stage2] loading Base NF4 + Reward v2.1 checkpoint: {checkpoint}", flush=True)
    model, processor, audit = load_qwen_lora(config, adapter_path=checkpoint)
    if audit.trainable_visual_parameter_count != 0:
        raise RuntimeError("Stage-2 visual parameters are trainable")
    print("[stage2] model ready; optimizer state is fresh", flush=True)
    return model, processor, {
        "adapter_hash": expected,
        "base_model_hash": sha256_tree(config.model.path),
        "base_model_path": str(config.model.path),
        "model_audit": audit.to_dict(),
    }


def _validate_contract(
    project_root: Path, config: Mapping[str, Any], config_path: Path,
) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    contract_path = project_root / str(config["contract"])
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if contract.get("marker") != CONTRACT_MARKER or contract.get("passed") is not True:
        raise RuntimeError("Stage-2 mathematical contract is not ready")
    checkpoint, v21_manifest = resolve_v21_selected_checkpoint(project_root)
    if str(checkpoint) != contract.get("stage2_init_checkpoint"):
        raise RuntimeError("Stage-2 init checkpoint differs from contract")
    if sha256_tree(checkpoint) != contract.get("stage2_init_checkpoint_tree_sha256"):
        raise RuntimeError("Stage-2 init checkpoint hash differs from contract")
    reward_path = project_root / str(config["reward_config"])
    hashes = dict(contract.get("config_sha256", {}))
    for path in (config_path, reward_path):
        relative = path.relative_to(project_root).as_posix()
        if hashes.get(relative) != sha256_file(path):
            raise RuntimeError(f"Stage-2 config changed after contract: {relative}")
    _validate_frozen_hashes(project_root, contract)
    return contract, checkpoint, v21_manifest


def _validate_frozen_hashes(
    project_root: Path, contract: Mapping[str, Any],
) -> None:
    for section in ("source_sha256", "frozen_input_sha256"):
        values = dict(contract.get(section, {}))
        if not values:
            raise RuntimeError(f"Stage-2 contract has no {section}")
        for relative, expected in values.items():
            path = (project_root / relative).resolve()
            try:
                path.relative_to(project_root)
            except ValueError as exc:
                raise RuntimeError("Stage-2 frozen path escapes project") from exc
            if sha256_file(path) != expected:
                raise RuntimeError(
                    f"Stage-2 frozen artifact changed: {relative}"
                )


def _validate_step0(project_root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    path = project_root / str(config["step0_evaluation"])
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("marker") != "GRPO_STAGE2_STEP0_FROZEN_DEV_READY":
        raise RuntimeError("Stage-2 Step-0 evaluation is not ready")
    if value.get("reproduction_passed") is not True:
        raise RuntimeError("Stage-2 Step-0 did not reproduce Reward v2.1")
    if value.get("frozen_test_accessed") is not False:
        raise RuntimeError("Stage-2 Step-0 accessed Frozen Test")
    return value


def run_stage2(
    *, project_root: Path, config_path: Path, output_dir: Path,
) -> dict[str, Any]:
    started = time.monotonic()
    project_root = Path(project_root).resolve()
    config_path = Path(config_path).resolve()
    output_dir = Path(output_dir).resolve()
    config = load_stage2_config(config_path)
    canonical_output_dir = (project_root / str(config["output_dir"])).resolve()
    try:
        canonical_output_dir.relative_to(project_root)
    except ValueError as exc:
        raise RuntimeError("Stage-2 canonical output escapes project") from exc
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("Stage-2 output must start empty")
    output_dir.mkdir(parents=True, exist_ok=True)
    contract, checkpoint, v21_manifest = _validate_contract(
        project_root, config, config_path
    )
    step0 = _validate_step0(project_root, config)
    reward_path = project_root / str(config["reward_config"])
    reward_config = load_reward_v2_config(reward_path)
    if reward_config.mode != "answer_dominant_positive":
        raise RuntimeError("Stage-2 did not retain Reward v2.1 terminal")
    _validate_pool(project_root)
    all_prompts = [
        PromptPoolItem.from_dict(row)
        for row in read_jsonl(project_root / reward_config.paths.prompt_pool)
    ]
    prompt_count = int(config["scale"]["prompt_count"])
    prompts = all_prompts[:prompt_count]
    if len(prompts) != prompt_count:
        raise RuntimeError("Stage-2 prompt pool is too small")
    write_json(output_dir / "prompt_ids.json", {
        "selection": "first_n_from_frozen_v21_train_pool",
        "prompt_count": prompt_count,
        "prompt_ids": [prompt.prompt_uid for prompt in prompts],
        "frozen_dev_accessed": False,
        "frozen_test_accessed": False,
    })
    write_json(output_dir / "initialization_checkpoint.json", {
        "step": 0,
        "checkpoint_path": str(checkpoint),
        "checkpoint_tree_sha256": contract[
            "stage2_init_checkpoint_tree_sha256"
        ],
        "source": "reward_v21_selected_checkpoint",
        "weights_copied": False,
        "reason": "immutable formal v2.1 checkpoint is the Stage-2 Step-0 checkpoint",
    })

    manager = HierarchicalGroundedSearchRewardV2(
        reward_config,
        coverage_cache=rows_by_prompt(read_cache_jsonl(
            project_root / reward_config.paths.coverage_cache
        )),
        question_baseline_cache=rows_by_prompt(read_cache_jsonl(
            project_root / reward_config.paths.question_baseline_cache
        )),
    )
    model, processor, runtime = _load_stage2_runtime(project_root, checkpoint)
    if runtime["adapter_hash"] != contract["stage2_init_checkpoint_tree_sha256"]:
        raise RuntimeError("Stage-2 runtime did not load Reward v2.1")
    environment = _load_environment(project_root)
    runtime["environment"] = _environment_manifest(project_root, environment)
    image_store = PromptImageStore(project_root / "data/raw/fvqa/fvqa_train.parquet")
    runner = TransformersTrajectoryRunner(
        model=model, processor=processor, environment=environment,
        image_store=image_store, base_model_hash=runtime["base_model_hash"],
        adapter_hash=runtime["adapter_hash"],
    )
    reload_runner = TransformersTrajectoryRunner(
        model=model, processor=processor, environment=environment,
        image_store=image_store, base_model_hash=runtime["base_model_hash"],
        adapter_hash=runtime["adapter_hash"],
    )
    optimizer = build_paged_adamw_8bit(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        learning_rate=float(config["optimizer"]["actor_learning_rate"]),
        weight_decay=float(config["optimizer"]["weight_decay"]),
    )
    tokenizer = getattr(processor, "tokenizer", processor)
    trainable_initial = _snapshot_trainable_content_hashes(model)
    checkpoint_hashes = trainable_initial
    visual_before = _snapshot_visual_versions(model)
    projector_before = _snapshot_projector_versions(model)
    for name in (
        "rollouts.jsonl", "reward_breakdown.jsonl", "group_statistics.jsonl",
        "training_metrics.jsonl", "continuous_contract.jsonl",
        "checkpoint_audit.jsonl", "text_query_advantage_trace.jsonl",
    ):
        (output_dir / name).touch()
    (output_dir / "config_resolved.yaml").write_text(
        config_path.read_text(encoding="utf-8"), encoding="utf-8"
    )
    (output_dir / "reward_config_resolved.yaml").write_text(
        reward_path.read_text(encoding="utf-8"), encoding="utf-8"
    )

    buffer_records: list[Any] = []
    buffer_advantages: list[torch.Tensor] = []
    buffer_breakdowns: list[dict[str, Any]] = []
    buffer_public: list[dict[str, Any]] = []
    buffer_groups: list[dict[str, Any]] = []
    all_breakdowns: list[dict[str, Any]] = []
    all_groups: list[dict[str, Any]] = []
    update_rows: list[dict[str, Any]] = []
    checkpoint_rows: list[dict[str, Any]] = []
    action_counts: Counter[str] = Counter()
    scheduled_updates = int(config["scale"]["scheduled_updates"])
    checkpoint_updates = set(int(value) for value in config["checkpoint_updates"])
    update_index = 0
    processed_prompt_count = 0
    run_seed = int(config["run_seed"])

    for group_index, prompt in enumerate(prompts):
        model.eval()
        if hasattr(model, "gradient_checkpointing_disable"):
            model.gradient_checkpointing_disable()
        if hasattr(model, "config"):
            model.config.use_cache = True
        records = runner.rollout_group(
            prompt, run_seed=run_seed, pool_pass=2, group_size=4,
            settings=GenerationSettings(do_sample=True),
        )
        scored = manager.score_group(records, prompt.to_dict(), tokenizer=tokenizer)
        current = []
        for record, row, token_values in zip(
            records, scored.breakdowns, scored.token_advantages
        ):
            if token_values is None or not torch.isfinite(
                torch.as_tensor(token_values)
            ).all() or not numeric_values_finite(row):
                raise RuntimeError("STAGE2_NONFINITE_REWARD_OR_ADVANTAGE")
            record.reward_total = float(row["terminal_reward"])
            record.answer_em = int(row["em_v1"])
            record.answer_f1 = float(row["token_f1_v1"])
            record.reward_components = dict(row)
            buffer_records.append(record)
            buffer_advantages.append(torch.as_tensor(token_values))
            buffer_public.append(_public_rollout(record))
            buffer_breakdowns.append(dict(row))
            current.append(dict(row))
            for action in record.actions:
                action_counts[str(action.get("action_type") or "none")] += 1
        group_row = group_summary(current)[0]
        buffer_groups.append(group_row)
        all_groups.append(group_row)
        all_breakdowns.extend(current)
        runner.release_visual(prompt.prompt_uid)
        if len(buffer_records) != 16:
            continue

        update = update_records_v2(
            model=model, trajectory_runner=runner, optimizer=optimizer,
            records=buffer_records, token_advantages=buffer_advantages,
            clip_ratio=float(config["optimizer"]["clip_ratio"]),
            max_grad_norm=float(config["optimizer"]["max_grad_norm"]),
            verify_alignment=True,
        )
        update_index += 1
        processed_prompt_count = group_index + 1
        update.update({
            "scheduled_update": update_index,
            "prompt_count_processed": group_index + 1,
            "learning_rate": float(config["optimizer"]["actor_learning_rate"]),
            "exploration_epsilon": 0.0,
            "exploration_logit_bias": 0.0,
            "optimizer_state_loaded": False,
            "scheduler_state_loaded": False,
        })
        gates = validate_update_contract(update)
        if _visual_change_count(visual_before, model) != 0:
            raise RuntimeError("STAGE2_CONTRACT_FAILED: visual_frozen")
        if _version_change_count(projector_before, model) != 0:
            raise RuntimeError("STAGE2_CONTRACT_FAILED: projector_frozen")
        update_rows.append(dict(update))
        _append_jsonl(output_dir / "training_metrics.jsonl", [update])
        _append_jsonl(output_dir / "continuous_contract.jsonl", [{
            "scheduled_update": update_index,
            **gates, "visual_frozen": True, "projector_frozen": True,
        }])
        _append_jsonl(output_dir / "rollouts.jsonl", buffer_public)
        _append_jsonl(output_dir / "reward_breakdown.jsonl", buffer_breakdowns)
        _append_jsonl(output_dir / "group_statistics.jsonl", buffer_groups)
        _append_jsonl(output_dir / "text_query_advantage_trace.jsonl", [
            {"prompt_id": row["prompt_id"], "rollout_uid": record.rollout_uid,
             **dict(action), "terminal_advantage": row["terminal_advantage"]}
            for record, row in zip(buffer_records, buffer_breakdowns)
            for action in row.get("search_actions", [])
            if action.get("tool") == "text_search"
        ])

        if update_index in checkpoint_updates:
            _validate_frozen_hashes(project_root, contract)
            current_hashes = _snapshot_trainable_content_hashes(model)
            changed = _content_hash_change_count(checkpoint_hashes, current_hashes)
            if changed <= 0:
                raise RuntimeError("STAGE2_CONTRACT_FAILED: cumulative_lora_change")
            checkpoint_hashes = current_hashes
            checkpoint_path = output_dir / "checkpoints" / f"step_{update_index:04d}"
            _save_adapter(model, checkpoint_path)
            _verify_adapter_checkpoint(checkpoint_path)
            reload_audit = _reload_checkpoint_and_restore_training_adapter(
                model=model, runner=reload_runner, checkpoint=checkpoint_path,
                prompt=prompts[0], run_seed=run_seed,
                checkpoint_index=update_index,
            )
            checkpoint_row = {
                "step": update_index,
                # The server wrapper writes into a temporary directory and atomically
                # renames it after success.  Persist the post-rename path, not the
                # temporary path used while saving/reloading the checkpoint.
                "checkpoint_path": (
                    canonical_output_dir / "checkpoints"
                    / f"step_{update_index:04d}"
                ).relative_to(project_root).as_posix(),
                "checkpoint_tree_sha256": sha256_tree(checkpoint_path),
                "lora_parameter_change_count_since_previous_checkpoint": changed,
                **reload_audit,
            }
            checkpoint_rows.append(checkpoint_row)
            _append_jsonl(output_dir / "checkpoint_audit.jsonl", [checkpoint_row])
        print(
            f"[{config['experiment_id']}] update={update_index}/{scheduled_updates} "
            f"prompts={group_index + 1}/{prompt_count}", flush=True,
        )
        buffer_records = []
        buffer_advantages = []
        buffer_breakdowns = []
        buffer_public = []
        buffer_groups = []

    if buffer_records:
        raise RuntimeError("Stage-2 stopped with a partial optimizer batch")
    if update_index != scheduled_updates:
        raise RuntimeError("Stage-2 scheduled update count differs")
    final_hashes = _snapshot_trainable_content_hashes(model)
    total_changes = _content_hash_change_count(trainable_initial, final_hashes)
    if total_changes <= 0:
        raise RuntimeError("Stage-2 LoRA parameters did not change")
    expected_checkpoint_steps = [
        int(step) for step in config["checkpoint_updates"]
        if int(step) <= update_index
    ]
    if [row["step"] for row in checkpoint_rows] != expected_checkpoint_steps:
        raise RuntimeError("Stage-2 checkpoints differ from schedule")
    _validate_frozen_hashes(project_root, contract)
    aggregate = aggregate_metrics(all_breakdowns, all_groups)
    marker = f"GRPO_STAGE2_{config['strategy']}_{config['mode'].upper()}_COMPLETE"
    summary = {
        "schema_version": "grpo-stage2-run-v1",
        "marker": marker,
        "status": "passed",
        "experiment_id": config["experiment_id"],
        "strategy": config["strategy"],
        "mode": config["mode"],
        "initialization": "reward_v21_selected_checkpoint",
        "stage2_init_checkpoint": str(checkpoint),
        "stage2_init_checkpoint_tree_sha256": runtime["adapter_hash"],
        "stage1_manifest_selected_checkpoint": v21_manifest["selected_checkpoint"],
        "optimizer_state_loaded": False,
        "scheduler_state_loaded": False,
        "stage1_learning_rate": config["optimizer"]["stage1_learning_rate"],
        "stage2_learning_rate": config["optimizer"]["actor_learning_rate"],
        "exploration_restarted": False,
        "exploration_state": {"epsilon": 0.0, "logit_bias": 0.0},
        "prompt_count": prompt_count,
        "prompt_count_processed": processed_prompt_count,
        "rollout_count": processed_prompt_count * 4,
        "max_scheduled_updates": scheduled_updates,
        "scheduled_updates": update_index,
        "effective_optimizer_steps": update_index,
        "checkpoint_steps": [row["step"] for row in checkpoint_rows],
        "step0_checkpoint": {
            "step": 0,
            "checkpoint_path": str(checkpoint),
            "checkpoint_tree_sha256": runtime["adapter_hash"],
        },
        "checkpoint_count": len(checkpoint_rows),
        "lora_parameter_change_count": total_changes,
        "lora_parameter_fingerprint_before": _hash_root(trainable_initial),
        "lora_parameter_fingerprint_after": _hash_root(final_hashes),
        "visual_parameter_change_count": 0,
        "projector_parameter_change_count": 0,
        "continuous_engineering_contract_passed": True,
        "reward_metrics": aggregate,
        "action_counts": dict(sorted(action_counts.items())),
        "step0_reproduction_manifest_sha256": sha256_file(
            project_root / str(config["step0_evaluation"])
        ),
        "frozen_dev_accessed_during_training": False,
        "frozen_test_accessed": False,
        "elapsed_seconds": time.monotonic() - started,
        **runtime,
    }
    write_json(output_dir / "run_manifest.json", summary)
    (output_dir / "report.md").write_text(
        f"# {config['experiment_id']}\n\n"
        f"- Initialization: `{checkpoint}`\n"
        f"- Strategy: {config['strategy']}\n"
        f"- Processed prompts/rollouts/updates: "
        f"{processed_prompt_count}/{processed_prompt_count * 4}/{update_index}\n"
        "- Exploration: disabled, inherited v2.1 terminal state\n"
        "- Optimizer/scheduler state loaded: false/false\n"
        "- Frozen Test accessed: false\n\n"
        f"{marker}\n",
        encoding="utf-8",
    )
    files = sorted(
        path for path in output_dir.rglob("*")
        if path.is_file() and path.name != "files.sha256"
    )
    (output_dir / "files.sha256").write_text(
        "".join(
            f"{sha256_file(path)}  {path.relative_to(output_dir).as_posix()}\n"
            for path in files
        ), encoding="utf-8",
    )
    print(marker, flush=True)
    print("FROZEN_TEST_NOT_ACCESSED", flush=True)
    return summary
