from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import subprocess
import time

import torch

from multimodal_web_agent.training.grpo.optimizer import build_paged_adamw_8bit
from multimodal_web_agent.training.grpo.schema import PromptPoolItem
from multimodal_web_agent.training.grpo.server_runner import (
    GenerationSettings,
    PromptImageStore,
    TransformersTrajectoryRunner,
    _code_provenance,
    _content_hash_change_count,
    _environment_manifest,
    _load_environment,
    _load_runtime,
    _public_rollout,
    _reload_adapter_and_generate,
    _save_adapter,
    _save_tensor_group,
    _select_prompts,
    _snapshot_trainable_content_hashes,
    _snapshot_visual_versions,
    _validate_pool,
    _verify_adapter_checkpoint,
    _visual_change_count,
    read_jsonl,
    sha256_file,
    write_json,
    write_jsonl,
)

from .config import load_reward_v2_config
from .coverage_cache import read_jsonl as read_cache_jsonl, rows_by_prompt
from .hierarchical_grounded_search_v2 import HierarchicalGroundedSearchRewardV2
from .reward_breakdown import aggregate_metrics, group_summary
from .training_integration import update_records_v2


def _git_commit(project_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=project_root, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable_not_git_worktree"


def _reward_v2_code_provenance(project_root: Path, output_dir: Path) -> dict:
    rewards_root = (
        project_root / "src/multimodal_web_agent/training/grpo/rewards"
    )
    files = list(rewards_root.glob("*.py"))
    files.extend(
        project_root / path
        for path in (
            "configs/grpo/reward_v2_hierarchical_grounded_search.yaml",
            "scripts/build_reward_v2_coverage_cache.py",
            "scripts/replay_reward_v2_offline.py",
            "scripts/run_reward_v2_contract_smoke.py",
            "scripts/run_grpo_reward_v2_smoke.py",
            "scripts/run_grpo_reward_v2_smoke_server.sh",
            "scripts/run_reward_v2_text_exploration_contract.py",
            "scripts/run_grpo_reward_v2_text_exploration_smoke.py",
            "scripts/run_grpo_reward_v2_text_exploration_smoke_server.sh",
        )
    )
    lines = [
        f"{sha256_file(path)}  {path.relative_to(project_root).as_posix()}"
        for path in sorted(files)
    ]
    payload = "\n".join(lines) + "\n"
    (output_dir / "reward_v2_code_files.sha256").write_text(
        payload, encoding="utf-8"
    )
    return {
        "reward_v2_code_file_count": len(lines),
        "reward_v2_code_manifest_sha256": hashlib.sha256(
            payload.encode("utf-8")
        ).hexdigest(),
    }


def run_reward_v2_smoke(*, project_root: Path, output_dir: Path, config_path: Path, run_seed: int = 20260730) -> dict:
    started = time.monotonic()
    config = load_reward_v2_config(config_path)
    _validate_pool(project_root)
    train_path = project_root / config.paths.prompt_pool
    train_rows = read_jsonl(train_path)
    prompts = _select_prompts(train_rows, search_free=12, search_required=20)
    if len(prompts) != 32:
        raise RuntimeError("Reward v2 smoke requires exactly 32 prompts")
    coverage_path = project_root / config.paths.coverage_cache
    baseline_path = project_root / config.paths.question_baseline_cache
    manager = HierarchicalGroundedSearchRewardV2(
        config,
        coverage_cache=rows_by_prompt(read_cache_jsonl(coverage_path)),
        question_baseline_cache=rows_by_prompt(read_cache_jsonl(baseline_path)),
    )
    model, processor, runtime = _load_runtime(project_root)
    environment = _load_environment(project_root)
    runtime["environment"] = _environment_manifest(project_root, environment)
    runner = TransformersTrajectoryRunner(
        model=model, processor=processor, environment=environment,
        image_store=PromptImageStore(project_root / "data/raw/fvqa/fvqa_train.parquet"),
        base_model_hash=runtime["base_model_hash"], adapter_hash=runtime["adapter_hash"],
    )
    runtime["processor_hash"] = runner.processor_hash
    runtime["chat_template_hash"] = runner.chat_template_hash
    optimizer = build_paged_adamw_8bit(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        learning_rate=5e-7, weight_decay=0.0,
    )
    before = _snapshot_trainable_content_hashes(model)
    visual_before = _snapshot_visual_versions(model)
    tokenizer = getattr(processor, "tokenizer", processor)
    public_rows, breakdowns, group_rows, update_rows = [], [], [], []
    buffer_records, buffer_advantages = [], []
    optimizer_step = 0
    for group_index, prompt in enumerate(prompts):
        model.eval()
        if hasattr(model, "gradient_checkpointing_disable"):
            model.gradient_checkpointing_disable()
        if hasattr(model, "config"):
            model.config.use_cache = True
        records = runner.rollout_group(
            prompt, run_seed=run_seed, pool_pass=1, group_size=4,
            settings=GenerationSettings(do_sample=True),
        )
        scored = manager.score_group(records, prompt.to_dict(), tokenizer=tokenizer)
        for record, row, token_values in zip(records, scored.breakdowns, scored.token_advantages):
            if token_values is None:
                raise RuntimeError("Reward v2 smoke did not produce token advantages")
            record.reward_total = float(row["terminal_reward"])
            record.answer_em = int(row["em_v1"])
            record.answer_f1 = float(row["token_f1_v1"])
            record.reward_components = dict(row)
            buffer_advantages.append(token_values)
            public_rows.append(_public_rollout(record))
            breakdowns.append(row)
        group_rows.append(scored.group_metrics)
        _save_tensor_group(output_dir, group_index, records)
        runner.release_visual(prompt.prompt_uid)
        buffer_records.extend(records)
        if len(buffer_records) == 16:
            update = update_records_v2(
                model=model, trajectory_runner=runner, optimizer=optimizer,
                records=buffer_records, token_advantages=buffer_advantages,
                clip_ratio=0.2, verify_alignment=True,
            )
            optimizer_step += 1
            update["optimizer_step"] = optimizer_step
            update["prompt_count_processed"] = group_index + 1
            update_rows.append(update)
            print(f"[reward_v2_smoke] optimizer_step={optimizer_step}/8 prompts={group_index + 1}/32", flush=True)
            buffer_records, buffer_advantages = [], []
    if optimizer_step != 8:
        raise RuntimeError("Reward v2 smoke did not execute eight optimizer steps")
    terminal_values = [float(row["terminal_reward"]) for row in breakdowns]
    if len(set(round(value, 12) for value in terminal_values)) < 2:
        raise RuntimeError("Reward v2 smoke terminal reward is degenerate")
    if not all(math.isfinite(value) for value in terminal_values):
        raise RuntimeError("Reward v2 smoke terminal reward is non-finite")
    nonzero_local = any(
        abs(float(action.get("local_advantage", 0.0))) > 0
        for row in breakdowns for action in row.get("search_actions", [])
    )
    if not nonzero_local:
        raise RuntimeError("Reward v2 smoke did not exercise a real nonzero local advantage")
    if not all(row["max_logprob_alignment_error"] < 1e-3 for row in update_rows):
        raise RuntimeError("Reward v2 smoke logprob alignment failed")
    after = _snapshot_trainable_content_hashes(model)
    lora_change_count = _content_hash_change_count(before, after)
    if lora_change_count <= 0:
        raise RuntimeError("Reward v2 smoke LoRA parameters did not change")
    visual_change_count = _visual_change_count(visual_before, model)
    if visual_change_count != 0:
        raise RuntimeError("Reward v2 smoke visual/projector parameters changed")
    _save_adapter(model, output_dir / "checkpoint")
    _verify_adapter_checkpoint(output_dir / "checkpoint")
    reload_validation = _reload_adapter_and_generate(
        model=model, runner=runner, checkpoint=output_dir / "checkpoint",
        prompt=prompts[0], run_seed=run_seed,
    )
    if not all(bool(reload_validation[key]) for key in ("checkpoint_reload_success", "reload_generation_success")):
        raise RuntimeError("Reward v2 smoke checkpoint save/reload failed")
    groups = group_summary(breakdowns)
    aggregate = aggregate_metrics(breakdowns, groups)
    write_jsonl(output_dir / "rollout_records.jsonl", public_rows)
    write_jsonl(output_dir / "trajectory_reward_breakdown.jsonl", breakdowns)
    write_jsonl(output_dir / "group_reward_summary.jsonl", group_rows)
    write_jsonl(output_dir / "update_metrics.jsonl", update_rows)
    write_json(output_dir / "reward_metrics.json", aggregate)
    config_bytes = config_path.read_bytes()
    reward_v2_provenance = _reward_v2_code_provenance(project_root, output_dir)
    manifest = {
        "schema_version": "hierarchical-grounded-search-reward-v2-smoke-v1",
        "reward_version": "hierarchical_grounded_search_v2",
        "git_commit": _git_commit(project_root),
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "sft_adapter_path": config.paths.sft_adapter,
        "sft_adapter_sha256": runtime["adapter_hash"],
        "environment_manifest_path": config.paths.environment_manifest,
        "environment_manifest_sha256": sha256_file(project_root / config.paths.environment_manifest),
        "text_corpus_sha256": json.loads((project_root / config.paths.coverage_manifest).read_text(encoding="utf-8"))["text_corpus_sha256"],
        "coverage_cache_sha256": sha256_file(coverage_path),
        "question_baseline_cache_sha256": sha256_file(baseline_path),
        "prompt_pool_sha256": sha256_file(train_path),
        "model_inference_rerun": True,
        "training_performed": True,
        "full_training_performed": False,
        "unified_frozen_test_accessed": False,
        "prompt_count": 32,
        "group_size": 4,
        "rollout_count": 128,
        "optimizer_steps": 8,
        "reward_nondegenerate": True,
        "terminal_advantage_finite": True,
        "local_advantage_real_trigger": True,
        "query_token_alignment_checked_when_generated": True,
        "image_token_alignment_checked_when_generated": True,
        "loss_and_gradient_finite": all(row["gradient_finite"] and math.isfinite(row["loss"]) for row in update_rows),
        "lora_parameter_change_count": lora_change_count,
        "visual_and_projector_change_count": visual_change_count,
        "input_identity_failure_count": sum(row["input_identity_failure_count"] for row in update_rows),
        "max_logprob_alignment_error": max(row["max_logprob_alignment_error"] for row in update_rows),
        "checkpoint_saved_and_reloaded": True,
        "behavior_gates_excluded": ["text_search_generation", "em_gain", "f1_gain", "tool_call_reduction"],
        "elapsed_seconds": time.monotonic() - started,
        "marker": "HIERARCHICAL_GROUNDED_SEARCH_REWARD_V2_SMOKE_READY",
        **reload_validation,
        **runtime,
        **reward_v2_provenance,
        **_code_provenance(project_root, output_dir),
    }
    write_json(output_dir / "run_manifest.json", manifest)
    print("HIERARCHICAL_GROUNDED_SEARCH_REWARD_V2_SMOKE_READY", flush=True)
    return manifest
