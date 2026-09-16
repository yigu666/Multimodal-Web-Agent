from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics
import time
from typing import Any, Mapping, Sequence

import torch
import yaml

from multimodal_web_agent.training.grpo.optimizer import build_paged_adamw_8bit
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
from .exploration_runner import TextSearchExplorationTrajectoryRunner
from .hierarchical_grounded_search_v2 import HierarchicalGroundedSearchRewardV2
from .reward_breakdown import aggregate_metrics, group_summary
from .smoke_runner import _git_commit, _reward_v2_code_provenance
from .text_search_exploration import EXPLORATION_METHOD, EXPLORATION_VERSION
from .training_integration import update_records_v2


SMOKE_MARKER = "HIERARCHICAL_REWARD_V2_TEXT_EXPLORATION_SMOKE_READY"
QUERY_HEAD_MARKER = "HIERARCHICAL_REWARD_V2_TEXT_QUERY_HEAD_ACTIVE"


def _finite(values: Sequence[float]) -> bool:
    return all(math.isfinite(float(value)) for value in values)


def _numeric_leaves(value: Any):
    if isinstance(value, bool) or value is None:
        return
    if isinstance(value, (int, float)):
        yield float(value)
        return
    if isinstance(value, Mapping):
        for item in value.values():
            yield from _numeric_leaves(item)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            yield from _numeric_leaves(item)


def _snapshot_projector_versions(model: Any) -> dict[str, int]:
    return {
        name: int(getattr(parameter, "_version", 0))
        for name, parameter in model.named_parameters()
        if any(
            token in name.casefold()
            for token in ("projector", "projection", "merger")
        )
    }


def _parameter_version_change_count(
    before: Mapping[str, int], model: Any
) -> int:
    current = dict(model.named_parameters())
    return sum(
        name not in current
        or int(getattr(current[name], "_version", 0)) != int(version)
        for name, version in before.items()
    )


def _failure_classification(gates: Mapping[str, bool]) -> str:
    if not gates.get("text_search_attempt_count", False) or not gates.get(
        "text_search_prompt_group_count", False
    ) or not gates.get("exploration_text_search_count", False):
        return "EXPLORATION_DID_NOT_PRODUCE_TEXT_SEARCH"
    if not gates.get("nonzero_text_query_advantage_count", False):
        return "TEXT_SEARCH_PRODUCED_BUT_QUERY_ADVANTAGE_ZERO"
    if not gates.get("behavior_logprob_contract", False):
        return "BEHAVIOR_LOGPROB_CONTRACT_FAILED"
    if not gates.get("text_token_advantage_alignment", False):
        return "ACTION_SPAN_ALIGNMENT_FAILED"
    if not gates.get("finite_training_signal", False):
        return "NONFINITE_TRAINING_SIGNAL"
    if not gates.get("lora_updated", False):
        return "MODEL_UPDATE_FAILED"
    return "OTHER"


def _source_hashes(
    project_root: Path, config_path: Path, config: Any
) -> dict[str, str]:
    paths = [
        config_path,
        project_root / config.paths.prompt_pool,
        project_root / config.paths.coverage_cache,
        project_root / config.paths.coverage_manifest,
        project_root / config.paths.question_baseline_cache,
        project_root / config.paths.question_baseline_manifest,
        project_root / config.paths.environment_manifest,
    ]
    return {
        path.relative_to(project_root).as_posix(): sha256_file(path)
        for path in paths if path.is_file()
    }


def _write_common_outputs(
    *, output_dir: Path, config_path: Path, public_rows: Sequence[Mapping[str, Any]],
    breakdowns: Sequence[Mapping[str, Any]], exploration_rows: Sequence[Mapping[str, Any]],
    text_rows: Sequence[Mapping[str, Any]], update_rows: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any], behavior_audit: Mapping[str, Any],
    token_audit: Mapping[str, Any], identity_audit: Mapping[str, Any],
    frozen_audit: Mapping[str, Any], reload_audit: Mapping[str, Any],
    source_hashes: Mapping[str, str], manifest: Mapping[str, Any],
) -> None:
    (output_dir / "config_resolved.yaml").write_text(
        config_path.read_text(encoding="utf-8"), encoding="utf-8"
    )
    write_json(output_dir / "source_hashes.json", dict(source_hashes))
    write_jsonl(output_dir / "rollouts.jsonl", public_rows)
    write_jsonl(output_dir / "reward_breakdown.jsonl", breakdowns)
    write_jsonl(output_dir / "exploration_trace.jsonl", exploration_rows)
    write_jsonl(output_dir / "text_query_advantage_trace.jsonl", text_rows)
    write_jsonl(output_dir / "training_metrics.jsonl", update_rows)
    write_json(output_dir / "smoke_summary.json", dict(summary))
    write_json(output_dir / "behavior_logprob_audit.json", dict(behavior_audit))
    write_json(output_dir / "token_advantage_audit.json", dict(token_audit))
    write_json(output_dir / "input_identity_audit.json", dict(identity_audit))
    write_json(output_dir / "frozen_parameter_audit.json", dict(frozen_audit))
    write_json(output_dir / "checkpoint_reload_audit.json", dict(reload_audit))
    write_json(output_dir / "run_manifest.json", dict(manifest))
    table = [
        "# Reward v2 Text Exploration Smoke 128", "",
        f"- Status: `{summary['status']}`",
        f"- Failure classification: `{summary.get('failure_classification')}`",
        f"- Text Search attempts: {summary['text_search_attempt_count']}",
        f"- Nonzero Query Advantage: {summary['nonzero_text_query_advantage_count']}",
        f"- Behavior log-prob max error: {summary['exploration_behavior_logprob_error_max']}",
        "", "## Hard gates", "",
    ]
    table.extend(
        f"- {key}: `{value}`" for key, value in summary["hard_gates"].items()
    )
    (output_dir / "smoke_summary.md").write_text(
        "\n".join(table) + "\n", encoding="utf-8"
    )
    (output_dir / "report.md").write_text(
        "\n".join(table + [
            "", "Full training was not performed. Unified Frozen Test was not accessed.",
        ]) + "\n",
        encoding="utf-8",
    )
    files = sorted(
        path for path in output_dir.iterdir()
        if path.is_file() and path.name != "files.sha256"
    )
    (output_dir / "files.sha256").write_text(
        "\n".join(
            f"{sha256_file(path)}  {path.name}" for path in files
        ) + "\n",
        encoding="utf-8",
    )


def run_text_exploration_smoke(
    *, project_root: Path, output_dir: Path, config_path: Path,
    run_seed: int = 20260730,
    attempt_label: str = "initial",
    supersedes_failed_dir: str | None = None,
) -> dict[str, Any]:
    started = time.monotonic()
    config = load_reward_v2_config(config_path)
    _validate_pool(project_root)
    train_path = project_root / config.paths.prompt_pool
    train_rows = read_jsonl(train_path)
    prompts = _select_prompts(train_rows, search_free=12, search_required=20)
    if len(prompts) != 32:
        raise RuntimeError("Text exploration smoke requires exactly 32 prompts")
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
    image_store = PromptImageStore(
        project_root / "data/raw/fvqa/fvqa_train.parquet"
    )
    runner = TextSearchExplorationTrajectoryRunner(
        model=model, processor=processor, environment=environment,
        image_store=image_store,
        base_model_hash=runtime["base_model_hash"],
        adapter_hash=runtime["adapter_hash"],
        exploration_config=config.text_search_exploration,
        global_seed=run_seed,
    )
    runtime["processor_hash"] = runner.processor_hash
    runtime["chat_template_hash"] = runner.chat_template_hash
    optimizer = build_paged_adamw_8bit(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        learning_rate=5e-7, weight_decay=0.0,
    )
    before = _snapshot_trainable_content_hashes(model)
    visual_before = _snapshot_visual_versions(model)
    projector_before = _snapshot_projector_versions(model)
    tokenizer = getattr(processor, "tokenizer", processor)
    public_rows: list[dict[str, Any]] = []
    breakdowns: list[dict[str, Any]] = []
    group_rows: list[dict[str, Any]] = []
    update_rows: list[dict[str, Any]] = []
    exploration_rows: list[dict[str, Any]] = []
    text_rows: list[dict[str, Any]] = []
    all_records = []
    all_token_advantages_finite = True
    buffer_records, buffer_advantages = [], []
    optimizer_step = 0
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
        for record, row, token_values in zip(
            records, scored.breakdowns, scored.token_advantages
        ):
            if token_values is None:
                raise RuntimeError("Text exploration smoke has no token advantages")
            all_token_advantages_finite = (
                all_token_advantages_finite
                and bool(torch.isfinite(torch.as_tensor(token_values)).all())
            )
            search_by_turn = {
                int(action["turn"]): action
                for action in row.get("search_actions", [])
            }
            for trace in record.exploration_trace:
                action = search_by_turn.get(int(trace["turn"]))
                if action is not None and action.get("tool") == "text_search":
                    trace.update({
                        "coverage_mask": action["coverage_mask"],
                        "actual_rank_utility": action["actual_rank_utility"],
                        "question_baseline_rank_utility": action[
                            "question_baseline_rank_utility"
                        ],
                        "query_improvement": action["query_improvement"],
                        "text_query_advantage": action["text_query_advantage"],
                        "terminal_advantage": row["terminal_advantage"],
                        "combined_text_token_advantage": action[
                            "combined_text_token_advantage"
                        ],
                    })
                trace["rollout_uid"] = record.rollout_uid
            for action in row.get("search_actions", []):
                if action.get("tool") != "text_search":
                    continue
                text_rows.append({
                    "prompt_id": row["prompt_id"],
                    "rollout_uid": record.rollout_uid,
                    "exploration_selected": bool(
                        record.exploration_metadata.get("exploration_selected")
                    ),
                    **dict(action),
                    "terminal_advantage": row["terminal_advantage"],
                })
            record.reward_total = float(row["terminal_reward"])
            record.answer_em = int(row["em_v1"])
            record.answer_f1 = float(row["token_f1_v1"])
            record.reward_components = dict(row)
            buffer_advantages.append(token_values)
            public_rows.append(_public_rollout(record))
            breakdowns.append(dict(row))
        group_rows.append(scored.group_metrics)
        _save_tensor_group(output_dir, group_index, records)
        runner.release_visual(prompt.prompt_uid)
        all_records.extend(records)
        buffer_records.extend(records)
        if len(buffer_records) == 16:
            update = update_records_v2(
                model=model, trajectory_runner=runner, optimizer=optimizer,
                records=buffer_records, token_advantages=buffer_advantages,
                clip_ratio=0.2, verify_alignment=True,
            )
            optimizer_step += 1
            update.update({
                "optimizer_step": optimizer_step,
                "prompt_count_processed": group_index + 1,
            })
            update_rows.append(update)
            print(
                f"[text_exploration_smoke] optimizer_step={optimizer_step}/8 "
                f"prompts={group_index + 1}/32", flush=True,
            )
            buffer_records, buffer_advantages = [], []
    if optimizer_step != 8:
        raise RuntimeError("Text exploration smoke did not run eight updates")

    after = _snapshot_trainable_content_hashes(model)
    lora_change_count = _content_hash_change_count(before, after)
    visual_change_count = _visual_change_count(visual_before, model)
    projector_change_count = _parameter_version_change_count(
        projector_before, model
    )
    # Build this after optimizer updates so every decision includes π_new and
    # the audited PPO ratio π_new / μ_old written by update_records_v2.
    exploration_rows = [
        dict(trace)
        for record in all_records
        for trace in record.exploration_trace
    ]
    text_actions = [
        action for row in breakdowns for action in row.get("search_actions", [])
        if action.get("tool") == "text_search"
    ]
    text_prompt_groups = {
        row["prompt_id"] for row in breakdowns
        if any(action.get("tool") == "text_search" for action in row.get("search_actions", []))
    }
    positive = sum(float(action["text_query_advantage"]) > 0 for action in text_actions)
    negative = sum(float(action["text_query_advantage"]) < 0 for action in text_actions)
    zero = sum(float(action["text_query_advantage"]) == 0 for action in text_actions)
    nonzero = positive + negative
    selected_records = [
        record for record in all_records
        if record.exploration_metadata.get("exploration_selected")
    ]
    exploration_text = sum(record.used_text_search for record in selected_records)
    non_exploration_text = sum(
        record.used_text_search for record in all_records
        if not record.exploration_metadata.get("exploration_selected")
    )
    boundary_keys = {
        (row["rollout_uid"], int(row["turn"])) for row in exploration_rows
    }
    behavior_errors = [
        float(row.get("behavior_logprob_abs_error", 0.0))
        for row in exploration_rows if row.get("exploration_applied")
    ]
    behavior_errors.extend(
        float(row["exploration_behavior_logprob_error_max"])
        for row in update_rows
    )
    behavior_errors.extend(
        max(
            float(record.exploration_metadata.get(
                "behavior_logprob_error_max", 0.0
            )),
            float(record.exploration_metadata.get(
                "base_replay_error_max", 0.0
            )),
            float(record.exploration_metadata.get(
                "unmodified_token_generation_error_max", 0.0
            )),
        )
        for record in selected_records
    )
    behavior_error_max = max(behavior_errors, default=0.0)
    behavior_error_mean = statistics.mean(behavior_errors) if behavior_errors else 0.0
    text_alignment_errors = [
        abs(
            float(action["token_advantage"])
            - (
                0.35 * float(row["terminal_advantage"])
                + 0.65 * float(action["text_query_advantage"])
            )
        )
        for row in breakdowns for action in row.get("search_actions", [])
        if action.get("tool") == "text_search"
    ]
    image_alignment_errors = [
        abs(float(action["token_advantage"]) - float(row["terminal_advantage"]))
        for row in breakdowns for action in row.get("search_actions", [])
        if action.get("tool") == "image_search"
    ]
    numeric_values = list(_numeric_leaves(breakdowns))
    numeric_values.extend(_numeric_leaves(update_rows))
    numeric_values.extend(_numeric_leaves(exploration_rows))
    policy_logprobs_finite = all(
        bool(torch.isfinite(torch.as_tensor(values)).all())
        for record in all_records
        for values in (
            record.old_log_probs,
            record.base_policy_log_probs,
            record.behavior_policy_log_probs,
        )
    )
    finite_signal = (
        _finite(numeric_values)
        and all_token_advantages_finite
        and policy_logprobs_finite
        and all(bool(row["gradient_finite"]) for row in update_rows)
    )
    gates = {
        "text_search_attempt_count": len(text_actions) >= 4,
        "text_search_prompt_group_count": len(text_prompt_groups) >= 2,
        "nonzero_text_query_advantage_count": nonzero >= 2,
        "positive_or_negative_query_advantage": positive > 0 or negative > 0,
        "exploration_selected_count": len(selected_records) > 0,
        "exploration_text_search_count": exploration_text > 0,
        "behavior_logprob_contract": behavior_error_max < 1e-3,
        "finite_training_signal": finite_signal,
        "text_token_advantage_alignment": max(text_alignment_errors, default=0.0) < 1e-8,
        "image_terminal_only_alignment": max(image_alignment_errors, default=0.0) < 1e-8,
        "environment_information_masked": all(
            int(row["information_token_train_mask_sum"]) == 0 for row in update_rows
        ),
        "input_identity": all(
            int(row[key]) == 0
            for row in update_rows
            for key in (
                "input_identity_failure_count",
                "response_token_mismatch_count",
                "processor_hash_mismatch_count",
                "target_truncation_count",
            )
        ),
        "visual_frozen": visual_change_count == 0,
        "projector_frozen": projector_change_count == 0,
        "lora_updated": lora_change_count > 0,
    }
    passed_before_reload = all(gates.values())
    failure = None if passed_before_reload else _failure_classification(gates)
    aggregate = aggregate_metrics(breakdowns, group_summary(breakdowns))
    summary = {
        "status": "pre_reload_pass" if passed_before_reload else "failed",
        "failure_classification": failure,
        "prompt_count": 32,
        "group_size": 4,
        "rollout_count": 128,
        "optimizer_steps": optimizer_step,
        "exploration_selected_count": len(selected_records),
        "exploration_action_boundary_count": len(boundary_keys),
        "exploration_text_search_count": exploration_text,
        "non_exploration_text_search_count": non_exploration_text,
        "text_search_attempt_count": len(text_actions),
        "successful_text_search_count": sum(
            bool(action.get("executed")) and not bool(action.get("tool_execution_failure"))
            for action in text_actions
        ),
        "coverage_masked_text_search_count": sum(
            float(action["coverage_mask"]) == 0 for action in text_actions
        ),
        "unmasked_text_search_count": sum(
            float(action["coverage_mask"]) > 0 for action in text_actions
        ),
        "positive_text_query_advantage_count": positive,
        "negative_text_query_advantage_count": negative,
        "zero_text_query_advantage_count": zero,
        "nonzero_text_query_advantage_count": nonzero,
        "mean_text_query_advantage": statistics.mean(
            [float(action["text_query_advantage"]) for action in text_actions]
        ) if text_actions else 0.0,
        "min_text_query_advantage": min(
            [float(action["text_query_advantage"]) for action in text_actions],
            default=0.0,
        ),
        "max_text_query_advantage": max(
            [float(action["text_query_advantage"]) for action in text_actions],
            default=0.0,
        ),
        "exploration_behavior_logprob_error_max": behavior_error_max,
        "exploration_behavior_logprob_error_mean": behavior_error_mean,
        "image_local_head_used_in_loss": False,
        "lora_parameter_change_count": lora_change_count,
        "visual_and_projector_change_count": (
            visual_change_count + projector_change_count
        ),
        "projector_parameter_change_count": projector_change_count,
        "hard_gates": gates,
        "reward_metrics": aggregate,
    }
    behavior_audit = {
        "contract_passed": gates["behavior_logprob_contract"],
        "max_abs_error": behavior_error_max,
        "mean_abs_error": behavior_error_mean,
        "ratio_denominator": "behavior_policy_logprob_mu_old",
        "unmodified_tokens_use_base_policy": True,
        "query_content_biased_token_count": 0,
    }
    token_audit = {
        "text_token_alignment_error_max": max(text_alignment_errors, default=0.0),
        "image_token_alignment_error_max": max(image_alignment_errors, default=0.0),
        "image_local_head_used_in_loss": False,
        "environment_information_token_advantage": 0.0,
    }
    identity_audit = {
        "passed": gates["input_identity"] and gates["environment_information_masked"],
        "input_identity_failure_count": sum(
            int(row["input_identity_failure_count"]) for row in update_rows
        ),
        "information_token_train_mask_sum": sum(
            int(row["information_token_train_mask_sum"]) for row in update_rows
        ),
    }
    frozen_audit = {
        "lora_parameter_change_count": lora_change_count,
        "visual_and_projector_change_count": (
            visual_change_count + projector_change_count
        ),
        "projector_parameter_count": len(projector_before),
        "projector_parameter_change_count": projector_change_count,
        "visual_frozen": gates["visual_frozen"],
        "projector_frozen": gates["projector_frozen"],
    }
    reload_audit: dict[str, Any] = {
        "checkpoint_saved": False, "checkpoint_reload_success": False,
        "reload_generation_success": False,
    }
    source_hashes = _source_hashes(project_root, config_path, config)
    manifest = {
        "schema_version": "reward-v2-text-exploration-smoke-128-v1",
        "reward_version": "hierarchical_grounded_search_v2",
        "exploration_version": EXPLORATION_VERSION,
        "exploration_method": EXPLORATION_METHOD,
        "exploration_train_only": True,
        "git_commit": _git_commit(project_root),
        "config_sha256": sha256_file(config_path),
        "sft_adapter_path": config.paths.sft_adapter,
        "sft_adapter_sha256": runtime["adapter_hash"],
        "environment_manifest_path": config.paths.environment_manifest,
        "environment_manifest_sha256": sha256_file(
            project_root / config.paths.environment_manifest
        ),
        "text_corpus_sha256": json.loads((
            project_root / config.paths.coverage_manifest
        ).read_text(encoding="utf-8"))["text_corpus_sha256"],
        "coverage_cache_sha256": sha256_file(coverage_path),
        "question_baseline_cache_sha256": sha256_file(baseline_path),
        "prompt_pool_sha256": sha256_file(train_path),
        "training_performed": True,
        "full_training_performed": False,
        "unified_frozen_test_accessed": False,
        "smoke_attempt_label": str(attempt_label),
        "supersedes_failed_dir": supersedes_failed_dir,
        "automatic_retry_performed": False,
        "prompt_count": 32, "group_size": 4, "rollout_count": 128,
        "optimizer_steps": optimizer_step,
        "status": "pre_reload_pass" if passed_before_reload else "failed",
        "failure_classification": failure,
        "image_local_head_used_in_loss": False,
        "elapsed_seconds": time.monotonic() - started,
        **runtime,
    }
    manifest.update({
        **_reward_v2_code_provenance(project_root, output_dir),
        **_code_provenance(project_root, output_dir),
    })
    if not passed_before_reload:
        _write_common_outputs(
            output_dir=output_dir, config_path=config_path,
            public_rows=public_rows, breakdowns=breakdowns,
            exploration_rows=exploration_rows, text_rows=text_rows,
            update_rows=update_rows, summary=summary,
            behavior_audit=behavior_audit, token_audit=token_audit,
            identity_audit=identity_audit, frozen_audit=frozen_audit,
            reload_audit=reload_audit, source_hashes=source_hashes,
            manifest=manifest,
        )
        raise RuntimeError(failure or "OTHER")

    try:
        _save_adapter(model, output_dir / "checkpoint")
        _verify_adapter_checkpoint(output_dir / "checkpoint")
        reload_runner = TransformersTrajectoryRunner(
            model=model, processor=processor, environment=environment,
            image_store=image_store,
            base_model_hash=runtime["base_model_hash"],
            adapter_hash=runtime["adapter_hash"],
        )
        reload_audit = {
            "checkpoint_saved": True,
            **_reload_adapter_and_generate(
                model=model, runner=reload_runner,
                checkpoint=output_dir / "checkpoint", prompt=prompts[0],
                run_seed=run_seed,
            ),
        }
    except Exception as exc:
        gates["checkpoint_save_reload"] = False
        reload_audit = {
            "checkpoint_saved": (output_dir / "checkpoint").is_dir(),
            "checkpoint_reload_success": False,
            "reload_generation_success": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        summary.update({
            "status": "failed",
            "failure_classification": "MODEL_UPDATE_FAILED",
            "hard_gates": gates,
        })
        manifest.update({
            "status": "failed",
            "failure_classification": "MODEL_UPDATE_FAILED",
            "elapsed_seconds": time.monotonic() - started,
        })
        _write_common_outputs(
            output_dir=output_dir, config_path=config_path,
            public_rows=public_rows, breakdowns=breakdowns,
            exploration_rows=exploration_rows, text_rows=text_rows,
            update_rows=update_rows, summary=summary,
            behavior_audit=behavior_audit, token_audit=token_audit,
            identity_audit=identity_audit, frozen_audit=frozen_audit,
            reload_audit=reload_audit, source_hashes=source_hashes,
            manifest=manifest,
        )
        raise RuntimeError("MODEL_UPDATE_FAILED") from exc
    gates["checkpoint_save_reload"] = all(
        bool(reload_audit[key]) for key in (
            "checkpoint_saved", "checkpoint_reload_success",
            "reload_generation_success",
        )
    )
    if not gates["checkpoint_save_reload"]:
        summary.update({
            "status": "failed",
            "failure_classification": "MODEL_UPDATE_FAILED",
            "hard_gates": gates,
        })
        manifest.update({
            "status": "failed",
            "failure_classification": "MODEL_UPDATE_FAILED",
            "elapsed_seconds": time.monotonic() - started,
        })
        _write_common_outputs(
            output_dir=output_dir, config_path=config_path,
            public_rows=public_rows, breakdowns=breakdowns,
            exploration_rows=exploration_rows, text_rows=text_rows,
            update_rows=update_rows, summary=summary,
            behavior_audit=behavior_audit, token_audit=token_audit,
            identity_audit=identity_audit, frozen_audit=frozen_audit,
            reload_audit=reload_audit, source_hashes=source_hashes,
            manifest=manifest,
        )
        raise RuntimeError("MODEL_UPDATE_FAILED")
    summary.update({
        "status": "passed", "failure_classification": None,
        "hard_gates": gates,
        "markers": [QUERY_HEAD_MARKER, SMOKE_MARKER],
    })
    manifest.update({
        "status": "passed", "failure_classification": None,
        "behavior_logprob_contract_passed": True,
        "marker": SMOKE_MARKER,
        "query_head_marker": QUERY_HEAD_MARKER,
        "elapsed_seconds": time.monotonic() - started,
        **reload_audit,
    })
    _write_common_outputs(
        output_dir=output_dir, config_path=config_path,
        public_rows=public_rows, breakdowns=breakdowns,
        exploration_rows=exploration_rows, text_rows=text_rows,
        update_rows=update_rows, summary=summary,
        behavior_audit=behavior_audit, token_audit=token_audit,
        identity_audit=identity_audit, frozen_audit=frozen_audit,
        reload_audit=reload_audit, source_hashes=source_hashes,
        manifest=manifest,
    )
    print(QUERY_HEAD_MARKER, flush=True)
    print(SMOKE_MARKER, flush=True)
    return manifest
